from scipy.optimize import bisect, root, minimize_scalar
from scipy import integrate
from scipy.integrate import ode, quad
from scipy.optimize import bisect, root, minimize_scalar

import cmath
import numpy as np
import math
from math import sqrt, factorial, pi

from . import functions as func

#===================================================================
# DiskParam
#===================================================================

class DiskParam:
    """
    Parameters of the underlying disk model.

    Atributes
    ---------
    cs : float
        Constant sound speed [c].

    Sigma : functions.IntervalFunction
        Column density [arb. units]. 

    rin : float
        Inner edge of the disk [M].

    rout : float
        Outer edge of the disk [M]. 
    """

    def __init__(self, cs, Sigma, rin, rout):

        self._cs = cs
        self._Sigma = Sigma
        self._rin = rin
        self._rout = rout

    @property
    def cs(self):
        """ Disk sound speed [c] """ 
        return self._cs

    @property
    def Sigma(self):
        """ Column density profile (arb. units) """
        return self._Sigma

    @property
    def rin(self):
        """ Inner edge of the disk [M] """
        return self._rin

    @property
    def rout(self):
        """ Outer edge of the disk [M] """
        return self._rout

    def check_range(self, r):
        """ raises ValueError exception if r is outside 
            the radial range of the disk """
        if (r < self._rin) or (r > self._rout):
            raise ValueError('Radial coordinate outside the disk.\n' + 
                             'r    = {}\n'.format(r) + 
                              'rmin = {}\n'.format(self._rin) +
                             'rmax = {}'.format(self._rout))


    def dlnSigma(self, r):
        """ Logaritmic derivative of the density profile [ln(Sigma)]' as function of r."""
        return self._Sigma.der(r)/self._Sigma(r)


    def frequencies(self, r):
        """
        Orbital, radial and vertical frequencies at given radius
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        Returns
        -------
        result : list
            [Omega, kappa, Omegav] in units of [M^-1]
        
        """
        raise NotImplementedError
        

    def Omega(self, r):
        """
        Orbital frequency at given radius
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        Returns
        -------
        Omega : float
        """
        raise NotImplementedError

    
    def kappa(self, r):
        """
        Radial frequency at given radius
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        Returns
        -------
        kappa : float
        """
        raise NotImplementedError
    
    
    def dkappa(self, r):
        """
        Derivative of the radial frequency
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        Returns
        -------
        dkappa : float
        """
        raise NotImplementedError
        
    @property	
    def r_kappa_max(self):
        """	Position of the maximum of the epicyclic frequency [M] """
        raise NotImplementedError

        
    def Omegav(self, r):
        """
        Vertical frequency at given radius
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        Returns
        -------
        Omegav : float
        """
        raise NotImplementedError
    

    def dOmegav(self, r):
        """
        Derivative of the vertical frequency

        Parameter
        ---------
        r : float
            radial coordinate [M]

        Returns
        -------
        dOmegav : float
        """
        raise NotImplementedError



#===================================================================
# ModeParam
#===================================================================

        
class ModeParam:
    """
    Parameters of the mode and related methods.

    Atributes
    ---------
    disk : DiskParam
        Underlying disk model.

    omega : complex
        Oscillation frequency.

    m : int
        Azimuthal wavenumber.

    n : positive int
        Vertical quantum number 
    """
        
    def __init__(self, disk, omega, m, n):
        
        self._disk = disk
        self._omega = omega
        self._m = m
        self._n = n
        self._cr = None
        self._lr = [None, None]
        self._vr = [None, None]
        self._calculate_radii()

    @property
    def disk(self):
        """ Underlying disk model: DiskParam """
        return self._disk

    @property
    def omega(self):
        """ Oscillation frequency: complex """
        return self._omega
    
    @property
    def m(self):
        """ Azimuthal wavenumber: int """
        return self._m

    @property
    def n(self):
        """ Vertical quantum number: positive int """
        return self._n

    @property
    def LRs(self):
        """	Positions of the Lindblad resonances [rILR, rOLR]: [float, float] """
        return self._lr

    @property
    def CR(self):
        """	Position of the corotation resonance: float """
        return self._cr
        
    @property
    def VRs(self):
        """ Position of the vertical resonance [rIVR, rOVR]: [float, float] """
        return self._vr


    def __repr__(self):
        """	String representation """
        return 'ModeParam(omega={}, m={}, n={})'.format(self.omega, self.m, self.n)


    def omegat(self, r, useim=True):
        """
        Osillation frequency measured in the corotation system.
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        useim : bool
            Set False for calculation based just on real part of omega.

        Returns
        -------
        omega : complex
            frequency in units of [M^-1]

        """
        self._disk.check_range(r)
        if useim:
            return self._omega - self._m*self._disk.Omega(r)
        else:
            return self._omega.real - self._m*self._disk.Omega(r)

    
    def D(self, r, useim=True):
        """
        Determinant of the horizontal-velocity block of 
        the linear equations.
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        useim : bool
            Set False for calculation based just on real part of omega.
        
        Returns
        -------
        D : float

        """
        self._disk.check_range(r)
        return self._disk.kappa(r)**2 - self.omegat(r, useim)**2
        
        
    def k2(self, r, useim=True):
        """
        Squared radial wavevector using WKBJ approximation
        
        Parameter
        ---------
        r : float
            radial coordinate [M]
        
        useim : bool
            Set False for calculation based just on real part of omega.
        
        Returns
        -------
        k2 : float

        """
        self._disk.check_range(r)
        kappa = self._disk.kappa(r)
        Omegav = self._disk.Omegav(r)
        omegat = self.omegat(r, useim)
        return (omegat**2 - kappa**2)*(omegat**2 - self._n*Omegav**2)/(omegat*self._disk.cs)**2
        

    def propagation_regions(self):
        """
        List of limits describing propagation diagrams according to WKBJ approximation.
        """
        R = sorted([r for r in (self._lr + self._vr + [self._disk.rin, self._disk.rout]) if r!= None])
        result = []
        for i in range(len(R)-1):
            if self.k2(0.5*(R[i] + R[i+1])).real > 0:
                result.append((R[i], R[i+1]))
        return result

        
    def _calculate_radii(self):
        # This calculates all resonance radii and store it in the variables _cr, _lr, _vr

        disk = self._disk

        # Corotation resonance:

        f = lambda r: self.omegat(r, useim=False).real
        if f(disk.rin)*f(disk.rout) < 0:
            self._cr = bisect(f, disk.rin, disk.rout)
        
        # Lindblad resonances:
        
        self._lr = [None, None]

        f = lambda r: -self.D(r, useim=False).real
        rmax = minimize_scalar(f, bounds=(disk.rin, disk.rout), method='bounded').x
        if f(disk.rin)*f(rmax) < 0 :
            self._lr[0] = bisect(f, disk.rin, rmax)
        if f(rmax)*f(disk.rout) < 0:
            self._lr[1] = bisect(f, rmax, disk.rout)

        # Vertical resonances:
        
        self._vr = [None, None]

        f = lambda r: -(self._n*disk.Omegav(r)**2 - self.omegat(r, useim=False).real**2)
        rmax = minimize_scalar(f, bounds=(disk.rin, disk.rout), method='bounded').x
        if f(disk.rin)*f(rmax) < 0 :
            self._vr[0] = bisect(f, disk.rin, rmax)
        if f(rmax)*f(disk.rout) < 0:
            self._vr[1] = bisect(f, rmax, disk.rout)

#===================================================================
# rspace:
#===================================================================

def rspace(mode, rmin, rmax, freq=12):
    """
    Radial range sampled according to WKBJ wavelength of the mode. 

    Parameters
    ----------
    mode : ModeParam
        Parameters of the mode.
    rmin : float
        Innermost radius.
    rmax : float
        Outermost radius.
    freq : int
        Number of evaluation points per wavelength

    Returns
    -------
    np.array
        Array of evaluation points.
    """
    radii = []
    r = rmin
    while r < rmax:
        radii.append(r)
        dr = 2*pi/(freq*sqrt(abs(mode.k2(r))))
        r += dr
    radii.append(rmax)
    return np.array(radii)


def effective_potential(mode, radii):
    """
    Effective potential of the wave propagation for the given mode.

    The function returns sampled potential at given radii.

    Parameters
    ----------
    mode : ModeParam
        Parameters of the mode.

    radii : 1d numpy.array
        Radii at which the potential is evaluated.
    
    Returns
    -------
    result : functions.SampledFunction(dim=1, dtype=complex)
        Potential at given radii.

    """
    
    def V(r):
        kappa = mode.disk.kappa(r)
        Omega = mode.disk.Omega(r)
        dlnZeta = 2*mode.disk.dkappa(r)/kappa - (kappa**2/(2*r*Omega**2) - 2/r) - mode.disk.dlnSigma(r)
        return mode.D(r)/mode.disk.cs**2 + (mode.m/r)**2 - 2*mode.m*Omega/(r*mode.omegat(r))*dlnZeta
    
    Vfunc = func.IntervalFunction(lambda r: V(r), radii[0], radii[-1])
    return func.SampledFunction.sample(radii, Vfunc)

#===================================================================
# Boundary conditions
#===================================================================

class bcGeneral:
    """
    General boundary conditions

    Atributes
    ---------
    disk : DiskParam
        Underlying disk model

    r : float
        location of the boundary
    """

    def __init__(self, disk, r):
        self._r = r
        self._disk = disk

    def __repr__(self):
        return 'bcGeneral(disk = {}, r = {})'.format(self._r, self._disk)

    @property
    def r(self):
        """ Position of the boundary. """
        return self._r
    
    @property
    def disk(self):
        """ Underlying disk model. """
        return self._disk


    def Y0(self, h, mode):
        """
        Applying the homogeneous boundary conditions. 
        
        Parameters
        ----------
        h : complex
            The enthalpy perturbation value.

        mode : ModeParam
            Properties of the mode to which the boudary is applied
        
        Returns
        -------
        Y : numpy.array
            The 2d solution vector for given value of h
        """
        raise NotImplementedError('Y0() is not implemented in the base class.')


    def YF(self, mode, F):
        """
        Inhomogeneous boundary conditions for the particular solution used 
        in function find_response().
        
        Parameters
        ----------
        h : complex
            The enthalpy perturbation value.

        mode : ModeParam
            Properties of the mode to which the boudary is applied
        
        Returns
        -------
        YF : numpy.array
            The 2d solution vector for given F
        """
        raise NotImplementedError('YF() is not implemented in the base class.')


class bcZeroVelocity(bcGeneral):
    """	
    Zero-velocity boundary condition at given radius
    
    Atributes
    ---------
    disk : DiskParam
        Underlying disk model

    r : float
        location of the boundary
    """
    pass

    def __repr__(self):
        return 'bcZeroVelocity(disk = {}, r = {})'.format(self._r, self._disk)

    def Y0(self, h, mode):
        return np.array([h, 0])
        
    def YF(self, mode, F):
        return np.array([0, 0])



class bcWarp(bcGeneral):
    """
    Initial condition for the Warp: 
        Theta = 1 & Theta'(r) = 0, 
    where Theta is a tilt angle of the disk.
    """
    def __repr__(self):
        return 'bcWarp(disk = {}, r = {})'.format(self._r, self._disk)

    def __init__(self, disk, r):
        super().__init__(disk, r)

    def Y0(self, h, mode):
        return np.array([h, -1j*h/(self.r*self.disk.Omega(self.r))])


class bcWave(bcGeneral):
    """
    Boundary conditions that mimics a behavior of infinite disk behind
    this boundary.

    Atributes
    ---------
    disk : DiskParam
        Underlying disk model.

    r : float
        Location of the boundary.

    which : str 
        either 'ingoing' or 'outgoing' for positive/negative 
        radial group-velocity of the wave is positive.

    order : non-negative int 
        Order of calculation of non-wave response.

    Notes
    -----
    One has to carefully avoid any wave reflections (that would be nonphysical). 
    This is particularly important in the case of inhomogeneous problems,
    where a non-wave response is also part of the solution. We treat 
    this case by putting the non-wave response as an boudary condition
    for the particular solution of the response (method YF).
    """

    def __init__(self, disk, r, which, order=2):
        super().__init__(disk, r)
        if which == 'outgoing':
            self._outgoing = True
        elif which == 'ingoing':
            self._outgoing = False
        else:
            raise ValueError('parameter which has to be either "positive" or "negative", but "{}" provided.'.format(dir))
        self._order = order
        
    @property
    def outgoing(self):
        """ Outgoing/ingoing wave boundary. """
        return self._outgoing

    @property
    def order(self):
        """ Order in calculations of non-wave response. """
        return self._order

    def __repr__(self):
        return 'bcWave(disk = {}, r = {}, outgoing = {}, order = {})'.format(self._disk, self._r, 
                                                                             self._outgoing, self._order)

    def Y0(self, h, mode):

        r = self._r
        omegat = mode.omegat(r)
        k2 = mode.k2(r)
        
        if k2.real < 0:
            raise ValueError('Wave boundary conditions ({}) applied in evanescent domain.\n'.format(self))

        if (omegat.real > 0):
            if self.outgoing:
                k = cmath.sqrt(k2)
            else:
                k = -cmath.sqrt(k2)
        else:
            if self.outgoing:
                k = -cmath.sqrt(k2)
            else:
                k = cmath.sqrt(k2)
        
        # squared wkb amplitude: 
        amp2 = func.IntervalFunction(lambda r: abs(mode.D(r))/(r*self._disk.Sigma(r)*sqrt(abs(mode.k2(r)))), 
                              self._disk.rin, self._disk.rout)

        # radial derivative of the enthalpy:
        dh = (1j*k + 0.5*amp2.logder(r))*h
        # velocity:
        v  = 1j*(omegat*dh - (2.*mode.m*self._disk.Omega(r)/r)*h)/mode.D(r)
        
        return np.array([h, v])


    def YF(self, mode, F):

        return nonwave_resp(self.r, mode, F, self._order)



class bcEvanescent(bcGeneral):
    """
    Spatially growing/damped perturbation with increasing radius.

    Atributes
    ---------
    disk : DiskParam
        Underlying disk model.

    r : float
        Location of the boundary.

    dir : str 
        'growing'/'decaying' for growing/decaying perturbations
        with increasing radius behind this boundary.

    order : non-negative int 
        Order of calculation of non-wave response.

    Notes
    -----
    One has to carefully avoid any wave reflections (that would be nonphysical). 
    This is particularly important in the case of inhomogeneous problems,
    where a non-wave response is also part of the solution. We treat 
    this case by putting the non-wave response as an boudary condition
    for the particular solution of the response (method YF).
    """

    def __init__(self, disk, r, which, order=2):
        super().__init__(disk, r)
        if which == 'growing':
            self._growing = True
        elif which == 'decaying':
            self._growing = False
        else:
            raise ValueError('parameter "which" has to be "growing" or "decaying", but "{}" provided.'.format(which))
        self._order = order

    @property
    def growing(self):
        """ growing/decaying perturbations behind this boundary. """
        return self._growing

    @property
    def order(self):
        """ Order in calculations of non-wave response. """
        return self._order

    def __repr__(self):
        return 'bcEvanescent(disk = {}, r = {}, outgoing = {}, order = {})'.format(self._disk, self._r, 
                                                                                   self._growing, self._order)


    def Y0(self, h, mode):
        
        r = self._r
        Omega = self._disk.Omega(r)
        omegat = mode.omegat(r)
        k2 = mode.k2(r)
        
        if k2.real > 0:
            raise ValueError('Evanscent boundary conditions applied in wave domain.')
            
        if self.growing:
            k = -cmath.sqrt(k2)
        else:
            k = cmath.sqrt(k2)
        
        # squared wkb amplitude: 
        amp2 = func.IntervalFunction(lambda r: abs(mode.D(r))/(r*self._disk.Sigma(r)*sqrt(abs(mode.k2(r)))), 
                              self._disk.rin, self._disk.rout)

        # radial derivative of the enthalpy:
        dh = (1j*k + 0.5*amp2.logder(r))*h

        # velocity:
        v  = 1j*(omegat*dh - (2.*mode.m*Omega/r)*h)/mode.D(r)
        
        return np.array([h, v])


    def YF(self, mode, F):

        return nonwave_resp(self.r, mode, F, self._order)


    
#===================================================================
# ODE 
#===================================================================
    

def ode_jac(r, Y, mode):
    """
    Jacobian of the rhs.
    
    Parameters
    ----------
    r : float
        Radial BL coordinate [M]

    Y : numpy.array
        Solution vector, Y = numpy.array([h, vr]).

    mode : ModeParam
        Parameters of the mode.
    
    Returns
    -------
    J : numpy.array    
        Jacobian numpy.array([[dh'/dh, dh'/dv], [dv'/dh, dv'/dv]])
    """
    
    if r < mode.disk.rin:
        r = mode.disk.rin

    disk = mode.disk
    [Omega, kappa, Omegav] = disk.frequencies(r)
    omegat = mode.omegat(r)

    m = mode.m
    n = mode.n
    c = disk.cs

    J00 = (2*m*Omega)/(omegat*r)
    J01 = (1j*(-kappa**2 + omegat**2))/omegat	
    J10 = -((1j*(c**2*m**2 - omegat**2*r**2 + n*Omegav**2*r**2))/(c**2*omegat*r**2))
    J11 = -(1/r) - (kappa**2*m)/(2.*Omega*omegat*r) - disk.Sigma.logder(r)
        
    return np.array([[J00, J01], [J10, J11]])

    

def ode_f(r, mode, F):
    """
    Force term on the rhs of the governing equation.
    
    Parameters
    ----------
    r : float
        Radial coordinate [M]

    mode : ModeParam
        Parameters of the mode

    F : functions.IntervalFunction
        Forcing full vector
    
    Returns
    -------
    ode_f : numpy.array shape=(2)
    
    ode_f[0]
        Force term in the enthalpy equation.
    
    ode_f[1]
        Force term in the radial-velocity equation.
    """
    
    if r < mode.disk.rin:
        r = mode.disk.rin

    [f, Fr, Fphi, Fz] = F(r) 
    Omega  = mode.disk.Omega(r)
    Omegav = mode.disk.Omegav(r)
    omegat = mode.omegat(r)
    c = mode.disk.cs
    
    F_h  = Fr + 2j*r*Omega*Fphi/omegat
    F_vr = f/c**2 + (mode.m*Fphi + 1j*Omegav*Fz/c)/omegat
    
    return np.array([F_h, F_vr])



def ode_rhs(r, Y, mode, F):
    """
    Right-hand side of the governing equation.
    
    Parameters
    ----------
    r : float
        Radial coordinate.

    Y : numpy.array shape=(2) 
        Solution vector Y = [h, h'].

    mode : ModeParam
        Parameters of the mode.
        
    Returns
    -------
    Y' : numpy.array, shape=(2) 
        Radial derivative of Y, 
        Y' = [h', h''].

    """
    if r < mode.disk.rin:
        r = mode.disk.rin

    J  = ode_jac(r, Y, mode)
    FF = ode_f(r, mode, F)

    return np.array([J[0][0]*Y[0] + J[0][1]*Y[1] + FF[0], J[1][0]*Y[0] + J[1][1]*Y[1] + FF[1]])

    

def nonwave_resp(r, mode, F, order=1):
    """
    Solves for the non-wave response of the disk to the forcing.
    Works only outside the resonance. 
    
    Parameters
    ----------
    r : float
        Radial coordinate.

    mode : ModeParam
        Parameters of the mode.

    F : functions.IntervalFunction
        Forcing term.

    order : int (positive)
        Order of recursive corrections
    
    Returns:  
    --------
    Y : numpy.array, shape = (2) 
        Solution vector Y of long-wave perturbations.

    Notes
    -----
    Finds numerically the solution vector Y such that
    the ode is satisfied. The algorithm starts with Y0 
    for which ode_rhs vanishes, 
        RHS(Y0) = 0 
    then it is recursively improved by solving
        RHS(Y1) = Y0',
        RHS(Y2) = Y1',
        ...
    This way approximations of different orders are found.	
    """
    from numpy.linalg import inv

    def Y(n, r):
        if n == 0:
            # zeroth-order approximation:
            J = np.array(ode_jac(r, [], mode))
            f = np.array(ode_f(r, mode, F))
            Jinv = inv(J)
            return -1*np.dot(Jinv, f)

        if n > 0:
            # calculate derivative of the approximation of the order (n-1) 
            # by creating the functions.IntervalFunction:
            dY = func.IntervalFunction(lambda r: Y(n-1, r), F.rmin, F.rmax).der(r)

            #dY = (Y(n-1, r+dr) - Y(n-1, r-dr))/(dr + dr)
            # multiply by the inverse of jacobian:
            J = np.array(ode_jac(r, [], mode))
            return np.dot(inv(J), dY)

        raise ValueError("The order has to be >= 0.")
    
    result = 0
    for n in range(order+1):
        result = result + Y(n, r)
    
    return result



def wkbj_amplitudes(mode, r, Y):
    """
    """
    raise NotImplementedError


#===================================================================
# Full vectors
#===================================================================


def Y2W(r, Y, mode, F): #im ana, and i want to make some mess in my boyfriend's code :)
    """
    For given vector Y = [h, vr] calculates the full 
    solution vector W = [h, vr, vphi, vz].
    
    Parameters
    ----------
    r : float
        Radial coordinate

    Y : numpy.array, shape = (2)
        Solution vector [h, r]

    mode : ModeParam
        Parameters of the mode

    F : func.IntervalFunction
        Forcing function. 
    
    Returns  [W, dW]
    -------
    W : numpy.array, shape = (4)
        Full solution [h, vr, vphi, vz]

    dW : numpy.array, shape = (4)
        Radial derivative of W
    """

    [h, vr] = Y
    Fphi, Fz = F(r)[2:]

    kappa  = mode.disk.kappa(r)
    Omega  = mode.disk.Omega(r)
    Omegav = mode.disk.Omegav(r)
    dkappa = mode.disk.dkappa(r)
    c = mode.disk.cs
    
    m, n = mode.m, mode.n
    omegat = mode.omegat(r)
        
    [h, vr] = Y
    vphi = -(-2*h*m*Omega + 1j*(-2*Fphi*Omega*r**2 + kappa**2*r*vr))/(2.*Omega*omegat*r**2) 
    vz   = 1j*(c*Fz - h*n*Omegav)/(c*omegat) 	  
    
    # solution vector:
    W = np.array([h, vr, vphi, vz])
    
    # derivatives of h and vr are calculated by calling ode_rhs:
    [dh, dvr] = ode_rhs(r, Y, mode, F)

    # derivative of the force:
    dFphi, dFz = F.der(r)[2:]
        
    # derivatives of the two other velocities are: 
    
    dvphi = (-(-2*dh*m*Omega - 2*h*m*(kappa**2/(2.*Omega*r) - (2*Omega)/r) + 
            1j*(dvr*kappa**2*r - 4*Fphi*Omega*r - 2*dFphi*Omega*r**2 - 
            2*Fphi*(kappa**2/(2.*Omega*r) - (2*Omega)/r)*r**2 + kappa**2*vr + 2*dkappa*kappa*r*vr))/
            (2.*Omega*omegat*r**2) + (-2*h*m*Omega + 1j*(-2*Fphi*Omega*r**2 + kappa**2*r*vr))/
            (Omega*omegat*r**3) - (m*(kappa**2/(2.*Omega*r) - (2*Omega)/r)*
            (-2*h*m*Omega + 1j*(-2*Fphi*Omega*r**2 + kappa**2*r*vr)))/(2.*Omega*omegat**2*r**2) + 
            ((kappa**2/(2.*Omega*r) - (2*Omega)/r)*(-2*h*m*Omega + 1j*(-2*Fphi*Omega*r**2 + kappa**2*r*vr)))/
            (2.*Omega**2*omegat*r**2))

    dvz = (0.5j*(c*(Fz*m*(kappa**2 - 4*Omega**2) + 2*dFz*Omega*omegat*r) - 
          n*Omegav*(h*m*(kappa**2 - 4*Omega**2) + 2*dh*Omega*omegat*r)))/(c*Omega*omegat**2*r)
    
    dW =np.array([dh, dvr, dvphi, dvz])
    
    return [W, dW]


def conjugate(mode, W):
    """
    The 'complex conjugated' mode (the solution with omega -> -omega, m -> -m).

    Parameters
    ----------
    mode : ModeParam
        Parameters of the input mode.

    W : functions.SampledFunction
        Eigenfunction of the input mode

    Returns   [mode, W]
    ------- 
    mode : ModeParam
        Parameters of the 'complex conjugated' mode.

    W : functions.SampledFunction
        Sampled eigenfunction of the 'complex-conjugated' mode.
    """
    result_mode = ModeParam(disk = mode.disk, omega = -np.conjugate(mode.omega), m = -mode.m, n = mode.n)
    result_W = func.SampledFunction(W.sample_r, np.conjugate(W.sample_y), np.conjugate(W.sample_dy))
    return [result_mode, result_W]


def linop(mode, W):
    """
    The linear operator L of the governing equations

    Parameters
    ----------
    mode : ModeParam
        Paramters of the mode.
    
    W : functions.SampledFunction
        Eigenfunction of the mode

    Returns
    -------
    LW : functions.SampledFunction
        The result of the operation L(W). The resulting function
        is sampled at the same point as the eigenfunction `W`.
    
    """

    def LWfunc(r):
        kappa  = mode.disk.kappa(r)
        Omega  = mode.disk.Omega(r)
        Omegav = mode.disk.Omegav(r)
        dlnSigma = mode.disk.dlnSigma(r)
        c = mode.disk.cs
        
        m, n = mode.m, mode.n
        
        [h, vr, vphi, vz] = W(r)
        dW = W.der(r)
        dh  = dW[0]
        dvr = dW[1]

        LWh   = 1j*m*Omega*h + c**2*(dvr + vr/r + dlnSigma*vr) + (1j*m*c**2)*vphi - c*Omegav*vz
        LWr   = 1j*m*Omega*vr - 2*r*Omega*vphi + dh
        LWphi = 1j*m*Omega*vphi + (kappa**2/(2*r*Omega))*vr + (1j*m/r**2)*h
        LWz   = 1j*m*Omega*vz + (n*Omegav/c)*h

        #print(LWh)
        return np.array([LWh, LWr, LWphi, LWz])

    fLW = func.IntervalFunction(lambda r: LWfunc(r), W.rmin, W.rmax)

    return func.SampledFunction.sample(W.sample_r, fLW)


def linop_adjoint(mode, U):
    """
    The adjoint operator to the operator `linop`

    Parameters
    ----------
    mode : ModeParam
        Paramters of the mode.
    
    U : functions.SampledFunction
        function from the adjoint space

    Returns
    -------
    LU : functions.SampledFunction
        The result of the operation L(W). The resulting function
        is sampled at the same point as the eigenfunction `W`.
    
    """
    def LUfunc(r):
        kappa  = mode.disk.kappa(r)
        Omega  = mode.disk.Omega(r)
        Omegav = mode.disk.Omegav(r)
        dlnSigma = mode.disk.dlnSigma(r)
        c = mode.disk.cs
        
        m, n = mode.m, mode.n
        
        [eta, ur, uphi, uz] = U(r)
        dU = U.der(r)
        deta  = dU[0]
        dur = dU[1]
        
        LUeta = 1j*m*Omega*eta - c**2*(dur + ur/r + ur*dlnSigma) + (1j*m*c**2)*uphi + c*Omegav*uz
        LUr   = 1j*m*Omega*ur + (r*kappa**2/(2*Omega))*uphi - deta
        LUphi = 1j*m*Omega*uphi - (2*Omega/r)*ur - (1j*m/r**2)*eta
        LUz   = 1j*m*Omega*uz - (n*Omegav/c)*eta

        return np.array([LUeta, LUr, LUphi, LUz])

    fLU = func.IntervalFunction(lambda r: LUfunc(r), U.rmin, U.rmax)

    return func.SampledFunction.sample(U.sample_r, fLU)


def scalar_product_integrand(mode, U, W):
    """
    The integrand of the scalar product < U | W >. 
    
    The scalar product is given by the integration of this function 
    over the common range of `U` and `W`. Both perturbations are assumed
    to share their vertical and azimuthal numbers (contained in `mode`).  

    Parameters
    ----------
    mode : ModeParam
        Parameters of the mode.
    
    U : functions.SampledFunction
        Function from the adjoint space (complex conjugated).
    
    W : functions.SampledFunction
        Function from the original space.

    Returns
    -------
    result : funtion.SampledFunction (signature r -> complex)
        Sampled integrand in the scalar product.

    """
    disk = mode.disk

    def integrand_local(r):
        
        [h, vr, vphi, vz] = W(r)
        [dh, dvr, dvphi, dvz] = W.der(r)
        [eta, ur, uphi, uz] = U(r)
        [deta, dur, duphi, duz] = U.der(r)

        if mode.n == 0:
            y  = 2*pi*math.factorial(mode.n)*(eta*h/disk.cs**2 + ur*vr + r**2*uphi*vphi)*r*disk.Sigma(r)
            dy = 2*pi*math.factorial(mode.n)*( 
                    (deta*h + eta*dh)/disk.cs**2 + (dur*vr + ur*dvr) + 2*r*uphi*vphi 
                    + r**2*(duphi*vphi + uphi*dvphi) 
                    + (eta*h/disk.cs**2 + ur*vr + r**2*uphi*vphi)*(1/r + disk.dlnSigma(r))
                    )*r*disk.Sigma(r)

        else:
            y  = 2*pi*math.factorial(mode.n)*(eta*h/disk.cs**2 + ur*vr + r**2*uphi*vphi + uz*vz/mode.n)*r*disk.Sigma(r)	
            dy = 2*pi*math.factorial(mode.n)*( 
                    (deta*h + eta*dh)/disk.cs**2 + (dur*vr + ur*dvr) + 2*r*uphi*vphi 
                    + r**2*(duphi*vphi + uphi*dvphi) + (duz*vz + uz*dvz)/mode.n 
                    + (eta*h/disk.cs**2 + ur*vr + r**2*uphi*vphi)*(1/r + disk.dlnSigma(r))
                    )*r*disk.Sigma(r)

        return [y, dy]

    radii = func.common_range(U.sample_r, W.sample_r)
    n = len(radii)
    ys = np.zeros((n,1), dtype=complex)
    dys = np.zeros((n,1), dtype=complex)
    for i, r in enumerate(radii):
        y, dy = integrand_local(r)
        ys[i] = y
        dys[i] = dy
    
    return func.SampledFunction(radii, ys, dys) 


def scalar_product(mode, U, W):
    """
    The scalar product < U | W >. 
    
    In the integration, the common range of `U` and `W`. Both perturbations 
    are assumed to share their vertical and azimuthal numbers (contained in `mode`).  

    Parameters
    ----------
    mode : ModeParam
        Parameters of the mode.
    
    U : functions.SampledFunction
        Function from the adjoint space (complex conjugated).
    
    W : functions.SampledFunction
        Function from the original space.

    Returns
    -------
    result : complex
        Value of the scalar product < U | W >.

    """

    integrand = scalar_product_integrand(mode, U, W)
    return integrand.integ()[0]


def governing_equation_residuals(mode, r, W, dW, F):
    """
    Residuals in the full governing equations for given solution.
    The residuals are evaluated in a single point. Both, the solution vector 
    and its derivative (and force expression) has to be supplied.
    
    Parameters
    ----------
    mode : ModeParam
        Parameters of the perturbation.

    r : float
        Point of the evaluation.

    W : numpy.array, shape = (4)
        Full solution vector [h, vr, vphi, vz].

    dW : numpy.array, shape = (4)
        Derivative [h', vr', vphi', vz'].

    F : functions.IntervalFunction 
        Forcing term (function r -> [f, Fr, Fphi, Fz])
    
    Returns 
    -------   
    Delta : numpy.array, shape = (4)
        Residuals (i.e., RHS_i(W)) for each
        component of the governing equation.
    """
    kappa  = mode.disk.kappa(r)
    Omega  = mode.disk.Omega(r)
    Omegav = mode.disk.Omegav(r)
    dlnSigma = mode.disk.dlnSigma(r)
    c = mode.disk.cs
    
    omegat = mode.omegat(r)
    m, n = mode.m, mode.n
    
    [f, Fr, Fphi, Fz] = F(r)
    [h, vr, vphi, vz] = W
    dh  = dW[0]
    dvr = dW[1]
    
    res_h    = -(1j*h*omegat*r - c**2*(dvr*r + 1j*m*r*vphi + vr + dlnSigma*r*vr) + c*Omegav*r*vz + r*f)/r
    res_vr   = dh - Fr - 2*Omega*r*vphi - 1j*omegat*vr 
    res_vphi = -Fphi + (1j*h*m)/r**2 - 1j*omegat*vphi + (kappa**2*vr)/(2.*Omega*r)
    res_vz   = (Omegav*(h*n*Omegav - c*(Fz + 1j*omegat*vz)))/c**2

    #norm = 1/omegat
    
    return np.array([res_h, res_vr, res_vphi, res_vz])
    

#===================================================================
# find_mode 
#===================================================================
    

def find_mode(trialmode, bc1, bc2, rm, radii=[], norm = {}, R=float('inf')):
    """
    This function adjusts the frequency of the trial mode in such way 
    that the resulting mode satisfies inner and outer boundary
    conditions. It implements the shooting method. The wave equation is 
    intergated from the inner boundary outward and from the outer 
    boundary inward. The two solutions are then matched at radius `rm`.
    
    Parameters
    -----------
    trialmode : ModeParam
        Parameters of the trial mode.

    bc1 : bcGeneral
        Inner boundary condition.

    bc2 : bcGeneral
        Outer boundary condition.

    rm : float
        Matching radius.

    radii : list or 1d numpy array, oprional
        Evaluation points, the eigenfunction will be evaluated 
        at these radii.

    norm : dict
        normalization of the solution, dict shoud have one of these
        keyword: 'hmax', 'vmax' 
    
    Returns  [mode, W]
    -------
    mode : ModeParam
        Parameters of the mode with adjusted frequency.

    W : func.SampledFunction
        The eigenfunction of the mode.
    """
    
    m = trialmode.m
    n = trialmode.n
    disk = trialmode.disk
    solver = ode(ode_rhs, ode_jac).set_integrator('zvode', method='bdf', rtol=1e-10, nsteps=10000)
    F = func.IntervalFunction(lambda r: np.array([0, 0, 0, 0]), rmin=0, rmax=float('inf'))
    
    if (bc1.r >= bc2.r):
        raise ValueError('Invalid solution domain (r2 smaller than r1). Here r1 = {}, r2 = {}'.format(
                         bc1.r, bc2.r))
    
    if not ((bc1.r <= rm) and (rm <= bc2.r)):
        raise ValueError('Matching point is out of the solution domain. Here r1 = {}, r2 = {}, rm = {}'.format(
                         bc1.r, bc2.r, rm))

    def omega2w(omega):
        if R == float('inf'):
            return omega - trialmode.omega
        else:
            omega_r, theta = cmath.polar(omega - trialmode.omega)
            w_r = math.tan(omega_r/R*math.pi/2)
            return cmath.rect(w_r, theta) 
        
    def w2omega(w):
        if R == float('inf'):
            return trialmode.omega + w 
        else:
            w_r, theta = cmath.polar(w)
            omega_r = 2/math.pi*R*math.atan(w_r)
            return trialmode.omega + cmath.rect(omega_r, theta)


    def Delta(X):
        # this function evalute discontinuity at the matching point	
        
        mode = ModeParam(disk, w2omega(X[0] + 1j*X[1]), m, n)
        
        # integration from inner boundary to rm
        Y10 = bc1.Y0(1, mode)
        solver.set_initial_value(Y10, bc1.r).set_f_params(mode, F).set_jac_params(mode)
        if rm > bc1.r:
            Y1 = solver.integrate(rm)
        else:
            Y1 = Y10


        # integration from the outer boundary to rm
        Y20 = bc2.Y0(1, mode)
        solver.set_initial_value(Y20, bc2.r).set_f_params(mode, F).set_jac_params(mode)
        Y2 = solver.integrate(rm)
        
        # evaluation of the discontinuity and return as 2d array:
        norm1 = Y1[0]*Y1[0]/disk.cs**2 + Y1[1]*Y1[1]
        norm2 = Y2[0]*Y2[0]/disk.cs**2 + Y2[1]*Y2[1]

        D = (Y1[0]*Y2[1] - Y2[0]*Y1[1])/sqrt(abs(norm1*norm2))
        return np.array([D.real, D.imag])
    
    
    w0 = omega2w(trialmode.omega)
    X0 = [w0.real, w0.imag] 
    sol = root(Delta, X0, method='hybr', options={'xtol': 1e-16})
    result_mode = ModeParam(disk, w2omega(sol.x[0] + 1j*sol.x[1]), m, n)

    n = len(radii)

    if n > 0:

        Y = np.zeros((n, 2), dtype=complex)

        # forward integration from inner boundary to rm:
        Y01 = bc1.Y0(1, result_mode)
        solver.set_initial_value(Y01, bc1.r).set_f_params(result_mode, F).set_jac_params(result_mode)

        # the integrator has troubles to start to integrate to the same point
        # (we have to take care about the boundary):
        if radii[0] == bc1.r:
            Y[0, :] = Y01
        else:
            if radii[0] < rm:
                Y[0, :] = solver.integrate(radii[0])			

        for i in range(1, n):
            if radii[i] > rm: 
                break
            Y[i, :] = solver.integrate(radii[i]) 

        if radii[i-1] < rm:
            YL = solver.integrate(rm)
        else:
            YL = Y[i-1]
        
        # backward integration from outer boundary to rm. 
        Y02 = bc2.Y0(1, result_mode)
        solver.set_initial_value(Y02, bc2.r).set_f_params(result_mode, F).set_jac_params(result_mode)

        if (radii[-1] == bc2.r) and not (rm == bc2.r):
            Y[-1, :] = Y02
        else:
            if radii[-1] > rm:
                Y[-1, :] = solver.integrate(radii[-1])

        for i in reversed(range(n-1)):
            if radii[i] <= rm:
                break
            Y[i, :] = solver.integrate(radii[i])
        
        YR = solver.integrate(rm)

        # continuity at rm & renormalization of the right solution: we want YL - a*YR = 0
        # calculation of the other quantities..
        
        # arithmetic mean of the two ways of renormalizing
        a = (YL[0]*YR[1] + YL[1]*YR[0])/(2*YR[0]*YR[1]) 

        # filling the sample: 
        W  = np.zeros((n, 4), dtype=complex)
        dW = np.zeros((n, 4), dtype=complex)
    
        for i, (r, y) in enumerate(zip(radii, Y)):
            if r <= rm:
                yres = y
            else:
                yres = a*y	

            local_W, local_dW = Y2W(r, yres, result_mode, F)
            W[i,:]  = local_W
            dW[i,:] = local_dW

        # renormalization:

        if 'hmax' in norm:
            b = norm['hmax']/max(abs(W[:, 0]))
        elif 'vmax' in norm:
            b = norm['vmax']/max(abs(W[:, 1]))
        else:
            b = 1.
            
    return [result_mode, func.SampledFunction(radii, b*W, b*dW)]


def find_mode_bvp(trialmode, bc1, bc2, rm, radii=[], norm = {}, R=float('inf')):

    solver = ode(ode_rhs, ode_jac).set_integrator('zvode', method='bdf', rtol=1e-10, nsteps=10000)
    F = func.IntervalFunction(lambda r: np.array([0, 0, 0, 0]), rmin=0, rmax=float('inf'))
    
    if (bc1.r >= bc2.r):
        raise ValueError('Invalid solution domain (r2 smaller than r1). Here r1 = {}, r2 = {}'.format(
                         bc1.r, bc2.r))
    
    if not ((bc1.r <= rm) and (rm <= bc2.r)):
        raise ValueError('Matching point is out of the solution domain. Here r1 = {}, r2 = {}, rm = {}'.format(
                         bc1.r, bc2.r, rm))

    # prepare the initial guess:

    n = radii.size
    Y = np.zeros((2, n), dtype=complex)
    
    YL0 = bc1.Y0(1, trialmode)
    solver.set_initial_value(YL0, bc1.r).set_f_params(trialmode, F).set_jac_params(trialmode)
    Y[:,0] = YL0
    for i, r in enumerate(radii[1:]):
        if r <= rm:
            Y[:,1+i] = solver.integrate(r)
        else:
            break
    if r < rm:
        YLm = solver.integrate(rm)
    else:
        YLm = Y[:,i]

    YR0 = bc2.Y0(1, trialmode)
    solver.set_initial_value(YR0, bc2.r).set_f_params(trialmode, F).set_jac_params(trialmode)
    Y[:,n-1] = YR0
    for i, r in enumerate(reversed(radii[:-1])):
        if r > rm:
            Y[:, n-2-i] = solver.integrate(r)
        else:
            break
    if r > rm:
        YRm = solver.integrate(rm)
    else:
        YRm = Y[:, n-i-1]

    for i, r in enumerate(radii):
        if r > rm:
            Y[:,i] = Y[:,i]*YLm[0]/YRm[0]

    # now call the bvp routine:
    def omega2w(omega):
        if R == float('inf'):
            return omega - trialmode.omega
        else:
            omega_r, theta = cmath.polar(omega - trialmode.omega)
            w_r = math.tan(omega_r/R*math.pi/2)
            return cmath.rect(w_r, theta) 
        
    def w2omega(w):
        if R == float('inf'):
            return trialmode.omega + w 
        else:
            w_r, theta = cmath.polar(w)
            omega_r = 2/math.pi*R*math.atan(w_r)
            return trialmode.omega + cmath.rect(omega_r, theta)

    def bvp_bc(YL, YR, p):
        mode = ModeParam(disk=trialmode.disk, omega=w2omega(p[0]), m=trialmode.m, n=trialmode.n)
        Y0L = bc1.Y0(1, mode)
        Y0R = bc2.Y0(1, mode)
        return np.array([YL[0]*Y0L[1] - YL[1]*Y0L[0], YR[0]*Y0R[1] - YR[1]*Y0R[0], YL[0] - Y0L[0]])

    def bvp_ode_rhs(rs, Ys, p):
        mode = ModeParam(disk=trialmode.disk, omega=w2omega(p[0]), m=trialmode.m, n=trialmode.n)
        n = rs.size
        dY = np.zeros((2, n), dtype=complex)
        for i, (r, Y) in enumerate(zip(rs, Ys)):
            dY[:,i] = ode_rhs(r, Y, mode, F)
        return dY	

    p = [omega2w(trialmode.omega)]
    bvp_sol = integrate.solve_bvp(bvp_ode_rhs, bvp_bc, radii, Y, p, tol=1e-2, max_nodes=100000, verbose=2)

    if (bvp_sol.status != 0):
        raise RuntimeError('solve_bvp returned status {}.'.format(bvp_sol.status))

    # sample the solution and store it as SampledFunction

    result_mode = ModeParam(disk=trialmode.disk, omega=w2omega(p[0]), m=trialmode.m, n=trialmode.n)

    # filling the sample: 
    W  = np.zeros((n, 4), dtype=complex)
    dW = np.zeros((n, 4), dtype=complex)

    Y = np.transpose(bvp_sol.sol(radii))
    print(Y.shape)

    for i, (r, y) in enumerate(zip(radii, Y)): 
        local_W, local_dW = Y2W(r, y, result_mode, F)
        W[i,:]  = local_W
        dW[i,:] = local_dW

    if 'hmax' in norm:
        b = norm['hmax']/max(abs(W[:, 0]))
    elif 'vmax' in norm:
        b = norm['vmax']/max(abs(W[:, 1]))
    else:
        b = 1.
            
    return [result_mode, func.SampledFunction(radii, b*W, b*dW)]



def adjoint_mode(mode, W):
    """
    Adjoint to the given mode.

    Parameters
    ----------
    mode : ModeParam
        Parameters of the mode
    W : functions.SampledFunction
        Eigenfunction

    Returns
    -------
    U : functions.SampledFunction
        'Eigenfunction' of the adjoint
     
    Note
    ----
    This function makes sense only when applied to
    the real oscillation modes.

    """

    def funcU(r):
        m = mode.m
        omegat = mode.omegat(r)
        Omega = mode.disk.Omega(r)
        kappa = mode.disk.kappa(r)
        [h, vr, vphi, vz] = W(r)

        eta = h/omegat
        ur = -vr/omegat
        uphi = 4*Omega**2/(omegat*kappa**2)*(vphi - m*h/(omegat*r**2)) + m*h/(r*omegat)**2
        uz = -vz/omegat

        return np.array([eta, ur, uphi, uz])


    return func.SampledFunction.sample(W.sample_r, 
                                       func.IntervalFunction(lambda r:funcU(r), W.rmin, W.rmax))



#===================================================================
# find_response 
#===================================================================

def response00(mode, F, radii, order=1):
    """
    Response for stationary (omega=0) and axisymmetric (m=0) force. 

    These modes cause problem because the corotation frequency vanishes 
    
    Parameters
    ----------
    mode : ModeParam
        Parameters of the mode. 

    F : functions.IntervalFunction
        Perturbing force.

    radii: numpy.array or list
        Response of the disk is evaluted at these radii.

    order: int, positive
        Order of approximation in mode.omega
    
    Returns  W
    -------
    W : functions.SampledFunction
        Response of the disk
    """

    print('Enetering response00()')
    
    disk = mode.disk
    c = disk.cs
    
    # the problem is algebraic, there are explicit analytic expressions for the response
    # they are implemented in this function:

    def Wfunc(r, force):
        Omega = disk.Omega(r)
        kappa = disk.kappa(r)
        dkappa = disk.dkappa(r)
        Omegav = disk.Omegav(r)
        dOmegav = disk.dOmegav(r)
        dlnSigma = disk.dlnSigma(r)
        [f, Fr, Fphi, Fz] = force(r)
        [dFphi, dFz] = force.der(r)[2:]

        if mode.n > 0:
            h  = (c*Fz)/(mode.n*Omegav)
            vphi = (c*dFz/(mode.n*Omegav) - c*Fz*dOmegav/(mode.n*Omegav**2) - Fr)/(2*r*Omega)
            vz = c*(-f/c**2 + 2*dFphi*Omega*r/kappa**2 
                    + Fphi*(1/Omega + 2*Omega*r*(-2*dkappa + dlnSigma*kappa)/kappa**3))/Omegav
        else:
            h = 0
            vphi = -Fr/(2*r*Omega)
            vz = 0
                
        vr = (2*Fphi*Omega*r)/kappa**2

        return np.array([h, vr, vphi, vz])

    # rhs (force) or lower-order approximations:
    Fn = F
    W = func.SampledFunction(radii, np.zeros((len(radii), 4)), np.zeros((len(radii), 4)))

    for n in range(order+1):
        print('response00: order', n)
        fWn = func.IntervalFunction(lambda r: Wfunc(r, Fn), Fn.rmin, Fn.rmax, dr=1e-4*(radii[1]-radii[0]))
        Wn = func.SampledFunction.sample(radii, fWn)
        W += Wn
        if n < order:
            Fn = (1j*mode.omega)*Wn

    return W


def find_response(mode, F, bc1, bc2, rm, radii, check00=True, order00=1):
    """
    This function finds the response of the disk to the forcing function.
    The response has to satisfy inner and outer boundary
    conditions. The function uses both forward and backward integration
    from the inner and outer bondary (resp.). The two solutions are 
    matched at the the radius rm.
    
    Parameters
    ----------
    mode : ModeParam
        Parameters characterizing the forcing.

    F : functions.IntervalFunction
        Perturbing force.

    bc1 : bcGeneral
        The inner boundary condition.

    bc2 : bcGeneral
        The outer boundary condition.

    rm : float
        The matching radius.

    radii : numpy.array or list
        Response of the disk is evaluated at these radii

    check00 : bool, optional
        Set to False if the response should be calculated always by
        ODE solver, even when omega = m = 0. Otherwise, some check are
        performed.
    
    Returns 
    -------
    result : functions.SampledFunction
        Response of the disk.

    TODO
    ----
    * care about the case when omega = m = 0. DONE
    """

    if len(radii) < 2:
        raise ValueError('Size of radii array must be at least (while it is {}).'.format(len(radii)))

    if mode.m == 0 and check00:
        Omegam = mode.disk.Omega(rm)
        if (abs(mode.omega.real) < 1e-5*Omegam) and (abs(mode.omega.imag) < 1e-1*Omegam):
            return response00(mode, F, radii, order=order00)

    if bc1.r >= bc2.r:
        raise ValueError('bc1.r must be smaller than bc2.r\n' + 
                         'bc1.r = {}\n'.format(bc1.r) + 'bc2.r = {}'.format(bc2.r))

    if (rm < bc1.r) or (rm > bc2.r):
        raise ValueError('Matching point (rm) has to be between bc1.r and bc2.r\n' +
                         'bc1.r = {}\n'.format(bc1.r) + 'bc2.r = {}\n'.format(bc2.r) +
                         'rm    = {}\n'.format(rm))		

    F0 = func.IntervalFunction(lambda r: np.array([0, 0, 0, 0]), F.rmin, F.rmax)
    r1 = bc1.r
    r2 = bc2.r

    n = len(radii)
    dr = func.minimal_step(radii)

    Y0 = np.zeros((n, 2), dtype=complex)
    YF = np.zeros((n, 2), dtype=complex)
    
    solver = ode(ode_rhs, ode_jac).set_integrator('zvode', method='bdf', rtol=1e-14, nsteps=1000, max_step=dr)

    # forward integration from r1 to rm. Particular solution with [0, 0] as initial condition:
    YF1 = bc1.YF(mode, F)
    solver.set_initial_value(YF1, r1).set_f_params(mode, F).set_jac_params(mode)

    if radii[0] == bc1.r:
        YF[0, :] = YF1
    else:
        if radii[0] < rm:
            YF[0, :] = solver.integrate(radii[0])			

    for i in range(1, n):
        if radii[i] > rm: 
            break
        YF[i, :] = solver.integrate(radii[i]) 

    [hLF, vLF] = solver.integrate(rm)
    
    # forward integration from r1 to r2. Fundamental solution with proper initial condition:
    Y01 = bc1.Y0(1, mode)
    solver.set_initial_value(Y01, r1).set_f_params(mode, F0).set_jac_params(mode)

    if radii[0] == bc1.r:
        Y0[0, :] = Y01
    else:
        if radii[0] < rm:
            Y0[0, :] = solver.integrate(radii[0])			

    for i in range(1, n):
        if radii[i] > rm: 
            break
        Y0[i, :] = solver.integrate(radii[i]) 

    [hL0, vL0] = solver.integrate(rm)
    
    # backward integration from r2 to r1. Particular solution with [0, 0] as initial condition:
    YF2 = bc2.YF(mode, F)
    solver.set_initial_value(YF2, r2).set_f_params(mode, F).set_jac_params(mode)

    if (radii[-1] == bc2.r) and not (rm == bc2.r):
        YF[-1, :] = YF2
    else:
        if radii[-1] > rm:
            YF[-1, :] = solver.integrate(radii[-1])

    for i in reversed(range(n-1)):
        if radii[i] <= rm:
            break
        YF[i, :] = solver.integrate(radii[i])

    [hRF, vRF] = solver.integrate(rm)
    
    # backward integration from r2 to r1. Fundamental solution with proper initial condition:
    Y02 = bc2.Y0(1, mode)
    solver.set_initial_value(Y02, r2).set_f_params(mode, F0).set_jac_params(mode)

    if (radii[-1] == bc2.r) and not (rm == bc2.r):
        Y0[-1, :] = Y02
    else:
        if radii[-1] > rm:
            Y0[-1, :] = solver.integrate(radii[-1])

    for i in reversed(range(n-1)):
        if radii[i] <= rm:
            break
        Y0[i, :] = solver.integrate(radii[i])
    
    [hR0, vR0] = solver.integrate(rm)
    
    # matching:
    alpha = ((hLF - hRF)*vR0 + hR0*(-vLF + vRF))/(hR0*vL0 - hL0*vR0)
    beta  = ((hLF - hRF)*vL0 + hL0*(-vLF + vRF))/(hR0*vL0 - hL0*vR0) 

    # filling the resulting sample: 
    W  = np.zeros((n, 4), dtype=complex)
    dW = np.zeros((n, 4), dtype=complex)

    for i, (r, y0, yF) in enumerate(zip(radii, Y0, YF)):
        if r < rm:
            yres = alpha*y0 + yF
        else:
            yres = beta*y0 + yF

        local_W, local_dW = Y2W(r, yres, mode, F)
        W[i,:]  = local_W
        dW[i,:] = local_dW
    
    return func.SampledFunction(radii, W, dW)


#===================================================================
# Nonlinear coupling
#===================================================================


def heProductCoeff(n1, n2, s):
    """
    Coefficient in the expansion of product of two Hermite polynomials:

        He(n1, y) He(n2, y) = c(n1, n2, 0) He(n1 + n2, y) + 
                              c(n1, n2, 1) He(n1 + n2 - 2, y) + 
                              c(n1, n2, 2) He(n1 + n2 - 4, y) +
                              ...

    Parameters
    ----------
    n1 : int (non-negative)
        Order of He1.

    n2 : int (non-negative)
        Order of He2.

    s : int (non-negative)
        Corresponding He is of the order n1 + n2 - 2*s.

    Returns
    -------
    c : float
        Coefficient c(n1, n2, s).
    
    Notes
    -----
    * Symmetry: c(n1, n2, s) == c(n2, n1, s)
    * for more, see Pattaroyo (2019): https://arxiv.org/pdf/1901.01648.pdf	
    
    """

    if (s < 0) or (n1 < 0) or (n2 < 0) or (s > min(n1, n2)):
        return 0
    else:
        return factorial(n1)*factorial(n2)/(factorial(n1-s)*factorial(n2-s)*factorial(s))



def cpl(r, mode1, W1, mode2, W2, a00, a01, a11, a12):
    """
    Local coupling. Calculates value of the forcing function at given radius. 

    Parameters
    ----------
    r : float
        Point of evaluation.

    mode1 : ModeParam
        Parameters of the mode 1.

    W1 : Functions.IntervalFunction
        Eigenfunction of the mode 1.

    mode2 : ModeParam
        Parameters of the mode 2.

    W2 : Functions.IntervalFunction
        Eigenfunction of the mode 2.

    a00 : float
        c(n1, n2, s)

    a01 : float
        c(n1 - 1, n2, s)

    a11 : float
        c(n1 - 1, n2 - 1, s - 1)

    a12 : float
        c(n1 - 2, n2 - 1, s - 1) 

    Returns 
    -------
    F : numpy.array, shape = (4)
        Value of the forcing function [f, Fr, Fphi, Fz]
    """
    n1, m1 = mode1.n, mode1.m
    n2, m2 = mode2.n, mode2.m

    [h1, vr1, vphi1, vz1] = W1(r)
    [h2, vr2, vphi2, vz2] = W2(r)
    [dh1, dvr1, dvphi1, dvz1] = W1.der(r)
    [dh2, dvr2, dvphi2, dvz2] = W2.der(r)

    f    = -(a00*(1j*h2*m2*vphi1 + 1j*h1*m1*vphi2 + dh2*vr1 + dh1*vr2) + 
            a11*(h2*n2*vz1 + h1*n1*vz2))/2

    Fr   = -(a00*(-2*r*vphi1*vphi2 + (dvr2 + 1j*m1*vphi2)*vr1 + (dvr1 + 1j*m2*vphi1)*vr2) + 
            a11*(n2*vr2*vz1 + n1*vr1*vz2))/2.

    Fphi = -(a00*(1j*(m1 + m2)*r*vphi1*vphi2 + (dvphi2*r + 2*vphi2)*vr1 + (dvphi1*r + 2*vphi1)*vr2)/r +
            a11*(n2*vphi2*vz1 + n1*vphi1*vz2))/2.

    Fz   = -(vz1*(1j*a01*m1*vphi2 + a12*(n1+n2-2)*vz2) + 
            a01*(dvz2*vr1 + dvz1*vr2 + 1j*m2*vphi1*vz2))/2.

    # if (a12 == a01 == 0):
    # 	print(Fz)

    return np.array([f, Fr, Fphi, Fz])


def nonlinear_coupling(mode1, W1, mode2, W2, radii=[]):
    """
    Calculates the nonlinear force arising from the coupling of two perturbations.
    
    Parameters
    ----------
    mode1 : ModeParam
        Parameters of the perturbation 1.

    W1 : functions.IntervalFunction
        Eigenfunction of the perturbation 1.

    mode2 : ModeParam
        Parameters of the perturbation 2.

    W2 : functions.IntervalFunction
        Eigenfunction of the perturbation 2.

    radii : numpy array (1d)
        Evaluation points. If it is not provided, the function uses the evaluation points of W1. 
        The range is restricted to the overlap of W1 and W2.
    
    Returns:  [component_params, component_forces] 
    --------
    components : list of ModeParam
        parameters of the perturbations of different vertical quantum numbers.

    forcings : list of functions.SampledFunction
        Sampled nonlinear force for each component. 
    """

    # create array of evaluation points:

    if radii == []:
        rbase = W1.sample_r
    else:
        rbase = radii

    r1 = min(rbase[0], W1.sample_r[0], W2.sample_r[0])
    r2 = max(rbase[-1], W1.sample_r[-1], W2.sample_r[-1])
    
    if (r1 > r2):
        if radii == []:
            raise ValueError('The intervals [{},{}] and [{},{}] does not overlap.'.format(
                              W1.sample_r[0], W1.sample_r[-1], W2.sample_r[0], W2.sample_r[-1]))
        else:
            raise ValueError('The intervals [{},{}], [{},{}] and [{},{}] do not overlap.'.format(
                              W1.sample_r[0], W1.sample_r[-1], W2.sample_r[0], W2.sample_r[-1], rbase[0], rbase[-1]))

    rads = []
    for r in rbase:
        if r > r2:
            break
        if r >= r1:
            rads.append(r)

    radii_used = np.array(rads) 
    N = len(radii_used)

    dr = max(W1.dr, W2.dr)

    # loop over the comonents (different n):

    component_params = []
    component_forces = []

    for s in range(mode1.n + mode2.n + 1):
        n = mode1.n + mode2.n - 2*s
        
        if n < 0:
             break
    
        # coefficients of the different n-components:
        a00 = heProductCoeff(mode1.n, mode2.n, s)
        a01 = heProductCoeff(mode1.n - 1, mode2.n, s)
        a11 = heProductCoeff(mode1.n - 1, mode2.n - 1, s - 1)
        a12 = heProductCoeff(mode1.n - 2, mode2.n - 1, s - 1)

        # print('n: ', n, 'a00:', a00, 'a01:', a01, 'a11:', a11, 'a12:', a12)

        F  = np.zeros((N, 4), dtype=complex)
        dF = np.zeros((N, 4), dtype=complex)

        for i, r in enumerate(radii_used):
            F[i, :] = cpl(r, mode1, W1, mode2, W2, a00, a01, a11, a12)
            
            # if n == 0 and mode1.m + mode2.m == 0:
            # 	print(F[:,3]) 


            drL = dr
            drR = dr
            if i == 0:
                drL = 0
            if i == N-1:
                drR = 0

            dF[i, :] = (cpl(r+drR, mode1, W1, mode2, W2, a00, a01, a11, a12) - 
                        cpl(r-drL, mode1, W1, mode2, W2, a00, a01, a11, a12))/(drR+drL)

        component_params.append(ModeParam(mode1.disk, mode1.omega + mode2.omega, mode1.m + mode2.m, n))
        component_forces.append(func.SampledFunction(radii_used, F, dF))
    
    return [component_params, component_forces]


def radial_resonances(mode1, mode2, mode12):
    """
    Radii, where three perturbations satisfy the WKBJ resonance condition `k1 + k2 = k12`.
    Negative values of the vawevectors are also taken into account. The wavevectors
    are calculated using WKBJ approximation.

    Parameters:
    -----------
    mode1, mode2 : disko.ModeParam
        Two 'daughter' perturbations with frequencies `omega1` and `omega2`

    mode12 : disko.ModeParam
        The 'parent' perturbation with frequency omega12 = omega1 + omega2

    Returns:
    --------
        List of radii where the radial resonances occur.
    """

    
    # mode, r --> abs(wavevector):
    def k(mode, r):
        eps = 1e-5    
        k2 = mode.k2(r).real
        if k2 + eps < 0:
            raise ValueError('Mode {} is in evanescent domain (k2 = {}, r = {})'.format(mode.__repr__(), k2, r))
        return math.sqrt(k2 + eps)

    # take all radii where modes have LRs or VRs: (k = 0 there)
    R = [mode1.disk.rin] + [mode1.disk.rout] + \
        [r for r in (mode1.LRs + mode1.VRs) if r!= None] + \
        [r for r in (mode2.LRs + mode2.VRs) if r!= None] + \
        [r for r in (mode12.LRs + mode12.VRs) if r!= None]
    R = sorted(R)
    
    res = []
    
    for i in range(len(R)-1):

        # check if the segment (R[i], R[i+1]) is in propagation regions of all modes:
        
        r = 0.5*(R[i] + R[i+1])
        if (mode1.k2(r).real > 0) and (mode2.k2(r).real > 0) and (mode12.k2(r).real > 0):

            # functions that are zero at resonances:
            # f1 = 0 when  k1 + k2 - k12 = 0
            # f2 = 0 when  k1 - k2 - k12 = 0  or -k1 + k2 - k12 = 0
            # the condition -k1-k2-k12 = 0 never happens
            
            f1 = lambda r: k(mode1, r) + k(mode2, r) - k(mode12, r)
            f2 = lambda r: abs(k(mode1, r) - k(mode2, r)) - k(mode12, r)
            
            for f in [f1, f2]:
                # look for roots of function f (as it can has complicated behavior
                # we add also minimum and maximum, this way we can reveal at most 
                # 3 roots, hope it is sufficient...)
                
                Rint = sorted([R[i], 
                               minimize_scalar(f, bounds=(R[i], R[i+1]), method='bounded').x, 
                               minimize_scalar(lambda r: -f(r), bounds=(R[i], R[i+1]), method='bounded').x, 
                               R[i+1]])

                for j in range(len(Rint)-1):
                    if f(Rint[j])*f(Rint[j+1]) < 0:
                        res.append(bisect(f, Rint[j], Rint[j+1]))

    return res
