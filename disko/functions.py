"""
Module implementing the radial functions on the disk. 
"""

import numpy as np
import math

def common_range(range1, range2, eps=0):
    """
    Given two ranges, this function returns the union on the overlap.

    Parameters
    ----------
    range1 : numpy.array
        First range.

    range2 : numpy.array
        Second range.
    
    eps : float, optional
        if the two points in the resulting range are closer than 
        this distance they are merged.

    Returns
    -------
    range : numpy.array
        The result consisting of evaluation points of both input ranges
        in their common domain.
    """
    if (range1[-1] < range2[0]) or (range2[-1] < range1[0]):
        return None
    i1 = 0
    i2 = 0
    result = []
    while (i1 < len(range1)) and (i2 < len(range2)):
        if abs(range1[i1] - range2[i2]) < eps:
            result.append((range1[i1] + range2[i2])/2)
            i1 += 1
            i2 += 1
        elif (range1[i1] < range2[i2]):
            if i2 > 0:
                result.append(range1[i1])
            i1 += 1
        else:
            if i1 > 0:
                result.append(range2[i2])
            i2 += 1

    return result


#===================================================================
# IntervalFunction
#===================================================================

class IntervalFunction:
    """
    General class describing radial functions on the disk. The radial 
    coordinate are bounded in the range rmin <= r <= rmax.

    Atributes
    ---------
    func : callable
        The function expression of the form f(r). The result may be scalar 
        or numpy array.

    rmin : float
        Lower bound of the radial coordinate.

    rmax : float
        Upper bound of the radial coordinate.

    dr : float, optional
        Increment for calculation radial derivatives.
    """

    def __init__(self, expr, rmin, rmax, dr=1e-4):
        self._expr = expr
        self._rmin = rmin
        self._rmax = rmax
        self._dr   = dr


    def __repr__(self):
        return 'IntervalFunction(expr = {}, rmin = {}, rmax = {})'.format(self._expr, self._rmin, self._rmax)

    @property
    def f(self):
        """ Function expression. """
        return self._expr
    
    @property
    def rmin(self):
        """ Lower limit of the radial coordinate. """ 
        return self._rmin


    @property
    def rmax(self):
        """ Upper limit of the radial coordinate. """
        return self._rmax

    @property
    def dr(self):
        """ Increment in calculations of numerical derivative. """
        return self._dr


    def _check_range(self, r):
        if (r < self._rmin) or (r > self._rmax):
            raise ValueError('r out of bounds [rmin, rmax]\n' +
                             'rmin = {}\n'.format(self._rmin) + 
                             'rmax = {}\n'.format(self._rmax) +
                             'r    = {}'.format(r))

    def _eval(self, r):
        return self._expr(r)


    def __call__(self, r):
        """
        Evaluation of the function at a given radius.

        Parameters
        ----------
        r : float
            The point of evaluaion.
        """
        self._check_range(r)
        return self._eval(r)


    def __rmul__(self, a):
        """ Multiply by a constant complex number a """
        return IntervalFunction(lambda r: a*self._expr(r), self._rmin, self._rmax, self._dr)


    def der(self, r):
        """
        First derivative of the function at given radius.

        Parameters
        ----------
        r : float
            The point of evaluation.
        """
        self._check_range(r)
        dr1 = min(self._dr, self._rmax - r)
        dr2 = min(self._dr, r - self._rmin)
        return (self._expr(r + dr1) - self._expr(r - dr2))/(dr1 + dr2)


    def logder(self, r):
        """
        Logaritmic derivative. [log(f)]'.

        Parameters
        ----------
        r : float
            The point of evaluation.
        """
        return self.der(r)/self._eval(r)		


#===================================================================
# SampledFunction
#===================================================================

def minimal_step(X):
    """ 
    Minimal distance of two consecutive points in the array.

    Parameters
    ----------
    radii : numpy.array (1d)
        Array of point.  
    
    Returns
    -------
    dr : float
        Minimal distance between X[i+1] and X[i], 
        where i is between 0 and len(X)-1. 
    """
    return min(abs(X[i+1] - X[i]) for i in range(len(X)-1))



class SampledFunction(IntervalFunction):
    """
    Radial functions (mode eigenfunctions, forcing, responses) on the disk. 
    This class stores both values and its radial derivaties in a given set of points.
    Approximation of functions given by its values and derivatives using piecewise 
    cubic polynomial.

    Atributes
    ---------
    r : numpy.array, shape = (n)
        Array of points of evaluation

    y : numpy.array shape = (n, ...)
        Array of values, y[j] = y(r[j]). Each value may be itself numpy.array

    dy : numpy.array, shape = (n, ...)
        Array of derivatives dy[j] = y'(r[j])

    der_method : str
        Method of calculating derivatives.
        der_method = 'linear' for linear approximation on the segments
        der_method = 'quadratic' for qudratic approximation on the segments

    Note
    ----
    The values at given r may be also numpy arrays.

    TODO
    ----
    Calculate the reasonable dr based on cubic approximation (what is reasonable dr
    so that simple derivative formula gives the same as derivative approximation?)
    """
    
    def __init__(self, r, y, dy, eval_method=('cubic', 'quadratic')): 
        self._r = np.array(r)
        self._y = np.array(y)
        self._dy = np.array(dy)
        self._rmin = r[0]
        self._rmax = r[-1]
        self._dr = 1e-8*minimal_step(self._r)
        self.set_eval(('cubic', 'quadratic'))
            
    
    @classmethod
    def sample(cls, radii, f):
        """ Create SampledFunction by sampling the IntervalFunction f """
        y = [f(r) for r in radii]
        dy = [f.der(r) for r in radii]
        return cls(radii, np.array(y), np.array(dy))	

    @classmethod
    def zero(cls, radii, valshape=(), dtype=float):
        """ Samples zero function """
        n = len(radii)
        y = np.zeros((n,)+valshape, dtype=dtype)
        dy = np.zeros((n,)+valshape, dtype=dtype)
        return cls(radii, y, dy)

    @property
    def n(self):
        """ Number of points in the sample. """
        return len(self._r)

    @property
    def valshape(self):
        """ shape of the function values in the sample. """
        return self._y.shape[1:]

    @property
    def dtype(self):
        """ type of the individual components. """
        return self._y.dtype

    @property
    def sample_r(self):
        """	Radial sampling points.	"""
        return self._r

    @property
    def sample_y(self):
        """	Values of the function on the sample. """
        return self._y
    
    @property
    def sample_dy(self):
        """ Values of the function derivative on the sample. """
        return self._dy


    def	__call__(self, r):
        return self._eval(r)


    def __repr__(self):
        return 'SampledFunction(rmin={}, rmax={}, n={}, valshape={}, dtype={})'.format(
                self._rmin, self._rmax, self.n, self.valshape, self.dtype)


    def __rmul__(self, a):
        """ Multiply by a constant complex number (from the left). """
        return SampledFunction(self._r, a*self._y, a*self._dy)


    def __add__(self, other):
        """ Add two sampled functions """
        if (type(other) == SampledFunction) and (other.valshape == self.valshape):
            if not (self._r == other._r).all():
                radii = common_range(self._r, other._r)
                y = np.array([self._eval(r) + other._eval(r) for r in radii])
                dy = np.array([self.der(r) + other.der(r) for r in radii])
                return SampledFunction(radii, y, dy)
            else:
                return SampledFunction(self._r, self._y + other._y, self._dy + other._dy)
        elif type(other) == IntervalFunction:
            sampled_other = SampledFunction.sample(self._r, other)
            return SampledFunction(self._r, self._y + sampled_other._y, self._dy + sampled_other._dy)
        else:
            raise NotImplementedError('Two incompatible types (SampledFunction and {})'.format(type(other)))

    def _segment_approx_value_cubic(self, r, i1, i2):
        """ Cubic approximation at given segment. """

        Dr = self._r[i2] - self._r[i1]
        a0 = self._y[i1]
        a1 = self._dy[i1]*Dr
        a2 = -(2.*self._dy[i1] + self._dy[i2])*Dr + 3*(self._y[i2] - self._y[i1])
        a3 = (self._dy[i1] + self._dy[i2])*Dr - 2*(self._y[i2] - self._y[i1])
        
        X = (r - self._r[i1])/Dr
        
        return a0 + a1*X + a2*X**2 + a3*X**3

    def _segment_approx_value_linear(self, r, i1, i2):
        """ Linear approximation at given segment. """

        Dr = self._r[i2] - self._r[i1]
        X = (r - self._r[i1])/Dr
        
        return self._y[i1] + (self._y[i2] - self._y[i1])*X

    def _segment_approx_der_quadratic(self, r, i1, i2):
        """ Derivative of the cubic approximation at given segment. """

        Dr = self._r[i2] - self._r[i1]
        a1 = self._dy[i1]*Dr
        a2 = -(2.*self._dy[i1] + self._dy[i2])*Dr + 3*(self._y[i2]- self._y[i1])
        a3 = (self._dy[i1] + self._dy[i2])*Dr - 2*(self._y[i2] - self._y[i1])
        
        X = (r - self._r[i1])/Dr
        
        return (a1 + 2*a2*X + 3*a3*X**2)/Dr

    def _segment_approx_der_linear(self, r, i1, i2):
        """ Derivatve at the segment, linear aproximation """
        Dr = self._r[i2] - self._r[i1]
        X = (r - self._r[i1])/Dr
        return self._dy[i1] + (self._dy[i2] - self._dy[i1])*X
        
    

    def _segment_approx_int(self, i, X=1):
        """ Approximation of the integral over the cubic segment. """

        Dr = self._r[i+1] - self._r[i]
        a0 = self._y[i]
        a1 = self._dy[i]*Dr
        a2 = -(2.*self._dy[i] + self._dy[i+1])*Dr + 3*(self._y[i+1] - self._y[i])
        a3 = (self._dy[i] + self._dy[i+1])*Dr - 2*(self._y[i+1] - self._y[i])
        
        return (a0*X + a1*X**2/2 + a2*X**3/3 + a3*X**4/4)*Dr


    def _get_segment_index(self, r):
        """ Find the the sample segment index `i` so that _r[i] <= r <= _r[i+1]. """
        found = False
        for i in range(len(self._r)-1):
            if (self._r[i] <= r) and (r <= self._r[i+1]):
                found = True
                break

        if not found:
            raise ValueError('No sample segment found. Perhaps r is out of the sample. Use `_check_range()')

        return i


    def _evaluate_triple(self, i, i1, i2):
        """ evaluate triple adjecent to ith evaluation point """
        y = self._segment_approx_value(self._r[i], i1, i2)
        dy = self._segment_approx_der(self._r[i], i1, i2)
        return np.linalg.norm(y - self._y[i]) + np.linalg.norm(dy - self._dy[i])			

    def _evaluate_sample_points(self):
        """ Evaluates how important the points are """
        result = np.zeros(len(self._r))
        for i in range(1, len(self._r) - 1):
            result[i] = self._evaluate_triple(i, i-1, i+1)
        return result

    def set_eval(self, method=('cubic', 'quadratic')):
        """ Set the evaluation method. """

        if method[0] == 'cubic':
            self._segment_approx_value = self._segment_approx_value_cubic
        elif method[0] == 'linear':
            self._segment_approx_value = self._segment_approx_value_linear
        else:
            raise ValueError('Method of evaluation can by either "linear" or "qudratic" but "{}" provided.'.format(der_method))
        
        if method[1] == 'quadratic':
            self._segment_approx_der = self._segment_approx_der_quadratic
        elif method[1] == 'linear':
            self._segment_approx_der = self._segment_approx_der_linear
        else:
            raise ValueError('Method for the derivative approximation can by either "linear" or "qudratic" but "{}" provided.'.format(der_method))

                               
    def remove_points(self, nrem=0):
        if nrem == 0:
            return

        if nrem > len(self._r) - 2:
            raise ValueError('You want to remove more points ({}) than is inside the sample ({}).'.format(nrem, len(self._r)-2))
        
        # evaluate the importance of points of the sample:
        # importance of the boundaries is infinite.
        importance = np.concatenate(([float('inf')], self._evaluate_sample_points()[1:-1], [float('inf')]))
        # flags for the point removal (keep = False means the point will be removed;
        # active point = point that has not been assigned for removing yet)
        keep = np.full(len(self._r), True)
        
        n = 0
        while n < nrem: 
            # sort the importance array so that the least important point are
            # at the begining: 
            indices = np.argsort(importance)
            # take the least important active point
            j = 0
            while keep[indices[j]] == False:
                j += 1
            # take the point from indices and remove flag it for removing:
            i = indices[j]
            keep[i] = False
            # find indices of nearby active points (i1 < i and i2 > i): 
            i1 = i - 1
            while keep[i1] == False: 
                i1 -= 1
            i2 = i + 1
            while keep[i2] == False:
                i2 += 1
            # find yet another active points before i1 and after i2 to form a triple
            i0 = i1 - 1
            while (i0 > -1) and (keep[i0] == False):
                i0 -= 1
            i3 = i2 + 1
            while (i3 < len(self._r)) and (keep[i3] == False):
                i3 += 1
            # Now, we have i0 < i1 < i <i2 < i3 where i0, i1, i2, i3 are closest active 
            # points. Update the evaluations of points i1 and i2 after removing the point i:
            if i1 > 0:
                importance[i1] = self._evaluate_triple(i1, i0, i2)
            if (i2 < len(self._r) - 1):
                importance[i2] = self._evaluate_triple(i2, i1, i3)  

            n += 1

        # Execute remove the point:	
        self._r = np.compress(keep, self._r)
        self._y = np.compress(keep, self._y, axis=0)
        self._dy = np.compress(keep, self._dy, axis=0)


    def optimize(self):
        
        # evaluate the importance of points of the sample:
        # importance of the boundaries is infinite.
        importance = self._evaluate_sample_points()[1:-1]
        maximp = max(importance)
        importance = np.concatenate(([float('inf')], importance, [float('inf')]))

        # flags for the point removal (keep = False means the point will be removed;
        # active point = point that has not been assigned for removing yet)
        keep = np.full(len(self._r), True)
        
        indices = np.argsort(importance)
        marked = False
        j = 0
        while j < len(self._r) - 2: 
            # sort the importance array so that the least important point are
            # at the begining: 
            if marked:
                indices = np.argsort(importance)
                marked = False
                j = 0
            # go over the indices array and try to remove the least important point.
            # to remove it, 2 conditions has to be satisfied:
            # (1) the point is still active
            # (2) removing the point does not increase importance of nearby points more than maximp.
            while not keep[indices[j]]:
                j += 1
            # i is active point.
            i = indices[j]
            # find indices of nearby active points (i1 < i and i2 > i): 
            i1 = i - 1
            while keep[i1] == False: 
                i1 -= 1
            i2 = i + 1
            while keep[i2] == False:
                i2 += 1
            # find yet another active points before i1 and after i2 to form a triple
            i0 = i1 - 1
            while (i0 > -1) and (keep[i0] == False):
                i0 -= 1
            i3 = i2 + 1
            while (i3 < len(self._r)) and (keep[i3] == False):
                i3 += 1
            # Now, we have i0 < i1 < i <i2 < i3 where i0, i1, i2, i3 are closest active 
            # points. Calculate the importance of the points i1, i2 when i is removed:
            if i1 > 0:
                imp1 = self._evaluate_triple(i1, i0, i2)
            else:
                imp1 = float('inf')
            if (i2 < len(self._r) - 1):
                imp2 = self._evaluate_triple(i2, i1, i3)  
            else:
                imp2 = float('inf')
            # if the importance of either of i1 and i2 does not exceed maximp mark,
            # update importance of i1 and i2 and mark i for removing
            if max(imp1, imp2) < maximp:
                importance[i1] = imp1
                importance[i2] = imp2
                keep[i] = False
                marked = True
            j += 1

        # Execute remove the point:	
        self._r = np.compress(keep, self._r)
        self._y = np.compress(keep, self._y, axis=0)
        self._dy = np.compress(keep, self._dy, axis=0)




    def _eval(self, r):
        """
        Evaluation at a given point

        Parameters
        ----------
        r : float
            The point of evaluation.
        """

        self._check_range(r)
        i = self._get_segment_index(r)
        return self._segment_approx_value(r, i, i+1)


    def der(self, r):
        """
        Derivative at a given point.

        Parameters
        ----------
        r : float
            The point of evaluation.
        """

        self._check_range(r)
        i = self._get_segment_index(r)		
        return self._segment_approx_der(r, i, i+1)


    def integ(self, r1=None, r2=None):
        """
        Integral over the range r1 and r2.

        Parameters
        ----------
        r1 : float, optional
            The Lower limit of the range. 
            If `None`, the lower limit of the sample is used.

        r2 : float, optional
            The upper limit of the range.
            If `None`, the upper limit of the sample is used.

        Returns
        -------
            The integral of all components of the function
        
        """
        if r1 == None:
            r1 = self._r[0]

        if r2 == None:
            r2 = self._r[-1]

        try:
            self._check_range(r1)
            self._check_range(r2)
        except ValueError:
            raise ValueError('Integration range ({} to {})'.format(r1, r2) 
                           + 'exceeds the sample of the function ({} to {}).'.format(self._r[0], self._r[1]))

        i1 = self._get_segment_index(r1)
        i2 = self._get_segment_index(r2)
        X1 = (r1 - self._r[i1])/(self._r[i1+1] - self._r[i1])
        X2 = (r2 - self._r[i2])/(self._r[i2+1] - self._r[i2])
        if (i2 == i1):
            result = self._segment_approx_int(i2, X2) - self._segment_approx_int(i2, X1)
        else:
            result = (self._segment_approx_int(i1) - self._segment_approx_int(i1, X1) 
                     + self._segment_approx_int(i2, X2))
            for i in range(i1+1, i2):
                result += self._segment_approx_int(i)
        
        return result

