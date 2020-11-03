from math import sqrt
from numpy import sign

from . import functions as func
from . import base


class Disk(base.DiskParam):
	'''
	Thin disk in Kerr spacetime with constant radial density profile.

	Atributes
	---------
	cs : float
		constant sound speed [c]

	a : float
		Kerr spin parameter

	rin : float
		Inner edge of the disk. if rin is less than 1, rin is set to ISCO.

	rout : float
		Outer edge of the disk.
	'''
	
	def __init__(self, cs, a, rin=0, rout=100):
		'''
		Initialization.

		'''		
		if rin < 1:
			Z1 = 1 + pow(1. - a*a, 1./3.)*(pow(1. + a, 1./3.) + pow(1. - a, 1./3.))
			Z2 = (3.*a*a + Z1*Z1)**0.5
			rin = 3. + Z2 - sign(a)*((3. - Z1)*(3. + Z1 + 2*Z2))**0.5 + 1e-10

		self._a = a
		super().__init__(cs, func.IntervalFunction(lambda r: 1, rin, float('inf')), rin, rout)

		
	def Omega(self, r):
		return 1.0/(r**1.5 + self._a)

	
	def kappa(self, r):
		return self.Omega(r)*(1.0 - 6.0/r + 8.0*self._a/(r**1.5) - 3.0*(self._a/r)**2)**0.5
	
	
	def dkappa(self, r):
		a = self._a
		return 3*(2*a**3 + 2*a*(1 - 6*r)*r - (-8 + r)*r**2.5 + a**2*sqrt(r)*(-4 + 5*r))/(
				2.*r**3*sqrt((-3*a**2 + 8*a*sqrt(r) + (-6 + r)*r)/r**2)*(a + r**1.5)**2)   
		
		
	def r_kappa_max(self):
		from scipy.optimize import bisect
		a = self._a
		f = lambda r: 2*a**3 + 2*a*(1 - 6*r)*r - (-8 + r)*r**2.5 + a**2*r**0.5*(-4 + 5*r) 
		return bisect(f, 2, 12)
			
		
	def Omegav(self, r):
		return self.Omega(r)*(1.0 - 4.0*self._a/r**1.5 + 3.0*(self._a/r)**2)**0.5


	def frequencies(self, r):
		return [self.Omega(r), self.kappa(r), self.Omegav(r)]
		
