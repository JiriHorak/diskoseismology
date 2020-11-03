import math
import cmath 
import numpy as np
from scipy import optimize 

from . import functions as func
from . import base


class Disk(base.DiskParam):
	'''
	Thin disk in Paczynski-Wiita potential with constant radial density profile.

	Atributes
	---------
	cs : float
		constant sound speed [c]

	rin : float
		Inner edge of the disk. if rin is less than 1, rin is set to ISCO.

	rout : float
		Outer edge of the disk.
	'''
	
	def __init__(self, cs, rin=0, rout=100):
		'''
		Initialization.

		'''		
		if rin < 1:
			rin = 6 + 1e-10
		super().__init__(cs, func.IntervalFunction(lambda r: 1, rin, float('inf')), rin, rout)

		
	def Omega(self, r):
		return 1.0/(math.sqrt(r)*(r-2))
	
	def kappa(self, r):
		# if r < self.rin:
		# 	raise RuntimeWarning('kappa is not real bellow disk.rin (r = {}).'.format(r))
		return cmath.sqrt((r - 6)/((r-2)**3 * r))

	def dkappa(self, r):
		return (-3*(4 + (-8 + r)*r))/(2.*math.sqrt(-6 + r)*(-2 + r)**2.5*r**1.5)
		
	def r_kappa_max(self):
		return 2*(2 + math.sqrt(3))
			
	def Omegav(self, r):
		return self.Omega(r)

	def dOmegav(self, r):
		return -(3*r - 2)/(2 * r**1.5 * (r - 2)**2)

	def frequencies(self, r):
		return [self.Omega(r), self.kappa(r), self.Omegav(r)]
		

	def vortensity(self, r):
		"""
		Profile of the vortensity

		Parameter
		---------
		r : float
			radial coordinate [M]

		Returns
		-------
		zeta : float

		"""
		return self.kappa(r)**2/(2*self.Omega(r)*self.Sigma(r))

	def rvort(self):
		return optimize.minimize_scalar(lambda r: -abs(self.vortensity(r)), bounds=(self.rin, self.rout), method='bounded').x

	def Omega_rvort(self):
		return self.Omega(self.rvort())

