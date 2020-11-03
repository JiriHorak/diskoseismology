"""
Module providing graphical output.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from . import functions as func
from . import base

def complex2str(z, ndigit):
	raise NotImplementedError

 
def propagation_diagram(ax, mode, radii, plot_grid=True, **kwargs):

    disk = mode.disk
    
    ccolor, mcolor = 'black', 'black'
    rlab = 0.5*(radii[0] + radii[-1])
    
    if 'ccolor' in kwargs:
        ccolor = kwargs['ccolor']
    if 'mcolor' in kwargs:
        mcolor = kwargs['mcolor']
    if 'rlab' in kwargs:
        rlab = kwargs['rlab']

    
    # limiting curves:
    ilr = lambda r: mode.m*disk.Omega(r).real - disk.kappa(r).real
    olr = lambda r: mode.m*disk.Omega(r).real + disk.kappa(r).real
    ivr = lambda r: mode.m*disk.Omega(r).real - math.sqrt(mode.n)*disk.Omegav(r).real
    ovr = lambda r: mode.m*disk.Omega(r).real + math.sqrt(mode.n)*disk.Omegav(r).real
    cr  = lambda r: mode.m*disk.Omega(r).real

    # labels of the curves:
    if mode.m == 0:
        lab_mOmega = ''
    elif mode.m == 1:
        lab_mOmega = r'\Omega'
    elif mode.m == -1:
        lab_mOmega = r'-\Omega'
    else:
        lab_mOmega = r'{}\Omega'.format(mode.m)
    
    if mode.n == 0:
        lab_nOmegav = ''
    elif mode.n == 1:
        lab_nOmegav = r'\Omega_\perp'
    elif (int(math.sqrt(mode.n)))**2 == mode.n:
        lab_nOmegav = r'{}'.format(int(math.sqrt(mode.n)))
    else:
        lab_nOmegav = r'\sqrt{' + str(mode.n) + r'}\Omega_\perp'
    
    ilr_label = r'{} - \kappa'.format(lab_mOmega)
    ivr_label = r'{} - {}'.format(lab_mOmega, lab_nOmegav)
    if mode.m != 0:
        olr_label = r'{} + \kappa'.format(lab_mOmega)
        ovr_label = r'{} + {}'.format(lab_mOmega, lab_nOmegav)
    else:
        olr_label = r'\kappa'
        ovr_label = lab_nOmegav
    
    # making the array of resonance dictionaries and sort it according to r:
    resdicts = []
    for r in mode.LRs:
        if (r == None) or (r < radii[0]) or (r > radii[-1]):
            continue
        if mode.omegat(r).real < 0:
            resdicts.append({'r': r, 'type': 'ILR', 'curve': ilr, 'crvlabel': ilr_label, 'reslabel': r'$r_\mathrm{ILR}$'})
        else:
            resdicts.append({'r': r, 'type': 'OLR', 'curve': olr, 'crvlabel': olr_label, 'reslabel': r'$r_\mathrm{OLR}$'})
    for r in mode.VRs:
        if (r == None) or (r < radii[0]) or (r > radii[-1]):
            continue
        if mode.omegat(r).real < 0:
            resdicts.append({'r': r, 'type': 'IVR', 'curve': ivr, 'crvlabel': ivr_label, 'reslabel': r'$r_\mathrm{IVR}$'})
        else:
            resdicts.append({'r': r, 'type': 'OVR', 'curve': ovr, 'crvlabel': ovr_label, 'reslabel': r'$r_\mathrm{OVR}$'})
    if (mode.CR != None) and (r > radii[0]) and (r < radii[-1]):
        resdicts.append({'r': r, 'type': 'CR', 'curve': cr, 'crvlabel': lab_mOmega, 'reslabel': r'$r_\mathrm{CR}$'})
    
    resdicts = sorted(resdicts, key = lambda item: item['r'])
    
    # draw them (dont draw twice):
    drawn = {'ILR': False, 'OLR': False, 'IVR': False, 'OVR': False, 'CR': False}
        
    for res in resdicts:
        if not drawn[res['type']]: 
            freq = [res['curve'](r) for r in radii]
            ax.plot(radii, freq, color=ccolor)
            drawn[res['type']] = True
    
    ax.plot([res['r'] for res in resdicts], len(resdicts)*[mode.omega.real], marker='.', ls='', color=mcolor)

    # draw the oscillation propagation:
    R = [radii[0]] + [res['r'] for res in resdicts] + [radii[-1]]
    for i in range(len(R)-1):
        if mode.k2(0.5*(R[i] + R[i+1]), useim=False).real > 0:
            p = matplotlib.rcParams['path.sketch']
            matplotlib.rcParams['path.sketch'] = (2, 15, 1)
            ax.plot([R[i], R[i+1]], 2*[mode.omega.real], ls='-', color=mcolor)
            matplotlib.rcParams['path.sketch'] = p
        else:
            ax.plot([R[i], R[i+1]], 2*[mode.omega.real], ls=':', color=mcolor)
       
    
    ax.set_xlabel(r'Radius $r[M]$')
    ax.set_ylabel(r'Frequency $\omega[M^{-1}]$')
    ax.set_xlim(radii[0], radii[-1])
    
    if plot_grid:
        axtw = ax.twiny()
        axtw.set_xlim(ax.get_xlim())
        axtw.set_xticks([res['r'] for res in resdicts])
        axtw.set_xticklabels([res['reslabel'] for res in resdicts])
        axtw.grid(ls=':', lw=1, color='gray')

    if rlab != None:
        drawn = {'ILR': False, 'OLR': False, 'IVR': False, 'OVR': False, 'CR': False}
        eps = 0.1
        for res in resdicts:
            if not drawn[res['type']]: 
                p1 = ax.transData.transform_point((rlab-eps, res['curve'](rlab-eps)))
                p2 = ax.transData.transform_point((rlab+eps, res['curve'](rlab+eps)))
                angle = 180.0 * math.atan2(p2[1] - p1[1], p2[0] - p1[0])/math.pi
                if angle > 0:
                    horalig = 'right'
                else:
                    horalig = 'left' 
                
                ax.text(rlab, res['curve'](rlab), r'$' + res['crvlabel'] + r'$', rotation=angle, 
                        ha=horalig, va='bottom', fontsize='large', color=ccolor)
                drawn[res['type']] = True


def eigenfunction_plot(panels, mode, W, radii=[], components=[0,1,2,3], norm=[1,1,1,1], units='', reslabels=True,
                       ylabels=[r'$h$', r'$v^r$', r'$v^\phi$', r'$v^z$'], **kwargs):
    """
    Plots various components of the solution. 

    The figure consists several	panels, each of them for one solution variable. 
    The panels should be iniciated outside the function.

    Parameters:
    -----------
    panels : axes 
        Panels to be filled
    mode : ModeParam
        Parameters of the mode (instance of the ModeParam class)
    W : functions.IntervalFunction
        Eigenfunction or the response of the disk
    """

    recolor, imcolor = 'black', 'black'
    rels, imls = '-', '-'
    
    if 'recolor' in kwargs:
        recolor = kwargs['recolor']
    if 'imcolor' in kwargs:
        imcolor = kwargs['imcolor']
    if recolor == imcolor:
        imls = '--'
        
    if len(radii) == 0:
        radii = W.sample_r
        
    r1 = max(W.rmin, radii[0])
    r2 = min(W.rmax, radii[-1])

    resonances = mode.LRs + mode.VRs + [mode.CR]
    resonance_labels = [r'$r_\mathrm{ILR}$', r'$r_\mathrm{OLR}$', r'$r_\mathrm{IVR}$', 
                        r'$r_\mathrm{OVR}$', r'$r_\mathrm{CR}$']

    rticks = []
    rlabels = []
    for r, label in zip(resonances, resonance_labels):
        if not(r == None) and ((r1 <= r) and (r <= r2)):
            rticks.append(r)
            if reslabels:
                rlabels.append(label)

    vals = np.array([W(r) for r in radii]) 
    for i, comp in enumerate(components):
        panels[i].plot(radii, norm[comp]*vals[:,comp].real, ls=rels, label='Re', color=recolor)
        panels[i].plot(radii, norm[comp]*vals[:,comp].imag, ls=imls, label='Im', color=imcolor)
        panels[i].set_xlim(left=r1, right=r2)
        if i < len(components)-1:
            panels[i].set_xticklabels([])
        panels[i].axhline(0, ls='--', color='black', lw=0.7)
        panels[i].set_ylabel(ylabels[comp])
        for r in rticks:
            panels[i].axvline(r, ls=':', color='gray', lw=1)

    panels[components[-1]].set_xlabel(r'Radius $r [M]$')
    #panels[0].annotate((r'$m={}$' + '\n' + r'$n={}$').format(mode.m, mode.n), xy=(0.99, 0.95), 
    #                    xycoords='axes fraction', va='top', ha='right')

    raxis2 = panels[0].twiny()
    raxis2.set_xlim(panels[0].get_xlim())
    if reslabels:
        raxis2.set_xticks(rticks)
    else:
        raxis2.set_xticks([])
        
    raxis2.set_xticklabels(rlabels)

    

def eigenfunction_multiplot(modes, Ws, radii=[], components=[0,1,2,3], norm=[1,1,1,1],  
                            ylabels=[r'$h$', r'$v^r$', r'$v^\phi$', r'$v^z$']):

	n = len(modes)
	if len(Ws) != n:
		raise ValueError('Different number of modes and eigenfunctions.')

	fig, panels = plt.subplots(4, n, figsize=(6*n, 6), sharex=True)

	for i, (mode, W) in enumerate(zip(modes, Ws)):
		if n > 1:
			eigenfunction_plot(panels[:,i], mode, W, radii, components, norm, ylabels)
		else:
			eigenfunction_plot(panels, mode, W, radii, components, norm, ylabels)
		for j, label in enumerate(ylabels):
			if n > 1:
				panels[j,i].set_ylabel(label)
			else:
				panels[j].set_ylabel(label)

	fig.tight_layout(w_pad=3, h_pad=0.2)
	fig.align_ylabels(panels)
	fig.align_xlabels(panels)

	return fig, panels




def residuals_plot(panels, mode, W, F, components=[0,1,2,3], norm=None, rel=True):
	"""
	Plots various the residual of the solution in the differential equations. 
	
	The figure consists of several	panels, each of them for one solution variable. 
	The panels should be iniciated outside the function.
	
	Parameters:
	-----------
	panels : axes 
		Panels to be filled
	mode : ModeParam
		Parameters of the mode (instance of the ModeParam class)
	W : functions.IntervalFunction
		Eigenfunction or the response of the disk
	F : functions.IntervalFunction
		Forcing terms
	"""

	r1 = W.rmin
	r2 = W.rmax

	radii = W.sample_r

	# calculate normalize residuals
	# res[i] are sampled residuals of i-th component of the solution, 
	# i.e., res[0] corresponds to the enthalpy, res[1] is vr, etc..

	if norm is None:
		res = np.array([1/mode.omegat(r)*base.governing_equation_residuals(mode, r, W(r), W.der(r), F) for r in radii])
	else:
		res = np.array([norm(r)*base.governing_equation_residuals(mode, r, W(r), W.der(r), F) for r in radii])

	val = W.sample_y

	if rel:
		plot_res = np.abs(res)/(np.abs(val)+1e-15)
	else:
		plot_res = np.abs(res)

	resonances = mode.LRs + mode.VRs + [mode.CR]
	resonance_labels  = [r'$r_\mathrm{ILR}$', r'$r_\mathrm{OLR}$', r'$r_\mathrm{IVR}$', 
	                     r'$r_\mathrm{OVR}$', r'$r_\mathrm{CR}$']
	
	rticks = []
	rlabels = []
	for r, label in zip(resonances, resonance_labels):
		if not(r == None) and ((r1 <= r) and (r <= r2)):
			rticks.append(r)
			rlabels.append(label)
	
	ylabels = [r'$|\Delta_h|\,/\,|\tilde{\omega}h|$', r'$|\Delta_r|\,/\,|\tilde{\omega} v^r|$', 
	           r'$|\Delta_\phi|\,/\,|\tilde{\omega}v^\phi|$', r'$|\Delta_z|\,/\,|\tilde{\omega}v^z|$']
	
	for i, comp in enumerate(components):	
		panels[i].plot(radii, plot_res[:,comp], ls='-', label='abs')
		#panels[i].plot(radii, res[comp].real, ls='-', color='blue', label='Re')
		#panels[i].plot(radii, res[comp].imag, ls='-', color='grey', label='Im')
		panels[i].set_xlim(left=r1, right=r2)
		panels[i].set_ylim(bottom=0)
		panels[i].axhline(0, ls='--', color='black', lw=0.7)
		panels[i].set_ylabel(ylabels[comp])
		for r in rticks:
			panels[i].axvline(r, ls='--', color='black', lw=0.7)

	panels[components[-1]].set_xlabel(r'$r [M]$')
	raxis2 = panels[0].twiny()
	raxis2.set_xlim(panels[0].get_xlim())
	raxis2.set_xticks(rticks)
	raxis2.set_xticklabels(rlabels)

	
	
def plot_radial_resonances(ax, mode1, mode2, mode12, radii, colors=None, labels=None):
    
    if colors == None: 
        colors = 3*[None]
    if labels == None:
        labels = 3*['']
    
    
    for i, mode in enumerate([mode1, mode2, mode12]):
        for (r1, r2) in mode.propagation_regions():
            first_line = True
            if (r2 > radii[0]) and (r1 < radii[-1]):
                rad = np.linspace(max(r1, radii[0]), min(r2, radii[-1]))
                k = [math.sqrt(abs(mode.k2(r).real)) for r in rad]
                ax.plot(rad, k, label=labels[i], color=colors[i])
                if first_line:
                    labels[i] = ''
    
    for r in base.radial_resonances(mode1, mode2, mode12):
        ax.axvline(r, ls=':', color='gray')
    
    ax.set_xlabel(r'Radius $r [M]$')
    ax.set_ylabel(r'Wavevector $k [M^{-1}]$')
                
    ax.set_xlim(radii[0], radii[-1])
    ax.set_ylim(bottom=0)