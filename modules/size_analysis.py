# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:13:35 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import matplotlib.colors as col
import PDMS_LCE_dipole_statics as model




def analyze_diagonal_dependence(h_val, ratio_val, thetaL_val, a_range, T_range):
    ''' Map stability along three parameter axes: thickness, LCE:PDMS ratio, initial angle '''
    q = np.radians(np.arange(-45.0,45.5,0.05))# Could increase the resolution here...
    
    n_a = max(a_range.shape)
    n_T = max(T_range.shape)
    angleT_array = np.zeros((n_a,n_T))
    angle0_array = np.zeros((n_a,n_T))
    numMin_array = np.zeros((n_a,n_T))
    phase_array = np.zeros((n_a,n_T))
   
    for i_a in range(n_a):
        sample = model.MetamaterialModel(h_total=h_val, ratio=ratio_val,
                                         thetaL=thetaL_val, a=a_range[i_a])
        for i_t in range(n_T):
            sample.update_T(T_range[i_t])
            U_total, U_m, U_k, U_lim = sample.calculate_total_energy(q)
            thetaT = sample.thetaT
            
            numMin, diff, barrier, phase = sample.analyze_energy_local_extrema(U_total, q, thetaL_val)

            phase_array[i_a,i_t] = phase
            numMin_array[i_a,i_t] = numMin
            angleT_array[i_a,i_t] = np.degrees(thetaT)
            angle0_array[i_a,i_t] = np.degrees(thetaT + thetaL_val)

    return numMin_array#, phase_array, angleT_array, angle0_array

def plot_a_T_relation(T_range, a_range, h_val, r_val, thetaL_val, min_array,
                      saveFlag=True, figdir=''):
    ''' Plot the stability relationship bewteen diagonal length $a$ and temperature $T$'''
    colors = ['xkcd:light red', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:electric blue', 'xkcd:blue purple', 'xkcd:electric blue']
    
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    X = 1000*a_range
    Y = T_range
    Z = np.transpose(min_array)
    diagram = ax.imshow((Z).astype(np.uint8), extent=[X.min(), X.max(), Y.min(), Y.max() ], origin='lower', aspect='auto', cmap=cm.viridis)#col.ListedColormap(colors))           
    
    # Formatting
    plt.title(r'$h$ = {0}, $\theta_0$ = {1}$^\circ$, $r$ = {2}'.format(h_val, thetaL_val, r_val), size=18)
    plt.xlabel(r'Diagonal length $a$ (mm)')
    plt.ylabel(r'Temperature $T$ ($^\circ$C)')
       
    #plt.ylim([np.amin(angle0_vals), np.amax(angle0_vals)])
    if saveFlag:
        title = 'angle_scaled_0_{0}_r_{1:0.2f}'.format(int(T_range[iT]), ratio_range[ir])
        plt.savefig(os.path.join(figdir,'{0}.png'.format(title)),dpi=200)
        plt.savefig(os.path.join(figdir,'{0}.svg'.format(title)))

    return


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw")
    cleandir = os.path.join(cwd,"data/cleaned")
    tmpdir = os.path.join(cwd,"tmp")
    savedir = os.path.join(cwd,"results/figures_from_script")


    h = 0.0012
    r = 0.5
    thetaL = 0.0
    a_range = np.arange(0.015, 0.0255, 0.0005)
    T_range = np.arange(20.0, 101.0, 1.0)
    
    sample = model.analyze_parameter_energy(h, r, 25.0, thetaL)
    
    numMin_array = analyze_diagonal_dependence(h_val=h,ratio_val=r,thetaL_val=thetaL,
                                               a_range=a_range, T_range=T_range)
    
    plot_a_T_relation(T_range, a_range, h, r, thetaL, numMin_array,
                      saveFlag=True, figdir=savedir)