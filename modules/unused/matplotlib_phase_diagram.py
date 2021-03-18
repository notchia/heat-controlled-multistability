# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:57:28 2020

@author: Lucia
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

import pltformat


''' Plot 3D phase diagram for given ratio index'''
def plot_phase_boundaries(boundaries, boundaryVals,
                          T_range, h_range, ir, thetaL_range,
                          minima, phases, angleT_vals, angle0_vals,
                          T_plane=0):
    if T_plane:
        iT = np.argwhere(T_range > T_plane)[0][0]
    colors = ['xkcd:light red', 'xkcd:red', 'xkcd:apple green', 'xkcd:dark lime green', 'xkcd:dark lime green', 'xkcd:electric blue','xkcd:electric blue']
    colors3 = ['xkcd:light red', 'xkcd:apple green', 'xkcd:electric blue']
    customcm = col.ListedColormap(colors3[::-1])
      
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-105)
    x = 1000*h_range
    y = np.degrees(thetaL_range)
    z = T_range
    X, Y = np.meshgrid(x, y)

    x_below = np.full_like(boundaryVals, np.nan)
    x_above = np.full_like(boundaryVals, np.nan)
    y_below = np.full_like(boundaryVals, np.nan)
    y_above = np.full_like(boundaryVals, np.nan)
    z_below = np.full_like(boundaryVals, np.nan)
    z_above = np.full_like(boundaryVals, np.nan)
    nPoints = max(boundaries.shape)
    for pt in range(nPoints):
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        if loc[0] <= iT:
            x_below[pt] = x[loc[1]] 
            y_below[pt] = y[loc[2]] 
            z_below[pt] = z[loc[0]] 
        else:
            x_above[pt] = x[loc[1]] 
            y_above[pt] = y[loc[2]] 
            z_above[pt] = z[loc[0]]             

    iL = np.argwhere(thetaL_range > 0)[0][0]

    x_rb = np.full_like(boundaryVals, np.nan)
    x_rt = np.full_like(boundaryVals, np.nan)
    y_rb = np.full_like(boundaryVals, np.nan)
    y_rt = np.full_like(boundaryVals, np.nan)
    z_rb = np.full_like(boundaryVals, np.nan)
    z_rt = np.full_like(boundaryVals, np.nan)
    for pt in range(nPoints):
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        if loc[2] >= iL:
            if loc[0] <= iT:
                x_rb[pt] = x[loc[1]] 
                y_rb[pt] = y[loc[2]] 
                z_rb[pt] = z[loc[0]] 
            else:
                x_rt[pt] = x[loc[1]] 
                y_rt[pt] = y[loc[2]] 
                z_rt[pt] = z[loc[0]]  
                
    diagram = ax.scatter(x_below, y_below, z_below, c=boundaryVals, 
                         marker='.', cmap=col.ListedColormap(colors), zorder=1)        
    if T_plane:
        ax.plot_surface(X, Y, np.full_like(Y, T_range[iT]), rstride=1, cstride=1, facecolors=customcm(np.transpose((minima[iT,:,ir,:])-1)/3), alpha=0.075, zorder=2)
#        diagram = ax.scatter(x_above, y_above, z_above, marker='.', c=boundaryVals, cmap=col.ListedColormap(colors), zorder=3)
    diagram = ax.scatter(x_rt, y_rt, z_rt, marker='.', c=boundaryVals, cmap=col.ListedColormap(colors), zorder=3)        

    ax.set_xlabel(r'$h$ (mm)', fontsize=14)
    ax.set_ylabel(r'$\theta_L$ (deg)', fontsize=14, rotation=150)
    ax.set_zlabel(r'$T$ ($^{\circ}$C)', fontsize=14, rotation=60)
    bar = fig.colorbar(diagram, aspect=10)