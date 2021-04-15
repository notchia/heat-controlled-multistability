"""
Define MetamaterialModel class and h-T phase space analysis for curvature
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time
from datetime import datetime
import os
import sys
import matplotlib.colors as col

from modules import metamaterialmodel as metamaterial


#%%
def analyze_main_parameter_composite_phases(thetaL, h_range=1e-3*np.arange(0.5,2.1,0.01),
                             r_range=np.radians(np.arange(0.0, 1.0, 0.01)),
                             T_range=np.arange(25.0,75.1,25.0),
                             b=[], k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                             bilayerDict={}):
    """ For a given composite with ratio r, compute number of minima across isotherms """
    n_h = len(h_range)
    n_r = len(r_range)
    n_T = len(T_range)
    angleT_array = np.zeros((n_h, n_r, n_T))
    angle0_array = np.zeros((n_h, n_r, n_T))
    numMin_array = np.zeros((n_h, n_r, n_T))
    phases_array = np.zeros((n_h, n_r, n_T))    

    t0 = time.perf_counter()
    for i_h in range(n_h):
        sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} h values at {2:0.1f} minutes".format(i_h + 1, n_h, (time.perf_counter()-t0)/60.0))
        sys.stdout.flush()
        h = h_range[i_h]
        for i_r in range(n_r):
            r = r_range[i_r]
            sample = metamaterial.MetamaterialModel(h, r, thetaL, T=25.0,
                                                    k_sq=k_sq, m=m, p_lim=p_lim,
                                                    **bilayerDict)
            for i_T in range(n_T):
                sample.update_T(T_range[i_T])              
                
                numMin_array[i_h, i_r, i_T] = sample.numMin
                phases_array[i_h, i_r, i_T] = sample.phases
                angleT_array[i_h, i_r, i_T] = np.degrees(sample.hinge.thetaT)
                angle0_array[i_h, i_r, i_T] = np.degrees(sample.total_angle)
                    
    return numMin_array, phases_array, angleT_array, angle0_array
    
    
def analyze_composite_phases(r, h_range=1e-3*np.arange(0.5,2.1,0.01),
                             thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                             T_range=np.arange(25.0,75.1,25.0),
                             b=[], k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                             bilayerDict={}):
    """ For a given composite with ratio r, compute number of minima across isotherms """
    n_h = len(h_range)
    n_L = len(thetaL_range)
    n_T = len(T_range)
    angleT_array = np.zeros((n_h, n_L, n_T))
    angle0_array = np.zeros((n_h, n_L, n_T))
    numMin_array = np.zeros((n_h, n_L, n_T))
    phases_array = np.zeros((n_h, n_L, n_T))    

    t0 = time.perf_counter()
    for i_h in range(n_h):
        sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} h values at {2:0.1f} minutes".format(i_h + 1, n_h, (time.perf_counter()-t0)/60.0))
        sys.stdout.flush()
        h = h_range[i_h]
        for i_L in range(n_L):
            thetaL = thetaL_range[i_L]
            sample = metamaterial.MetamaterialModel(h, r, thetaL, T=25.0, k_sq=k_sq, m=m, p_lim=p_lim, **bilayerDict)
            for i_T in range(n_T):
                sample.update_T(T_range[i_T])              
                
                numMin_array[i_h, i_L, i_T] = sample.numMin
                phases_array[i_h, i_L, i_T] = sample.phases
                angleT_array[i_h, i_L, i_T] = np.degrees(sample.hinge.thetaT)
                angle0_array[i_h, i_L, i_T] = np.degrees(sample.total_angle)
                    
    return numMin_array, phases_array, angleT_array, angle0_array
 
    
def run_composite_phase_boundary_analysis(r_val,
                       h_range=1e-3*np.arange(0.5,2.1,0.01),
                       thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                       T_range=np.arange(25.0,105.0,25.0),
                       k_sq=0.0, m=0, limFlag='exp', p_lim=[],
                       bilayerDict={},
                       savedir='', datestr='', closeFlag=True):
    """ Full parameter space analysis for h-theta_L-T given r """
    
    if datestr == '':
        datestr = datetime.today().strftime('%Y%m%d')
        
    parameter_file = os.path.join(savedir, '{0}_parameters.csv'.format(datestr))
    metamaterial_file = os.path.join(savedir, '{0}_metamaterialObject.npy'.format(datestr))
    minima_file = os.path.join(savedir, '{0}_r{1:.3f}_minima.npy'.format(datestr, r_val))
    phases_file = os.path.join(savedir, '{0}_r{1:.3f}_phases.npy'.format(datestr, r_val))
    thetaT_file = os.path.join(savedir, '{0}_r{1:.3f}_thetaT.npy'.format(datestr, r_val))
    theta0_file = os.path.join(savedir, '{0}_r{1:.3f}_theta0.npy'.format(datestr, r_val))
    
    # Load existing data if previously run; otherwise, run analysis
    try:
        print("Try loading previous parameter-space analysis...")
        #r_val, T_range, h_range, thetaL_range = import_parameters(parameter_file)
        minima = np.load(minima_file)
        phases = np.load(phases_file)
        thetaT = np.load(thetaT_file)
        theta0 = np.load(theta0_file)
        sampleModel = np.load(metamaterial_file, allow_pickle=True)
        print("\tLoaded parameters, minima, phases, theta_T, and theta_0")
    except IOError:
        export_parameters(parameter_file, r_val, T_range, h_range, thetaL_range) #UPDATE THIS FUNCTION
        sampleModel = metamaterial.MetamaterialModel(1e-3, r_val, 0.0, T=25.0,
                                                     k_sq=k_sq, m=m, p_lim=p_lim,
                                                     **bilayerDict)
        np.save(metamaterial_file, sampleModel)
        print("\tRunning new parameter-space analysis...")
        minima, phases, thetaT, theta0 = analyze_composite_phases(
            r_val, h_range=h_range, thetaL_range=thetaL_range, T_range=T_range,
            k_sq=k_sq, m=m, limFlag=limFlag, p_lim=p_lim,
            bilayerDict=bilayerDict)
        for T in T_range:
            plot_isotherm(r_val, T, phases, theta0, h_range=h_range, thetaL_range=thetaL_range,
                          T_range=T_range, savedir=savedir, closeFlag=closeFlag)
        print("\tSaving parameters, minima, phases, theta_T, and theta_0")
        np.save(minima_file, minima, allow_pickle=False)
        np.save(phases_file, phases, allow_pickle=False)
        np.save(thetaT_file, thetaT, allow_pickle=False)
        np.save(theta0_file, theta0, allow_pickle=False)

    if datestr == '':
        return minima, phases, thetaT, theta0
    else:
        paramDict = {'r':r_val, 'T':T_range, 'h':h_range, 'theta_L':thetaL_range}
        return minima, phases, thetaT, theta0, paramDict, sampleModel


def run_main_parameter_phase_boundary_analysis(thetaL_val,
                       h_range=1e-3*np.arange(0.5,2.1,0.01),
                       r_range=np.radians(np.arange(0.0,1.0,0.01)),
                       T_range=np.arange(25.0,105.0,25.0),
                       k_sq=0.0, m=0, limFlag='exp', p_lim=[],
                       bilayerDict={},
                       savedir='', datestr='', closeFlag=True):
    """ Full parameter space analysis for h-r-T given theta_L """
    
    if datestr == '':
        datestr = datetime.today().strftime('%Y%m%d')
        
    parameter_file = os.path.join(savedir, f'{datestr}_mainParams_parameters.csv')
    metamaterial_file = os.path.join(savedir, f'{datestr}_mainParams_metamaterialObject.npy')
    minima_file = os.path.join(savedir, f'{datestr}_mainParams_minima.npy')
    phases_file = os.path.join(savedir, f'{datestr}_mainParams_phases.npy')
    thetaT_file = os.path.join(savedir, f'{datestr}_mainParams_thetaT.npy')
    theta0_file = os.path.join(savedir, f'{datestr}_mainParams_theta0.npy')
    
    # Load existing data if previously run; otherwise, run analysis
    try:
        print("Try loading previous parameter-space analysis...")
        #thetaL_val, T_range, h_range, r_range = import_parameters(parameter_file)
        minima = np.load(minima_file)
        phases = np.load(phases_file)
        thetaT = np.load(thetaT_file)
        theta0 = np.load(theta0_file)
        sampleModel = np.load(metamaterial_file, allow_pickle=True)
        print("\tLoaded parameters, minima, phases, theta_T, and theta_0")
    except IOError:
        export_parameters(parameter_file, 0.0, T_range, h_range, r_range) #UPDATE THIS FUNCTION
        sampleModel = metamaterial.MetamaterialModel(1e-3, 0.25, 0.0, T=25.0,
                                                     k_sq=k_sq, m=m, p_lim=p_lim, **bilayerDict)
        np.save(metamaterial_file, sampleModel)
        print("\tRunning new parameter-space analysis...")
        minima, phases, thetaT, theta0 = analyze_main_parameter_composite_phases(
            thetaL=0.0, h_range=h_range, r_range=r_range, T_range=T_range,
            k_sq=k_sq, m=m, limFlag=limFlag, p_lim=p_lim,
            bilayerDict=bilayerDict)
        for T in T_range:
            plot_main_parameter_isotherm(thetaL_val, T, phases, theta0,
                                         h_range=h_range, r_range=r_range, T_range=T_range,
                                         savedir=savedir, closeFlag=closeFlag)
        print("\tSaving parameters, minima, phases, theta_T, and theta_0")
        np.save(minima_file, minima, allow_pickle=False)
        np.save(phases_file, phases, allow_pickle=False)
        np.save(thetaT_file, thetaT, allow_pickle=False)
        np.save(theta0_file, theta0, allow_pickle=False)

    if datestr == '':
        return minima, phases, thetaT, theta0
    else:
        paramDict = {'r':r_range, 'T':T_range, 'h':h_range, 'theta_L':thetaL_val}
        return minima, phases, thetaT, theta0, paramDict, sampleModel


def analyze_h_T_relation(r, thetaL=0.0, h_range=1e-3*np.arange(0.5,2.1,0.001),
                         T_range=np.arange(20.0,100.1,1.0),
                         k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                         bilayerDict={},
                         saveFlag=False, figdir=''):
    """ For a given composite with ratio r, compute number of minima across isotherms """
    n_h = len(h_range)
    n_T = len(T_range)
    angleT_array = np.zeros((n_h, n_T))
    angle0_array = np.zeros((n_h, n_T))
    numMin_array = np.zeros((n_h, n_T))
    phases_array = np.zeros((n_h, n_T))

    # Find minima, phases, and bending angles ---------------------------------
    t0 = time.perf_counter()
    for i_h in range(n_h):
        sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} h values at {2:0.1f} minutes".format(i_h + 1, n_h, (time.perf_counter()-t0)/60.0))
        sys.stdout.flush()
        h = h_range[i_h]
        sample = metamaterial.MetamaterialModel(h, r, thetaL, T=25.0,
                                                k_sq=k_sq, m=m, p_lim=p_lim, **bilayerDict)
        for i_T in range(n_T):
            sample.update_T(T_range[i_T])              
            
            numMin_array[i_h, i_T] = sample.numMin
            phases_array[i_h, i_T] = sample.phases
            angleT_array[i_h, i_T] = np.degrees(sample.hinge.thetaT)
            angle0_array[i_h, i_T] = np.degrees(sample.total_angle)
                    
    # Plot --------------------------------------------------------------------
    colors = ['xkcd:light red', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:electric blue', 'xkcd:blue purple', 'xkcd:electric blue']
    title = 'h-T_r{1:0.2f}_thetaL{1:0.1f}'.format(r, np.degrees(thetaL))
    fig = plt.figure(title, dpi=200)
    plt.xlabel(r'hinge thickness $h$ (mm)')
    plt.ylabel(r'Temperature ($^\circ$C)')
    ax = fig.gca()

    X = 1000*h_range
    Y = T_range
    Z = np.transpose(phases_array)
    diagram = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', cmap=col.ListedColormap(colors))           
    
    if saveFlag:
        plt.savefig(os.path.join(figdir,"{0}.png".format(title)), dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(title)), transparent=True)   
    plt.close()


def analyze_diagonal_dependence(h_val, ratio_val, thetaL_val,
                                a_range=np.arange(0.010, 0.0255, 0.00005),
                                T_range = np.arange(20.0, 100.1, 0.5),
                                k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                                bilayerDict={},
                                saveFlag=False, figdir=''):
    """ Map stability along three parameter axes: thickness, LCE:PDMS ratio, initial angle """
   
    n_a = max(a_range.shape)
    n_T = max(T_range.shape)
    angleT_array = np.zeros((n_a,n_T))
    angle0_array = np.zeros((n_a,n_T))
    numMin_array = np.zeros((n_a,n_T))
    phases_array = np.zeros((n_a,n_T))
    
    # Find minima, phases, and bending angles ---------------------------------
    t0 = time.perf_counter()
    for i_a in range(n_a):
        sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} h values at {2:0.1f} minutes".format(
            i_a + 1, n_a, (time.perf_counter()-t0)/60.0))
        sys.stdout.flush()
        sample = metamaterial.MetamaterialModel(h_val, ratio_val, thetaL_val,
                                   T=25.0, k_sq=k_sq, m=m,
                                   p_lim=p_lim, limFlag=limFlag,
                                   d=a_range[i_a],
                                   **bilayerDict, showWarnings=False)
        for i_T in range(n_T):
            sample.update_T(T_range[i_T])
            
            numMin_array[i_a, i_T] = sample.numMin
            phases_array[i_a, i_T] = sample.phases
            angleT_array[i_a, i_T] = np.degrees(sample.hinge.thetaT)
            angle0_array[i_a, i_T] = np.degrees(sample.total_angle)

    # Plot --------------------------------------------------------------------
    colors = ['xkcd:electric blue','xkcd:apple green','xkcd:light red']    
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    X = 1000*a_range
    Y = T_range
    Z = np.transpose(numMin_array)
    diagram = ax.imshow((Z).astype(np.uint8), extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', cmap=col.ListedColormap(colors))        
    
    # Formatting
    plt.title(r'$h$ = {0}, $\theta_0$ = {1}$^\circ$, $r$ = {2}'.format(
        h_val, thetaL_val, ratio_val))
    plt.xlabel(r'Diagonal length $a$ (mm)')
    plt.ylabel(r'Temperature $T$ ($^\circ$C)')
    cbar = plt.colorbar(diagram)
    cbar.ax.get_yaxis().set_ticks([])
       
    if saveFlag:
        title = 'diagonal_dependence_h{0:.2f}_r{1:.2f}_thetaL{2:.1f}'.format(
            h_val, ratio_val, thetaL_val)
        plt.savefig(os.path.join(figdir,'{0}.png'.format(title)),dpi=200)
        plt.savefig(os.path.join(figdir,'{0}.svg'.format(title)))

    return


def find_3D_phase_boundaries(r, h_range=1e-3*np.arange(0.5,2.1,0.01),
                             thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                             T_range=np.arange(25.0,105.0,25.0),
                             minima=[], phases=[], angleT_vals=[], angle0_vals=[]):   
    """ Find locations of phase boundaries for a given composite
        With h, thetaL, T ranges of length N_h, N_thetaL, N_T
        Returns:
            boundaries      [N, 3]? array of points on boundary
            boundaryVals    [N]
            boundaryData    [N, 4]? """

    # Find where changes in phase occur *along temperature axis*
    diffs = np.diff(phases)
    boundaries = np.argwhere(diffs != 0)
    print(boundaries.shape)
    N = max(boundaries.shape) # Number of points found located at boundary
    boundaryVals = np.zeros(N)
    for pt in range(N):
        # Get h, thetaL, T, and phase value [0-6] at boundary point
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        val = phases[loc[0],loc[1],loc[2]]
        # Find maximum phase difference between boundary point and all nearest
        # neighbors *in isotherm* (ignore points at end of range)
        max_diff = 0
        check = [-1,0,1]
        vals = np.zeros(9,dtype=int)
        if not ((-1 in loc) or loc[0] == phases.shape[0]-1 or loc[1] == phases.shape[1]-1 or loc[2] == phases.shape[2]-1):
            count = 0
            for i in check:
                for j in check:
                    val_temp = phases[loc[0]+i,loc[1]+j,loc[2]]
                    val_diff = np.abs(val - val_temp)
                    if val_diff > max_diff:
                        max_diff = val_diff
                    vals[count] = val_temp
                    count += 1
        if max_diff == 0:
            boundaryVals[pt] = np.nan
        elif (0 in vals) and (1 in vals):
            boundaryVals[pt] = 0
        elif (0 in vals) and (1 not in vals):
            boundaryVals[pt] = 1
        elif ((2 in vals) or (3 in vals)) and (5 not in vals):
            boundaryVals[pt] = 3
        elif ((4 in vals) or (6 in vals)) and (5 not in vals):
            boundaryVals[pt] = 2
        elif (max_diff == 2 or max_diff == 3 or max_diff == 5):
            boundaryVals[pt] = 6
        elif 1 in vals:
            boundaryVals[pt] = 3
        else:
            boundaryVals[pt] = 5

    # Join and sort boundary information by value
    boundaryT = np.array([T_range[i] for i in boundaries[:,2]]).reshape((-1,1))
    boundaryh = np.array([1000*h_range[i] for i in boundaries[:,0]]).reshape((-1,1))
    boundarytheta = np.array([np.degrees(thetaL_range[i]) for i in boundaries[:,1]]).reshape((-1,1))

    boundaryVals = boundaryVals.reshape((-1,1))
    boundaryData = np.concatenate((boundaryT, boundaryh, boundarytheta, boundaryVals), axis=1)
    boundaryData = boundaryData[boundaryData[:,-1].argsort(kind='mergesort')]

    return boundaries, boundaryVals, boundaryData


def find_3D_phase_boundaries_main_params(thetaL, h_range=1e-3*np.arange(0.5,2.1,0.01),
                             r_range=np.arange(0.0,1.0,0.1),
                             T_range=np.arange(25.0,105.0,25.0),
                             minima=[], phases=[], angleT_vals=[], angle0_vals=[]):   
    """ Find locations of phase boundaries for a given composite
        With h, r, T ranges of length N_h, N_r, N_T
        Returns:
            boundaries      [N, 3]? array of points on boundary
            boundaryVals    [N]
            boundaryData    [N, 4]? """

    # Find where changes in phase occur *along temperature axis*
    diffs = np.diff(phases)
    boundaries = np.argwhere(diffs != 0)
    N = max(boundaries.shape) # Number of points found located at boundary
    boundaryVals = np.zeros(N)
    for pt in range(N):
        # Get h, r, T, and phase value [0-6] at boundary point
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        val = phases[loc[0],loc[1],loc[2]]
        # Find maximum phase difference between boundary point and all nearest
        # neighbors *in isotherm* (ignore points at end of range)
        max_diff = 0
        check = [-1,0,1]
        vals = np.zeros(9,dtype=int)
        if not ((-1 in loc) or loc[0] == phases.shape[0]-1 or loc[1] == phases.shape[1]-1 or loc[2] == phases.shape[2]-1):
            count = 0
            for i in check:
                for j in check:
                    val_temp = phases[loc[0]+i,loc[1]+j,loc[2]]
                    val_diff = np.abs(val - val_temp)
                    if val_diff > max_diff:
                        max_diff = val_diff
                    vals[count] = val_temp
                    count += 1
        if max_diff == 0:
            boundaryVals[pt] = np.nan
        elif (0 in vals) and (1 in vals):
            boundaryVals[pt] = 0
        elif (0 in vals) and (1 not in vals):
            boundaryVals[pt] = 1
        elif ((2 in vals) or (3 in vals)) and (5 not in vals):
            boundaryVals[pt] = 3
        elif ((4 in vals) or (6 in vals)) and (5 not in vals):
            boundaryVals[pt] = 2
        elif (max_diff == 2 or max_diff == 3 or max_diff == 5):
            boundaryVals[pt] = 6
        elif 1 in vals:
            boundaryVals[pt] = 3
        else:
            boundaryVals[pt] = 5

    # Join and sort boundary information by value
    boundaryT = np.array([T_range[i] for i in boundaries[:,2]]).reshape((-1,1))
    boundaryh = np.array([1000*h_range[i] for i in boundaries[:,0]]).reshape((-1,1))
    boundaryr = np.array([r_range[i] for i in boundaries[:,1]]).reshape((-1,1))
    boundaryVals = boundaryVals.reshape((-1,1))
    
    boundaryData = np.concatenate((boundaryT, boundaryh, boundaryr, boundaryVals), axis=1)
    boundaryData = boundaryData[boundaryData[:,-1].argsort(kind='mergesort')]

    return boundaries, boundaryVals, boundaryData
    

# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------
def plot_isotherm(r, T, phases, angle0_vals,
                  h_range=1e-3*np.arange(0.5,2.1,0.01),
                  thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                  T_range=np.arange(25.0,105.0,25.0), savedir='', closeFlag=True):
    """ Plot isotherm phase diagram for a given composite r:
        h vs total equilibrium angle at a particular temperature T """
    
    # Find where in simulation data the desired T value occurs
    i_T = np.argwhere(T_range == T)[0][0]
    
    # Plot
    colors = ['xkcd:light red', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:electric blue', 'xkcd:electric blue', 'xkcd:electric blue']
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    X = 1000*h_range
    Y = np.degrees(thetaL_range)
    Z = np.transpose(phases[:,:,i_T])
    diagram = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', cmap=col.ListedColormap(colors))           
    
    # Format
    plt.title(r'$T$ = {0}$^\circ$C, $r$ = {1:0.2f}'.format(int(T), r), size=18)
    plt.xlabel(r'$h$ (mm)')
    plt.ylabel(r'$\theta_L$ (degrees)')
    
    # Save and close
    if savedir != '':
        figname = 'isotherm_{0}C_r{1:0.2f}.png'.format(int(T), r)
        plt.savefig(os.path.join(savedir, figname),dpi=200)
    if closeFlag:
        plt.close()

    return


def plot_main_parameter_isotherm(thetaL, T, phases, angle0_vals,
                                 h_range=1e-3*np.arange(0.5,2.1,0.01),
                                 r_range=np.radians(np.arange(0.0, 1.0, 0.01)),
                                 T_range=np.arange(25.0,105.0,25.0),
                                 savedir='', closeFlag=True):
    """ Plot isotherm phase diagram for a given value of theta_L: h vs r
        at a particular temperature T"""
    
    # Find where in simulation data the desired T value occurs
    i_T = np.argwhere(T_range == T)[0][0]
    
    # Plot
    colors = ['xkcd:light red', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:electric blue', 'xkcd:electric blue', 'xkcd:electric blue']
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    X = 1000*h_range
    Y = r_range
    Z = np.transpose(phases[:,:,i_T])
    diagram = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', cmap=col.ListedColormap(colors))           
    
    # Format
    plt.title(r'$T$ = {0}$^\circ$C, $\theta_L$ = {1:0.1f}'.format(int(T), thetaL), size=18)
    plt.xlabel(r'$h$ (mm)')
    plt.ylabel(r'$r$')
    
    # Save and close
    if savedir != '':
        figname = 'isotherm_mainParams_{0}C.png'.format(int(T))
        plt.savefig(os.path.join(savedir, figname),dpi=200)
    if closeFlag:
        plt.close()

    return


# -----------------------------------------------------------------------------
# File loading and saving functions
# -----------------------------------------------------------------------------
def import_parameters(filename):
    """ Import parameters which were previously exported by export_parameters
        (used for previously-calculated parameter-space analysis) """

    with open(filename, 'r') as f:
        p = f.readlines()
        r = p[0].strip('\n')
        T = p[1].strip('\n').split(', ')
        h = p[2].strip('\n').split(', ')
        thetaL = p[3].strip('\n').split(', ')

    print("Imported the following parameters:")
    print(f"thetaL \t ({thetaL[0]}:{thetaL[-1]}), length {len(thetaL)}")
    print(f"T \t ({T[0]}:{T[-1]}), length {len(T)}")
    print(f"h \t ({h[0]}:{h[-1]}), length {len(h)}")
    print(f"r = {r}")

    thetaL = [float(x) for x in thetaL]
    T = [float(x) for x in T]
    h = [float(x) for x in h]
    r = float(r)

    return r, T, h, thetaL


def export_parameters(filename, r, T, h, thetaL):
    """ Export parameters used to map parameter space, for future import by 
        import_parameters """

    with open(filename, 'w') as f:
        f.write("{0:.3f}\n".format(r))
        f.write(", ".join(map(str, T)) + "\n")
        f.write(", ".join(map(str, h)) + "\n")
        f.write(", ".join(map(str, thetaL)) + "\n")


def export_to_csv(filename, data, variables=[], units=[], fmt='%0.18e'):
    """ Export columns with variable names/units if provided as 1 or 2 rows """
    
    if variables != [] and units != []:
        headerVars = ", ".join(map(str, variables))
        headerUnits = ", ".join(map(str, units))
    else: 
        headerVars = ''
        headerUnits = ''
    headerStr = headerVars + '\n' + headerUnits
    
    np.savetxt(filename, data, delimiter=', ', fmt=fmt, header=headerStr)
    
    return filename


def save_isotherms(datestr, resdir, minima, i_isotherm, h_range, thetaL_range, T_range, tag=''):
    """ Given i_isotherm corresponding to index in T_range for which to plot,
        save individual isothersm to csv and vtk """

    x = np.array([1000*h for h in h_range])
    if tag=='':
        y = np.array([np.degrees(theta) for theta in thetaL_range])
    else:
        y = thetaL_range
        
    for count, index in enumerate(i_isotherm):
        isotherm_vals = minima[:,:,index]
        isotherm_file_base = os.path.join(resdir, '{0}_isotherm{1}_{2}'.format(datestr, tag, count))
        data2vtk_rectilinear(x, y, T_range[index], isotherm_vals, isotherm_file_base)
    
    return


def save_boundaries(datestr, cwd, boundaries, boundaryData, boundaryVals, tag=''):
    """ Given boundary values, save to csv and vtk """
    
    boundaries_file = os.path.join(cwd, '{0}_boundaries{1}.csv'.format(datestr, tag))
    boundaryVals_file = os.path.join(cwd, '{0}_boundaryVals{1}.csv'.format(datestr, tag))
    boundaryData_file = os.path.join(cwd, '{0}_boundaryData{1}.csv'.format(datestr, tag))
    
    # Export files containing all data
    export_to_csv(boundaries_file, boundaries, fmt='%d',
                  variables=["T index", "h index", "thetaL index"],
                  units=["index","index","index"])
    export_to_csv(boundaryData_file, boundaryData, fmt=['%0.1f','%0.4f','%0.2f','%0.1f'],
                  variables=["T", "h", "thetaL", "value"],
                  units=["degrees C","mm","degrees","id number"])
    np.savetxt(boundaryVals_file, boundaryVals, delimiter=', ', fmt='%0.1f', header='Boundary values')
    
    # Export a file for each surface, except for points labeled nan
    boundarySurfaces = np.split(boundaryData, np.where(np.diff(boundaryData[:,-1]))[0]+1)
    nanCounter = 0
    for surface in boundarySurfaces:
        value = surface[0,-1]
        if not np.isnan(value):
            fname = os.path.join(cwd, '{0}_boundaryData{1}_{2}.csv'.format(datestr,tag,int(value)))
            export_to_csv(fname, surface[:,0:-1], fmt=['%0.1f','%0.4f','%0.3f'],
                          variables=["T", "h", "thetaL"],
                          units=["degrees C","mm","degrees"])
            csv2vtk_unstructured(fname)
        else:
            nanCounter += 1
            
    print("nan count: {0} ({1:.3f}%)".format(nanCounter, 100*nanCounter/len(boundarySurfaces)))

   
def data2vtk_rectilinear(x, y, z_val, values, basename):
    """ Create vtk file of rectilinear grid type from x, y, array csv file """  
    nx = len(x)
    ny = len(y)
    nz = 1
    
    with open(basename + ".vtk", "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write(basename + "\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write("DIMENSIONS {0} {1} {2}\n".format(nx, ny, nz))
        f.write("X_COORDINATES {0} float\n".format(nx))
        f.write(' '.join([str(val) for val in x]) + "\n")
        f.write("Y_COORDINATES {0} float\n".format(ny))
        f.write(' '.join([str(val) for val in y]) + "\n")
        f.write("Z_COORDINATES {0} float\n".format(nz))
        f.write("{0}\n".format(z_val))
        f.write("POINT_DATA {0}\n".format(nx*ny))
        f.write("SCALARS scalars float\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write("{0} ".format(values[i,j]))
        f.write("\n")


def csv2vtk_unstructured(fname):
    """ Create vtk file of unstructured grid type from csv array """
    
    basename = os.path.splitext(fname)[0]
    points = np.genfromtxt(fname,delimiter=", ",skip_header=1)
    value = int(basename[-1]) # Get surface number from filename
    z, x, y = points[:,0], points[:,1], points[:,2]
    nPoints = len(x)
    
    with open(basename + ".vtk", "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write(basename + "\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS {0} float\n".format(nPoints))
        for i in range(nPoints):
            f.write("{0} {1} {2}\n".format(x[i],y[i],z[i]))
        f.write("CELLS {0} {1}\n".format(nPoints, 2*nPoints))
        for i in range(nPoints):
            f.write("1 {0}\n".format(i))
        f.write("CELL_TYPES {0}\n".format(nPoints))
        for i in range(nPoints):
            f.write("1\n")
        f.write("POINT_DATA {0}\n".format(nPoints))
        f.write("SCALARS point_scalars float\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(0, nPoints):
            f.write("{0}\n".format(value))

