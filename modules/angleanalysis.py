"""
Fit experimental data for unit cell bending angle (no magnets).
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt

import modules.bilayermodel as bl


def fit_arclength_s(bilayer, T_vals, dtheta_avg):
    """ Least-squares optimization of arc length """
   
    s_guess = bilayer.s
    s_fit = opt.least_squares(residual_bilayer_s, s_guess,
                              args=(dtheta_avg, T_vals, bilayer))
    return s_fit.x


def residual_bilayer_s(s, angle_exp, T_vals, bilayer):
    """ Residual to minize for fit_arclength_s """
    
    # Update bilayer model with current guess for best-fit arc length
    bilayer.s = s
    bilayer.update_temperature(24.3)
    thetaRT = bilayer.thetaT
    
    # Calcualte updated bilayer model values for change in angle
    angle_model = np.zeros_like(T_vals)
    for i, T in enumerate(T_vals):
        bilayer.update_temperature(T)
        angle_model[i] = bilayer.thetaT - thetaRT
        
    residual = angle_exp - angle_model
    return residual


def quadratic_pure(x, b):
    return b*x**2


def analyze_bending_angles(datapath, parampath, nSamples=3,
                           LCE_modulus_params=[], LCE_strain_params=[],
                           saveFlag=False, figdir='', titlestr=''):
    """ Analyze ImageJ unit cell change-in-bending-angle data (three samples,
        two temperature setpoints above RT) 
        The samples should be nominally "fixed h" or "fixed r", indicated in
        titlestr; some analysis, including returning a best-fit b, is only done
        for the fixed-r case. """
    
    # Import bending data (divide by 2 to match BilayerModel definition)
    arr = np.genfromtxt(datapath, dtype=float, delimiter=',', skip_header=3)
    T = arr[:,0]
    dtheta_avg = np.transpose(arr[:,1:nSamples+1])/2
    dtheta_std = np.transpose(arr[:,nSamples+1:8])/2

    # Import parameters corresponding to above data
    arr = np.genfromtxt(parampath,delimiter=',',skip_header=2)
    h_avg = arr[0:nSamples,2] # measured h
    h_std = arr[nSamples:2*nSamples,2]
    r_avg = arr[0:nSamples,3] # measured r
    r_std = arr[nSamples:2*nSamples,3]
    s_avg = arr[0:nSamples,4] # measured hinge (arc) length
    s_std = arr[nSamples:2*nSamples,4]
    d_avg = arr[0:nSamples,5] # measured unit cell lattice spacing
    d_std = arr[nSamples:2*nSamples,5]
    
    # Find values for s which results in best fit to the change angle at 
    # elevated temperatures relative to RT ------------------------------------
    T_range = np.arange(25, 100, 1)
    s_fit = np.zeros(nSamples)
    for i in range(nSamples):
        # For each sample, calculate best-fit s
        dtheta_rad = np.radians(dtheta_avg[i,:])
        h_val = h_avg[i]
        r_val = r_avg[i]
        s_val = s_avg[i]
        bilayer = bl.BilayerModel(h_val*1e-3, r_val, s=s_val*1e-3, T=24.3,
                                  LCE_modulus_params=LCE_modulus_params,
                                  LCE_strain_params=LCE_strain_params)
        s_fit[i] = 1e3*fit_arclength_s(bilayer, T, dtheta_rad) # [mm]
    
    print('Arc length fit (all values [mm]):')
    print('\th:\t\t\t' + ', '.join(['{0:.2f}'.format(h) for h in h_avg]))
    print('\tnominal s:\t' + ', '.join(['{0:.2f}'.format(s) for s in s_avg]))
    print('\tbest fit s:\t' + ', '.join(['{0:.2f}'.format(s) for s in s_fit]))

    # Find best-fit parameters (for quadratic model!) to describe relationship
    # between measured h and best-fit s
    hsfit, _ = opt.curve_fit(quadratic_pure, h_avg, s_fit)
    b_fit = 1e3*hsfit
    
    # Plot h-s relationship showing quadratic model for arc length ------------
    h_range = np.arange(0.0, 2.01, 0.01)
    s_model = quadratic_pure(h_range, hsfit)

    plt.figure(dpi=200)
    plt.xlabel('$h$ (mm)')
    plt.ylabel('$s$ (mm)')
    plt.xlim([0.0, 2.0])      

    plt.plot(h_avg, s_fit, 'ok', label='best fit, {}'.format(titlestr))
    plt.plot(h_range, s_model, 'k', label='quadratic fit $s = {0:.3f}h^2$'.format(b_fit[0]))
    plt.legend()

    if saveFlag:
        filename = 'r-const_arclengthfit'
        plt.savefig(os.path.join(figdir, "{0}.png".format(filename)), dpi=200)
        plt.savefig(os.path.join(figdir, "{0}.svg".format(filename)))   
    
    # Find resulting modeled change in angle over temperature ------------------
    dtheta_model_update = np.zeros((3,len(T_range)))
    if 'h' not in titlestr:
        for i in range(3):
            h_val = h_avg[i]
            r_val = r_avg[i]
            s_val = s_avg[i]
            bilayer = bl.BilayerModel(h_val*1e-3, r_val, T=24.3,
                                      b=b_fit, bFlag='quad', 
                                      LCE_modulus_params=LCE_modulus_params)
            theta0 = bilayer.thetaT
            for j, T_val in enumerate(T_range):
                bilayer.update_temperature(T_val)
                dtheta_model_update[i,j] = np.degrees(bilayer.thetaT - theta0)    

        # Plot modeled and measured changes in angle with temperature -------------
        plt.figure(dpi=200)
        plt.title(titlestr)
        plt.xlabel("Temperature ($^\circ$C)")
        plt.ylabel(r"Change in angle $\theta_T$ from RT ($^\circ$)")
        letters = ['A','B','C']
        labels = ['{0}: h={1:.2f}, r={2:.2f}'.format(letters[i],h_avg[i],r_avg[i]) for i in range(3)]
        colors = ['r','g','b']
        markers = ['o','^']
        for i in range(3):
            plt.errorbar(T, dtheta_avg[i,:], yerr=dtheta_std[i,:], 
                         fmt=colors[i]+markers[0], capsize=4, label=labels[i])
            plt.plot(T_range, dtheta_model_update[i,:], colors[i],
                     linestyle='--', label='model: '+labels[i])
        plt.legend()
        
        if saveFlag:
            filename = 'r-const_angle'
            plt.savefig(os.path.join(figdir, "{0}.png".format(filename)), dpi=200)
            plt.savefig(os.path.join(figdir, "{0}.svg".format(filename)))   
    
    if 'h' in titlestr:
        return None
    else:
        return b_fit