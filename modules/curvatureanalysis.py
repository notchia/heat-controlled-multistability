"""
Fit experimental data for bilayer hinge curvature.
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import modules.bilayermodel as bl


def analyze_bending_data(paramfile, datafile, nSamples=5,
                         LCE_modulus_params=[], LCE_strain_params=[],
                         saveFlag=False, figdir='', verboseFlag=False):
    """ Process bending data for bilayer samples 1-6 (note samples 4 & 5 are
        nominally the same, so these were combined). """
    print("Analyzing bilayer curvature as a function of temperature:")
                                                     
    # Import data -------------------------------------------------------------
    # Import parameter information                                               
    arr = np.genfromtxt(paramfile, dtype=float, delimiter=',', skip_header=2)
    h = arr[0,1:nSamples+1]
    r = arr[1,1:nSamples+1]
    h_std = arr[0,nSamples+1:]
    r_std = arr[0,nSamples+1:]
    print('h: ' + ', '.join(['{:.2f}'.format(val) for val in h]) + ' (mm)')
    print('r: ' + ', '.join(['{:.2f}'.format(val) for val in r]) + ' (mm/mm)')
    
    # Import curvature for above parameters, as measured in ImageJ using the
    # Three Point Circle tool (1/radius of curvature)
    arr = np.genfromtxt(datafile, dtype=float, delimiter=',', skip_header=1)
    T = arr[:,0]
    curvature_avg = arr[:,1:nSamples+1]
    curvature_std = arr[:,nSamples+1:(2*nSamples + 1)]
    
    nTemps = curvature_avg.shape[0]
    nCombos = curvature_avg.shape[1]

    # Compute modeled curvature values for these parameters -------------------
    # Note: arc length = thickness (s = 0 + 1h) as normalization parameter
    T_range = np.arange(T[0], T[-1], 1)
    nTempRange = len(T_range)
    curvature_model = np.zeros((nTempRange, nCombos))
    curvature_change_model = np.zeros((nTempRange, nCombos))
    for i in range(nCombos):
        h_val = h[i]
        r_val = r[i]
        bilayer = bl.BilayerModel(h_val*1e-3, r_val, b=[0,1], bFlag='lin',
                                  LCE_modulus_params=LCE_modulus_params,
                                  LCE_strain_params=LCE_strain_params)
        for j in range(nTempRange):
            bilayer.update_temperature(T_range[j])
            curvature_model[j,i] = bilayer.curvature
        # Change in curvature with respect to room temperature
        curvature_change_model[:,i] = curvature_model[:,i] - curvature_model[0,i]

    # Analyze and plot data ---------------------------------------------------    
    def plot_values_with_temperature(y_data_avg, y_data_std, y_model=np.array([]),
                                     title='', ylabel='', RTindex=1):
        """ Plot temperature-series values, with model if applicable """
        
        colors = ['blue','orange','red','green','purple','brown','pink','gray','olive','cyan','magenta']        
        fig = plt.figure(title, dpi=200)
        plt.xlabel("Temperature ($^{{\circ}}C$)")
        plt.ylabel(ylabel)
        for i in range(nCombos):
            labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
            plt.errorbar(T[RTindex:], y_data_avg[RTindex:,i], yerr=y_data_std[RTindex:,i],
                         fmt='o', capsize=2, label=labelstr, color=colors[i])
            if y_model.any():
                plt.plot(T_range, y_model[:,i], linestyle='dashed',
                         linewidth=2, color=colors[i],label='model: '+labelstr)
        plt.legend()
        plt.tight_layout()
        if saveFlag:
            plt.savefig(os.path.join(figdir,"{0}.png".format(title)), dpi=200)
            fig.patch.set_facecolor('None')
            plt.savefig(os.path.join(figdir,"{0}.svg".format(title)), transparent=True)
        
        return

    # Plot measured curvature as a function of temperature for all samples      
    plot_values_with_temperature(curvature_avg, curvature_std,
                                 title='temperature-curvature_raw',
                                 ylabel='Measured curvature $\kappa$ (1/m)',
                                 RTindex=0)

    # Change in curvature
    curvature_change_avg = curvature_avg - curvature_avg[0]

    # Change in angle (curvature normalized by h): 
    # Use arc length s = thickness h to compute normalized angle
    angle_avg = 1e-3*np.multiply(h,curvature_change_avg)
    angle_std = 1e-3*np.multiply(h,curvature_std)
    angle_model = 1e-3*np.multiply(h,curvature_model)
    
    plot_values_with_temperature(angle_avg, angle_std,
                                 y_model=angle_model,
                                 title='bilayer_temperature-angle_model',
                                 ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)$')    

    # Change in normalized curvature (curvature normalized by multiplication by r^2): 
    normalized_curvature_r2_avg = np.multiply(curvature_change_avg, np.square(r))
    normalized_curvature_r2_std = np.multiply(curvature_std, np.square(r))
    normalized_curvature_r2_model = np.multiply(curvature_change_model, np.square(r))  

    # Change in normalized angle (curvature normalized by h and r): 
    normalized_angle_r2_avg = 1e-3*np.multiply(h, normalized_curvature_r2_avg)
    normalized_angle_r2_std = 1e-3*np.multiply(h, normalized_curvature_r2_std)
    normalized_angle_r2_model = 1e-3*np.multiply(h, normalized_curvature_r2_model)

    plot_values_with_temperature(normalized_angle_r2_avg, normalized_angle_r2_std,
                                 y_model=normalized_angle_r2_model,
                                 title='bilayer_temperature-angle_model_norm',
                                 ylabel='Normalized curvature change\n$hr^2(\kappa - \kappa_0)$')

    # Can the curvature be normalized by some values? -------------------------
    def plot_values_with_parameter(x_avg, x_std, y_avg, y_std,
                                   title='', xlabel='', ylabel='',
                                   fitFlag=False):
        """ Plot values w.r.t. parameter values, with fit to T=90C data """
        
        colors = ['blue','orange','red','green','purple','brown','pink','gray','olive','cyan','magenta']   
        plt.figure(title, dpi=200)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for i in range(nTemps):
            plt.errorbar(x_avg, np.transpose(y_avg[i,:]), yerr=np.transpose(y_std[i,:]),
                         xerr=np.transpose(x_std), fmt='o', capsize=2,
                         label="T = {0}".format(T[i]), color=colors[i])
            fit_to_x = np.polyfit(x_avg, y_avg[-2,:],1)
        if fitFlag:
            plt.plot(x_avg, np.polyval(fit_to_x, x_avg), color=colors[nTemps-2], label='linear fit')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.tight_layout()
        
        if saveFlag:
            plt.savefig(os.path.join(figdir,"{0}.svg".format(title)), transparent=True)
            plt.savefig(os.path.join(figdir,"{0}.png".format(title)), dpi=200)    
        
        return
        
    # Relation to composite ratio r
    plot_values_with_parameter(r, r_std, angle_avg, angle_std,
                               title='bilayer_ratio-normcurvature',
                               xlabel='LCE:total ratio $r$ (mm/mm)',
                               ylabel='Normalized curvature change $h(\kappa - \kappa_0)$')

    # Relation to total thickness h, showing independence
    plot_values_with_parameter(h, h_std, angle_avg, angle_std,
                               title='bilayer_thickness-curvature',
                               xlabel='Total thickness $h$ (mm)',
                               ylabel='Normalized curvature change $h(\kappa - \kappa_0)$')
    
    return
