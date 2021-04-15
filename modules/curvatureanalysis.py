"""
Fit experimental data for bilayer hinge curvature.
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt

import modules.bilayermodel as bl


# -----------------------------------------------------------------------------
# Main function, to be called in data_analysis_script
# -----------------------------------------------------------------------------
def analyze_bending_data(paramfile, datafile, nSamples=5, rBestFit=False,
                         LCE_modulus_params=[], LCE_strain_params=[],
                         saveFlag=False, figdir='', verboseFlag=False):
    """Analyze bilayer curvature as a function of temperature, from experiment"""
                                                     
    # =========================================================================
    # Import data
    # =========================================================================
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

    b = [0,1]
    bFlag = 'lin'

    # =========================================================================
    # Compute modeled curvature values (no fitting)
    # ========================================================================= 
    # Uses arc length = thickness (s = 0 + 1h) as normalization parameter
    T_range = np.arange(T[0], T[-1], 1)
    nTempRange = len(T_range)
    curvature_model = np.zeros((nTempRange, nCombos))
    curvature_change_model = np.zeros((nTempRange, nCombos))
    for i in range(nCombos):
        h_val = h[i]
        r_val = r[i]
        bilayer = bl.BilayerModel(h_val*1e-3, r_val, b=b, bFlag=bFlag,
                                  LCE_modulus_params=LCE_modulus_params,
                                  LCE_strain_params=LCE_strain_params)
        for j in range(nTempRange):
            bilayer.update_temperature(T_range[j])
            curvature_model[j,i] = bilayer.curvature
        curvature_change_model[:,i] = curvature_model[:,i] - curvature_model[0,i]



    # =========================================================================
    # Analyze and plot data
    # ========================================================================= 
    colors = ['blue','orange','red','green','purple','brown','pink','gray','olive','cyan','magenta']
    def plot_values_with_temperature(y_data_avg, y_data_std, y_model=np.array([]),
                                     title='', ylabel='', RTindex=1):
        """Plot temperature-series values, with model if applicable"""
        
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
    curvature_change_avg = curvature_avg - curvature_avg[0,:]

    # Change in angle (curvature normalized by h): 
    # Use arc length s = thickness h to compute normalized angle
    angle_avg = 1e-3*np.multiply(h,curvature_change_avg)
    angle_std = 1e-3*np.multiply(h,curvature_std)
    angle_model = 1e-3*np.multiply(h,curvature_model)
    
    plot_values_with_temperature(angle_avg, angle_std, y_model=angle_model,
                                 title='bilayer_temperature-angle_model',
                                 ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)$')    

    # =========================================================================
    # Could it make sense to make r a fitting parameter?
    # =========================================================================
    def find_best_fit_r():
        """Find r value for each sample which results in best fit to curvature data"""
        r_fits = np.zeros(nCombos)
        for i in range(nCombos):
            p_given = [h[i], b, bFlag, LCE_modulus_params, LCE_strain_params]  
            r_fits[i] = fit_r(r[i], p_given, curvature_change_avg[:,i], T)
    
        T_range = np.arange(T[0], T[-1], 1)
        nTempRange = len(T_range)
        curvature_model_r = np.zeros((nTempRange, nCombos))
        curvature_change_model_r = np.zeros((nTempRange, nCombos))
        for i in range(nCombos):
            h_val = h[i]
            r_val = r_fits[i]
            bilayer = bl.BilayerModel(h_val*1e-3, r_val, b=b, bFlag=bFlag,
                                      LCE_modulus_params=LCE_modulus_params,
                                      LCE_strain_params=LCE_strain_params)
            for j in range(nTempRange):
                bilayer.update_temperature(T_range[j])
                curvature_model_r[j,i] = bilayer.curvature
            curvature_change_model_r[:,i] = curvature_model_r[:,i] - curvature_model_r[0,i]
    
        plot_values_with_temperature(angle_avg, angle_std,
                                     y_model=1e-3*np.multiply(h,curvature_change_model_r),
                                     title='bilayer_temperature-angle_model_best-fit-r',
                                     ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)$')    
    
        # Plot relation between measured and best-fit r
        r_relation, residuals, _, _, _ = np.polyfit(r, r_fits, 1, full=True)
        fig = plt.figure('best-fit r', dpi=200)
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        plt.xlabel('r, measured')
        plt.ylabel('r, best-fit')
        plt.errorbar(r, r_fits, xerr=r_std, fmt='ko', capsize=2)
        plt.plot(r, np.polyval(r_relation, r), 'r',
                 label=f'linear fit, y = {r_relation[0]:.3f}x + {r_relation[1]:.3f}\n(RSS = {residuals[0]:.2e})')
        plt.xlim([0,0.75])
        plt.ylim([0,0.75])
        plt.legend()
        plt.tight_layout()
        
        return r_relation

    if rBestFit:
        r_relation = find_best_fit_r()
    else:
        r_relation = [1, 0] # use measured r
    print(f"r = {r_relation[0]}*r_measured + {r_relation[1]}")


    # =========================================================================
    # How does curvature depend on individual parameters?
    # ========================================================================= 
    def plot_values_with_parameter(x_avg, x_std, y_avg, y_std,
                                   title='', xlabel='', ylabel='',
                                   fitFlag=False):
        """Plot values w.r.t. parameter values, with fit to T=90C data"""
          
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
        
    # Relation to composite ratio r (does curvature increase with r?)
    plot_values_with_parameter(r, r_std, angle_avg, angle_std,
                               title='bilayer_ratio-normcurvature',
                               xlabel='LCE:total ratio $r$ (mm/mm)',
                               ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)$')

    # Relation to total thickness h (is curvature independent of h?)
    plot_values_with_parameter(h, h_std, angle_avg, angle_std,
                               title='bilayer_thickness-curvature',
                               xlabel='Total thickness $h$ (mm)',
                               ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)$')
    
    return r_relation 
    

# -----------------------------------------------------------------------------
# Other functions
# -----------------------------------------------------------------------------
def fit_r(p_guess, p_given, curvatures, temperatures):
    """Find best-fit r for a particular sample from curvature change"""
    if np.isnan(curvatures[-1]):
        curvatures = curvatures[:-1]
        temperatures = temperatures[:-1]
    
    p_fit = opt.least_squares(residual_bilayer, p_guess,
                              args=(curvatures, temperatures, p_given),
                              ftol=1e-10, gtol=1e-10, xtol=1e-10)
    
    return p_fit.x[0]

        
def residual_bilayer(p, curvatures, temperatures, p_given):
    """Residual for finding best fit r"""
    h, b, bFlag, LCE_modulus_params, LCE_strain_params = p_given    
    bilayer = bl.BilayerModel(1e-3*h, p, b=b, bFlag=bFlag,
                              LCE_modulus_params=LCE_modulus_params,
                              LCE_strain_params=LCE_strain_params)
    
    curvatures_fit = np.zeros_like(curvatures)
    
    for i, T in enumerate(temperatures):
        bilayer.update_temperature(T)
        curvatures_fit[i] = bilayer.curvature
    curvatures_fit = curvatures_fit - curvatures_fit[0]
    
    residual = curvatures - curvatures_fit
    
    return residual
        
    
    