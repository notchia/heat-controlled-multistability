# -*- coding: utf-8 -*-
'''
Define BilayerModel class and fit experimental data for bilayer bending.
    
@author: Lucia Korpas
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.optimize as opt

import modules.LCEmodel as LCE
import modules.PDMSmodel as PDMS


class BilayerModel:
    ''' Model for a PDMS-LCE bilayer hinge of total thickness h_total and
        LCE:total thickness ratio r. Initialized at room temperature, but can
        be updated for any temperature. Specify only one of s or b.'''
    def __init__(self, h_total, ratio, T=25.0,
                 LCE_strain_params=[0.039457,142.37,6.442e-3],
                 LCE_modulus_params=[-8.43924097e-02,2.16846882e+02,3.13370660e+05],
                 w=0.010, b=[], s=0, bFlag='lin'):
        
        assert T > 20.0, "Invalid T: lower than room temperature"
        assert h_total > 0, "Invalid h: must be positive"
        assert h_total < 0.01, "Make sure h is in meters, not millimeters"
        assert (ratio >= 0 and ratio < 1), "Invalid r: must be in [0,1)"
        
        # Main model parameters
        self.T = T                  # [Celcius]     Temperature
        self.h_total = h_total      # [m]           Total bilayer hinge thickness
        self.ratio = float(ratio)          # [m/m]         Bilayer hinge LCE:PDMS thickness ratio
        
        # Material parameters
        self.LCE_strain_params = LCE_strain_params
        self.LCE_modulus_params = LCE_modulus_params
        self.E_PDMS = PDMS.model_elastic_modulus(T) # [Pa] traditional AML PDMS; default from 08/07/20
        self.E_LCE = LCE.model_elastic_modulus(T, *LCE_modulus_params) # [Pa] Rui's formulation; default from 08/17/20
        
        # Geometric parameters: out-of-plane width w, and arc length s
        self.w = w
        if s == 0:
            assert b, 'Specify either arc length s or parameters b'
            self.bFlag = bFlag
            self.s = self._calculate_arc_length(b, bFlag)
        else:
            assert s < 0.01, "Make sure s is in meters, not millimeters"
            self.s = s
        
        # Geometric parameters
        self.h_PDMS = h_total*(1-ratio)    # [m] hinge thickness, PDMS
        self.h_LCE =  h_total*ratio        # [m] hinge thickness, LCE. Assume LCE is on "inside" of hinge
        
        self.I_PDMS, self.I_LCE = self._calculate_2nd_moments()
        self.k = self._calculate_stiffness() # [m] stiffness of composite hinge
        self.curvature = self._calculate_curvature(T)
        self.thetaT = self._calculate_angle(T)


    def update_temperature(self, T):
        ''' Update hinge properties given new temperature '''  
        self.T = T
        self.E_LCE = LCE.model_elastic_modulus(T, *(self.LCE_modulus_params))
        self.I_PDMS, self.I_LCE = self._calculate_2nd_moments()
        self.k = self._calculate_stiffness()
        self.curvature = self._calculate_curvature(T)
        self.thetaT = self._calculate_angle(T)
        return
             
    def _calculate_arc_length(self, b, bFlag):
        ''' Using arc length parameters b, compute effective arc length s from
            hinge thickness h '''
        if bFlag == 'lin':
            s = b[1]*self.h_total + b[0]
        elif bFlag == 'quad':
            s = b[0]*self.h_total**2
        else:
            assert False, "choose 'lin' or 'quad' for arc length fitting"
        return s
        
    def _calculate_stiffness(self):
        ''' Using physical parameters, calculate effective linear spring constant 
            from beam theory, assuming a rectangular beam bending in thickness. '''
        k0 = (self.E_PDMS*self.I_PDMS + self.E_LCE*self.I_LCE)/self.s
        return k0  


    def _calculate_2nd_moments(self):
        ''' Calculate 2nd moments of area of PDMS and LCE layers individually  '''
        # Location of effective neutral axis w.r.t. PDMS surface
        h0 = ((self.E_PDMS*self.h_PDMS**2/2 + self.E_LCE*self.h_LCE**2/2
               + self.E_LCE*self.h_PDMS*self.h_LCE)/(self.E_PDMS*self.h_PDMS
                                                     + self.E_LCE*self.h_LCE))
        # 2nd moments of area w.r.t. neutral axis (parallel axis theorem)                                            
        I_PDMS = (self.w*self.h_PDMS**3/12) + self.w*self.h_PDMS*(
            h0 - self.h_PDMS/2)**2
        I_LCE = (self.w*self.h_LCE**3/12) + self.w*self.h_LCE*(
            self.h_total - h0 - self.h_LCE/2)**2
        
        return I_PDMS, I_LCE
    
    def _calculate_curvature(self, T):
        ''' Calculate curvature [1/m] at a given temperature '''
        if self.ratio == 0:
            kappa = 0
        else:
            numer = LCE.model_strain(T, *(self.LCE_strain_params))
            denom = ((2/self.h_total)*(self.E_LCE*self.I_LCE + self.E_PDMS*self.I_PDMS)
                     *(1/(self.E_LCE*self.h_LCE*self.w) + 1/(self.E_PDMS*self.h_PDMS*self.w))
                     + (self.h_total/2))
            kappa = numer/denom
        return -kappa
    
    def _calculate_angle(self, T):
        ''' Calculate hinge angle at a given temperature as s*kappa
            Currently multiplying by 2, but that's really a unit cell property,
            so should probably rearrange this correspondingly...'''
        angle = self.s*self.curvature
        return angle

#%%
def test_BilayerModel():
    ''' Speed test and sanity checks on BilayerModel '''
    
    h_total = 0.001
    b = [3]
    
    bilayer = BilayerModel(h_total, 0.0, b=b)
    kappa0 = bilayer.curvature
    bilayer.update_temperature(100)
    kappa1 = bilayer.curvature
    assert kappa0 == kappa1, 'r = 0 should not bend with temperature change'

    t0 = time.perf_counter()
    bilayer = BilayerModel(h_total, 0.5, b=b)
    kappa0 = bilayer.curvature
    t1 = time.perf_counter()
    bilayer.update_temperature(100)
    kappa1 = bilayer.curvature
    t2 = time.perf_counter()
    print(" Create:\t {0:.2e} s\n Update:\t {1:0.2e} s".format(t1-t0, t2-t1))
    assert kappa0 < kappa1, 'change in curvature should be positive with increased temperature'

    print("BilayerModel passes tests")
    
    return True

#%% Find arc length for best fit to angle data
def fit_arclength_s(bilayer, T_vals, dtheta_avg):
    s_guess = bilayer.s
    s_fit = opt.least_squares(residue_bilayer_s, s_guess,
                                      args=(dtheta_avg, T_vals, bilayer))
    return s_fit.x

def fit_arclength_b(bilayer, T_vals, dtheta_avg):
    b_guess = [0, 1]
    b_fit = opt.least_squares(residue_bilayer_b, b_guess,
                                      args=(dtheta_avg, T_vals, bilayer))
    return b_fit.x

def residue_bilayer_s(s, angle_exp, T_vals, bilayer):
    bilayer.s = s
    angle_model = np.zeros_like(T_vals)
    bilayer.update_temperature(24.3)
    thetaRT = bilayer.thetaT
    for i, T in enumerate(T_vals):
        bilayer.update_temperature(T)
        angle_model[i] = bilayer.thetaT - thetaRT
        
    residue = angle_exp - angle_model
    return residue

def residue_bilayer_b(b, angle_exp, T_vals, bilayer):
    bilayer.s = b[0] + b[1]*bilayer.h_total
    angle_model = np.zeros_like(T_vals)
    thetaRT = bilayer.thetaT
    for i, T in enumerate(T_vals):
        bilayer.update_temperature(T)
        angle_model[i] = bilayer.thetaT - thetaRT
        
    residue = angle_exp - angle_model
    return residue

#%% Import and analyze experiments relating to this model -----------------------
def analyze_data_200814(filepath):
    ''' Process samples 1-6. Note samples 4 & 5 are nominally the same, so compiled into one'''
    
    h_PDMS = [0.57, 0.57, 0.87, 0.91, 1.34] 
    h_LCE  = [0.30, 0.99, 0.99, 0.30, 0.30]
    h = [h_PDMS[i]+h_LCE[i] for i in range(len(h_PDMS))]
    r = [h_LCE[i]/h[i] for i in range(len(h_PDMS))]
    print('h: ' + ', '.join(['{:.2f}'.format(val) for val in h]))
    print('r: ' + ', '.join(['{:.2f}'.format(val) for val in r]))
    
    arr = np.genfromtxt(filepath,dtype=float,delimiter=',',skip_header=2)
    T = arr[:,0]
    curvature_avg = arr[:,1:6]
    curvature_std = arr[:,6:11]
    
    nTemps = curvature_avg.shape[0]
    nCombos = curvature_avg.shape[1]
    
    s = h # arc length = thickness
    angle_avg = np.degrees(1e-3*np.multiply(s,curvature_avg))
    angle_std = np.degrees(1e-3*np.multiply(s,curvature_std))
    angle_avg = angle_avg - angle_avg[0]
    
    colors = ['blue','orange','red','green','purple']
    
    plt.figure('temp-curvature', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Measured curvature $\kappa$ (1/m)")
    for i in range(nCombos):
        plt.errorbar(T, curvature_avg[:,i], yerr=curvature_std[:,i], 
                     label="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i]), 
                     fmt='o', capsize=2, color=colors[i])
    plt.legend()

    # Compute model values for these parameters (with arc length = thickness)
    curvature_model = np.zeros_like(curvature_avg)
    angle_model = np.zeros_like(curvature_avg)
    for i in range(nCombos):
        h_val = h[i]
        r_val = r[i]
        bilayer = BilayerModel(h_val*1e-3, r_val, b=[0,1])
        for j in range(nTemps):
            bilayer.update_temperature(T[j])
            curvature_model[j,i] = bilayer.curvature
            angle_model[j,i] = bilayer.thetaT
    
    # Plot curvature and angle, comparing model and experiments
    plt.figure('temp-curvature-model', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Change in curvature $\kappa - \kappa_0$ (1/m)")
    curvature_change_avg = curvature_avg - curvature_avg[0]

    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, curvature_change_avg[:,i], yerr=curvature_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T, curvature_model[:,i], marker='.', linestyle='dashed',
                 linewidth=2, color=colors[i],label='model: '+labelstr)
    plt.legend()

    plt.figure('t-a', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Change in angle $h(\kappa - \kappa_0)$ (deg)")
    colors = ['blue','orange','red','green','purple']
    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, angle_avg[:,i], yerr=angle_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T, np.degrees(angle_model[:,i]), marker='.', linestyle='dashed',
                 linewidth=2, color=colors[i],label='model: '+labelstr)        
    plt.legend()

    # Can the curvature be normalized by some values?
    # Relation to LCE:total thickness ratio r:
    plt.figure('ratio-curvature', dpi=200)
    plt.xlabel("LCE:total ratio $r$ (mm/mm)")
    plt.ylabel("Change in angle $h(\kappa - \kappa_0)$ ($^\circ$)")
    for i in range(nTemps):
        plt.errorbar(r, np.transpose(angle_avg[i,:]), yerr=np.transpose(angle_std[i,:]), fmt='o', capsize=2,
                     label="T = {0}".format(T[i]), color=colors[i])
        #plt.plot(r, np.transpose(np.degrees(angle_model[i,:])), marker='^', linewidth=0, color=colors[i],label="T = {0}, model".format(T[i]))  
    plt.legend()
    # Relation to total thickness h:
    plt.figure('thickness-curvature', dpi=200)
    plt.xlabel("Total thickness h (mm/mm)")
    plt.ylabel("Normalized curvature $(\kappa - \kappa_0)/r$ (1/m)")
    normalized_curvature_avg = np.divide(curvature_change_avg, r)
    normalized_curvature_std = np.divide(curvature_std, r)
    for i in range(nTemps):
        plt.errorbar(h, np.transpose(normalized_curvature_avg[i,:]), yerr=np.transpose(normalized_curvature_std[i,:]), fmt='o', capsize=2,
                     label="T = {0}".format(T[i]), color=colors[i])
    plt.legend()

    return


def analyze_data_200819(paramfile, datafile, LCE_modulus_params=[], LCE_strain_params=[],
                        saveFlag=False, figdir='', verboseFlag=False):
    ''' Process bending data for bilayer samples 1-6 (note samples 4 & 5 are
        nominally the same, so these were combined).'''
    
    # Import data -------------------------------------------------------------
    # Import parameter information                                               
    arr = np.genfromtxt(paramfile,dtype=float,delimiter=',',skip_header=2)
    h = arr[0,1:6]
    r = arr[1,1:6]
    h_std = arr[0,6:]
    r_std = arr[0,6:]
    print('h: ' + ', '.join(['{:.2f}'.format(val) for val in h]) + ' (mm)')
    print('r: ' + ', '.join(['{:.2f}'.format(val) for val in r]) + ' (mm/mm)')
    
    # Import curvature for above parameters, as measured in ImageJ using the
    # Three Point Circle tool (1/radius of curvature)
    arr = np.genfromtxt(datafile,dtype=float,delimiter=',',skip_header=1)
    T = arr[:,0]
    curvature_avg = arr[:,1:6]
    curvature_std = arr[:,6:11]
    
    nTemps = curvature_avg.shape[0]
    nCombos = curvature_avg.shape[1]

    # Compute model values for these parameters -------------------------------
    # Note: arc length = thickness, as normalization parameter
    T_range = np.arange(T[0],T[-1],1)
    nTempRange = len(T_range)
    curvature_model = np.zeros((nTempRange,nCombos))
    curvature_change_model = np.zeros((nTempRange,nCombos))
    for i in range(nCombos):
        h_val = h[i]
        r_val = r[i]
        bilayer = BilayerModel(h_val*1e-3, r_val, b=[0,1], bFlag='lin',
                               LCE_modulus_params=LCE_modulus_params,
                               LCE_strain_params=LCE_strain_params)
        for j in range(nTempRange):
            bilayer.update_temperature(T_range[j])
            curvature_model[j,i] = bilayer.curvature
        curvature_change_model[:,i] = curvature_model[:,i] - curvature_model[0,i]

    # Analyze and plot data ---------------------------------------------------    

    def plot_values_with_temperature(y_data_avg, y_data_std, y_model=np.array([]),
                                     title='', ylabel='', RTindex=1):
        ''' Plot temperature-series values, with model if applicable '''
        
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

    # Change in normalized curvature (curvature normalized by r): 
    normalized_curvature_avg = np.divide(curvature_change_avg, r)
    normalized_curvature_std = np.divide(curvature_std, r)
    normalized_curvature_model = np.divide(curvature_change_model, r)        

    # Change in normalized angle (curvature normalized by h and r): 
    normalized_angle_avg = 1e-3*np.multiply(h, normalized_curvature_avg)
    normalized_angle_std = 1e-3*np.multiply(h, normalized_curvature_std)
    normalized_angle_model = 1e-3*np.multiply(h, normalized_curvature_model)

    # Change in normalized curvature (curvature normalized by ln(r)): 
    log_r = -np.log(r)
    normalized_curvature_log_avg = np.divide(curvature_change_avg, log_r)
    normalized_curvature_log_std = np.divide(curvature_std, log_r)
    normalized_curvature_log_model = np.divide(curvature_model, log_r)

    # Change in normalized angle (curvature normalized by h and ln(r)): 
    normalized_angle_log_avg = 1e-3*np.multiply(h, normalized_curvature_log_avg)
    normalized_angle_log_std = 1e-3*np.multiply(h, normalized_curvature_log_std)
    normalized_angle_log_model = 1e-3*np.multiply(h, normalized_curvature_log_model)
    plot_values_with_temperature(normalized_angle_log_avg, normalized_angle_log_std,
                                 y_model=normalized_angle_log_model,
                                 title='bilayer_temperature-logangle_model',
                                 ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)/(-\ln(r))$')

    if verboseFlag:
        plot_values_with_temperature(curvature_change_avg, curvature_std,
                                     y_model=curvature_change_model,
                                     title='bilayer_temperature-curvature_model',
                                     ylabel='Change in curvature $\kappa-\kappa_0$ (1/m)')
        plot_values_with_temperature(normalized_curvature_avg, normalized_curvature_std,
                                     y_model=normalized_curvature_model,
                                     title='bilayer_temperature-normcurvature_model',
                                     ylabel='Normalized curvature change\n$(\kappa - \kappa_0)/r$ (1/m)')
        plot_values_with_temperature(normalized_angle_avg, normalized_angle_std,
                                     y_model=normalized_angle_model,
                                     title='bilayer_temperature-normangle_model',
                                     ylabel='Normalized curvature change\n$h(\kappa - \kappa_0)/r$')
        plot_values_with_temperature(normalized_curvature_log_avg, normalized_curvature_log_std,
                                     y_model=normalized_curvature_log_model,
                                     title='bilayer_temperature-logcurvature_model',
                                     ylabel='Normalized curvature change\n$(\kappa - \kappa_0)/(-\ln(r))$ (1/m)')

    # Can the curvature be normalized by some values? -------------------------
    # Relation to LCE:total thickness ratio r:

    def plot_values_with_parameter(x_avg, x_std, y_avg, y_std,
                                   title='', xlabel='', ylabel='',
                                   fitFlag=False):
        ''' Plot values w.r.t. parameter values, with fit to T=90C data '''
        
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

    # Relation to total thickness h
    plot_values_with_parameter(h, h_std, normalized_curvature_avg, normalized_curvature_std,
                               title='bilayer_thickness-curvature1',
                               xlabel='Total thickness $h$ (mm)',
                               ylabel='Normalized curvature change $(\kappa - \kappa_0)/r$')

    # Relation to total thickness h
    plot_values_with_parameter(h, h_std, normalized_angle_avg, normalized_angle_std,
                               title='bilayer_thickness-curvature2',
                               xlabel='Total thickness $h$ (mm)',
                               ylabel='Normalized curvature change $h(\kappa - \kappa_0)/r$')
    
    """
    # Scale the model by a fit to h. Really what should be done is finding fit_to_h
    # for each temperature, then fitting a curve to how the slope of the fit
    # changes with temperature. Right now the fit is probably good enough that
    # this is unnecessary.
    normalized_adjusted_curvature = adjust_by_h(h, r, normalized_curvature_model, fit_to_h=fit_to_h, h_ref=h[0])
    plt.figure('temp-curvature-model-adjust-norm', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Normalized change in curvature $(\kappa - \kappa_0)/r$,\n     adjusted model (1/m)")
    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, normalized_curvature_avg[:,i], yerr=normalized_curvature_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T_range, normalized_adjusted_curvature[:,i], linestyle='dashed',
                 linewidth=2, color=colors[i],label='model: '+labelstr)
    plt.legend()
    plt.tight_layout()"""

    return

"""
def adjust_by_h(h_vals, r_vals, modeled_curvature, fit_to_h=[1,0], h_ref=0):
    ''' Take array of N h_vals and N r_vals corresponding to the (N,M) cuvatures
    (for M temperature points) and the coefficients of a linear polyfit fit_to_h
    and return the adjusted curvatures'''
    assert modeled_curvature.shape[1] == len(h_vals) and modeled_curvature.shape[1] == len(r_vals)
    nSamples = modeled_curvature.shape[1]
    adjusted_curvature = np.zeros_like(modeled_curvature)

    for i in range(nSamples):
        h = h_vals[i]
        r = r_vals[i]
        adjustment = get_h_adjustment_coeff(h, r, fit_to_h, h_ref)
        adjusted_curvature[:,i] = modeled_curvature[:,i]*adjustment
    return adjusted_curvature

def get_h_adjustment_coeff(h, r, fit_to_h, h_ref):
    ''' Using a fit to normalized_curvature as a function of h at a certain r value'''
    normalized_adjustment = (fit_to_h[0]*(h - h_ref))/(fit_to_h[0]*h_ref + fit_to_h[1])
    adjustment = 1 + normalized_adjustment*0.7
    return adjustment"""

    
def analyze_curvature_change_with_temp(LCE_modulus_params=[],LCE_strain_params=[],
                                   saveFlag=False, figdir='', verboseFlag=False):
    ''' 2D color plot of angle on h-T axes '''
    #h_range = 1e-3*np.arange(0.5,2.5,0.5) #[m]
    h_range = 1e-3*np.array([1.0])
    r_range = np.arange(0.01,1.0,0.01)
    T_range = np.arange(25,101,1)
    
    for k, h in enumerate(h_range):
        kappaT_vals = np.zeros((len(T_range), len(r_range)))
        thetaT_vals = np.zeros((len(T_range), len(r_range)))
        normT_vals = np.zeros((len(T_range), len(r_range)))
        for j, r in enumerate(r_range):
            bilayer = BilayerModel(h, r, LCE_modulus_params=LCE_modulus_params, s=h)
            for i, T in enumerate(T_range):
                bilayer.update_temperature(T)
                kappaT_vals[i,j] = bilayer.curvature
                thetaT_vals[i,j] = bilayer.thetaT
                normT_vals[i,j] = h*bilayer.curvature/(-np.log(r))
        indices = np.unravel_index(np.argmax(thetaT_vals), thetaT_vals.shape)
        print(indices)
        print("Maximum normalized curvature is {0:.4f} at r={1:.3f} and T={2:.1f}".format(
            thetaT_vals[indices], r_range[indices[1]], T_range[indices[0]]))
        colorplot_change_with_temp(h, h, r_range, T_range, thetaT_vals,
                                   barLabel='Normalized curvature $h\kappa$',
                                   figLabel='thetaT',
                                   saveFlag=saveFlag, figdir=figdir)
        if verboseFlag:
            colorplot_change_with_temp(h, h, r_range, T_range, kappaT_vals,
                                       barLabel='Curvature $\kappa$ (1/m)', figLabel='kappaT',
                                       saveFlag=saveFlag, figdir=figdir)
            colorplot_change_with_temp(h, h, r_range, T_range, normT_vals,
                                       barLabel='Normalized curvature $h\kappa/(-ln(r))$', figLabel='normT',
                                       saveFlag=saveFlag, figdir=figdir)
    
    return

def colorplot_change_with_temp(h, s, r_range, T_range, curvatureT_vals, 
                               barLabel='', figLabel='', saveFlag=False, figdir=''):
    ''' 2D color plot of angle on h-T axes '''
    r_range, T_range = np.meshgrid(r_range, T_range)
    minVal = np.amin(curvatureT_vals)
    maxVal = np.amax(curvatureT_vals)
    level = maxVal/100
    colorlevels = np.arange(minVal, maxVal, level)
    
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.contourf(r_range, T_range, curvatureT_vals, levels=colorlevels)    
    plt.xlabel('$r$ (mm/mm)')
    plt.ylabel('$T$ ($^\circ$C)')
    plt.title('$h$ = {0:0.2f} mm, $s$ = {1:0.2f} mm'.format(1e3*h, 1e3*s))
    bar = fig.colorbar(surf, aspect=8)
    bar.set_label(barLabel)
    plt.tight_layout()
    if saveFlag:
        filename = '{2}_h{0:0.2f}_s{1:0.2f}'.format(1e3*h, 1e3*s, figLabel)
        plt.savefig(os.path.join(figdir,"{0}.png".format(filename)),dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(filename)))

    return

#%% Bending angle of unit cells

def quadratic_pure(x, b):
    return b*x**2

def analyze_bending_angles(datapath, parampath,
                           LCE_modulus_params=[], LCE_strain_params=[],
                           saveFlag=False, figdir='', titlestr=''):
    ''' Analyze ImageJ bend data '''
    # Import data
    arr = np.genfromtxt(datapath,dtype=float,delimiter=',',skip_header=3)
    T = arr[:,0]
    dtheta_avg = np.transpose(arr[:,1:4])/2
    dtheta_std = np.transpose(arr[:,4:8])/2
    nTemps = len(T)

    # Import parameters file
    arr = np.genfromtxt(parampath,delimiter=',',skip_header=2)
    h_avg = arr[0:3,2]
    h_std = arr[3:6,2]
    r_avg = arr[0:3,3]
    r_std = arr[3:6,3]
    s_avg = arr[0:3,4]
    s_std = arr[3:6,4]
    d_avg = arr[0:3,5]
    d_std = arr[3:6,5]
    
    # Compare to model
    T_range = np.arange(25,100,1)
    dtheta_model = np.zeros((3,len(T_range)))
    s_fit = np.zeros(3)
    b_fit = np.zeros_like(dtheta_avg)
    
    for i in range(3):
        h_val = h_avg[i]
        r_val = r_avg[i]
        s_val = s_avg[i]
        bilayer = BilayerModel(h_val*1e-3, r_val, s=s_val*1e-3, T=24.3,
                               LCE_modulus_params=LCE_modulus_params,
                               LCE_strain_params=LCE_strain_params)
        theta0 = bilayer.thetaT
        for j, T_val in enumerate(T_range):
            bilayer.update_temperature(T_val)
            dtheta_model[i,j] = np.degrees(bilayer.thetaT - theta0)
        dtheta_rad = np.radians(dtheta_avg[i,:])
        s_fit[i] = 1e3*fit_arclength_s(bilayer, T, dtheta_rad)
        b_fit[i] = fit_arclength_b(bilayer, T, dtheta_rad)
    
    print('s values')
    print(', '.join(['{0:.2f}'.format(s) for s in s_avg]))
    print(', '.join(['{0:.2f}'.format(s) for s in s_fit]))

    # Fit quadratic function to best-fit h-s relation
    hsfit, _ = opt.curve_fit(quadratic_pure, h_avg, s_fit)
    b_fit = 1e3*hsfit
    h_range = np.arange(0.0, 2.01, 0.01)
    s_model = quadratic_pure(h_range, hsfit)

    plt.figure(dpi=200)
    plt.xlabel('$h$ (mm)')
    plt.ylabel('$s$ (mm)')
    plt.xlim([0.0,2.0])      
    #plt.plot(h_avg, s_avg, 'or', label='measured, {}'.format(titlestr))
    plt.plot(h_avg, s_fit, 'ok', label='best fit, {}'.format(titlestr))
    plt.plot(h_range, s_model, 'k', label='quadratic fit $s = {0:.3f}h^2$'.format(b_fit[0]))
    plt.legend()

    if saveFlag:
        filename = 'r-const_arclengthfit'
        plt.savefig(os.path.join(figdir,"{0}.png".format(filename)),dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(filename)))   
    
    # Update value
    dtheta_model_update = np.zeros((3,len(T_range)))
    if 'h' not in titlestr:
        for i in range(3):
            h_val = h_avg[i]
            r_val = r_avg[i]
            s_val = s_avg[i]
            bilayer = BilayerModel(h_val*1e-3, r_val, b=b_fit, bFlag='quad', T=24.3,
                                   LCE_modulus_params=LCE_modulus_params)
            theta0 = bilayer.thetaT
            for j, T_val in enumerate(T_range):
                bilayer.update_temperature(T_val)
                dtheta_model_update[i,j] = np.degrees(bilayer.thetaT - theta0)    

    plt.figure(dpi=200)
    plt.title(titlestr)
    plt.xlabel("Temperature ($^\circ$C)")
    plt.ylabel(r"Change in angle $\theta_T$ from RT ($^\circ$)")
    letters = ['A','B','C']
    labels = ['{0}: h={1:.2f}, r={2:.2f}'.format(letters[i],h_avg[i],r_avg[i]) for i in range(3)]
    colors = ['r','g','b']
    markers = ['o','^']
    for i in range(3):
        plt.errorbar(T, dtheta_avg[i,:], fmt=colors[i]+markers[0], yerr=dtheta_std[i,:], capsize=4, label=labels[i])
        plt.plot(T_range, dtheta_model_update[i,:], colors[i], linestyle='--', label='model: '+labels[i])
    plt.legend()
    if saveFlag:
        filename = 'r-const_angle'
        plt.savefig(os.path.join(figdir,"{0}.png".format(filename)),dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(filename)))   
    
    if 'h' in titlestr:
        return None
    else:
        return b_fit

#%% 
if __name__ == "__main__":
   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/bilayer_properties")
    tmpdir = os.path.join(cwd,"tmp")

    test_BilayerModel()