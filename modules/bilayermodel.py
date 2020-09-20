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
                 w=0.010, b=[], s=0):
        
        assert T > 20.0, "Invalid T: lower than room temperature"
        assert h_total > 0, "Invalid h: must be positive"
        assert h_total < 0.01, "Make sure h is in meters, not millimeters"
        assert (ratio >= 0 and ratio < 1), "Invalid r: must be within [0,1)"
        
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
            assert b, 'Specify either arc length s or parameters b=[b0, b1]'
            self.s = quadratic_pure(h_total, b)
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
    ''' Sanity checks on BilayerModel '''
         
    bilayer = BilayerModel(0.001, 0.0, b=[0,1])
    kappa0 = bilayer.curvature
    bilayer.update_temperature(100)
    kappa1 = bilayer.curvature
    assert kappa0 == kappa1, 'r = 0 should not bend with temperature change'

    t0 = time.perf_counter()
    bilayer = BilayerModel(0.001, 0.5, b=[0,1])
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
    curvature_from_model = np.zeros_like(curvature_avg)
    angle_from_model = np.zeros_like(curvature_avg)
    for i in range(nCombos):
        h_val = h[i]
        r_val = r[i]
        bilayer = BilayerModel(h_val*1e-3, r_val, b=[0,1])
        for j in range(nTemps):
            bilayer.update_temperature(T[j])
            curvature_from_model[j,i] = bilayer.curvature
            angle_from_model[j,i] = bilayer.thetaT
    
    # Plot curvature and angle, comparing model and experiments
    plt.figure('temp-curvature-model', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Change in curvature $\kappa - \kappa_0$ (1/m)")
    curvature_avg_zeroed = curvature_avg - curvature_avg[0]

    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, curvature_avg_zeroed[:,i], yerr=curvature_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T, curvature_from_model[:,i], marker='.', linestyle='dashed',
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
        plt.plot(T, np.degrees(angle_from_model[:,i]), marker='.', linestyle='dashed',
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
        #plt.plot(r, np.transpose(np.degrees(angle_from_model[i,:])), marker='^', linewidth=0, color=colors[i],label="T = {0}, model".format(T[i]))  
    plt.legend()
    # Relation to total thickness h:
    plt.figure('thickness-curvature', dpi=200)
    plt.xlabel("Total thickness h (mm/mm)")
    plt.ylabel("Normalized curvature $(\kappa - \kappa_0)/r$ (1/m)")
    normalized_curvature_avg = np.divide(curvature_avg_zeroed, r)
    normalized_curvature_std = np.divide(curvature_std, r)
    for i in range(nTemps):
        plt.errorbar(h, np.transpose(normalized_curvature_avg[i,:]), yerr=np.transpose(normalized_curvature_std[i,:]), fmt='o', capsize=2,
                     label="T = {0}".format(T[i]), color=colors[i])
    plt.legend()

    return


def analyze_data_200819(paramfile, datafile, LCE_modulus_params=[-0.082755,237.64,1097398],
                        saveFlag=False, figdir=''):
    ''' Process samples 1-6. Note samples 4 & 5 are nominally the same, so compiled into one'''
    
    arr = np.genfromtxt(paramfile,dtype=float,delimiter=',',skip_header=2)
    h = arr[0,1:6]
    r = arr[1,1:6]
    h_std = arr[0,6:]
    r_std = arr[0,6:]
    print('h: ' + ', '.join(['{:.2f}'.format(val) for val in h]) + ' (mm)')
    print('r: ' + ', '.join(['{:.2f}'.format(val) for val in r]) + ' (mm/mm)')
    arr = np.genfromtxt(datafile,dtype=float,delimiter=',',skip_header=1)
    T = arr[:,0]
    curvature_avg = arr[:,1:6]
    curvature_std = arr[:,6:11]
    
    nTemps = curvature_avg.shape[0]
    nCombos = curvature_avg.shape[1]
    
    #s = h # use arc length = thickness
    angle_avg = np.degrees(1e-3*np.multiply(h,curvature_avg))
    angle_std = np.degrees(1e-3*np.multiply(h,curvature_std))
    angle_avg = angle_avg - angle_avg[0]
    
    colors = ['blue','orange','red','green','purple','brown','pink','gray','olive','cyan','magenta']
    
    # Plot measured curvature as a function of temperature for all samples
    plt.figure('temp-curvature1', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Measured curvature $\kappa$ (1/m)")
    for i in range(nCombos):
        plt.errorbar(T, curvature_avg[:,i], yerr=curvature_std[:,i], 
                     label="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i]), 
                     fmt='o', capsize=2, color=colors[i])
    plt.legend()

    # Compute model values for these parameters (with arc length = thickness)
    T_range = np.arange(T[0],T[-1],1)
    nTempRange = len(T_range)
    curvature_from_model = np.zeros((nTempRange,nCombos))
    angle_from_model = np.zeros((nTempRange,nCombos))

    for i in range(nCombos):
        h_val = h[i]
        r_val = r[i]
        bilayer = BilayerModel(h_val*1e-3, r_val, b=[0,1], LCE_modulus_params=LCE_modulus_params)
        for j in range(nTempRange):
            bilayer.update_temperature(T_range[j])
            curvature_from_model[j,i] = bilayer.curvature
            angle_from_model[j,i] = bilayer.thetaT
    
    # Plot curvature and angle, comparing model and experiments
    # Change in curvature: 
    plt.figure('temp-curvature-model1', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Change in curvature $\kappa - \kappa_0$ (1/m)")
    curvature_avg_zeroed = curvature_avg - curvature_avg[0]
    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, curvature_avg_zeroed[:,i], yerr=curvature_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T_range, curvature_from_model[:,i], linestyle='dashed',
                 linewidth=2, color=colors[i],label='model: '+labelstr)
    plt.legend()
    plt.tight_layout()
    if saveFlag:
        plt.savefig(os.path.join(figdir,"bilayer_curvature.svg"), transparent=True)
        plt.savefig(os.path.join(figdir,"bilayer_curvature.png"), dpi=200)
    # Change in angle: 
    fig = plt.figure('temp-angle1', dpi=200) 
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Change in angle $h(\kappa - \kappa_0)$ (deg)")
    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, angle_avg[:,i], yerr=angle_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T_range, np.degrees(angle_from_model[:,i]), linestyle='dashed',
                 linewidth=2, color=colors[i],label='model: '+labelstr)        
    plt.legend()
    plt.tight_layout()
    if saveFlag:
        plt.savefig(os.path.join(figdir,"bilayer_angle.svg"), transparent=True)
        plt.savefig(os.path.join(figdir,"bilayer_angle.png"),dpi=200)
    # Change in curvature, normalized by r: 
    normalized_curvature_avg = np.divide(curvature_avg_zeroed, r)
    normalized_curvature_std = np.divide(curvature_std, r)
    normalized_curvature_from_model = np.divide(curvature_from_model, r)
    plt.figure('temp-curvature-model-norm', dpi=200)
    plt.xlabel("Temperature ($^{{\circ}}C$)")
    plt.ylabel("Normalized change in curvature $(\kappa - \kappa_0)/r$ (1/m)")
    for i in range(nCombos):
        labelstr="h = {0:.1f} mm, r = {1:.2f}".format(h[i], r[i])
        plt.errorbar(T, normalized_curvature_avg[:,i], yerr=normalized_curvature_std[:,i], fmt='o', capsize=2,
                     label=labelstr, color=colors[i])
        plt.plot(T_range, normalized_curvature_from_model[:,i], linestyle='dashed',
                 linewidth=2, color=colors[i],label='model: '+labelstr)
    plt.legend()
    plt.tight_layout()
    if saveFlag:
        plt.savefig(os.path.join(figdir,"bilayer_normalized_curvature.png"), dpi=200)
        fig.patch.set_facecolor('None')
        plt.savefig(os.path.join(figdir,"bilayer_normalized_curvature.svg"), transparent=True)
        
    # Can the curvature be normalized by some values?
    # Relation to LCE:total thickness ratio r:
    plt.figure('ratio-curvature1', dpi=200)
    plt.xlabel("LCE:total ratio $r$ (mm/mm)")
    plt.ylabel("Change in angle $h(\kappa - \kappa_0)$ (deg)")
    for i in range(nTemps):
        plt.errorbar(r, np.transpose(angle_avg[i,:]), yerr=np.transpose(angle_std[i,:]), xerr=np.transpose(r_std), fmt='o', capsize=2,
                     label="T = {0}".format(T[i]), color=colors[i])
        fit_to_r = np.polyfit(r, angle_avg[-2,:],1)
    plt.plot(r, fit_to_r[0]*r + fit_to_r[1], color=colors[nTemps-2], label='linear fit')
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.tight_layout()
    if saveFlag:
        plt.savefig(os.path.join(figdir,"bilayer_r_dependence.svg"), transparent=True)
        plt.savefig(os.path.join(figdir,"bilayer_r_dependence.png"), dpi=200)
    # Relation to total thickness h:
    plt.figure('thickness-curvature1', dpi=200)
    plt.xlabel("Total thickness $h$ (mm)")
    plt.ylabel("Normalized curvature $(\kappa - \kappa_0)/r$ (1/m)")
    normalized_curvature_avg = np.divide(curvature_avg_zeroed, r)
    normalized_curvature_std = np.divide(curvature_std, r)
    for i in range(nTemps):
        plt.errorbar(h, np.transpose(normalized_curvature_avg[i,:]), yerr=np.transpose(normalized_curvature_std[i,:]), xerr=np.transpose(h_std), fmt='o', capsize=2,
                     label="T = {0}".format(T[i]), color=colors[i])
    fit_to_h = np.polyfit(h, normalized_curvature_avg[-2,:],1)
    plt.plot(h, fit_to_h[0]*h + fit_to_h[1], color=colors[nTemps-2], label='linear fit')
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.tight_layout()
    if saveFlag:
        plt.savefig(os.path.join(figdir,"bilayer_h_dependence1.svg"), transparent=True)
        plt.savefig(os.path.join(figdir,"bilayer_h_dependence1.png"), dpi=200)    
    # Another relation to total thickness h:
    normalized_angle_avg = np.divide(angle_avg, r)
    normalized_angle_std = np.divide(angle_std, r)
    plt.figure('thickness-angle1', dpi=200)
    plt.xlabel("Total thickness $h$ (mm)")
    plt.ylabel("Change in normalized angle $h(\kappa - \kappa_0)/r$ (deg)")
    for i in range(nTemps):
        plt.errorbar(h, np.transpose(normalized_angle_avg[i,:]), yerr=np.transpose(normalized_angle_std[i,:]), xerr=np.transpose(h_std), fmt='o', capsize=2,
                     label="T = {0}".format(T[i]), color=colors[i])
    fit_to_h_norm = np.polyfit(h, normalized_angle_avg[-2,:],1)
    plt.plot(h, fit_to_h_norm[0]*h + fit_to_h_norm[1], color=colors[nTemps-2], label='linear fit')
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.tight_layout()
    if saveFlag:
        plt.savefig(os.path.join(figdir,"bilayer_h_dependence2.svg"), transparent=True)
        plt.savefig(os.path.join(figdir,"bilayer_h_dependence2.png"), dpi=200)    
    # Scale the model by a fit to h. Really what should be done is finding fit_to_h
    # for each temperature, then fitting a curve to how the slope of the fit
    # changes with temperature. Right now the fit is probably good enough that
    # this is unnecessary.
    normalized_adjusted_curvature = adjust_by_h(h, r, normalized_curvature_from_model, fit_to_h=fit_to_h, h_ref=h[0])
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
    plt.tight_layout()

    return

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
    return adjustment


""" 2D color plot of angle on h-T axes """  
def analyze_angle_change_with_temp(LCE_modulus_params=[-0.082755,237.64,1097398],
                                   b=[0.0,1.5], saveFlag=False, figdir=''):
    
    h_range = 1e-3*np.arange(0.5,2.5,0.5) #[m]
    r_range = np.arange(0.01,1.0,0.01)
    T_range = np.arange(25,101,1)
    
    for k, h in enumerate(h_range):
        kappaT_vals = np.zeros((len(T_range), len(r_range)))
        thetaT_vals = np.zeros((len(T_range), len(r_range)))
        for j, r in enumerate(r_range):
            bilayer = BilayerModel(h, r, LCE_modulus_params=LCE_modulus_params, b=b)
            for i, T in enumerate(T_range):
                bilayer.update_temperature(T)
                kappaT_vals[i,j] = bilayer.curvature
                thetaT_vals[i,j] = np.degrees(bilayer.thetaT)
        s = b[0]+b[1]*h
        plot_angle_change_with_temp(h, s, r_range, T_range, thetaT_vals,
                                    saveFlag=saveFlag, figdir=figdir)
        plot_curvature_change_with_temp(h, s, r_range, T_range, kappaT_vals,
                                    saveFlag=saveFlag, figdir=figdir)
    
    return
  

def plot_angle_change_with_temp(h, s, r_range, T_range, thetaT_vals, 
                                saveFlag=False, figdir=''):
    ''' 2D color plot of angle on h-T axes '''
    r_range, T_range = np.meshgrid(r_range, T_range)
    colorlevels = np.arange(np.amin(thetaT_vals), np.amax(thetaT_vals), 0.5)
    
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.contourf(r_range, T_range, thetaT_vals, levels=colorlevels)    
    plt.xlabel('$r$ (mm/mm)')
    plt.ylabel('$T$ ($^\circ$C)')
    plt.title('$h$ = {0:0.2f} mm, $s$ = {1:0.2f} mm'.format(1000*h, 1000*s))
    bar = fig.colorbar(surf, aspect=8)
    bar.set_label('Angle (degrees)')
    plt.tight_layout()
    if saveFlag:
        filename = 'thetaT_h{0:0.2f}_s{1:0.2f}.png'.format(1000*h, 1000*s)
        plt.savefig(os.path.join(figdir,filename),dpi=200)
        plt.close()

    return

def plot_curvature_change_with_temp(h, s, r_range, T_range, curvatureT_vals, 
                                saveFlag=False, figdir=''):
    ''' 2D color plot of angle on h-T axes '''
    r_range, T_range = np.meshgrid(r_range, T_range)
    colorlevels = np.arange(np.amin(curvatureT_vals), np.amax(curvatureT_vals), 0.5)
    
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.contourf(r_range, T_range, curvatureT_vals, levels=colorlevels)    
    plt.xlabel('$r$ (mm/mm)')
    plt.ylabel('$T$ ($^\circ$C)')
    plt.title('$h$ = {0:0.2f} mm, $s$ = {1:0.2f} mm'.format(1000*h, 1000*s))
    bar = fig.colorbar(surf, aspect=8)
    bar.set_label('Curvature (1/m)')
    plt.tight_layout()
    if saveFlag:
        filename = 'kappaT_h{0:0.2f}_s{1:0.2f}.png'.format(1000*h, 1000*s)
        plt.savefig(os.path.join(figdir,filename),dpi=200)
        plt.close()

    return

#%%
def quadratic_with_x0(x, b2, b1, b0=0):
    return b2*x**2 + b1*x + b0

def quadratic_pure(x, b):
    return b*x**2

def analyze_bending_angles(datapath, parampath,
                           LCE_modulus_params=[-8.43924097e-02,2.16846882e+02,3.13370660e+05],
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
                               LCE_modulus_params=LCE_modulus_params)
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

    plt.figure('h-vs-s',dpi=200)
    if 'h' in titlestr:
        mkr='o'
    else:
        mkr='^'
        #hsfit, _ = opt.curve_fit(lambda x, b: b*x, h_avg, s_fit)
        #b_fit = [0, hsfit[0]]
        #plt.plot(h_avg, b_fit[1]*h_avg, 'r', label='linear fit to fixed-r data: s = {0:.2f}h + {1:.2f}'.format(*b_fit))

        hsfit, _ = opt.curve_fit(quadratic_pure, h_avg, s_fit)
        b_fit = 1e3*hsfit#1e-3*hsfit#[1e-3*hsfit[1], hsfit[0]]
        plt.plot(h_avg, quadratic_pure(h_avg, hsfit), 'r', label='fit to fixed-r data, with b = {0}'.format(b_fit))
        
    plt.plot(h_avg, s_avg, mkr+'k', label='measured, {}'.format(titlestr))
    plt.plot(h_avg, s_fit, mkr+'r', label='best fit, {}'.format(titlestr))
    plt.xlabel('h (mm)')
    plt.ylabel('s (mm)')
    plt.legend()
    
    # update value
    dtheta_model_update = np.zeros((3,len(T_range)))
    if 'h' not in titlestr:
        for i in range(3):
            h_val = h_avg[i]
            r_val = r_avg[i]
            s_val = s_avg[i]
            bilayer = BilayerModel(h_val*1e-3, r_val, b=b_fit, T=24.3,
                                   LCE_modulus_params=LCE_modulus_params)
            theta0 = bilayer.thetaT
            for j, T_val in enumerate(T_range):
                bilayer.update_temperature(T_val)
                dtheta_model_update[i,j] = np.degrees(bilayer.thetaT - theta0)    

    plt.figure(dpi=200)
    plt.title(titlestr)
    plt.xlabel("Temperature ($^\circ$C)")
    plt.ylabel("Change in angle from RT ($^\circ$)")
    letters = ['A','B','C']
    labels = ['{0}: h={1:.2f}, r={2:.2f}'.format(letters[i],h_avg[i],r_avg[i]) for i in range(3)]
    colors = ['r','g','b']
    markers = ['o','^']
    for i in range(3):
        plt.errorbar(T, dtheta_avg[i,:], fmt=colors[i]+markers[0], yerr=dtheta_std[i,:], capsize=4, label=labels[i])
        plt.plot(T_range, dtheta_model_update[i,:], colors[i], linestyle='--')
    plt.legend()
    
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