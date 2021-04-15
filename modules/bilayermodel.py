"""
Define BilayerModel class and h-T phase space analysis for curvature
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

import modules.LCEmodel as LCE
import modules.PDMSmodel as PDMS


class BilayerModel:
    """ Model for a PDMS-LCE bilayer hinge of total thickness h_total and
        LCE:total thickness ratio r. Initialized at room temperature, but can
        be updated for any temperature. Specify only one of s or b."""
    def __init__(self, h_total, ratio, T=25.0,
                 LCE_strain_params=[], LCE_modulus_params=[],
                 w=0.010, s=0, b=[], bFlag='lin'):
        
        assert T > 20.0, "Invalid T: lower than room temperature"
        assert h_total > 0, "Invalid h: must be positive"
        assert h_total < 0.01, "Make sure h is in meters, not millimeters"
        assert (ratio >= 0 and ratio < 1), "Invalid r: must be in [0,1)"
        
        # Main model parameters
        self.T = T                  # [Celcius]     Temperature
        self.h_total = h_total      # [m]           Total bilayer hinge thickness
        self.ratio = float(ratio)   # [m/m]         Bilayer hinge LCE:PDMS thickness ratio
        
        # Material parameters
        self.LCE_strain_params = LCE_strain_params
        self.LCE_modulus_params = LCE_modulus_params
        
        # Geometric parameters: out-of-plane width w, and arc length s
        self.w = w
        if s == 0:
            assert b, "Specify either arc length s or parameters b with bFlag"
            self.bFlag = bFlag
            self.s = self._calculate_arc_length(b, bFlag)
        else:
            assert s < 0.01, "Make sure s is in meters, not millimeters"
            self.s = s
        
        # Geometric parameters
        self.h_PDMS = h_total*(1-ratio)    # [m] hinge thickness, PDMS
        self.h_LCE =  h_total*ratio        # [m] hinge thickness, LCE. Assume LCE is on "inside" of hinge
        
        # Calculate all temperature-dependent parameters:
        # - material stiffness for each material
        # - second moments of area for each material, w.r.t. both neutral axis and centroid
        # - stiffness, curvature, and angle
        self._set_temperature_dependent_parameters()


    def update_temperature(self, T):
        """ Update hinge properties given new temperature """  
        self.T = T
        self._set_temperature_dependent_parameters()
        return
    
    def _set_temperature_dependent_parameters(self):
        """ Set all temperature-dependent parameters """  
        self.E_PDMS = PDMS.model_elastic_modulus(self.T) # [Pa] traditional AML PDMS; default from 08/07/20
        self.E_LCE = LCE.model_elastic_modulus(self.T, *(self.LCE_modulus_params)) # [Pa] Rui's formulation; default from 08/17/20
        self.I_PDMS, self.I_LCE = self._calculate_2nd_moments()  # [m^4] w.r.t individual centroids
        self.I_PDMS_NA, self.I_LCE_NA = self._calculate_2nd_moments_neutral_axis() # [m^4] w.r.t neutral axis
        self.k = self._calculate_stiffness() # [m] stiffness of composite hinge
        self.curvature = self._calculate_curvature() # [1/m]
        self.thetaT = self._calculate_angle() # [radians]     
    
    def _calculate_arc_length(self, b, bFlag):
        """ Using arc length parameters b, compute effective arc length s from
            hinge thickness h """
        if bFlag == 'lin':
            s = b[0] + b[1]*self.h_total
        elif bFlag == 'quad':
            s = b[0]*self.h_total**2
        else:
            assert False, "choose 'lin' or 'quad' for arc length fitting"
        return s
        
    def _calculate_stiffness(self):
        """ Using physical parameters, calculate effective linear spring constant 
            from beam theory, assuming a rectangular beam bending in thickness. 
            Second moments of area are defined about the neutral axis."""
        k0 = (self.E_PDMS*self.I_PDMS_NA + self.E_LCE*self.I_LCE_NA)/self.s
        return k0  

    def _calculate_2nd_moments(self):
        """ Calculate 2nd moments of area of PDMS and LCE layers individually  """
        # 2nd moments of area w.r.t. their own axes                                          
        I_PDMS = self.w*self.h_PDMS**3/12
        I_LCE = self.w*self.h_LCE**3/12
        
        return I_PDMS, I_LCE
     
    def _calculate_2nd_moments_neutral_axis(self):
        """ Shift 2nd moments of area of PDMS and LCE to be about neutral axis """
        # Location of effective neutral axis w.r.t. PDMS surface
        h0 = ((self.E_PDMS*self.h_PDMS**2/2 + self.E_LCE*self.h_LCE**2/2
               + self.E_LCE*self.h_PDMS*self.h_LCE)/(self.E_PDMS*self.h_PDMS
                                                     + self.E_LCE*self.h_LCE))
                                                     
        # 2nd moments of area w.r.t. neutral axis (parallel axis theorem)                                            
        I_PDMS_NA = self.I_PDMS + self.w*self.h_PDMS*(h0 - self.h_PDMS/2)**2
        I_LCE_NA  = self.I_LCE  + self.w*self.h_LCE*(self.h_total - h0 - self.h_LCE/2)**2
        
        return I_PDMS_NA, I_LCE_NA
    
    def _calculate_curvature(self):
        """ Calculate curvature [1/m] at a given temperature.
            Second moments of area are defined about each component's centroid. """
        if self.ratio == 0:
            kappa = 0
        else:
            numer = LCE.model_strain(self.T, *(self.LCE_strain_params))
            denom = ((2/self.h_total)*(self.E_LCE*self.I_LCE + self.E_PDMS*self.I_PDMS)
                     *(1/(self.E_LCE*self.h_LCE*self.w) + 1/(self.E_PDMS*self.h_PDMS*self.w))
                     + (self.h_total/2))
            kappa = numer/denom
        return -kappa
    
    def _calculate_angle(self):
        """ Calculate hinge angle at a given temperature as s*kappa. 
            Note that this is different from the unit cell definition of thetaT
            by a factor of 2."""
        angle = self.s*self.curvature
        return angle


#%%
def test_BilayerModel():
    """ Speed test and sanity checks on BilayerModel """
    
    h_total = 0.001
    b = [0, 1] # s = b[0] + b[1]h
    bFlag = 'lin'
    
    bilayer = BilayerModel(h_total, 0.0, b=b, bFlag=bFlag)
    kappa0 = bilayer.curvature
    bilayer.update_temperature(100)
    kappa1 = bilayer.curvature
    assert kappa0 == kappa1, 'r = 0 should not bend with temperature change'

    t0 = time.perf_counter()
    bilayer = BilayerModel(h_total, 0.5, b=b, bFlag=bFlag)
    kappa0 = bilayer.curvature
    t1 = time.perf_counter()
    bilayer.update_temperature(100)
    kappa1 = bilayer.curvature
    t2 = time.perf_counter()
    print(" Create:\t {0:.2e} s\n Update:\t {1:0.2e} s".format(t1-t0, t2-t1))
    assert kappa0 < kappa1, 'change in curvature should be positive with increased temperature'

    print("BilayerModel passes tests")
    
    return True


#%%   
def analyze_curvature_change_with_temp(LCE_modulus_params=[], LCE_strain_params=[],
                                       saveFlag=False, figdir='', verboseFlag=False):
    """ Generate 2D color plots of normalized angle (s = h) on h-T axes """
    
    h_range = [1e-3] #[m]
    r_range = np.arange(0.01,1.0,0.01)
    T_range = np.arange(25,101,1)
    
    for k, h in enumerate(h_range):
        kappaT_vals = np.zeros((len(T_range), len(r_range)))
        thetaT_vals = np.zeros((len(T_range), len(r_range)))
        for j, r in enumerate(r_range):
            bilayer = BilayerModel(h, r, s=h,
                                   LCE_modulus_params=LCE_modulus_params,
                                   LCE_strain_params=LCE_strain_params)
            for i, T in enumerate(T_range):
                bilayer.update_temperature(T)
                kappaT_vals[i,j] = bilayer.curvature
                thetaT_vals[i,j] = bilayer.thetaT
        indices = np.unravel_index(np.argmax(thetaT_vals), thetaT_vals.shape)
        
        print("Maximum normalized curvature is {0:.4f} at r={1:.3f} and T={2:.1f}".format(
            thetaT_vals[indices], r_range[indices[1]], T_range[indices[0]]))
        
        colorplot_change_with_temp(h, h, r_range, T_range, thetaT_vals,
                                   barLabel='Normalized curvature $h\kappa$',
                                   figLabel='thetaT',
                                   saveFlag=saveFlag, figdir=figdir)
        if verboseFlag:
            colorplot_change_with_temp(h, h, r_range, T_range, kappaT_vals,
                                       barLabel='Curvature $\kappa$ (1/m)',
                                       figLabel='kappaT',
                                       saveFlag=saveFlag, figdir=figdir)
    
    return


def colorplot_change_with_temp(h, s, r_range, T_range, curvatureT_vals, 
                               barLabel='', figLabel='', saveFlag=False, figdir=''):
    """ 2D color plot of curvature on h-T axes """
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


#%% 
if __name__ == "__main__":
   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd, "data/raw/bilayer_properties")
    tmpdir = os.path.join(cwd, "tmp")

    test_BilayerModel()