# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:43:38 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt

import LCEmodel as LCE
import PDMSmodel as PDMS

class BilayerModel:
    def __init__(self, h_total, ratio, T,
                 E_PDMS=0, E_LCE=0,
                 a=0.018, w=0.010):
        # Main model parameters
        self.T = T                  # [Celcius]     Temperature
        self.h_total = h_total      # [m]           Total bilayer hinge thickness
        self.ratio = ratio          # [m/m]         Bilayer hinge LCE:PDMS thickness ratio
        
        if E_PDMS == 0: 
            self.E_PDMS = PDMS.model_elastic_modulus()         # [Pa] traditional printable PDMS (measured)
        else:
            self.E_PDMS = E_PDMS
        if E_LCE == 0:
            self.E_LCE = LCE.model_elastic_modulus(T)           # [Pa] Rui's formulation. Previously, measured to be 1.75*10**6
        else:
            self.E_LCE = E_LCE
        
        self.w = w  # beam width
        self.b = [0, 2.0]           # hinge beam arc length = b[0] + b[1]*h_total
        
        self.h_PDMS = h_total*(1-ratio)    # [m] hinge thickness, PDMS
        self.h_LCE =  h_total*ratio        # [m] hinge thickness, LCE. Assume LCE is on "inside" of hinge
        self.I_PDMS, self.I_LCE = self.calculate_I_vals(T) # [m] hinge thickness, PDMS
        
        self.k = self.calculate_linear_torsion_stiffness() # [m] stiffness of composite hinge
        self.thetaT = self.calculate_angle(T)

    """ Update hinge properties given new temperature """       
    def update_properties(self, T):
        self.T = T
        self.E_LCE = LCE.model_elastic_modulus(T)
        self.I_PDMS, self.I_LCE = self.calculate_I_vals(T)
        self.k = self.calculate_linear_torsion_stiffness()
        self.thetaT = self.calculate_angle(T)
        
        
    """ Using physical parameters, calculate effective linear spring constant 
        from beam theory, assuming a rectangular beam bending in thickness. """        
    def calculate_linear_torsion_stiffness(self):
        s = self.b[0] + self.b[1]*self.h_total
        k0 = (self.E_PDMS*self.I_PDMS + self.E_LCE*self.I_LCE)/s
        return k0  

    """ Calculate second moments of area """
    def calculate_I_vals(self, T):
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
    
        """ Calculate angle at a given temperature """
    def calculate_angle(self, T):
        numer = self.calculate_LCEstrain(T)
        denom = ((2/self.h_total)*(self.E_LCE*self.I_LCE + self.E_PDMS*self.I_PDMS)
                 *(1/(self.E_LCE*self.h_LCE*self.w) + 1/(self.E_PDMS*self.h_PDMS*self.w))
                 + (self.h_total/2))
        kappa = numer/denom  # curvature [1/m]
        s = self.b[0] + self.b[1]*self.h_total # arc length [m]
        return -s*kappa