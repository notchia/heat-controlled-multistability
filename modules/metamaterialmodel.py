# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:27:45 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import argrelextrema

import pltformat  # module: no funcions, just standardized plot formatting
import magnetmodel as magnet
import bilayermodel as bilayer


class MetamaterialModel:
    def __init__(self, h_total, ratio, T, thetaL,
                 m=0, E_PDMS=0, E_LCE=0,
                 a=0.018, w=0.010,
                 limFlag='piecewise'):
        # Main model parameters
        self.T = T                  # [Celcius]     Temperature
        self.h_total = h_total      # [m]           Total bilayer hinge thickness
        self.ratio = ratio          # [m/m]         Bilayer hinge LCE:PDMS thickness ratio
        self.thetaL = thetaL        # [rad]         As-fabricated angle of hinge
        # Material parameters: fit to data
        if m == 0: 
            self.m = magnet.model_force()           # [J/Celcius] or [A m^2] magnetic moment (measured)
        else:
            self.m = m
        self.hinge = bilayer.BilayerModel(h_total, ratio, T,
                                          E_PDMS=E_PDMS, E_LCE=E_LCE, w=w)
        
        self.a =  a             # [m] diagonal defining square lattice  

        # Model fit parameters
        self.limFlag = limFlag
        self.p_lim = [0.2, np.radians(41)] # [Nm] (k_lim, q_lim) Effective stiffness of the squares when in contact



    """ Update all values which vary with temperature, in the appropriate order"""  
    def update_T(self, T):
        self.T = T
        self.hinge.update_properties(T) # Will need to check that this works
        return
    
    
    """ Return total equilibrium angle (sum of fabrication and temperature 
        contributions)"""  
    def get_total_angle(self):
        return self.thetaT + self.thetaL
    

    """ Calculate energy of collision of adjacent squares """
    def model_collision_energy(self, q):
        if self.limFlag == 'exp': #Exponential
            A, B =[5e-20, 50] #self.p_lim #[5e-20, 50]
            U_limit = A*(np.exp(B*2*(q)/2) + np.exp(-B*2*(q)/2))
        elif self.limFlag == 'piecewise': #Piecewise
            k_limit, q_limit = self.p_lim #[0.2, np.radians(41)]
            if q > q_limit:
                U_limit = 0.5*k_limit*(q-q_limit)**2
            elif (q < -q_limit):
                U_limit = 0.5*k_limit*(q+q_limit)**2
            else:
                U_limit = 0
        return U_limit

    """ Potential energy of magnet, torsion spring, and square deformation at
        a given angle. Returns both total energy and list of component energies.
        thetas: array"""    
    def calculate_total_energy(self, thetas):
        U_total = np.zeros(len(thetas))
        U_m_arr = np.zeros(len(thetas))
        U_k_arr = np.zeros(len(thetas))
        U_lim_arr = np.zeros(len(thetas))
        for i in range(len(thetas)):
            theta = thetas[i]
            d = (self.a*np.cos(theta))
            U_m = -(mu*self.m**2)/(2*np.pi*d**3)
            U_k = 2*self.k*(theta - self.thetaT - self.thetaL)**2
            U_limit = self.model_collision_energy(theta)
            U_total[i] = U_m + U_k + U_limit
            U_m_arr[i] = U_m
            U_k_arr[i] = U_ks
            U_lim_arr[i] = U_limit

        return U_total, U_m_arr, U_k_arr, U_lim_arr
    
    """ Plot total energy curve, contributions from magnet and hinge, and local extrema """   
    def plot_energy(self, q, U_total, U_m, U_k, U_lim):
        q_deg = np.degrees(q)
        plt.figure()

        plt.plot(q_deg,U_m,'b--',label="Magnet")
        plt.plot(q_deg,U_k,'r--',label="Beam")
        plt.plot(q_deg,U_lim,'g--',label="Limit")
        plt.plot(q_deg,U_total,'k',label="Total")
        
        minima = argrelextrema(U_total,np.less)
        maxima = argrelextrema(U_total,np.greater)
        plt.plot(q_deg[minima],U_total[minima],'gv')
        plt.plot(q_deg[maxima],U_total[maxima],'r^')
        plt.xlabel(r"Angle $\theta$ (degrees)")
        plt.ylabel("Energy (J)")
        plt.title(r"$h$ = {0}, $r$ = {1}, $T$ = {2}, $\theta_L$ = {3:0.1f}".format(self.h_total, self.ratio, self.T, np.degrees(self.thetaL)))
        plt.legend()
        plt.tight_layout()
        plt.savefig('h{0}_r{1}_T{2}_L{3:0.1f}.png'.format(self.h_total, self.ratio, self.T, np.degrees(self.thetaL)))
        plt.close()
        
        return
   
    """ Count local extrema of energy curve and find normalized energy barriers
        and differences between them """    
    def analyze_energy_local_extrema(self, energy, q=np.radians(np.arange(-45.0,45.5,0.5)), q0=0.0):
        minLoc = signal.argrelmin(energy)[0]
        maxLoc = signal.argrelmax(energy)[0]
        numMin = len(minLoc)
        phase = -1 # 0 = LCR; 1,2,3 = LR,LC,CR; 4,5,6 = L,C,R
        locGuess = [-np.pi/4, q0, np.pi/4]
        
        # Compute energy barriers for minima and energy differences between minima
        if numMin == 3:
            # Tristable: three local minima at left, center, and right (LCR)
            diff_CL = (energy[minLoc[1]] - energy[minLoc[0]])
            diff_CR = (energy[minLoc[1]] - energy[minLoc[2]])
            ratio_LR  = (diff_CL - diff_CR)/(diff_CL + diff_CR) #xi from paper    
            
            barrier_CL = (energy[maxLoc[0]] - energy[minLoc[1]])
            barrier_CR = (energy[maxLoc[1]] - energy[minLoc[1]])
            
            phase = 0
            
            return numMin, [diff_CL, diff_CR, ratio_LR], [barrier_CL, barrier_CR], phase
        
        elif numMin == 2:
            # Bistable: two local minima (CL, CR, or LR)
            diff = energy[minLoc[0]] - energy[minLoc[1]]
            barrier = energy[maxLoc[0]] - energy[minLoc[1]]

            iLoc1 = (np.abs(locGuess - q[minLoc[0]])).argmin()
            iLoc2 = (np.abs(locGuess - q[minLoc[1]])).argmin()
            if iLoc1 == 0 and iLoc2 == 2:
                phase = 1
            elif iLoc1 == 0:
                phase = 2
            elif iLoc2 == 2:
                phase = 3
            
            return numMin, [diff], [barrier], phase
        
        else:
            # Monostable: single local minimum (L, C, or R)   
            pvals = [4, 5, 6]
            iLoc = (np.abs(locGuess - q[minLoc[0]])).argmin()
            phase = pvals[iLoc]
            
            return numMin, [], [], phase    
    
    
#--------------------------------------------------------------------------
    
''' Given parameters, plot energy and return model, figure flag'''
def analyze_parameter_energy(h_total, ratio, T, thetaL):
    q = np.radians(np.arange(-45.0,45.5,0.5))
    sample = MetamaterialModel(h_total=h_total, ratio=ratio, T=T, thetaL=thetaL)
    U_total, U_m, U_k, U_lim = sample.calculate_total_energy(q)
    sample.plot_energy(q, U_total, U_m, U_k, U_lim)
    print("Equivalent torsional spring constant: {0}".format(sample.k))
    return sample
    
#--------------------------------------------------------------------------
    
if __name__ == "__main__":
    analyze_parameter_energy(0.001, 0.5, 25.0, 0.0)