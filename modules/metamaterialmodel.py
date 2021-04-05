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

from modules import bilayermodel as bilayer


class MetamaterialModel:
    """ Model for a PDMS-LCE bilayer hinge of total thickness h_total and
        LCE:total thickness ratio r. Initialized at room temperature, but can
        be updated for any temperature. Specify only one of s or b."""
    def __init__(self, h_total, ratio, thetaL, T=25.0,
                 LCE_strain_params=[], LCE_modulus_params=[],
                 d=0.018, w=0.010, b=[], s=0, m=0, bFlag='lin',
                 limFlag='exp', p_lim=[], k_sq=0.0,
                 plotFlag=False, verboseFlag=False, loadFlag=False, hasMagnets=True,
                 analysisFlag=True):
        
        # Main model parameters
        self.T = T                  # [Celcius]     Temperature
        self.h_total = h_total      # [m]           Total bilayer hinge thickness
        self.ratio = ratio          # [m/m]         Bilayer hinge LCE:PDMS thickness ratio
        self.thetaL = thetaL        # [rad]         As-fabricated angle of hinge
        
        # Material parameters: fit to data
        assert ((m != 0) if (hasMagnets) else True), 'Set a nonzero magnetic moment if the sample has magnets'
        self.m = m    # [J/Celcius] or [A m^2] magnetic moment (measured)
        self.hinge = bilayer.BilayerModel(h_total, ratio, T=T, 
                                          LCE_strain_params=LCE_strain_params,
                                          LCE_modulus_params=LCE_modulus_params,
                                          w=w, b=b, bFlag=bFlag, s=s)
        self.total_angle = self.hinge.thetaT + self.thetaL # total equilibrium angle (sum of fabrication and temperature contributions)
        
        # Geometric parameterss
        self.d = d  # [m] diagonal defining square lattice

        # Model for effective stiffness of the squares when in contact
        assert (limFlag == 'pcw' or limFlag == 'exp'), 'Choose piecewise (pcw) or exponential (exp) collision model'
        assert (len(p_lim) > 0), 'Set p_lim'
        self.limFlag = limFlag # 'pcw' OR 'exp'
        self.p_lim = p_lim # (k_lim, q_lim) OR (A, B)
        self.k_sq = k_sq
        self.k_eq = self._equivalent_torsional_spring()   

        # Run preliminary analysis: compute potential energy over a range of
        # angles, then find the local minima and corresponding phases
        self.plotFlag = plotFlag
        self.verboseFlag = verboseFlag
        self.loadFlag = loadFlag
        self.analysisFlag = analysisFlag
        self.hasMagnets = hasMagnets
        if analysisFlag:
            self._analyze_energy()

    
    # Functions callable from outside -----------------------------------------
    def update_T(self, T):
        """ Update all values which vary with temperature, in the appropriate order"""
        self.T = T
        self.hinge.update_temperature(T)
        self.total_angle = self.hinge.thetaT + self.thetaL
        self.k_eq = self._equivalent_torsional_spring()  
        if self.analysisFlag:
            self._analyze_energy()
        return

    def model_load_disp(self, disp):
        """ Given displacement x, return load corresponding to unit cell test """
        q = self._disp2rot(disp)
        M_k = self._torque_k(q)
        if self.hasMagnets:
            M_m = self._torque_magnet(q)
        else:
            M_m = 0
        M_lim = self._torque_lim(q)
        M = 4*(M_k + M_m + M_lim)
        F = -M/(self.d*np.cos(q))
        return F

    def model_energy(self, q_range=np.radians(np.arange(-50.0,50.5,0.5))):
        energy = self._calculate_total_energy(q_range)
        return energy, q_range

    # Forces and torques, for comparison to experimental data -----------------
    def _rot2disp(self, q):
        """ Displacement between centers of squares """
        return self.d*np.cos(q)
    
    def _disp2rot(self, x):
        """ Displacement between centers of squares """
        return np.arccos(x/self.d)
    
    def _equivalent_torsional_spring(self):
        if self.k_sq == 0:
            k_eq = self.hinge.k
        else:
            k_eq = 1/(1/self.hinge.k + 1/self.k_sq)
        return k_eq
    
    def _torque_k(self, q):
        """ Linear torsional spring representing hinge """
        return -self.k_eq*(2*(q - self.total_angle))
    
    def _torque_magnet(self, q):
        """ Point dipole at centers of each square """
        mu = 1.2566*10**(-6) # [N/A^2]
        return 3*mu*(self.m**2)*(self.d/2)*np.sin(q)/(2*np.pi*self._rot2disp(q)**4)
    
    def _torque_lim(self, q):
        """ Torque resulting from square collision, limiting max displacement """
        if self.limFlag=='pcw':
            f_lim = self._torque_lim_pcw_single
        elif self.limFlag=='exp':
            f_lim = self._torque_lim_exp_single
        
        if isinstance(q,(list,np.ndarray)):
            F = np.zeros_like(q)
            for i, q_val in enumerate(q):
                F[i] = f_lim(q_val)
        else:
            F = f_lim(q)
        return F
    
    def _torque_lim_pcw_single(self, q_value):
        """ Piecewise linear torsional spring model for square collision """
        q_lim, k_lim = self.p_lim
        if q_value > q_lim:
            M = -k_lim*(2*(q_value-q_lim))
        elif q_value < -q_lim:
            M = -k_lim*(2*(q_value+q_lim))
        else:
            M = 0
        return M    
    
    def _torque_lim_exp_single(self, q_value):
        """ Exponential model for square collision """
        A, B = self.p_lim
        M = -B*A*(np.exp(B*(q_value)) - np.exp(-B*(q_value)))
        return M
    
    # Enerties, for simulation and phase analysis -----------------------------
    def _analyze_energy(self):
        """ Calculate potential energy and find the minima and phases """
        q_range = np.radians(np.arange(-50.0,50.5,0.5))
        energy = self._calculate_total_energy(q_range)
        self.numMin, self.locMin, self.phases, self.diffs, self.barriers = self._analyze_energy_local_extrema(energy, q_range)
        
        return 
    
    def _model_U_limit(self, q):
        """ Calculate energy of collision of adjacent squares, given current angle between squares """
        if self.limFlag == 'exp': #Exponential
            A, B = self.p_lim #[1e-21, 50]
            U_limit = A*(np.exp(B*2*q/2) + np.exp(-B*2*q/2)) # 01/20/21: Fuck... just realized I somehow lost the -2 in here. Maybe that's why I was having numerical issues...
        elif self.limFlag == 'pcw': #Piecewise
            k_limit, q_limit = self.p_lim #[0.2, np.radians(41)]
            if q > q_limit:
                U_limit = 0.5*k_limit*(2*(q-q_limit))**2
            elif (q < -q_limit):
                U_limit = 0.5*k_limit*(2*(q+q_limit))**2
            else:
                U_limit = 0
        return U_limit

    def _model_U_magnet(self, q):
        """ Calculate energy of magnet interaction, given current angle between squares"""
        mu = 1.256637e-6 # magnetic permeability of free space
        dist = self.d*np.cos(q) # distance between square centers
        U_m = -(mu*self.m**2)/(2*np.pi*dist**3)
        return U_m

    def _model_U_spring(self, q):
        """ Calculate energy of hinge linear torsional spring deformation, given current angle between squares """
        U_k = 0.5*self.k_eq*(2*(q - self.total_angle))**2
        return U_k
 
    def _calculate_total_energy(self, q_vals):
        """ Potential energy of magnet, torsion spring, and square deformation
            at a given angle. Returns both total energy and list of component
            energies. thetas: array"""  
        assert isinstance(q_vals, np.ndarray), 'function takes array of q values'
        nVals = len(q_vals)
        
        U_total = np.zeros(nVals)
        if self.verboseFlag:
            U_m_arr = np.zeros(nVals)
            U_k_arr = np.zeros(nVals)
            U_lim_arr = np.zeros(nVals)
        for i in range(nVals):
            q = q_vals[i]
            U_m = self._model_U_magnet(q)
            U_k = self._model_U_spring(q)
            U_limit = self._model_U_limit(q)
            U_total[i] = U_m + U_k + U_limit
            if self.verboseFlag:
                U_m_arr[i] = U_m
                U_k_arr[i] = U_k
                U_lim_arr[i] = U_limit

        if self.plotFlag:
            if self.verboseFlag:
                self._plot_energy(q_vals, U_total, U_m_arr, U_k_arr, U_lim_arr)
            else:
                self._plot_energy(q_vals, U_total)

        return U_total
    
    def _plot_energy(self, q, U_total, U_m=[], U_k=[], U_lim=[], componentFlag=False):
        """ Plot total energy curve, individual contributions if specified, and
            local extrema """ 
        
        q = np.degrees(q)
        
        plt.figure(dpi=300)
        plt.title(r"$h$={0}mm, $r$={1}, $\theta_L$={2:0.1f}$^\circ$ @ $T$={3}$^\circ$C".format(self.h_total, self.ratio, np.degrees(self.thetaL), int(self.T)))
        plt.xlabel(r"Angle $\theta$ (degrees)")
        plt.ylabel("Energy (J)")        

        if componentFlag:
            if U_m.any():
                plt.plot(q, U_m, 'b--', label="Magnet")
            if U_k.any():
                plt.plot(q, U_k, 'r--', label="Spring")
            if U_lim.any():
                plt.plot(q, U_lim, 'g--', label="Collision")
            
        plt.plot(q, U_total, 'k', label="Total")
        
        minima = signal.argrelmin(U_total)[0]
        maxima = signal.argrelmax(U_total)[0]
        minU = U_total[minima]
        maxU = U_total[maxima]
        plt.plot(q[minima], minU, 'gv')
        plt.plot(q[maxima], maxU, 'r^')
        
        minU.sort()
        maxU.sort()
        if maxU.size > 0:
            plt.ylim(minU[0]-0.0001, maxU[-1]+0.0001)
        plt.ylim(-0.0015, 0.001)
        
        if componentFlag:
            if (U_m.any() or U_k.any() or U_lim.any()):
                plt.legend()
                
        plt.tight_layout()
        
        plt.savefig('h{0:.2f}_r{1:.2f}_L{2:0.1f}_T{3:.1f}.png'.format(
            self.h_total, self.ratio, np.degrees(self.thetaL), int(self.T)))
        #plt.close()
        
        return
    
    def _analyze_energy_local_extrema(self, energy, q_range, verboseFlag=False):
        """ Count local extrema of energy curve, find normalized energy
            barriers and differences between them, and determine what phases
            are present """  
        
        q_crit = np.arccos(1-0.24) # Consider only local maxima occuring before this point
        plimL = np.where(q_range > -q_crit)[0][0]
        plimR = np.where(q_range > q_crit)[0][0]
        if all(self.p_lim):
            minIndex = signal.argrelmin(energy)[0]
            maxIndex = signal.argrelmax(energy)[0]
        else:
            minIndex = signal.argrelmin(energy[plimL:plimR])[0]
            maxIndex = signal.argrelmax(energy[plimL:plimR])[0]            
        minLoc = q_range[minIndex]
        maxLoc = q_range[maxIndex]
        numMin = len(minIndex)
        numMax = len(maxIndex)
                
        phase = -1 # 0 = LCR; 1,2,3 = LR,LC,CR; 4,5,6 = L,C,R
        locGuess = [-np.pi/4, self.total_angle, np.pi/4]
      
        # If p_lim is defined, directly use local minima to determine stability
        if all(self.p_lim):
            if numMin == 0:
                print('WARNING: numMin = {0}... assuming LR bistable.'.format(numMin))
                return 2, [], 1, [], [] 
            elif numMin > 3:
                print('WARNING: numMin = {0}... will return empty lists instead of analyzing.'.format(numMin))
                return numMin, [], 1, [], []
            
            # Compute energy barriers for minima and energy differences between minima
            if numMin == 3:
                # Tristable: three local minima at left, center, and right (LCR)
                diff_CL = (energy[minIndex[1]] - energy[minIndex[0]])
                diff_CR = (energy[minIndex[1]] - energy[minIndex[2]])
                ratio_LR  = (diff_CL - diff_CR)/(diff_CL + diff_CR) #xi from paper    
    
                barrier_CL = (energy[maxIndex[0]] - energy[minIndex[1]])
                barrier_CR = (energy[maxIndex[1]] - energy[minIndex[1]])
    
                phase = 0
                
                diffs = [diff_CL, diff_CR, ratio_LR]
                barriers = [barrier_CL, barrier_CR]
            
            elif numMin == 2:
                # Bistable: two local minima (CL, CR, or LR)
                iLoc1 = (np.abs(locGuess - minLoc[0])).argmin()
                iLoc2 = (np.abs(locGuess - minLoc[1])).argmin()
                if iLoc1 == 0 and iLoc2 == 2:
                    phase = 1
                elif iLoc1 == 0:
                    phase = 2
                elif iLoc2 == 2:
                    phase = 3
                assert phase != -1, 'something is very wrong with this sorting'
                
                diff = energy[minIndex[0]] - energy[minIndex[1]]  
                barrier = energy[maxIndex[0]] - energy[minIndex[1]]
                diffs = [diff]
                barriers = [barrier]
            
            else:
                # Monostable: single local minimum (L, C, or R)   
                pvals = [4, 5, 6]
                iLoc = (np.abs(locGuess - q_range[minIndex[0]])).argmin()
                phase = pvals[iLoc]
                
                diffs = []
                barriers = []
        else: # If p_lim is not defined, check if local max exists before max angle
            if numMin == 0 and numMax == 0:
                # Monostable L or R
                pointL = np.where(q_range > -np.pi/8)[0][0]
                pointR = np.where(q_range > np.pi/8)[0][0]
                if energy[pointL] < energy[pointR]:
                    phase = 4
                else:
                    phase = 6
            elif numMin == 1 and numMax == 0:
                # Monostable C
                phase = 5
            elif numMin == 0 and numMax == 1:
                # Bistable LR
                phase = 1
            elif numMin == 1 and numMax == 1:
                # Bistable CR or CL
                pointL = np.where(q_range > -np.pi/8)[0][0]
                pointR = np.where(q_range > np.pi/8)[0][0]
                if energy[pointL] < energy[pointR]:
                    phase = 2
                else:
                    phase = 3
            elif numMin == 1 and numMax == 2:
                # Tristable: three local minima at left, center, and right (LCR)  
                phase = 0
            elif numMin == 1:
                # Bistable: two local minima (CL, CR, or LR)
                iLoc1 = (np.abs(locGuess - minLoc[0])).argmin()
                iLoc2 = (np.abs(locGuess - minLoc[1])).argmin()
                if iLoc1 == 0 and iLoc2 == 2:
                    phase = 1
                elif iLoc1 == 0:
                    phase = 2
                elif iLoc2 == 2:
                    phase = 3
                assert phase != -1, 'something is very wrong with this sorting'
                
                diff = energy[minIndex[0]] - energy[minIndex[1]]  
                barrier = energy[maxIndex[0]] - energy[minIndex[1]]
                diffs = [diff]
                barriers = [barrier]
                
            diffs = []
            barriers = []
        
        assert phase != -1, 'this should never happen'
        
        return numMin, minLoc, phase, diffs, barriers


# -----------------------------------------------------------------------------
# Other functions
# -----------------------------------------------------------------------------
def plot_energy_concept(h_total, ratio, thetaL, T_range, 
                        b=[], k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                        bilayerDict={}, figdir=''):
    """ Plot energy concept diagram for first figure in manuscript """
    
    sample = MetamaterialModel(h_total=h_total, ratio=ratio, thetaL=thetaL, T=T_range[0],
                               k_sq=k_sq, m=m, limFlag=limFlag, p_lim=p_lim,
                               **bilayerDict)
    
    plt.figure(figsize=(5, 8), dpi=200)
    plt.title(r"$h$={0}mm, $r$={1}, $\theta_L$={2:0.1f}$^\circ$".format(
        h_total, ratio, np.degrees(thetaL)))
    plt.xlabel(r"Angle $\theta$ (degrees)")
    plt.ylabel("Energy (J)")   
    
    q_range = np.radians(np.arange(-50.0,50.5,0.5))
    q_deg = np.degrees(q_range)
    colors = ['tab:orange','firebrick','yellowgreen','royalblue']
    for i, T in enumerate(T_range):
        sample.update_T(T)
        U_total = sample._calculate_total_energy(q_range)
        U_spring = sample._model_U_spring(q_range)
        plt.plot(q_deg, U_total, '-', color=colors[i], label=f'Total, T={T}')
        plt.plot(q_deg, U_spring, '--', color=colors[i], label=f'Spring, T={T}')
        minima = signal.argrelmin(U_total)[0]
        maxima = signal.argrelmax(U_total)[0]
        minU = U_total[minima]
        maxU = U_total[maxima]
        plt.plot(q_deg[minima], minU, 'v', color=colors[i])
        plt.plot(q_deg[maxima], maxU, '^', color=colors[i])
   
    plt.ylim(-0.0015,0.002)
    plt.legend()
    
    plt.savefig(os.path.join(figdir,'{0}.png'.format('energy_concept')))
    plt.savefig(os.path.join(figdir,'{0}.svg'.format('energy_concept')))

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_MetamaterialModel(h_total, ratio, thetaL, T):
    """ Sanity check performance of MetamaterialModel class """
    t0 = time.perf_counter() 
    sample = MetamaterialModel(h_total=h_total, ratio=ratio, thetaL=thetaL, T=T, b=[0.0,1.5])
    t1 = time.perf_counter() 
    sample.update_T(100.0)
    t2 = time.perf_counter() 
    sample = MetamaterialModel(h_total=h_total, ratio=ratio, thetaL=thetaL, T=T, b=[0.0,1.5],
                               plotFlag=True, verboseFlag=True)
    t3 = time.perf_counter() 
    print(" Base version:\t\t {0:.2e} s\n Updating T:\t\t {1:.02e} s\n Verbose version:\t {2:0.2e} s".format(t1-t0, t2-t1, t3-t2))
    print("h={0}mm, r={1}, theta_L={2:0.1f}deg @ T={3}C".format(h_total, ratio, np.degrees(thetaL), int(T)))
    print("Equivalent torsional spring constant: {0:.3f} Nmm".format(1e3*sample.hinge.k))
        
    return True

# -----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------- 
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    tmpdir = os.path.join(cwd,"tmp")

    h_val = 0.0009
    r_val = 0.25
    thetaL_val = 0.0
    T_val = 25.0
    test_MetamaterialModel(h_val, r_val, thetaL_val, T_val)

    T_range = [25.0, 35.0, 50.0, 80.0]
    plot_energy_concept(h_val, r_val, 0.0, T_range, figdir=tmpdir)
    