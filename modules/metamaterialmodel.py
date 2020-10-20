# -*- coding: utf-8 -*-
'''
Created on Fri Jun 19 11:27:45 2020

@author: Lucia
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time
from datetime import datetime
import os
import sys
import matplotlib.colors as col
from matplotlib import cm


from modules import bilayermodel as bilayer


class MetamaterialModel:
    ''' Model for a PDMS-LCE bilayer hinge of total thickness h_total and
        LCE:total thickness ratio r. Initialized at room temperature, but can
        be updated for any temperature. Specify only one of s or b.'''
    def __init__(self, h_total, ratio, thetaL, T=25.0,
                 LCE_strain_params=[0.039457,142.37,6.442e-3],
                 LCE_modulus_params=[-8.43924097e-02,2.16846882e+02,3.13370660e+05],
                 d=0.018, w=0.010, b=[], s=0, m=0.1471, bFlag='lin',
                 limFlag='exp', p_lim=[3e-22, 51], k_sq=0.0,
                 plotFlag=False, verboseFlag=False, loadFlag=False, hasMagnets=True,
                 analysisFlag=True):
        
        # Main model parameters
        self.T = T                  # [Celcius]     Temperature
        self.h_total = h_total      # [m]           Total bilayer hinge thickness
        self.ratio = ratio          # [m/m]         Bilayer hinge LCE:PDMS thickness ratio
        self.thetaL = thetaL        # [rad]         As-fabricated angle of hinge
        
        # Material parameters: fit to data
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
        ''' Update all values which vary with temperature, in the appropriate order'''
        self.T = T
        self.hinge.update_temperature(T)
        self.total_angle = self.hinge.thetaT + self.thetaL
        self.k_eq = self._equivalent_torsional_spring()  
        if self.analysisFlag:
            self._analyze_energy()
        return

    def model_load_disp(self, disp):
        ''' Given displacement x, return load corresponding to unit cell test '''
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
        ''' Displacement between centers of squares '''
        return self.d*np.cos(q)
    
    def _disp2rot(self, x):
        ''' Displacement between centers of squares '''
        return np.arccos(x/self.d)
    
    def _equivalent_torsional_spring(self):
        if self.k_sq == 0:
            k_eq = self.hinge.k
        else:
            k_eq = 1/(1/self.hinge.k + 1/self.k_sq)
        return k_eq
    
    def _torque_k(self, q):
        ''' Linear torsional spring representing hinge '''
        return -self.k_eq*(2*(q - self.total_angle))
    
    def _torque_magnet(self, q):
        ''' Point dipole at centers of each square '''
        mu = 1.2566*10**(-6) # [N/A^2]
        return 3*mu*(self.m**2)*(self.d/2)*np.sin(q)/(2*np.pi*self._rot2disp(q)**4)
    
    def _torque_lim(self, q):
        ''' Torque resulting from square collision, limiting max displacement '''
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
        ''' Piecewise linear torsional spring model for square collision '''
        q_lim, k_lim = self.p_lim
        if q_value > q_lim:
            M = -k_lim*(2*(q_value-q_lim))
        elif q_value < -q_lim:
            M = -k_lim*(2*(q_value+q_lim))
        else:
            M = 0
        return M    
    
    def _torque_lim_exp_single(self, q_value):
        ''' Exponential model for square collision '''
        A, B = self.p_lim
        M = -B*A*(np.exp(B*(q_value)) - np.exp(-B*(q_value)))
        return M
    
    # Enerties, for simulation and phase analysis -----------------------------
    def _analyze_energy(self):
        ''' Calculate potential energy and find the minima and phases '''
        q_range = np.radians(np.arange(-50.0,50.5,0.5))
        energy = self._calculate_total_energy(q_range)
        self.numMin, self.locMin, self.phases, self.diffs, self.barriers = self._analyze_energy_local_extrema(energy, q_range)
        
        return 
    
    def _model_U_limit(self, q):
        ''' Calculate energy of collision of adjacent squares, given current angle between squares '''
        if self.limFlag == 'exp': #Exponential
            A, B = self.p_lim #[1e-21, 50]
            U_limit = A*(np.exp(B*2*q/2) + np.exp(-B*2*q/2))
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
        ''' Calculate energy of magnet interaction, given current angle between squares'''
        mu = 1.256637e-6 # magnetic permeability of free space
        dist = self.d*np.cos(q) # distance between square centers
        U_m = -(mu*self.m**2)/(2*np.pi*dist**3)
        return U_m

    def _model_U_spring(self, q):
        ''' Calculate energy of hinge linear torsional spring deformation, given current angle between squares '''
        U_k = 0.5*self.k_eq*(2*(q - self.total_angle))**2
        return U_k
 
    def _calculate_total_energy(self, q_vals):
        ''' Potential energy of magnet, torsion spring, and square deformation
            at a given angle. Returns both total energy and list of component
            energies. thetas: array'''  
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
    
    def _plot_energy(self, q, U_total, U_m=[], U_k=[], U_lim=[]):
        ''' Plot total energy curve, individual contributions if specified, and
            local extrema ''' 
        
        q = np.degrees(q)
        
        plt.figure(dpi=300)
        plt.title(r"$h$={0}mm, $r$={1}, $\theta_L$={2:0.1f}$^\circ$ @ $T$={3}$^\circ$C".format(self.h_total, self.ratio, np.degrees(self.thetaL), int(self.T)))
        plt.xlabel(r"Angle $\theta$ (degrees)")
        plt.ylabel("Energy (J)")        

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
        plt.ylim(-0.0015,0.001)
        
        if (U_m.any() or U_k.any() or U_lim.any()):
            plt.legend()
        plt.tight_layout()
        #plt.savefig('h{0:.2f}_r{1:.2f}_L{2:0.1f}_T{3:.1f}.png'.format(self.h_total, self.ratio, np.degrees(self.thetaL), int(self.T)))
        #plt.close()
        
        return
    
    def _analyze_energy_local_extrema(self, energy, q_range, verboseFlag=False):
        ''' Count local extrema of energy curve, find normalized energy
            barriers and differences between them, and determine what phases
            are present '''  
        
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

    
#%%

def analyze_composite_phases(r, h_range=1e-3*np.arange(0.5,2.1,0.01),
                             thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                             T_range=np.arange(25.0,75.1,25.0),
                             b=[], k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                             bilayerDict={}):
    ''' For a given composite with ratio r, compute number of minima across isotherms '''
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
            sample = MetamaterialModel(h, r, thetaL, T=25.0, k_sq=k_sq, m=m, p_lim=p_lim, **bilayerDict)
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
                       b=[], k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                       bilayerDict={},
                       savedir='', datestr='', closeFlag=True):
    ''' FULL PHASE ANALYSIS '''
    
    if datestr == '':
        datestr = datetime.today().strftime('%Y%m%d')
        
    parameter_file = os.path.join(savedir, '{0}_parameters.csv'.format(datestr))
    metamaterial_file = os.path.join(savedir, '{0}_metamaterialObject.npy'.format(datestr))
    minima_file = os.path.join(savedir, '{0}_r{1:.3f}_minima.npy'.format(datestr, r_val))
    phases_file = os.path.join(savedir, '{0}_r{1:.3f}_phases.npy'.format(datestr, r_val))
    thetaT_file = os.path.join(savedir, '{0}_r{1:.3f}_thetaT.npy'.format(datestr, r_val))
    theta0_file = os.path.join(savedir, '{0}_r{1:.3f}_theta0.npy'.format(datestr, r_val))
    boundaries_file = os.path.join(savedir, '{0}_boundaries.csv'.format(datestr))
    boundaryVals_file = os.path.join(savedir, '{0}_boundaryVals.csv'.format(datestr))
    boundaryData_file = os.path.join(savedir, '{0}_boundaryData.csv'.format(datestr))
    
    # Load existing data if previously run
    try:
        print("Try loading previous parameter-space analysis...")
        r_val, T_range, h_range, thetaL_range = import_parameters(parameter_file)
        minima = np.load(minima_file)
        phases = np.load(phases_file)
        thetaT = np.load(thetaT_file)
        theta0 = np.load(theta0_file)
        sampleModel = np.load(metamaterial_file)
        print("\tLoaded parameters, minima, phases, theta_T, and theta_0")
    except IOError:
        export_parameters(parameter_file, r_val, T_range, h_range, thetaL_range) #UPDATE THIS FUNCTION
        sampleModel = MetamaterialModel(1e-3, r_val, 0.0, T=25.0, k_sq=k_sq, m=m, p_lim=p_lim, **bilayerDict)
        np.save(metamaterial_file, sampleModel)
        print("\tRunning new parameter-space analysis...")
        minima, phases, thetaT, theta0 = analyze_composite_phases(r_val, h_range=h_range, thetaL_range=thetaL_range, T_range=T_range,
                                                                  b=b, k_sq=k_sq, m=m, limFlag=limFlag, p_lim=p_lim, bilayerDict=bilayerDict)
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


def analyze_h_T_relation(r, h_range=1e-3*np.arange(0.5,2.1,0.001),
                             thetaL=0.0,
                             T_range=np.arange(20.0,100.1,1.0),
                             k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                             bilayerDict={},
                             saveFlag=False, figdir=''):
    ''' For a given composite with ratio r, compute number of minima across isotherms '''
    n_h = len(h_range)
    n_T = len(T_range)
    angleT_array = np.zeros((n_h, n_T))
    angle0_array = np.zeros((n_h, n_T))
    numMin_array = np.zeros((n_h, n_T))
    phases_array = np.zeros((n_h, n_T))

    t0 = time.perf_counter()
    for i_h in range(n_h):
        sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} h values at {2:0.1f} minutes".format(i_h + 1, n_h, (time.perf_counter()-t0)/60.0))
        sys.stdout.flush()
        h = h_range[i_h]
        sample = MetamaterialModel(h, r, thetaL, T=25.0, k_sq=k_sq, m=m, p_lim=p_lim, **bilayerDict)
        for i_T in range(n_T):
            sample.update_T(T_range[i_T])              
            
            numMin_array[i_h, i_T] = sample.numMin
            phases_array[i_h, i_T] = sample.phases
            angleT_array[i_h, i_T] = np.degrees(sample.hinge.thetaT)
            angle0_array[i_h, i_T] = np.degrees(sample.total_angle)
                    
    # Plot
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
    ''' Map stability along three parameter axes: thickness, LCE:PDMS ratio, initial angle '''
   
    n_a = max(a_range.shape)
    n_T = max(T_range.shape)
    angleT_array = np.zeros((n_a,n_T))
    angle0_array = np.zeros((n_a,n_T))
    numMin_array = np.zeros((n_a,n_T))
    phases_array = np.zeros((n_a,n_T))
    
    # Find minima
    t0 = time.perf_counter()
    for i_a in range(n_a):
        sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} h values at {2:0.1f} minutes".format(i_a + 1, n_a, (time.perf_counter()-t0)/60.0))
        sys.stdout.flush()
        sample = MetamaterialModel(h_val, ratio_val, thetaL_val,
                                   T=25.0, k_sq=k_sq, m=m, p_lim=p_lim,
                                   d=a_range[i_a],
                                   **bilayerDict)
        for i_T in range(n_T):
            sample.update_T(T_range[i_T])
            
            numMin_array[i_a, i_T] = sample.numMin
            phases_array[i_a, i_T] = sample.phases
            angleT_array[i_a, i_T] = np.degrees(sample.hinge.thetaT)
            angle0_array[i_a, i_T] = np.degrees(sample.total_angle)

    # Plot
    colors = ['xkcd:electric blue','xkcd:apple green','xkcd:light red']    
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    X = 1000*a_range
    Y = T_range
    Z = np.transpose(numMin_array)
    diagram = ax.imshow((Z).astype(np.uint8), extent=[X.min(), X.max(), Y.min(), Y.max() ],
                        origin='lower', aspect='auto', cmap=col.ListedColormap(colors))#cm.viridis          
    
    # Formatting
    plt.title(r'$h$ = {0}, $\theta_0$ = {1}$^\circ$, $r$ = {2}'.format(h_val, thetaL_val, ratio_val))
    plt.xlabel(r'Diagonal length $a$ (mm)')
    plt.ylabel(r'Temperature $T$ ($^\circ$C)')
    cbar = plt.colorbar(diagram)
    cbar.ax.get_yaxis().set_ticks([])
       
    #plt.ylim([np.amin(angle0_vals), np.amax(angle0_vals)])
    if saveFlag:
        title = 'diagonal_dependence_h{0:.2f}_r{1:.2f}_thetaL{2:.1f}'.format(h_val, ratio_val, thetaL_val)
        plt.savefig(os.path.join(figdir,'{0}.png'.format(title)),dpi=200)
        plt.savefig(os.path.join(figdir,'{0}.svg'.format(title)))

    return


''' Export parameters used to map parameter space '''
def export_parameters(filename, r, T, h, thetaL):
    with open(filename, 'w') as f:
        f.write("{0:.3f}\n".format(r))
        f.write(", ".join(map(str, T)) + "\n")
        f.write(", ".join(map(str, h)) + "\n")
        f.write(", ".join(map(str, thetaL)) + "\n")


''' Import parameters used for previously-calculated parameter-space analysis '''
def import_parameters(filename):
    with open(filename, 'r') as f:
        p = f.readlines()
        r = p[0]
        T = [ n.split(', ') for n in p[1] ]
        h = [ n.split(',') for n in p[2] ]
        thetaL = [ n.split(',') for n in p[3] ]

    print("Imported the following parameters:")
    print("thetaL \t ({0}:{1}), length {2}".format(thetaL[0],thetaL[-1],len(thetaL)))
    print("T \t ({0}:{1}), length {2}".format(T[0],T[-1],len(T)))
    print("h \t ({0}:{1}), length {2}".format(h[0],h[-1],len(h)))
    print("r = {0}".format(r))

    return thetaL, T, h, r

def export_to_csv(filename, data, variables=[], units=[], fmt='%0.18e'):
    if variables != [] and units != []:
        headerVars = ", ".join(map(str, variables))
        headerUnits = ", ".join(map(str, units))
    else: 
        headerVars = ''
        headerUnits = ''
    headerStr = headerVars + '\n' + headerUnits
    
    np.savetxt(filename, data, delimiter=', ', fmt=fmt, header=headerStr)
    return filename

def find_3D_phase_boundaries(r, h_range=1e-3*np.arange(0.5,2.1,0.01),
                             thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                             T_range=np.arange(25.0,105.0,25.0),
                             minima=[], phases=[], angleT_vals=[], angle0_vals=[]):   
    ''' Find locations of phase boundaries for a given composite '''

    # Find where changes in phase occur
    diffs = np.diff(phases)
    boundaries = np.argwhere(diffs != 0)
    print(boundaries.shape)
    N = max(boundaries.shape) # Number of points found located at boundary
    boundaryVals = np.zeros(N)
    for pt in range(N):
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        val = phases[loc[0],loc[1],loc[2]]
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
    

def plot_isotherm(r, T, phases, angle0_vals,
                         h_range=1e-3*np.arange(0.5,2.1,0.01),
                         thetaL_range=np.radians(np.arange(-15.0,15.1,0.1)),
                         T_range=np.arange(25.0,105.0,25.0), savedir='', closeFlag=True):
    ''' Plot isotherm phase diagram for a given composite r: h vs total equilibrium angle
        at a particular temperature T'''
    
    # Find where in simulation data the desired T value occurs
    i_T = np.argwhere(T_range == T)[0][0]
    #assert i_T, "Choose a value of T that is in T_range; specify T_range if not default"
    
    colors = ['xkcd:light red', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:electric blue', 'xkcd:electric blue', 'xkcd:electric blue']
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    X = 1000*h_range
    Y = np.degrees(thetaL_range)#angle0_vals
    Z = np.transpose(phases[:,:,i_T])
    diagram = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', cmap=col.ListedColormap(colors))           
    
    # Formatting
    plt.title(r'$T$ = {0}$^\circ$C, $r$ = {1:0.2f}'.format(int(T), r), size=18)
    plt.xlabel(r'hinge thickness $h$ (mm)')
    plt.ylabel(r'fabrication angle $\theta_L$ (degrees)')
    
    if savedir != '':
        figname = 'isotherm_{0}C_r{1:0.2f}.png'.format(int(T), r)
        plt.savefig(os.path.join(savedir, figname),dpi=200)
    if closeFlag:
        plt.close()

    return
   

#%%

def test_MetamaterialModel(h_total, ratio, thetaL, T):
    ''' Sanity check performance of MetamaterialModel class '''
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
    
def plot_energy_concept(h_total, ratio, thetaL, T_range, 
                        b=[], k_sq=0.0, m=0.1471, limFlag='exp', p_lim=[3e-22, 51],
                        bilayerDict={}, figdir=''):
    
    sample = MetamaterialModel(h_total=h_total, ratio=ratio, thetaL=thetaL, T=T_range[0],
                               k_sq=k_sq, m=m, limFlag=limFlag, p_lim=p_lim,
                               **bilayerDict)
    
    plt.figure(figsize=(5, 8), dpi=200)
    plt.title(r"$h$={0}mm, $r$={1}, $\theta_L$={2:0.1f}$^\circ$".format(h_total, ratio, np.degrees(thetaL)))
    plt.xlabel(r"Angle $\theta$ (degrees)")
    plt.ylabel("Energy (J)")   
    
    q_range = np.radians(np.arange(-50.0,50.5,0.5))
    q_deg = np.degrees(q_range)
    colors = ['tab:orange','firebrick','yellowgreen','royalblue']
    for i, T in enumerate(T_range):
        sample.update_T(T)
        U_total = sample._calculate_total_energy(q_range)
        U_spring = sample._model_U_spring(q_range)
        plt.plot(q_deg, U_total, '-', color=colors[i], label='Total, T={0}'.format(T))
        plt.plot(q_deg, U_spring, '--', color=colors[i], label='Spring, T={0}'.format(T))
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

#%%
    
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    tmpdir = os.path.join(cwd,"tmp")

    test_MetamaterialModel(0.001, 0.5, 0.0, 25.0)

    """try:
        print("Try loading previous parameter-space analysis...")
        thetaL_range, T_range, h_range, r_range = import_parameters(parameter_file)
        minima = np.load(minima_file)
        phases = np.load(phases_file)
        thetaT = np.load(thetaT_file)
        theta0 = np.load(theta0_file)
        print("\tLoaded parameters, minima, phases, theta_T, and theta_0")
    except IOError:"""
    '''print("\tRunning new parameter-space analysis...")    
    #r_range = np.arange(0.1,0.95,0.1)
    r_range = np.array([0.377]) #r_const average value
    h_range = 1e-3*np.arange(0.5,2.1,0.01)
    thetaL_range = np.radians(np.arange(-20.0,20.1,0.1))
    T_range = np.arange(25.0,105.0,5.0)
    analyze_composites(r_range=r_range,
                       h_range=h_range,
                       thetaL_range=thetaL_range,
                       T_range=T_range,
                       savedir=tmpdir)'''
    h_val = 0.0007
    r_val = 0.25
    sample = MetamaterialModel(h_total=h_val, ratio=r_val, thetaL=0.0, T=25.0, b=[0.0,1.5],
                           plotFlag=True, verboseFlag=True)
    h_val = 0.0009
    T_range = [25.0, 35.0, 50.0, 80.0]
    plot_energy_concept(h_val, r_val, 0.0, T_range, figdir=tmpdir)
    