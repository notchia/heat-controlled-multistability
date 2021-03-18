# -*- coding: utf-8 -*-
'''
Created on Mon Jul 20 13:21:05 2020

@author: Lucia
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.interpolate as interp
import scipy.optimize as opt
import re

from collections.abc import Iterable

from modules import bilayermodel as bilayer
from modules import tensiletest

DEBUG_FLAG = False  # Show more figures and print statements


#%% Model -------------------------------------------------------------------
def rot2disp(q, l):
    ''' Displacement between centers of squares '''
    return 2*l*np.cos(q)

def disp2rot(x, l):
    ''' Displacement between centers of squares '''
    return np.arccos(x/(2*l))

def torque_k(q, kq, q0):
    return -kq*(2*(q-q0))

def equivalent_springs_in_series(k_hinge, k_compliance):
    return 1/(1/k_hinge + 1/k_compliance)

def torque_magnet(q, moment, L):
    mu = 1.2566*10**(-6) # [N/A^2]
    return 3*mu*(moment**2)*L*np.sin(q)/(2*np.pi*rot2disp(q, L)**4)

def torque_lim(q, qlim, klim, flag='pcw'):
    assert flag=='pcw' or flag=='exp'
    if flag=='pcw':
        f_lim = torque_lim_pcw_single
    elif flag=='exp':
        f_lim = torque_lim_exp_single
    
    if isinstance(q,(list,np.ndarray)):
        F = np.zeros_like(q)
        for i, q_val in enumerate(q):
            F[i] = f_lim(q_val, qlim, klim)
    else:
        F = f_lim(q, qlim, klim)
    return F

def torque_lim_pcw_single(q_value, qlim, klim):
    if q_value > qlim:
        M = -klim*(2*(q_value-qlim))
    elif q_value < -qlim:
        M = -klim*(2*(q_value+qlim))
    else:
        M = 0
    return M    

def torque_morse(q, A, alpha, qMorse, L, q0):
    M = 2*alpha*A*(  np.exp(4*alpha*(q+q0-qMorse))
                   - np.exp(2*alpha*(q+q0-qMorse))
                   - np.exp(-4*alpha*(q+q0+qMorse))
                   + np.exp(-2*alpha*(q+q0+qMorse)) )
    return M

def torque_lim_exp_single(q_value, A, B):
    #M = A*B*(np.exp(B*2*q_value) - np.exp(-B*2*q_value))
    M = B*A*(np.exp(B*(q_value)) - np.exp(-B*(q_value)))
    return M

# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_magnet(x, p, q0, kq, L, pMorse=[], flag='pcw'):
    ''' Defines force(disp) function '''
    moment, qlim, klim = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    if flag=='morse':
        M_morse = torque_morse(q, *pMorse)
        M = 4*M_k + 4*M_morse
    else:
        M_m = torque_magnet(q, moment, L)
        M_lim = torque_lim(q, qlim, klim, flag=flag)
        M = 4*M_k + 4*M_m + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F

# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_nomagnet(x, p, q0, L, flag='pcw'):
    ''' Defines force(disp) function '''
    kq, qlim, klim = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_lim = torque_lim(q, qlim, klim, flag=flag)
    M = 4*M_k + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F

# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_spring(x, kq, q0, L):
    ''' Defines force(disp) function '''
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M = 4*M_k
    F = M/(2*L*np.cos(q))
    return F
    

#%% Analysis ------------------------------------------------------------------
def residue_force_displacement_spring(p, y, x, q0, L):
    return (y - force_displacement_spring(x, p, q0, L))

def residue_force_displacement_nomagnet(p, y, x, q0, L, flag='pcw'):
    return (y - force_displacement_nomagnet(x, p, q0, L, flag=flag))

def residue_force_displacement_magnet(p, y, x, q0, kq, L, flag='pcw'):
    return (y - force_displacement_magnet(x, p, q0, kq, L, flag=flag))

def find_force_zero(disp, force):
    ''' Finds the zero crossing of the force-displacement curve corresponding
        to open configuration '''
    fcn = interp.interp1d(disp, force)
    disp_offset = opt.newton(fcn, x0=-0.0006, tol=1e-8, maxiter=int(1e6))
    return disp_offset

def import_experiment(filepath, L, speed=0.1, zeroDisp=0, sampleLoad=0, tempLoadOffset=0):
    ''' Import one-shot test (compressive or tensile) '''
    cols = [0,1] #time, load
    arr = np.genfromtxt(filepath,dtype=float,delimiter=',',skip_header=17,usecols=cols)
    force = arr[:,1]
    startTrim = np.argwhere(np.abs(force-force[0])>0.05)[0][0]
    endTrim = np.argmax(force)
    disp_exp = 1e-3*(arr[startTrim:endTrim,0]*speed - arr[startTrim,0]*speed + zeroDisp)
    force_exp = force[startTrim:endTrim] + sampleLoad + tempLoadOffset
      
    # Set zero of displacement to be the zero force position
    #offset = find_force_zero(disp_exp, force_exp)
    #disp_exp = disp_exp - offset
    
    # Interpolation -------------------------------------
    f1 = interp.interp1d(disp_exp, force_exp)
    disp = np.linspace(min(disp_exp), max(disp_exp), 2000)
    load = f1(disp)
    
    if DEBUG_FLAG:
        plt.figure(dpi=200)
        filename = os.path.splitext(os.path.split(filepath)[-1])[0]
        plt.title(filename)
        plt.xlabel('Displacement')
        plt.ylabel('$F$ (N)')
        plt.plot(1e3*disp, -load, 'k', linewidth=2, label='Experiment')
        plt.tight_layout()
    
    return disp, load

def approximate_magnet(disp, force, p_guess, p_given, figFlag=True):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, kq, L = p_given
    
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residue_force_displacement_magnet, p_guess,
                                      args=(force, disp, q0, kq, L))
    moment, qlim, klim = p_fit.x
    paramstr = "moment: {0:.4f} A$\cdot$m$^2$\n$q_{{lim}}$: {1:.4f}$^\circ$\n$k_{{lim}}$: {2:.4f} Nm".format(moment, np.degrees(qlim), klim)

    force_fit = force_displacement_magnet(disp, p_fit.x, q0, kq, L)
    energy_fit = np.cumsum(force_fit)
    
    if figFlag:
        plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
                              titlestr='With magnet', notestr=paramstr)

    return moment, qlim, klim

def crop_to_compressed(disp, force, L):
    index = np.argwhere(disp>2*L)[0][0]
    force = force[0:index]
    disp = disp[0:index]
    return disp, force


def model_spring(disp, force, L, h, r, T, LCE_modulus_params, b, figFlag=True, titlestr=''):
    hinge = bilayer.BilayerModel(h, r, T=T, LCE_modulus_params=LCE_modulus_params, b=b)
    kq = hinge.k
    
    k_compliance = 0.003
    kq_adjusted = equivalent_springs_in_series(kq, k_compliance)
    
    if figFlag:
        q0 = hinge.thetaT
        force_model = force_displacement_spring(disp, kq_adjusted, q0, L)  
        energy_model = np.cumsum(-force_model)
        disp_plt = 2*L - disp
        paramstr = "$k_q$: {0:.3f} Nmm\n$q_0$ = {1:.3f} $^\circ$\n spring in series = {2:.1f} Nmm".format(kq*1e3, np.degrees(q0), k_compliance*1e3)
        plot_force_and_energy(disp_plt, force-force[-1], force_model, energy_model*1e-3,
                              titlestr='Estimate from bilayer model'+titlestr, notestr=paramstr)
    
    return kq

def approximate_spring(disp, force, p_guess, p_given, preCropped=False, figFlag=True):

    q0, L, flag = p_given

    # Crop data to where displacement exceeds 
    if not preCropped:
        disp, force = crop_to_compressed(disp, force, L)
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residue_force_displacement_nomagnet, p_guess,
                                      args=(force, disp, q0, L), kwargs={'flag':flag})
    params = p_fit.x
    if flag=='pcw':
        kq, qlim, klim = params
        paramstr = "$k_q$: {0:.4f} Nm\n$q_{{lim}}$: {1:.4f}$^\circ$\n$k_{{lim}}$: {2:.4f} Nm".format(kq, np.degrees(qlim), klim)
    elif flag=='exp':
        kq, A, B = params
        paramstr = "$k_q$: {0:.4f} Nm\n$A$: {1:.3e} Nm\n$B$: {2:.3e}".format(kq, A, B)

    force_fit = force_displacement_nomagnet(disp, params, q0, L, flag=flag)  
    energy_fit = np.cumsum(-force_fit)
    
    if figFlag:
        plot_force_and_energy(disp_plt, force, force_fit, energy_fit*1e-3,
                              titlestr='No magnet', notestr=paramstr)

    return params

#def compare_spring_theory_to_fit(disp, force)

def approximate_only_spring(disp, force, p_guess, p_given, preCropped=False, figFlag=True, titlestr=''):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, L = p_given

    # Crop data to where displacement exceeds 
    if not preCropped:
        disp, force = crop_to_compressed(disp, force, L)
    disp_plt = 2*L - disp
    cropIndex = np.argwhere(disp_plt<0.004)[0][0]
    disp_fit = disp[cropIndex:]
    force_fit = force[cropIndex:]
    

    # Fit to model
    p_fit = opt.least_squares(residue_force_displacement_spring, p_guess,
                                      args=(force_fit, disp_fit, q0, L))
    kq = p_fit.x[0]
    paramstr = "kq: {0:.4f} N/m".format(kq)
    
    force_model = force_displacement_spring(disp, kq, q0, L)  
    energy_model = np.cumsum(-force_model)
    
    if figFlag:
        plot_force_and_energy(disp_plt, force, force_model, energy_model*1e-3,
                              titlestr='Torsional spring estimate'+titlestr, notestr=paramstr)

    return kq

def plot_force_and_energy(disp, load_exp, load_model, energy_model, titlestr='', notestr=''):
    ''' Plot load-displacement for experiment and model, and energy for model, 
        including descriptive title and additional notes text'''
    disp = 1e3*disp
    
    fig, axleft = plt.subplots(dpi=200)
    plt.title(titlestr)
    plt.xlabel('Displacement from open (mm)')
    plt.ylabel('$F$ (N)', color='r')
    plt.tick_params(axis='y', labelcolor='r')
    plt.plot(disp, -load_exp, 'k', linewidth=2, label='Force from experiment')
    plt.plot(disp, -load_model, '--r', label='Force from model')
    plt.text(0, max(abs(load_exp))-1, notestr)
    axleft.twinx()
    plt.plot(disp, -energy_model, '--g', label='Energy from model')
    plt.ylabel('$E$ (mJ)', color='g')
    plt.tick_params(axis='y', labelcolor='g')
    plt.tight_layout()
    return

#%% Test --------------------------------------------------------------------
def test_force_model():
     
    q = np.radians(np.arange(0.01,45.5,0.01))
    q0 = np.radians(0)
    kq = 0.0008 #nominally, should be around 0.0004 for d = 0.075; artifically increasing to be able to observe minimum @C
    L = 0.009#(0.012 + 0.001*(np.cos(q0) + np.sin(q0)))/(2*np.cos(q0))
    x = rot2disp(q, L)
    u = np.array([2*L - x_val for x_val in x])
    params = [0.1517, np.radians(44), 0.1]#[0.07, np.radians(40), 0.03]

    F = force_displacement_magnet(x, params, q0, kq, L)
    E = np.cumsum(F)
    
    plt.figure()
    plt.plot(1000*u, -F, color='b')
    plt.ylabel("Force (N)")
    plt.xlabel("u (mm)")
    plt.axhline(color='k',linewidth=0.5)
    plt.plot(1000*u, -E/1000, color='g')

    plt.figure()
    plt.plot(np.degrees(q), -F, color='b')
    plt.ylabel("Force (N)")
    plt.xlabel("angle (deg)")
    plt.axhline(color='k',linewidth=0.5)
    plt.plot(np.degrees(q), -E/1000, color='g')
    
    minLoc = signal.argrelmin(E)[0]
    maxLoc = signal.argrelmax(E)[0]
    print('minima (u, mm): {}'.format(1000*u[minLoc]))
    print('maxima (u, mm): {}'.format(1000*u[maxLoc]))
    print('minima (q, deg): {}'.format(np.degrees(q[minLoc])))
    print('maxima (q, deg): {}'.format(np.degrees(q[maxLoc])))
    
    return

#%% Analyze specific data

def analyze_data_200729(filelist, sourcedir):
    ''' First force-displacement test; check if repeatable on same sample '''   
    
    print("*****************\n Analyzing data from 07/29/20...")
    
    q0 = np.radians(0)
    L = 0.009
    
    plt.figure(dpi=300)
    plt.title("PDMS hinge, $t = 1.40$, no magnets")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.axhline(color='gray')
    plt.vlines(2*L*1e3,-5,5,colors='gray')
    
    disp_n = np.zeros((2000, 3))
    load_n = np.zeros((2000, 3))
    disp_y = np.zeros((2000, 3))
    load_y = np.zeros((2000, 3))  

    for i, fname in enumerate(filelist):
        fullpath = os.path.join(sourcedir, fname)
        fnameParams = os.path.splitext(fname)[0].split('_')
        zeroDisp = 9.81

        if 'tension' in fnameParams and '200729' in fnameParams:
            num = int(fnameParams[-1])
            if 'N' in fnameParams:
                col = 'b'
                sampleLoad = 0.16
                label='no magnet, {0}'.format(num)
            elif 'Y' in fnameParams:
                col = 'r'
                sampleLoad = 0.20
                label='magnet, {0}'.format(num)
            print(label)
            disp, load = import_experiment(fullpath, L, speed=0.1, 
                                           zeroDisp=zeroDisp, sampleLoad=sampleLoad)
            if 'N' in fnameParams:
                disp_n[:,num-1] = disp
                load_n[:,num-1] = load
            elif 'Y' in fnameParams:
                disp_y[:,num-1] = disp
                load_y[:,num-1] = load
            if num == 1:
                plt.plot(disp*1e3, load, col, label=label)
         
    mean_n = np.mean(load_n,axis=1)
    mean_y = np.mean(load_y,axis=1)
    std_n = np.std(load_n,axis=1)
    std_y = np.std(load_y,axis=1)
    
    plt.plot(disp_n[:,0]*1e3, mean_n, 'b',label='no magnets')
    plt.plot(disp_y[:,0]*1e3, mean_y, 'r',label='magnets')
    plt.legend()
    
    p_guess = [0.001, np.radians(40), 0.03]
    p_given = [q0, L]
    kq, qlim, klim = approximate_spring(disp_n[:,0], mean_n, p_guess, p_given)

    moment = 0.1471
    
    index = np.argwhere(disp_y[:,0]>2*L)[0][0]
    force = mean_y[0:index]
    disp = disp_y[0:index,0]
    disp_plt = 2*L - disp
    
    force_fit = force_displacement_magnet(disp, [moment, qlim, klim], q0, kq, L)
    plt.figure()
    #plt.plot(disp_y[:,0], mean_y)
    #plt.plot(disp_y[:,0], force_fit)
    E = np.cumsum(force_fit)
    
    fig, axleft = plt.subplots(dpi=300)
    plt.title('With magnets')
    plt.plot(disp_plt*1e3, -force, 'k', linewidth=2, label='Experiment')
    plt.plot(disp_plt*1e3, -force_fit, '--r', label='Force fit')
    plt.ylabel('$F$ (N)', color='r')
    plt.tick_params(axis='y', labelcolor='r')
    plt.xlabel('Displacement from open (mm)')
    axright = axleft.twinx()
    plt.plot(disp_plt*1e3, E*1e-3, '--g', label='Energy from fit')
    plt.ylabel('$E$ (J)', color='g')
    plt.tick_params(axis='y', labelcolor='g')
    plt.tight_layout()  
    
    return

def analyze_data_200803(sourcedir):
    ''' Check dependence of vibration on displacement rate; find good filter '''
    
    print("*****************\n Analyzing data from 08/03/20...")
    
    q0 = np.radians(0)
    L = 0.0102
    
    plt.figure(dpi=300)
    plt.title("PDMS hinge, $t = 1.40$, no magnets")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.axhline(color='gray')
    plt.vlines(2*L*1e3,-5,5,colors='gray')
    
    disp_all = np.zeros((2000, 5))
    load_all = np.zeros((2000, 5))
    count = 0
    filelist = os.listdir(sourcedir)
    for i, fname in enumerate(filelist):
        fullpath = os.path.join(sourcedir, fname)
        fnameParams = os.path.splitext(fname)[0].split('_')
        zeroDisp = 11.88

        if 'tension' in fnameParams and '200803' in fnameParams:
            sampleLoad = 0.17
            speed = [s for s in fnameParams if "mmps" in s]
            speed = float(speed[0].strip('mps'))
            label='{0} mm/s'.format(speed)
            print(label)
            disp, load = import_experiment(fullpath, L, speed=speed, 
                                           zeroDisp=zeroDisp, sampleLoad=sampleLoad)
            disp_all[:,count] = disp
            load_all[:,count] = load
            count += 1
            plt.plot(disp*1e3, load, label=label, linewidth=0.5, markersize=1.0,
                     zorder=5-count)
    plt.legend()
    
    p_guess = [0.001, np.radians(40), 0.03]
    p_given = [q0, L, 'pcw']
    kq, qlim, klim = approximate_spring(disp_all[:,0], load_all[:,0], p_guess, p_given)
   
    Nlist = [5, 10, 25, 50, 100]
    ratelist = [0.05, 0.1, 0.2, 0.5, 1.0]
    timelist = [224, 114, 59, 26, 15]
    for i in range(5):
        disp = disp_all[:,i]
        load = load_all[:,i]
        plt.figure(dpi=300)    
        plt.title("Displacement rate: {0} mm/s ({1:.1f} min)".format(ratelist[i],timelist[i]/60))
        plt.xlabel("Displacement (mm)")
        plt.ylabel("Load (N)")
        plt.plot(disp*1e3, load, 'k', label='raw')
        for N in Nlist:
            mean = tensiletest.running_mean_centered(load, N)    
            plt.plot(disp[0:max(mean.shape)]*1e3, mean, label="{0}".format(N))
        plt.ylim((-2.5, 1.5))
        plt.legend()
    return
    
def analyze_data_200813(sourcedir):
    ''' First force-displacement test; check if repeatable on same sample '''   
    
    print("*****************\n Analyzing data from 08/13/20...")
    
    q0 = np.radians(0.1)
    L = 0.0189/2
    
    for index in range(3):
        plt.figure('200813_{0}'.format(index+1),dpi=200)
        plt.xlabel("Displacement (mm)")
        plt.ylabel("Load (N)")
        plt.axhline(color='gray')
        plt.vlines(2*L*1e3,-5,5,colors='gray')
    
    disp_n = np.zeros((1700, 3))
    load_n = np.zeros((1700, 3))
    disp_y = np.zeros((1700, 3))
    load_y = np.zeros((1700, 3))  

    filelist = os.listdir(sourcedir)
    for i, fname in enumerate(filelist):
        fullpath = os.path.join(sourcedir, fname)
        fnameParams = os.path.splitext(fname)[0].split('_')
        zeroDisp = 11.948

        if ('tension' in fnameParams) and ('200813' in fnameParams) and ('2' in fnameParams):
            #num = int(fnameParams[-1])
            if ('h0.96' in fnameParams):
                j = 1
                if ('N' in fnameParams):
                    col = 'b'    
                    sampleLoad = 0.16
                    label='h=0.96mm, r=0.40, no magnet'              
                if ('Y' in fnameParams):
                    col = 'r'    
                    sampleLoad = 0.230
                    label='h=0.96mm, r=0.40, magnet'             
            if ('h1.30' in fnameParams):
                j = 2
                if ('N' in fnameParams):
                    col = 'b'    
                    sampleLoad = 0.175
                    label='h=1.30mm, r=0.30, no magnet'              
                if ('Y' in fnameParams):
                    col = 'r'    
                    sampleLoad = 0.245
                    label='h=1.30mm, r=0.30, magnet'  
            if ('h1.73' in fnameParams):
                j = 3
                if ('N' in fnameParams):
                    col = 'b'    
                    sampleLoad = 0.165
                    label='h=1.73mm, r=0.22, no magnet'              
                if ('Y' in fnameParams):
                    col = 'r'    
                    sampleLoad = 0.230
                    label='h=1.73mm, r=0.22, magnet'  
            print(label)
            disp, load = import_experiment(fullpath, L, speed=0.2, 
                                           zeroDisp=zeroDisp, sampleLoad=sampleLoad)
            disp, load = tensiletest.filter_raw_data(disp, load, 50)

            plt.figure('200813_{0}'.format(j), dpi=200)
            plt.plot(disp[100:1800]*1e3, load[100:1800], col, label=label)
            plt.legend()
            
            if ('N' in fnameParams):
                disp_n[:,j-1] = disp[100:1800]
                load_n[:,j-1] = load[100:1800]
            if ('Y' in fnameParams):
                disp_y[:,j-1] = disp[100:1800]
                load_y[:,j-1] = load[100:1800]

    # Fit both models to data
    p_guess_pcw = [0.001, np.radians(40), 0.03] #kq, qlim, klim
    p_given_pcw = [q0, L, 'pcw']
    p_guess_exp = [0.001, 5e-20, 50] #kq, A, B
    p_given_exp = [q0, L, 'exp']
    params_pcw_n = np.zeros((3,3))
    params_exp_n = np.zeros((3,3))
    for i in range(3):    
        x = disp_n[:,i]
        y = load_n[:,i]
        params = approximate_spring(x, y, p_guess_pcw, p_given_pcw)
        params_pcw_n[:,i] = params
        plt.figure('200813_{0}'.format(i+1))
        plt.plot(x*1e3,force_displacement_nomagnet(x, params, q0, L, flag='pcw'), label='Piecewise fit')
        params = approximate_spring(x, y, p_guess_exp, p_given_exp)
        params_exp_n[:,i] = params
        plt.figure('200813_{0}'.format(i+1))
        plt.plot(x*1e3,force_displacement_nomagnet(x, params, q0, L, flag='exp'), label='Exponential fit')
        plt.legend()

    return


def get_temp_offset(T):
    return -1.29*((T-24.5)/(76.9-24.5))

def analyze_data_200825(sourcedir):
    print("*****************\n Analyzing data from 08/25/20...")
    
    
    LCE_modulus_params = [-8.43924097e-02, 2.16846882e+02, 3.13370660e+05]
    
    q0 = np.radians(0.1)
    
    
    fig = plt.figure('200825_comparison_low',dpi=200)
    plt.title('T = RT $^\circ$C')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.axhline(color='gray')

    fig = plt.figure('200825_comparison_med',dpi=200)
    plt.title('45 < T < 47 $^\circ$C')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.axhline(color='gray')

    fig = plt.figure('200825_comparison_high',dpi=200)
    plt.title('77 < T < 79 $^\circ$C')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.axhline(color='gray')

    p_guess_pcw = [0.001, np.radians(40), 0.03] #kq, qlim, klim
    p_guess_exp = [0.001, 5e-20, 50] #kq, A, B
    p_guess_mor = [0.001, 5e-20, 50] #kq, A, B

    filelist = os.listdir(sourcedir)
    for i, fname in enumerate(filelist):
        fullpath = os.path.join(sourcedir, fname)
        fnameParams = os.path.splitext(fname)[0].split('_')
        zeroDisp = 12.62

        if ('tension' in fnameParams) and ('200825' in fnameParams):
            T = float(re.search("_(\d+\.\d+)C", fname).group(1)) #[C]
            tempLoadOffset = get_temp_offset(T)
            if ('sample1' in fnameParams):
                j = 1
                L = 0.0178/2
                h = 1.65
                r = 0.18
                s = 1.57
                if ('N' in fnameParams):
                    col = 'b'    
                    sampleLoad = 0.13
                    label='Sample1, no magnet'              
                if ('Y' in fnameParams):
                    col = 'r'    
                    sampleLoad = 0.20
                    label='Sample1, magnet'             
            if ('sample2' in fnameParams):
                j = 2
                L = 0.0181/2
                h = 1.76
                r = 0.17
                s = 1.59
                if ('N' in fnameParams):
                    col = 'b'    
                    sampleLoad = 0.15
                    label='Sample2, no magnet'              
                if ('Y' in fnameParams):
                    col = 'r'    
                    sampleLoad = 0 #TBD
                    label='Sample2, magnet'  
            if ('sample3' in fnameParams):
                j = 3
                L = 0.0184/2
                h = 1.74
                r = 0.17
                s = 1.94
                if ('N' in fnameParams):
                    col = 'b'    
                    sampleLoad = 0.16
                    label='Sample3, no magnet'              
                if ('Y' in fnameParams):
                    col = 'r'    
                    sampleLoad = 0 #TBD
                    label='Sample3, magnet'  
            print(label)
            
            b = [0, s/h]
            #L = 0.0177/2
            p_given_pcw = [q0, L, 'pcw']
            p_given_exp = [q0, L, 'exp']
            p_given_mor = [q0, L, 'mor']
            
            disp, load = import_experiment(fullpath, L, speed=0.2, 
                                           zeroDisp=zeroDisp, sampleLoad=sampleLoad,
                                           tempLoadOffset=tempLoadOffset)
            disp, load = tensiletest.filter_raw_data(disp, load, 50, cropFlag=True)

            plt.figure(dpi=200)
            plt.plot(disp*1e3, load, col, label=label+', T={0}C'.format(T))
            plt.xlabel("Displacement (mm)")
            plt.ylabel("Load (N)")
            plt.axhline(color='gray')
            plt.vlines(2*L*1e3,-5,5,colors='gray')
            plt.legend()

            if T<30 and ('N' in fnameParams):
                plt.figure('200825_comparison_low')
                disp, load = crop_to_compressed(disp, load, L)
                #load = load - load[-1]
                plt.plot((2*L-disp)*1e3, -load, label='Sample{0}'.format(j))
                plt.legend()
            elif T<50 and ('N' in fnameParams):
                plt.figure('200825_comparison_med')
                disp, load = crop_to_compressed(disp, load, L)
                #load = load - load[-1]
                plt.plot((2*L-disp)*1e3, -load, label='Sample{0}'.format(j))
                plt.legend()
            elif T>50 and ('N' in fnameParams):
                plt.figure('200825_comparison_high')
                disp, load = crop_to_compressed(disp, load, L)
                #load = load - load[-1]
                plt.plot((2*L-disp)*1e3, -load, label='Sample{0}'.format(j))
                plt.legend()
            
            if 'N' in fnameParams:
                x = disp
                y = load
                y = y-y[-1]
                params_spr = approximate_only_spring(x, y, 0.001, [q0, L], preCropped=True, titlestr=', T={0}'.format(T))
                params_thy = model_spring(disp, load, L, h*1e-3, r, T, LCE_modulus_params, b, figFlag=True, titlestr=', T={0}'.format(T))
                params_pcw = approximate_spring(x, y, p_guess_pcw, p_given_pcw, preCropped=True)
                params_exp = approximate_spring(x, y, p_guess_exp, p_given_exp, preCropped=True)
                #params_mor = approximate_spring(x, y, p_guess_mor, p_given_mor, preCropped=True)
    return

#%%
if __name__ == "__main__":
    import os
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/unitCell_properties")
    tmpdir = os.path.join(cwd,"tmp")
    
    #analyze_data_200729(filelist, sourcedir)
    #sourcedir = os.path.join(rawdir,'200803')    
    #analyze_data_200803(sourcedir)
    #sourcedir = os.path.join(rawdir,'200813')
    #analyze_data_200813(sourcedir)
    sourcedir = os.path.join(rawdir,'200825')
    analyze_data_200825(sourcedir)
    