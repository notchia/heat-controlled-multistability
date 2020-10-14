# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:08:15 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt

from modules import bilayermodel as bilayer

DEBUG_FLAG = False  # Show more figures and print statements


#%% Model -------------------------------------------------------------------
def rot2disp(q, L):
    ''' Displacement between centers of squares '''
    return 2*L*np.cos(q)

def disp2rot(x, L):
    ''' Displacement between centers of squares '''
    return np.arccos(x/(2*L))

def equivalent_springs_in_series(k_hinge, k_squares):
    return 1/(1/k_hinge + 1/k_squares)

def torque_k(q, kq, q0):
    ''' Linear torsional spring representing hinge '''
    return -kq*(2*(q-q0))

def torque_magnet(q, moment, L):
    ''' Point dipole at centers of each square '''
    mu = 1.2566*10**(-6) # [N/A^2]
    return 3*mu*(moment**2)*L*np.sin(q)/(2*np.pi*rot2disp(q, L)**4)

def torque_lim(q, qlim, klim, flag='pcw'):
    ''' Torque resulting from square collision, limiting max displacement '''
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
    ''' Piecewise linear torsional spring model for square collision '''
    if q_value > qlim:
        M = -klim*(2*(q_value-qlim))
    elif q_value < -qlim:
        M = -klim*(2*(q_value+qlim))
    else:
        M = 0
    return M    

def torque_lim_exp_single(q_value, A, B):
    ''' Exponential model for square collision '''
    M = -B*A*(np.exp(B*(q_value)) - np.exp(-B*(q_value)))
    return M

def torque_morse(q, A, alpha, qMorse, L, q0):
    ''' Morse interatomic potential model for magnetic interaction + square collision '''   
    M = 2*alpha*A*(  np.exp(4*alpha*(q+q0-qMorse))
                   - np.exp(2*alpha*(q+q0-qMorse))
                   - np.exp(-4*alpha*(q+q0+qMorse))
                   + np.exp(-2*alpha*(q+q0+qMorse)) )
    return M

# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_magnet(x, p, q0, kq, L, pMorse=[], flag='exp'):
    ''' Defines force(disp) function '''
    moment, p1, p2 = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    if flag=='morse':
        M_morse = torque_morse(q, *pMorse)
        M = 4*M_k + 4*M_morse
    else:
        M_m = torque_magnet(q, moment, L)
        M_lim = torque_lim(q, p1, p2, flag=flag)
        M = 4*M_k + 4*M_m + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F

# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_only_magnet(x, moment, q0, kq, L):
    ''' Defines force(disp) function '''
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_m = torque_magnet(q, moment, L)
    M = 4*M_k + 4*M_m
    F = M/(2*L*np.cos(q))
    return F


# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_magnet_collision(x, p, q0, kq, L, flag='exp'):
    ''' Defines force(disp) function '''
    moment, p1, p2 = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_m = torque_magnet(q, moment, L)
    M_lim = torque_lim(q, p1, p2, flag=flag)
    M = 4*M_k + 4*M_m + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F

# three fitting parameters; assume q0, L are design parameters, and kq is known
def force_displacement_nomagnet(x, p, q0, L, flag='pcw'):
    ''' Defines force(disp) function '''
    kq, p1, p2 = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_lim = torque_lim(q, p1, p2, flag=flag)
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
    
def force_displacement_magnet_collision_for_fit(x, moment, p1, p2, q0=0, kq=0, L=0, flag='exp'):
    ''' Defines force(disp) function, with inputs rearranged for opt.curve_fit'''
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_m = torque_magnet(q, moment, L)
    M_lim = torque_lim(q, p1, p2, flag=flag)
    M = 4*M_k + 4*M_m + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F

#%% Analysis ------------------------------------------------------------------
def residue_force_displacement_spring(p, y, x, q0, L):
    return (y - force_displacement_spring(x, p, q0, L))

def residue_force_displacement_nomagnet(p, y, x, q0, L, flag='pcw'):
    return (y - force_displacement_nomagnet(x, p, q0, L, flag=flag))

def residue_force_displacement_magnet(p, y, x, q0, kq, L, flag='pcw'):
    return (y - force_displacement_magnet(x, p, q0, kq, L, flag=flag))

def residue_force_displacement_only_magnet(p, y, x, q0, kq, L):
    return (y - force_displacement_only_magnet(x, p, q0, kq, L))

def residue_force_displacement_only_collision(p, y, x, q0, kq, L, m):
    return (y - force_displacement_magnet(x, [m, p[0], p[1]], q0, kq, L))

def residue_force_displacement_magnet_collision(p, y, x, q0, kq, L, flag='exp'):
    return (y - force_displacement_magnet_collision(x, p, q0, kq, L, flag=flag))

def nested_function_all(q0, kq, L, flag):
    def nokw_fcn(x, m, p1, p2):
        return force_displacement_magnet_collision_for_fit(x, m, p1, p2,
                                                           q0=q0, kq=kq, L=L, flag=flag)
    return nokw_fcn

def nested_function_collision(q0, kq, L, m, flag):
    def nokw_fcn(x, p):
        return force_displacement_magnet_collision_for_fit(x, *p,
                                                           q0=q0, kq=kq, L=L,
                                                           m=m, flag=flag)
    return nokw_fcn

def fit_magnet_and_collision_weighted(disp, force, p_guess, p_given, weights,
                                      figFlag=False):
    
    q0, kq, L, flag = p_given

    fcn_to_fit = nested_function_all(q0, kq, L, flag)

    # Fit to model
    p_fit, _ = opt.curve_fit(fcn_to_fit, disp, force, #method='trf',
                             #bounds=([0.0,1e-30,1],[0.50,1e-1,100]),
                             p0=p_guess, sigma=weights, maxfev=12800)
    print(p_fit)
    moment, qlim, klim = p_fit
    paramstr = "moment: {0:.4f} A$\cdot$m$^2$\n$q_{{lim}}$: {1:.4f}$^\circ$\n$k_{{lim}}$: {2:.4f} Nm".format(moment, np.degrees(qlim), klim)

    force_fit = force_displacement_magnet_collision(disp, p_fit, q0, kq, L)
    energy_fit = np.cumsum(force_fit)
    
    if figFlag:
        disp_plt = 2*L - disp
        plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
                              titlestr='With magnet', notestr=paramstr)
   
    return p_fit

def approximate_magnet_collision(disp, force, p_guess, p_given, figFlag=False):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, kq, L = p_given
    
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residue_force_displacement_magnet_collision, p_guess,
                                      args=(force, disp, q0, kq, L))
    #moment, p1, p2 = p_fit.x
    #paramstr = "moment: {0:.4f} A$\cdot$m$^2$\n$q_{{lim}}$: {1:.4f}$^\circ$\n$k_{{lim}}$: {2:.4f} Nm".format(moment, np.degrees(qlim), klim)

    #force_fit = force_displacement_magnet_collision(disp, p_fit.x, q0, kq, L)
    #energy_fit = np.cumsum(force_fit)
    
    #if figFlag:
    #    plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
    #                          titlestr='With magnet', notestr=paramstr)

    
    return p_fit.x


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
    energy_fit = energy_fit - energy_fit[0]
    
    if figFlag:
        plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
                              titlestr='With magnet', notestr=paramstr)

    return moment, qlim, klim


def approximate_only_magnet(disp, force, p_guess, p_given, figFlag=False):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, kq, L = p_given
    
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residue_force_displacement_only_magnet, p_guess,
                                      args=(force, disp, q0, kq, L))
    moment = p_fit.x[0]
    paramstr = "moment: {0:.4f} A$\cdot$m$^2$".format(moment)

#    force_fit = force_displacement_only_magnet(disp, p_fit.x, q0, kq, L)
#    energy_fit = np.cumsum(force_fit)
#    energy_fit = energy_fit - energy_fit[0]
    
#    if figFlag:
#        plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
#                              titlestr='With magnet, no collision', notestr='')

    return moment

def approximate_only_collision(disp, force, p_guess, p_given, figFlag=False):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, kq, L, m = p_given
    
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residue_force_displacement_only_collision, p_guess,
                                      args=(force, disp, q0, kq, L, m))
    p_lim = p_fit.x
    paramstr = "p_lim: {0}".format(p_lim)

#    force_fit = force_displacement_magnet(disp, [m, p_lim[0], p_lim[1]], q0, kq, L)
#    energy_fit = np.cumsum(force_fit)
#    energy_fit = energy_fit - energy_fit[0]
    
#    if figFlag:
#        plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
#                              titlestr='With magnet, no collision', notestr=paramstr)

    return p_lim

def approximate_only_collision_weighted(disp, force, p_guess, p_given, figFlag=False):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, kq, L, m = p_given
    
    # Fit to model
    fcn_to_fit = nested_function_collision(q0, kq, L, m, flag)
    p_fit, _ = opt.curve_fit(fcn_to_fit, disp, force, 
                             p0=p_guess, sigma=weights, maxfev=1600)
    p_lim_fit = p_fit
    
    p_fit = opt.least_squares(residue_force_displacement_only_collision, p_guess,
                                      args=(force, disp, q0, kq, L, m))
    p_lim = p_fit.x
    paramstr = "p_lim: {0}".format(p_lim)

    force_fit = force_displacement_magnet(disp, [m, p_lim[0], p_lim[1]], q0, kq, L)
    energy_fit = np.cumsum(force_fit)
    energy_fit = energy_fit - energy_fit[0]
    
    if figFlag:
        disp_plt = 2*L - disp
        plot_force_and_energy(disp_plt, force, force_fit, energy_fit,
                              titlestr='With magnet, no collision', notestr=paramstr)

    return p_lim


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

def approximate_spring(disp, force, p_guess, p_given, figFlag=True):

    q0, L, flag = p_given

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
    energy_fit = energy_fit - energy_fit[0]
    
    if figFlag:
        plot_force_and_energy(disp_plt, force, force_fit, energy_fit*1e-3,
                              titlestr='No magnet', notestr=paramstr)

    return params

#def compare_spring_theory_to_fit(disp, force)

def approximate_only_spring(disp, force, p_guess, p_given, figFlag=False, titlestr=''):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, L = p_given

    # Crop data to where displacement exceeds 
    #if not preCropped:
    #    disp, force = crop_to_compressed(disp, force, L)
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
    energy_model = np.cumsum(force_model)
    energy_model = energy_model - energy_model[0]
    
    if figFlag:
        plot_force_and_energy(disp_plt, force, force_model, energy_model*1e-3,
                              titlestr='Torsional spring estimate'+titlestr)

    return kq

def approximate_ksq_and_s(disp, force, p_guess, p_given, preCropped=False, figFlag=True, titlestr=''):
    ''' Given some known parameters p_given, determine fitting parameters p_guess '''
    
    q0, L = p_given

    # Crop data to where displacement exceeds 
    #if not preCropped:
    #    disp, force = crop_to_compressed(disp, force, L)
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
    energy_model = energy_model - energy_model[0]
    
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