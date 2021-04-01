"""
Contains functions defining models for force-displacement curves

@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt

from modules import bilayermodel as bilayer

DEBUG_FLAG = False  # True to show more figures and print statements


# =============================================================================
# Force model
# =============================================================================
def rot2disp(q, L):
    """ Displacement between centers of squares """
    return 2*L*np.cos(q)

def disp2rot(x, L):
    """ Displacement between centers of squares """
    return np.arccos(x/(2*L))

def equivalent_springs_in_series(k_hinge, k_squares):
    return 1/(1/k_hinge + 1/k_squares)

def torque_k(q, kq, q0):
    """ Linear torsional spring representing hinge """
    return -kq*(2*(q-q0))

def torque_magnet(q, moment, L):
    """ Point dipole at centers of each square """
    mu = 1.2566*10**(-6) # [N/A^2]
    return 3*mu*(moment**2)*L*np.sin(q)/(2*np.pi*rot2disp(q, L)**4)

def torque_lim(q, qlim, klim, flag='pcw'):
    """ Torque resulting from square collision, limiting max displacement """
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
    """ Piecewise linear torsional spring model for square collision """
    if q_value > qlim:
        M = -klim*(2*(q_value-qlim))
    elif q_value < -qlim:
        M = -klim*(2*(q_value+qlim))
    else:
        M = 0
    return M    

def torque_lim_exp_single(q_value, A, B):
    """ Exponential model for square collision """
    M = -B*A*(np.exp(B*(q_value)) - np.exp(-B*(q_value)))
    return M

def torque_morse(q, A, alpha, qMorse, L, q0):
    """ Morse interatomic potential model for magnetic interaction + square collision """   
    M = 2*alpha*A*(  np.exp(4*alpha*(q+q0-qMorse))
                   - np.exp(2*alpha*(q+q0-qMorse))
                   - np.exp(-4*alpha*(q+q0+qMorse))
                   + np.exp(-2*alpha*(q+q0+qMorse)) )
    return M

# =============================================================================
# Constant-r analysis 
# =============================================================================
def force_displacement_all(x, p, q0, L, m, p_lim, flag='exp'):
    return force_displacement_magnet_collision(x, [m, p_lim[0], p_lim[1]],
                                               q0, p, L, flag=flag)


def force_displacement_magnet_collision(x, p, q0, kq, L, flag='exp'):
    """ Defines force(disp) function including magnets, with moment and
        collision parameters and fitting parameters and k_q as known """
    moment, p1, p2 = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_m = torque_magnet(q, moment, L)
    M_lim = torque_lim(q, p1, p2, flag=flag)
    M = 4*M_k + 4*M_m + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F


def force_displacement_only_magnet(x, moment, q0, kq, L):
    """ Defines force(disp) function, with moment as fitting parameter and 
        assuming k_q is known """
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_m = torque_magnet(q, moment, L)
    M = 4*M_k + 4*M_m
    F = M/(2*L*np.cos(q))
    return F


def force_displacement_magnet(x, p, q0, kq, L, pMorse=[], flag='exp'):
    """ Defines force(disp) function including magnets, with moment and
        collision parameters and fitting parameters and k_q as known;
        allows either Morse model or moment + collision function model """
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


def residual_force_displacement_spring_from_all(p, y, x, q0, L, m, p_lim, flag='exp'):
    return (y - force_displacement_all(x, p, q0, L, m, p_lim, flag=flag))


def residual_force_displacement_only_magnet(p, y, x, q0, kq, L):
    return (y - force_displacement_only_magnet(x, p, q0, kq, L))


def residual_force_displacement_only_collision(p, y, x, q0, kq, L, m):
    return (y - force_displacement_magnet(x, [m, p[0], p[1]], q0, kq, L))


def approximate_only_magnet(disp, force, p_guess, p_given, figFlag=False):
    """ Given some known parameters p_given, determine magnetic moment """
    
    q0, kq, L = p_given
    disp_plt = 2*L - disp
    p_fit = opt.least_squares(residual_force_displacement_only_magnet, p_guess,
                                      args=(force, disp, q0, kq, L))
    moment = p_fit.x[0]
    paramstr = "moment: {0:.4f} A$\cdot$m$^2$".format(moment)

    return moment


def approximate_only_collision(disp, force, p_guess, p_given, figFlag=False):
    """ Given some known parameters p_given, determine collision function parameters """
    
    q0, kq, L, m = p_given
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residual_force_displacement_only_collision, p_guess,
                                      args=(force, disp, q0, kq, L, m))
    p_lim = p_fit.x
    paramstr = "p_lim: {0}".format(p_lim)

    return p_lim

# =============================================================================
# NEW STUFF FOR CONSTANT-R!
# =============================================================================
def approximate_ksq_L(disp, force, p_guess, p_given, figFlag=False):
    """ Given some known parameters p_given, determine magnetic moment """
    
    q0, k_hinge, moment = p_given
    p_fit = opt.least_squares(residual_force_displacement_ksq_L, p_guess,
                              args=(force, disp, q0, k_hinge, moment))
    p_fit = p_fit.x

    return p_fit

def residual_force_displacement_ksq_L(p, y, x, q0, k_hinge, moment):
    return (y - force_displacement_ksq_L(x, p, q0, k_hinge, moment))

def force_displacement_ksq_L(x, p, q0, k_hinge, moment):
    """ Defines force(disp) function including magnets, with k_sq and L as
    fitting parameters and k_hinge, moment as known. Assume no collision! """
    k_squares, L = p
    kq = equivalent_springs_in_series(k_hinge, k_squares)
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_m = torque_magnet(q, moment, L)
    #M_lim = torque_lim(q, p1, p2, flag=flag)
    M = 4*M_k + 4*M_m# + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F


# =============================================================================
# Repeatability analysis 
# =============================================================================
def approximate_spring(disp, force, p_guess, p_given, figFlag=True):
    """ Given some known parameters p_given, determine k_q and collision params """
    
    q0, L, flag = p_given
    disp_plt = 2*L - disp

    # Fit to model
    p_fit = opt.least_squares(residual_force_displacement_nomagnet, p_guess,
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


def residual_force_displacement_nomagnet(p, y, x, q0, L, flag='pcw'):
    return (y - force_displacement_nomagnet(x, p, q0, L, flag=flag))


def force_displacement_nomagnet(x, p, q0, L, flag='pcw'):
    """ Defines force(disp) function without magnets, for fitting parameters 
        k_q and collision parameters"""
    kq, p1, p2 = p
    q = disp2rot(x, L)
    M_k = torque_k(q, kq, q0)
    M_lim = torque_lim(q, p1, p2, flag=flag)
    M = 4*M_k + 4*M_lim
    F = M/(2*L*np.cos(q))
    return F


def plot_force_and_energy(disp, load_exp, load_model, energy_model,
                          titlestr='', notestr=''):
    """ Plot load-displacement for experiment and model, and energy for model, 
        including descriptive title and additional notes text"""
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
