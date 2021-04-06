"""
Analyze data on constant-r samples
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.optimize as opt
import scipy.signal as signal

from modules import unitforcedata as experiment
from modules import unitforcemodel as model
from modules import metamaterialmodel as m3

ANGLE_NEAR_ZERO = 1e-8 # Since setting angle to exactly zero can give numerical trouble


# =============================================================================
# Main functions (to be called in script)
# =============================================================================
def import_rconst_data(sourcedir, bilayerDict={}, m=0, setStartLoadToZero=False):
    """ Check how repeatably we can manufacture and test unit cells with
        nominally identical hinge thickness ratio """
    
    ucdf = experiment.import_all_unit_cells(sourcedir, bilayerDict=bilayerDict, 
                                            setStartLoadToZero=setStartLoadToZero,
                                            m=m, figFlag=False)
    h_vals = ucdf.h.unique()
    for h in h_vals:
        experiment.plot_magnet_and_T_comparison(ucdf[ucdf["h"] == h], 
                                                title=f'rconst, $h$ = {1e3*h} mm')
    r_avg = ucdf["r"].mean()

    return r_avg, ucdf


def analyze_rconst_ksq(ucdf, bilayerDict):
    """Analysis of just no-magnet case: find best-fit square stiffness ksq
    (in series with hinge stiffness kq)"""
    
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1] 
    
    # Find individual best-fit values for k_q
    p_fit_spring_individual = np.zeros(len(ucdf_Y))
    count = 0
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              hasMagnets=bool(unitData.magnets), m=unitData.m,
                                              p_lim = [0,0], # Assuming no collision
                                              **bilayerDict)

        # Use subset of load-disp data before collision begins to fit k_q
        maxStrain = 0.15
        maxIndex = np.where(unitData.strain < maxStrain)[0][-1]
        #p_given = [unitModel.total_angle, unitData.d/2]
        q0 = unitModel.total_angle
        L = unitData.d/2
        m = unitData.m
        p_lim = [0,0]
        p_guess = 1e-3 # k_q guess

        p_fit = opt.least_squares(model.residual_force_displacement_spring_from_all, p_guess,
                                  args=(-unitData.load[:maxIndex], unitData.disp[:maxIndex], q0, L, m, p_lim))
        
        p_fit_spring_individual[count] = p_fit.x[0]
        count += 1

    # Find best-fit value for k_sq
    p_fit_all = fit_constant_ksq(ucdf_Y, p_fit_spring_individual, bilayerDict)
    k_sq_fit = p_fit_all.x[0]
    
    return k_sq_fit


def analyze_rconst_moment(ucdf, k_sq_fit, bilayerDict):
    """Analysis of just with-magnet case: find best-fit magnetic moment m"""
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  

    p_guess = 0.20
    
    # Find individual best-fit values for magnetic moment
    p_fit_moment_individual = np.zeros(len(ucdf_Y))
    count = 0
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              k_sq=k_sq_fit,
                                              hasMagnets=False, p_lim=[0,0], # Assuming no collision
                                              **bilayerDict) # Using this only for k_eq

        # Use only data near zero crossings, besides any due to collision!
        nearZeroIndices = find_near_zero_indices(unitData.load, window=100)
        maxStrain = 0.15
        maxIndex = np.where(unitData.strain > maxStrain)[0][0]
        nearZeroIndices = [i for i in nearZeroIndices if i < maxIndex]
        
        #plt.figure()
        #plt.plot(unitData.strain, unitData.load, 'k')
        #plt.plot(unitData.strain[nearZeroIndices], unitData.load[nearZeroIndices], 'ro', linewidth=0)

        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2]
        p_fit_moment_individual[count] = model.approximate_only_magnet(unitData.disp[:maxIndex],
                                                                       -unitData.load[:maxIndex],
                                                                       p_guess, p_given)
        count += 1

    # Find best-fit value for magnetic moment
    p_fit_all = fit_constant_moment(p_fit_moment_individual, p_guess)
    moment_fit = p_fit_all.x[0]
    
    return moment_fit


def plot_final_rconst_fit(ucdf, k_sq_fit, m_fit, p_lim_fit, bilayerDict, d_fit=0.018, limFlag='exp'):
    """Plot force-displacement curves for experiment data and best fit for all data,
    including markers at roots"""
    
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1] 
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        plt.figure(dpi=200)
        plt.xlabel("Strain, $\delta/d$")
        plt.ylabel("Load (N)")
        plt.title(unitData.label)
        
        # Plot experiment and model curves
        disp_plt = unitData.strain        
        plt.plot(disp_plt, unitData.load, 'k', label="experiment")
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              k_sq=k_sq_fit, m=m_fit, p_lim=p_lim_fit,
                                              limFlag=limFlag,
                                              hasMagnets=bool(unitData.magnets), analysisFlag=False,
                                              **bilayerDict)
        plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), 'r', label="model")
        
        # Plot experiment and model roots
        dataIndices, _ = find_zero_crossings(unitData.load, startSign=1)
        for i in dataIndices:
            plt.plot(disp_plt[i], 0, 'ko', fillstyle='none', label='roots of experiment')
        modelIndices, _ = find_zero_crossings(unitModel.model_load_disp(unitData.disp), startSign=1)
        for j in modelIndices:
            plt.plot(disp_plt[int(j)], 0, 'rx', label='roots of model') 
        
        plt.legend()
        
    return()

# =============================================================================
# Internal functions
# =============================================================================
def fit_constant_ksq(ucdf, p_fit_individual, bilayerDict, k_sq_guess=1e-3, p_lim=[0,0], limFlag='exp'):
    """ Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit """
       
    p_fit = opt.least_squares(residual_constant_ksq, k_sq_guess,
                              args=(p_fit_individual, ucdf, bilayerDict),
                              kwargs={'p_lim':p_lim, 'limFlag':limFlag})
    
    return p_fit


def residual_constant_ksq(p, ksq_fit_individual, ucdf, bilayerDict, p_lim=[0,0], limFlag='exp'):
    """ Cost function: minimize difference between equivalent spring constant
        for hinge in series with constant k_sq and the best-fit torsional
        spring constant, for all force-displacement relations """

    k_sq = p
    ksq_fit_all = np.zeros(len(ucdf))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                         T=unitData.T, d=unitData.d, s=unitData.s, 
                                         k_sq=k_sq, p_lim=p_lim, m=unitData.m,
                                         hasMagnets=bool(unitData.magnets),
                                         **bilayerDict)
        ksq_fit_all[count] = unitModel.k_eq
        count += 1

    residual = ksq_fit_individual - ksq_fit_all

    return residual


def fit_constant_moment(p_fit_individual, p_guess):
    """ Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit """
    
    p_fit = opt.least_squares(residual_constant_moment, p_guess,
                              args=(p_fit_individual,))
    
    return p_fit


def residual_constant_moment(p, moment_fit_individual):
    """ Cost function: minimize difference between constant magnetic moment and
        best-fit magnetic moment, for all force-displacement relations """

    return moment_fit_individual - p


def find_near_zero_indices(y_array, window=25):
    zeroIndices, _ = find_zero_crossings(y_array)
    nearZeroIndices = []
    for i in zeroIndices:
        imin = max(0, i-window)
        imax = min(len(y_array)-1, i+window)
        nearZeroIndices.extend(list(range(imin, imax)))
    return nearZeroIndices


def find_zero_crossings(y_array, startSign=1):
    """ Finds locations and signs of zero crossings present """
    zeroCrossings = []
    slopeAtZero = []
    nPoints = len(y_array)
    previousSign = startSign
    for i, y in enumerate(y_array[::-1]):
        currentSign = np.sign(y)
        if currentSign != previousSign:
            zeroCrossings.append(nPoints - i - 1)
            slopeAtZero.append(previousSign)
            previousSign = currentSign
    return zeroCrossings, slopeAtZero


# =============================================================================
# Not currently in use
# =============================================================================
def analyze_rconst_collision(ucdf, k_sq_fit, moment_fit, bilayerDict):
    """ Analysis of just with-magnet case: find best-fit collision parameters
        (A, B) for exponential fit"""
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
    
    p_fit_all = fit_constant_collision(ucdf_Y, [k_sq_fit, moment_fit, bilayerDict])
    p_lim_fit = p_fit_all.x
    
    return p_lim_fit


def fit_constant_collision(ucdf, p_given, p_lim_guess=[3e-22, 51], limFlag='exp'): #[3e-10, 18]
    """ Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit """
       
    p_fit = opt.least_squares(residual_constant_collision, p_lim_guess,
                              args=(ucdf, p_given),
                              kwargs={'limFlag':limFlag},
                              xtol=1e-10, gtol=1e-10)
    return p_fit


def residual_constant_collision(p, ucdf, params_given, limFlag='exp'):
    """ Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations """

    k_sq_fit, m_fit, bilayerDict = params_given   

    # Find individual best-fit values for k_q
    nSamples = len(ucdf)
    p_lim_fit_individual = np.zeros((nSamples,2))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=k_sq_fit, p_lim=p, limFlag=limFlag,
                                              m=m_fit,
                                              hasMagnets=bool(unitData.magnets),
                                              analysisFlag=False,
                                              **bilayerDict)

        # Use only data nera zero crossings!
        #nearZeroIndices = find_near_zero_indices(unitData.load, window=30)
        minStrain = 0.26
        minIndex = np.where(unitData.strain > minStrain)[0][0]
        
        #plt.figure()
        #plt.plot(unitData.strain, unitData.load, 'k')
        #plt.plot(unitData.strain[minIndex:], unitData.load[minIndex:], 'ro', linewidth=0)
        
        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2, m_fit]
        p_guess = p
        p_lim_fit_individual[count,:] = model.approximate_only_collision(unitData.disp[minIndex:],
                                                                       -unitData.load[minIndex:],
                                                                       p_guess, p_given)
        count += 1
        
    # Reshape and subtract
    p_lim_fit_individual = p_lim_fit_individual.flatten()
    p_lim_fit_individual = p_lim_fit_individual.reshape((-1,1))

    p_lim_fit_all = np.zeros((nSamples,2))
    p_lim_fit_all[:,0] = p[0]
    p_lim_fit_all[:,1] = p[1]
    p_lim_fit_all = p_lim_fit_all.flatten()
    p_lim_fit_all = p_lim_fit_all.reshape((-1,1))

    residual = p_lim_fit_individual - p_lim_fit_all
    residual = residual.flatten()

    return residual


# =============================================================================
# New fit??
# =============================================================================
def update_data_with_d(ucdf, d_fit):
    ucdf_update = pd.DataFrame()
    for index, row in ucdf.iterrows():
        data = row["data"]
        data.reset_d(d_fit)
        ucdf_update = ucdf_update.append(data.get_Series(), ignore_index=True)
    
    return ucdf_update

def analyze_rconst_ksq_and_d(ucdf, m_fit, bilayerDict):
    """ Analysis of just with-magnet case: find best-fit ksq and d parameters"""
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
    p_given = [m_fit, bilayerDict]
    p_fit_all = fit_constant_ksq_and_d(ucdf_Y, p_given)
    p_fit = p_fit_all.x
    
    return p_fit


def fit_constant_ksq_and_d(ucdf, p_given, p_guess=[0.005, 0.009], limFlag='exp'): 
    """ Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and diagonal length which minimizes the
        least-squares error for all fit """
       
    p_fit = opt.least_squares(residual_constant_ksq_and_d, p_guess,
                              args=(ucdf, p_given),
                              kwargs={'limFlag':limFlag})
    return p_fit


def residual_constant_ksq_and_d(p, ucdf, params_given, limFlag='exp'):
    """ Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations """

    m_fit, bilayerDict = params_given
    k_sq_fit, L_fit = p

    print(f"running... p = {p}")    

    # Find individual best-fit values for k_q
    nSamples = len(ucdf)
    p_fit_individual = np.zeros((nSamples,2))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitData.reset_d(2*L_fit)
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                         T=unitData.T, p_lim=[0,0], limFlag=limFlag,
                                         k_sq=k_sq_fit, d=2*L_fit, 
                                         m=m_fit, 
                                         hasMagnets=bool(unitData.magnets),
                                         analysisFlag=False,
                                         **bilayerDict)

        # Use only data before collision
        maxStrain = 0.15
        maxIndex = np.where(unitData.strain < maxStrain)[0][-1]
        
        #plt.figure()
        #plt.plot(unitData.strain, unitData.load, 'k')
        #plt.plot(unitData.strain[:maxIndex], unitData.load[:maxIndex], 'ro', linewidth=0)
        
        p_given = [unitModel.total_angle, unitModel.hinge.k, m_fit]
        p_guess = p
        p_fit_individual[count,:] = model.approximate_ksq_L(unitData.disp[:maxIndex],
                                                            -unitData.load[:maxIndex],
                                                            p_guess, p_given)
        count += 1
        
    # Reshape and subtract
    p_fit_individual = p_fit_individual.flatten()
    p_fit_individual = p_fit_individual.reshape((-1,1))

    p_fit_all = np.zeros((nSamples,2))
    p_fit_all[:,0] = p[0]
    p_fit_all[:,1] = p[1]
    p_fit_all = p_fit_all.flatten()
    p_fit_all = p_fit_all.reshape((-1,1))

    residual = p_fit_individual - p_fit_all
    residual = residual.flatten()

    return residual

# =============================================================================
# New new fit??
# =============================================================================
def analyze_rconst_ksq_and_m(ucdf, bilayerDict):
    """ Analysis of just with-magnet case: find best-fit ksq and d parameters"""
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
    p_given = bilayerDict
    p_fit_all = fit_constant_ksq_and_m(ucdf_Y, p_given)
    p_fit = p_fit_all.x
    
    return p_fit


def fit_constant_ksq_and_m(ucdf, p_given, p_guess=[0.005, 0.1465], limFlag='exp'): 
    """ Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and diagonal length which minimizes the
        least-squares error for all fit """
       
    p_fit = opt.least_squares(residual_constant_ksq_and_m, p_guess,
                              args=(ucdf, p_given),
                              kwargs={'limFlag':limFlag},
                              xtol=1e-16, gtol=1e-16)
    return p_fit


def residual_constant_ksq_and_m(p, ucdf, params_given, limFlag='exp'):
    """ Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations """

    bilayerDict = params_given
    k_sq_fit, m_fit = p

    print(f"running... p = {p}")    

    # Find individual best-fit values for k_q and m
    nSamples = len(ucdf)
    p_fit_individual = np.zeros((nSamples,2))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = m3.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                         T=unitData.T, p_lim=[0,0], limFlag=limFlag,
                                         k_sq=k_sq_fit, d=unitData.d, 
                                         m=m_fit, 
                                         hasMagnets=bool(unitData.magnets),
                                         analysisFlag=False,
                                         **bilayerDict)

        # Use only data before collision
        maxStrain = 0.15
        maxIndex = np.where(unitData.strain < maxStrain)[0][-1]
        
        p_given = [unitModel.total_angle, unitModel.hinge.k, unitData.d/2]
        p_guess = p
        p_fit_individual[count,:] = model.approximate_ksq_m(unitData.disp[:maxIndex],
                                                            -unitData.load[:maxIndex],
                                                            p_guess, p_given)
        count += 1
        
    # Reshape and subtract
    p_fit_individual = p_fit_individual.flatten()
    p_fit_individual = p_fit_individual.reshape((-1,1))

    p_fit_all = np.zeros((nSamples,2))
    p_fit_all[:,0] = p[0]
    p_fit_all[:,1] = p[1]
    p_fit_all = p_fit_all.flatten()
    p_fit_all = p_fit_all.reshape((-1,1))

    residual = p_fit_individual - p_fit_all
    residual = residual.flatten()

    return residual


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/unitCell_properties")
    tmpdir = os.path.join(cwd,"tmp")
    
    sourcedir = os.path.join(rawdir,"unitCell_tension_rconst")
    #analyze_rconst_data(sourcedir)