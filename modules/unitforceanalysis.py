# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:21:38 2020

@author: Lucia
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.signal as signal

from modules import unitforcedata as experiment
from modules import unitforcemodel as model
from modules import metamaterialmodel as metamat

ANGLE_NEAR_ZERO = 1e-8 # Since setting angle to exactly zero can give numerical trouble

#%% Anaylze repeatability data
def analyze_repeatability_data(sourcedir, bilayerDict, k_sq=4e-3, setStartLoadToZero=False,
                               m=[0.1431], p_lim=[1e-14, 30],
                               saveFlag=False, figdir=''):
    ''' Check how repeatably we can manufacture and test unit cells with
        nominally identical parameters '''
    
    ucdf = experiment.import_all_unit_cells(sourcedir, setStartLoadToZero=setStartLoadToZero,
                                            bilayerDict=bilayerDict)
    experiment.plot_magnet_and_T_comparison(ucdf, legendOut=True)    
    
    ksq = k_sq

    # Get UCDF for averaged data
    ucdf_avg_N = pd.DataFrame()
    ucdf_model = []
    for T_group in range(3):
        subframe = ucdf.loc[(ucdf["T_group"] == T_group) & (ucdf["magnets"] == 0)]
        unitCell_avg = experiment.compute_unitCell_mean_std(subframe)
        row = unitCell_avg.get_Series()
        ucdf_avg_N = ucdf_avg_N.append(row,ignore_index=True)
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=ksq, loadFlag=True, hasMagnets=bool(unitData.magnets),
                                              **bilayerDict)
        ucdf_model.append(unitModel.model_load_disp(unitData.disp))
    
    #experiment.plot_magnet_and_T_comparison(ucdf_avg_N, stdFlag=True)
    experiment.plot_no_magnet_model_T_comparison(ucdf_avg_N, ucdf_model, stdFlag=True)
    if saveFlag:
        title = 'unitCell_repeatability_force_N'
        plt.savefig(os.path.join(figdir,"{0}.png".format(title)), dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(title)), transparent=True)   
 
   
    for index, row in ucdf_avg_N.iterrows():
        unitData = row["data"]
        plt.figure(dpi=200)
        plt.xlabel("Strain, $\delta/d$")
        plt.ylabel("Load (N)")
        plt.title(unitData.label)
        disp_plt = unitData.strain
        
        plt.plot(disp_plt, unitData.load, 'k', label="experiment")
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=ksq, loadFlag=True, hasMagnets=bool(unitData.magnets),
                                              **bilayerDict)
        plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), 'r', label="model")
        plt.fill_between(disp_plt, unitData.load-unitData.std, unitData.load+unitData.std, color='k', alpha=0.2)
        print(r"T = {0}C, $\theta_T$ = {1:.2f}, k = {2:.2e}".format(unitData.T, np.degrees(unitModel.total_angle), unitModel.hinge.k))
        plt.legend()  
        
        p_given_exp = [unitModel.total_angle, unitModel.d/2, 'exp']
        p_guess_exp = [0.001, 5e-20, 50]
        params = model.approximate_spring(unitData.disp, -(unitData.load), p_guess_exp, p_given_exp)
        print(params)
        #print('{0},{1}'.format(params[1],params[2]))   

    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
#    experiment.plot_magnet_and_T_comparison(ucdf_Y, legendOut=False)

#    moment_fit = analyze_rconst_moment(ucdf_Y, k_sq, bilayerDict)
#    print("m_fit: {0}".format(moment_fit))

    # Analyze Y case
    colors = ['r','g','b']
    TLabels = ['RT', 'T~45$^\circ$', 'T~75$^\circ$']
    
    fig = plt.figure('unitCell_repeatability_energy', dpi=200)
    plt.xlabel("Angle $\theta$ ($^\circ$)")
    plt.ylabel("Energy (J)")
    
    fig = plt.figure('unitCell_repeatability_force_Y', dpi=200)
    plt.xlabel("Strain, $\delta/d$")
    plt.ylabel("Load (N)")
    plt.axhline(color='k',linewidth=1)
    
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        
        iplt = unitData.T_group
        
        # Plot individual curve with model
        # plt.figure(dpi=200)
        # plt.xlabel("Strain, $\delta/d$")
        # plt.ylabel("Load (N)")
        # plt.title(unitData.label)
        disp_plt = unitData.strain
        
        # plt.plot(disp_plt, unitData.load-unitData.load[0], 'k', label="experiment")
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=ksq, loadFlag=True, hasMagnets=bool(unitData.magnets),
                                              m=m, p_lim=p_lim,
                                              **bilayerDict)
        #plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), 'r', label="model")
        #plt.fill_between(disp_plt, unitData.load-unitData.load[0]-unitData.std, unitData.load-unitData.load[0]+unitData.std, color='k', alpha=0.2)
        print(r"T = {0}C, $\theta_T$ = {1:.2f}, k = {2:.2e}".format(unitData.T, np.degrees(unitModel.total_angle), unitModel.hinge.k))
        #plt.legend()  

        fig = plt.figure('unitCell_repeatability_force_Y')        
        plt.plot(disp_plt, unitData.load, colors[iplt], label="experiment, {0}".format(TLabels[iplt]))
        plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), colors[iplt]+'--',
                 label="model, {0}".format(TLabels[iplt]))
        
        fig = plt.figure('unitCell_repeatability_energy')
        U_total, q_range = unitModel.model_energy()
        zeroIndex = np.where(q_range > 0)[0][0]
        U_total = U_total - U_total[zeroIndex]
        plt.plot(np.degrees(q_range), U_total, colors[iplt], label=TLabels[iplt])

        minima = signal.argrelmin(U_total)[0]
        maxima = signal.argrelmax(U_total)[0]
        minU = U_total[minima]
        maxU = U_total[maxima]
        plt.plot(np.degrees(q_range[minima]), minU, colors[iplt]+'v')
        plt.plot(np.degrees(q_range[maxima]), maxU, colors[iplt]+'^')
        
        minU.sort()
        maxU.sort()
        if maxU.size > 0:
            plt.ylim(minU[0]-0.0001, 0.003)
        
        p_given_exp = [unitModel.total_angle, unitModel.d/2, 'exp']
        p_guess_exp = [0.001, 5e-20, 50]
        params = model.approximate_spring(unitData.disp, -(unitData.load-unitData.load[0]), p_guess_exp, p_given_exp)
    
    if saveFlag:
        title = 'unitCell_repeatability_energy'
        fig = plt.figure(title, dpi=200)
        plt.legend()
        plt.savefig(os.path.join(figdir,"{0}.png".format(title)), dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(title)), transparent=True)       
        
        title = 'unitCell_repeatability_force_Y'
        fig = plt.figure(title, dpi=200)
        plt.legend()
        plt.savefig(os.path.join(figdir,"{0}.png".format(title)), dpi=200)
        plt.savefig(os.path.join(figdir,"{0}.svg".format(title)), transparent=True)   

    return

#%% Anaylze rconstant data
def import_rconst_data(sourcedir, bilayerDict={}, setStartLoadToZero=False, m=0):
    ''' Check how repeatably we can manufacture and test unit cells with
        nominally identical parameters '''
    
    ucdf = experiment.import_all_unit_cells(sourcedir, setStartLoadToZero=setStartLoadToZero,
                                            figFlag=False, bilayerDict=bilayerDict, m=m)
    experiment.plot_magnet_and_T_comparison(ucdf, legendOut=True)

    r_avg = ucdf["r"].mean()

    return r_avg, ucdf

def analyze_rconst_nomagnets(ucdf, bilayerDict):
    ''' Analysis of just no-magnet case: find best-fit square stiffness ksq (in
        series with hinge stiffness kq) '''
    
    ucdf_N = ucdf.loc[ucdf["magnets"] == 0] 
    
    # Find individual best-fit values for k_q
    p_fit_spring_individual = np.zeros(len(ucdf_N))
    count = 0
    for index, row in ucdf_N.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              hasMagnets=bool(unitData.magnets),
                                              **bilayerDict)

        # Use subset of load-disp data before collision begins to fit k_q
        maxStrain = 0.15#0.225
        maxIndex = np.where(unitData.strain < maxStrain)[0][-1]
        p_given = [unitModel.total_angle, unitData.d/2]
        p_guess = 1e-3 # k_q guess
        p_fit_spring_individual[count] = model.approximate_only_spring(unitData.disp[:maxIndex],
                                                                       -unitData.load[:maxIndex],
                                                                       p_guess, p_given)
        count += 1

    # Find best-fit value for k_sq
    p_fit_all = fit_constant_ksq(ucdf_N, p_fit_spring_individual)
    k_sq_fit = p_fit_all.x[0]
    
    return k_sq_fit

def analyze_rconst_ksq(ucdf, bilayerDict):
    ''' Analysis of just no-magnet case: find best-fit square stiffness ksq (in
        series with hinge stiffness kq) '''
    
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1] 
    
    # Find individual best-fit values for k_q
    p_fit_spring_individual = np.zeros(len(ucdf_Y))
    count = 0
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              hasMagnets=bool(unitData.magnets), m=unitData.m,
                                              **bilayerDict)

        # Use subset of load-disp data before collision begins to fit k_q
        maxStrain = 0.15#0.225
        maxIndex = np.where(unitData.strain < maxStrain)[0][-1]
        #p_given = [unitModel.total_angle, unitData.d/2]
        q0 = unitModel.total_angle
        L = unitData.d/2
        m = unitData.m
        p_lim = [0,0]
        p_guess = 1e-3 # k_q guess

        p_fit = opt.least_squares(model.residue_force_displacement_spring_from_all, p_guess,
                                  args=(-unitData.load[:maxIndex], unitData.disp[:maxIndex], q0, L, m, p_lim))
        
        p_fit_spring_individual[count] = p_fit.x[0]
        count += 1

    # Find best-fit value for k_sq
    p_fit_all = fit_constant_ksq(ucdf_Y, p_fit_spring_individual)
    k_sq_fit = p_fit_all.x[0]
    
    return k_sq_fit

def analyze_rconst_moment(ucdf, k_sq_fit, bilayerDict):
    ''' Analysis of just with-magnet case: find best-fit magnetic moment m '''
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  

    p_guess = 0.20
    
    # Find individual best-fit values for k_q
    p_fit_moment_individual = np.zeros(len(ucdf_Y))
    count = 0
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              k_sq=k_sq_fit,
                                              hasMagnets=0, p_lim=[0,0],
                                              **bilayerDict) # Using this only for k_eq

        # Use only data nera zero crossings!
        nearZeroIndices = find_near_zero_indices(unitData.load, window=100)
        maxStrain = 0.175
        maxIndex = np.where(unitData.strain > maxStrain)[0][0]
        nearZeroIndices = [i for i in nearZeroIndices if i < maxIndex]
        
        plt.figure()
        plt.plot(unitData.strain, unitData.load, 'k')
        plt.plot(unitData.strain[nearZeroIndices], unitData.load[nearZeroIndices], 'ro', linewidth=0)

        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2]
        p_fit_moment_individual[count] = model.approximate_only_magnet(unitData.disp[:maxIndex],
                                                                       -unitData.load[:maxIndex],
                                                                       p_guess, p_given)
        count += 1

    # Find best-fit value for magnetic moment
    p_fit_all = fit_constant_moment(p_fit_moment_individual, p_guess)
    moment_fit = p_fit_all.x[0]
    
    return moment_fit

def analyze_rconst_collision(ucdf, k_sq_fit, moment_fit, bilayerDict):
    ''' Analysis of just with-magnet case: find best-fit collision parameters
        (A, B) for exponential fit'''
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
    
    p_fit_all = fit_constant_collision(ucdf_Y, [k_sq_fit, moment_fit, bilayerDict])
    p_lim_fit = p_fit_all.x
    
    return p_lim_fit

def analyze_rconst_moment_and_collision_test(ucdf, k_sq_fit, bilayerDict):
    ''' Analysis of just with-magnet case: find best-fit collision parameters
        (A, B) for exponential fit'''
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
    
    p_all = fit_constant_magnet_and_collision_test(ucdf_Y, [k_sq_fit, bilayerDict])
    p_fit = p_all.x
    
    return p_fit

def analyze_rconst_moment_and_collision(ucdf, k_sq_fit, bilayerDict, p_guess=[0.18,np.radians(44),1],
                                        limFlag='pcw', weightMagnitude=1):
    ''' Using curve fitting weighted to best match roots of load-displacement,
        find best-fit magnetic moment and collision parameters '''
    
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]     
    p_fit = fit_constant_magnet_and_collision(ucdf_Y, k_sq_fit, p_guess=p_guess,
                                              limFlag=limFlag, weightMagnitude=1,
                                              bilayerDict=bilayerDict)
    moment_fit = p_fit[0]
    p_lim_fit = p_fit[1], p_fit[2]
    
    return moment_fit, p_lim_fit

def find_zero_crossings(y_array, startSign=1):
    ''' Finds locations and signs of zero crossings present '''
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

def find_near_zero_indices(y_array, window=25):
    zeroIndices, _ = find_zero_crossings(y_array)
    nearZeroIndices = []
    for i in zeroIndices:
        imin = max(0, i-window)
        imax = min(len(y_array)-1, i+window)
        nearZeroIndices.extend(list(range(imin, imax)))
    return nearZeroIndices

#%% Fitting functions
def fit_constant_ksq(ucdf, p_fit_individual, k_sq_guess=1e-3, p_lim=[0,0], limFlag='exp'):
    ''' Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit '''
       
    p_fit = opt.least_squares(residue_constant_ksq, k_sq_guess,
                              args=(p_fit_individual, ucdf),
                              kwargs={'p_lim':p_lim, 'limFlag':limFlag})
    
    return p_fit

def residue_constant_ksq(p, ksq_fit_individual, ucdf, p_lim=[0,0], limFlag='exp'):
    ''' Cost function: minimize difference between equivalent spring constant
        for hinge in series with constant k_sq and the best-fit torsional
        spring constant, for all force-displacement relations '''

    k_sq = p
    ksq_fit_all = np.zeros(len(ucdf))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, s=unitData.s, 
                                              k_sq=k_sq, p_lim=p_lim, m=unitData.m,
                                              hasMagnets=bool(unitData.magnets))
        ksq_fit_all[count] = unitModel.k_eq
        count += 1

    residual = ksq_fit_individual - ksq_fit_all

    return residual


def fit_constant_moment(p_fit_individual, p_guess):
    ''' Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit '''
    
   
    p_fit = opt.least_squares(residue_constant_moment, p_guess,
                              args=(p_fit_individual,))
    
    return p_fit

def residue_constant_moment(p, moment_fit_individual):
    ''' Cost function: minimize difference between constant magnetic moment and
        best-fit magnetic moment, for all force-displacement relations '''

    return moment_fit_individual - p


def fit_constant_collision(ucdf, p_given, p_lim_guess=[3e-10, 18], limFlag='exp'):
    ''' Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit '''
       
    p_fit = opt.least_squares(residue_constant_collision, p_lim_guess,
                              args=(ucdf, p_given),
                              kwargs={'limFlag':limFlag})
    
    return p_fit

def residue_constant_collision(p, ucdf, params_given, limFlag='exp'):
    ''' Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations '''

    k_sq_fit, m_fit, bilayerDict = params_given

    # Find individual best-fit values for k_q
    nSamples = len(ucdf)
    p_lim_fit_individual = np.zeros((nSamples,2))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=k_sq_fit, p_lim=p, limFlag=limFlag,
                                              m = m_fit,
                                              hasMagnets=bool(unitData.magnets),
                                              analysisFlag=False,
                                              **bilayerDict)

        # Use only data nera zero crossings!
        nearZeroIndices = find_near_zero_indices(unitData.load, window=30)
        
        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2, m_fit]
        p_guess = p
        p_lim_fit_individual[count,:] = model.approximate_only_collision(unitData.disp[nearZeroIndices],
                                                                       -unitData.load[nearZeroIndices],
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


def fit_constant_magnet_and_collision_test(ucdf, p_given, p_lim_guess=[0.18, 3e-10, 18], limFlag='exp'):
    ''' Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit '''
       
    p_fit = opt.least_squares(residue_constant_magnet_and_collision_test, p_lim_guess,
                              args=(ucdf, p_given),
                              kwargs={'limFlag':limFlag})
    
    return p_fit

def residue_constant_magnet_and_collision_test(p, ucdf, params_given, limFlag='exp'):
    ''' Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations '''

    k_sq_fit, bilayerDict = params_given

    # Find individual best-fit values for k_q
    nSamples = len(ucdf)
    p_lim_fit_individual = np.zeros((nSamples,3))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=k_sq_fit, p_lim=p[1:], limFlag=limFlag,
                                              m = p[0],
                                              hasMagnets=bool(unitData.magnets),
                                              analysisFlag=False,
                                              **bilayerDict)

        # Use only data nera zero crossings!
        nearZeroIndices = find_near_zero_indices(unitData.load)
        
        plt.figure()
        plt.plot(unitData.strain, unitData.load, 'k')
        plt.plot(unitData.strain[nearZeroIndices], unitData.load[nearZeroIndices], 'ro', linewidth=0)
        
        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2]
        p_guess = p
        p_lim_fit_individual[count,:] = model.approximate_magnet_collision(unitData.disp[nearZeroIndices],
                                                                           -unitData.load[nearZeroIndices],
                                                                           p_guess, p_given)
        count += 1

    # Reshape and subtract
    p_lim_fit_individual = p_lim_fit_individual.flatten()
    p_lim_fit_individual = p_lim_fit_individual.reshape((-1,1))

    p_lim_fit_all = np.zeros((nSamples,3))
    p_lim_fit_all[:,0] = p[0]
    p_lim_fit_all[:,1] = p[1]
    p_lim_fit_all[:,2] = p[2]
    p_lim_fit_all = p_lim_fit_all.flatten()
    p_lim_fit_all = p_lim_fit_all.reshape((-1,1))

    residual = p_lim_fit_individual - p_lim_fit_all
    residual = residual.flatten()

    return residual


def residue_constant_collision_weighted(p, ucdf, params_given, limFlag='exp'):
    ''' Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations '''

    k_sq_fit, m_fit = params_given

    # Find individual best-fit values for k_q
    nSamples = len(ucdf)
    p_lim_fit_individual = np.zeros((nSamples,2))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, s=unitData.s, 
                                              k_sq=k_sq_fit, p_lim=p, limFlag=limFlag,
                                              m = m_fit,
                                              hasMagnets=bool(unitData.magnets),
                                              analysisFlag=False) # Using only for k_eq

        # Use subset of load-disp data before collision begins to fit moment
        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2, m_fit]
        p_guess = p
        weights, rootIndices, rootSigns = add_weights_at_roots(unitData, weightMagnitude=0.5, windowSize=5)
        p_lim_fit_individual[count,:] = model.approximate_only_collision_weighted(unitData.disp,
                                                                                  -unitData.load,
                                                                                  p_guess, p_given,
                                                                                  weights)
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


def add_weights_at_roots(unitData, weightMagnitude=1, windowSize=1):
    ''' make vector of weights, all 1 except for weightMagnitude < 1 around some
        window windowSize of '''

    # Use load and displacement corresponding just to cropped portion
    if unitData.cropFlag:
        load = unitData.load
        disp = unitData.disp
    else:
        load = unitData.load_crop
        disp = unitData.disp_crop         
    
    # Find zero crossings
    rootIndices, rootSigns = find_zero_crossings(load, startSign=1)

    weights = np.ones_like(disp)
    for index in rootIndices:
        start = max(index-windowSize, 0)
        stop = min(index+windowSize, len(disp))
        weights[start:stop] = weightMagnitude

    return weights, rootIndices, rootSigns

def fit_constant_magnet_and_collision(ucdf, p_given, 
                                      p_guess=[0.18, 3e-10, 18],
                                      limFlag='exp', weightMagnitude=1, 
                                      bilayerDict={}):
    ''' Takes dataframe containing all non-magnet samples and returns the 
        square stiffness and collision parameters which minimizes the
        least-squares error for all fit '''

    # Find roots and weight 
    all_weights = []
    all_indices = []
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        shift = 0
        load = unitData.load_full[unitData.zeroIndex-shift:]
        disp = unitData.disp_full[unitData.zeroIndex-shift:]
        strain = unitData.strain_full[unitData.zeroIndex-shift:]
        rootIndices, _ = find_zero_crossings(load, startSign=1)

        weights = np.ones_like(disp)
        window = 5
        for index in rootIndices:
            start = max(index-window, 0)
            stop = min(index+window, len(strain))
            weights[start:stop] = weightMagnitude

        weights = weights[shift:]

        all_weights.append(weights)
        all_indices.append(rootIndices)

    # Do the fitting!
    p_fit = opt.least_squares(residue_constant_magnet_and_collision, p_guess,
                              args=(ucdf, p_given, all_weights),
                              kwargs={'limFlag':limFlag, 'bilayerDict':bilayerDict})
    p_fit = p_fit.x
    m_fit = p_fit[0]
    p_lim_fit = [p_fit[1], p_fit[2]]
    
    k_sq_fit = p_given

    # Plot results
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        plt.figure(dpi=200)
        plt.xlabel("Strain, $\delta/d$")
        plt.ylabel("Load (N)")
        plt.title(unitData.label)
        disp_plt = unitData.strain
        
        plt.plot(disp_plt, unitData.load, 'k', label="experiment")
        for i in all_indices[count]:
            plt.plot(disp_plt[i], 0, 'gx')
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=k_sq_fit, m=m_fit, p_lim=p_lim_fit,
                                              limFlag=limFlag, hasMagnets=1, analysisFlag=False,
                                              **bilayerDict) # Using this only for k_eq
        plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), 'r', label="model")
        modelIndices, _ = find_zero_crossings(unitModel.model_load_disp(unitData.disp), startSign=1)
        #print(modelIndices)
        for j in modelIndices:
            #print(disp_plt[int(j)])
            plt.plot(disp_plt[int(j)], 0, 'bo')
        count += 1
    
    return p_fit

def residue_constant_magnet_and_collision(p, ucdf, params_given, all_weights,
                                          limFlag='exp', bilayerDict={}):
    ''' Cost function: minimize difference between constant collision parameters
        and best-fit collision parameters, for all force-displacement relations '''

    k_sq_fit = params_given
    m_guess = p[0]
    p_lim_guess = [p[1],p[2]]

    # Find individual best-fit values for k_q
    nSamples = len(ucdf)
    p_lim_fit_individual = np.zeros((nSamples,3))
    count = 0
    for index, row in ucdf.iterrows():
        unitData = row["data"]
        #print(unitData.label)
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=k_sq_fit, m=m_guess, 
                                              p_lim=p_lim_guess, limFlag=limFlag,
                                              hasMagnets=bool(unitData.magnets),
                                              analysisFlag=False,
                                              **bilayerDict) # Using only for k_eq

        # Use subset of load-disp data before collision begins to fit moment
        p_given = [unitModel.total_angle, unitModel.k_eq, unitData.d/2, limFlag]
        p_guess = p
        weights = all_weights[count]
        p_fit = model.fit_magnet_and_collision_weighted(unitData.disp,
                                                        -unitData.load,
                                                        p_guess, p_given, weights)
        p_lim_fit_individual[count,:] = p_fit
        
        count += 1

    # Reshape and subtract
    p_lim_fit_individual = p_lim_fit_individual.flatten()
    p_lim_fit_individual = p_lim_fit_individual.reshape((-1,1))

    p_lim_fit_all = np.zeros((nSamples,3))
    p_lim_fit_all[:,0] = p[0]
    p_lim_fit_all[:,1] = p[1]
    p_lim_fit_all[:,2] = p[2]
    p_lim_fit_all = p_lim_fit_all.flatten()
    p_lim_fit_all = p_lim_fit_all.reshape((-1,1))

    residual = p_lim_fit_individual - p_lim_fit_all
    residual = residual.flatten()

    return residual


def plot_final_rconst_fit(ucdf, k_sq_fit, m_fit, p_lim_fit, bilayerDict, limFlag='exp'):
    ''' Plot force-displacement curves for experiment data and best fit for all data,
        including markers at roots'''
    
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
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
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


#%% Main importing function
if __name__ == "__main__":   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/unitCell_properties")
    tmpdir = os.path.join(cwd,"tmp")
    
    sourcedir = os.path.join(rawdir,"unitCell_repeatability")
    #analyze_repeatability_data(sourcedir)
    
    sourcedir = os.path.join(rawdir,"unitCell_tension_rconst")
    #analyze_rconst_data(sourcedir)