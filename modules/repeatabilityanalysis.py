"""
Analyze repeatability data (three samples with nominally the same parameters)
    
@author: Lucia Korpas
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
def analyze_repeatability_data(sourcedir, bilayerDict, setStartLoadToZero=False,
                               k_sq=0, m=0, p_lim=[],
                               saveFlag=False, figdir=''):
    """How repeatably can we manufacture and test unit cells with
    nominally identical parameters? """
    
    # Import all sample test and corresponding information from the directory
    ucdf = experiment.import_all_unit_cells(sourcedir, setStartLoadToZero=setStartLoadToZero,
                                            bilayerDict=bilayerDict)   

    # Generate DataFrame for averaged no-magnet data and for model based on
    # r-const fit parameters
    ucdf_avg_N = pd.DataFrame()
    ucdf_model = []
    nGroups = 3
    print("Analysis of averaged no-magnet force-displacement data:")
    for T_group in range(nGroups):
        subframe = ucdf.loc[(ucdf["T_group"] == T_group) & (ucdf["magnets"] == 0)]
        unitCell_avg = experiment.compute_unitCell_mean_std(subframe)
        row = unitCell_avg.get_Series()
        ucdf_avg_N = ucdf_avg_N.append(row,ignore_index=True)
        unitData = row["data"]
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d,
                                              k_sq=k_sq, m=m, p_lim=p_lim,
                                              loadFlag=True, hasMagnets=bool(unitData.magnets),
                                              **bilayerDict)
        ucdf_model.append(unitModel.model_load_disp(unitData.disp))
    
    # Show raw data, compare between model and averaged data
    experiment.plot_magnet_and_T_comparison(ucdf[ucdf["magnets"] == 0])
    experiment.plot_magnet_and_T_comparison(ucdf_avg_N, modellist=ucdf_model,
                                            saveFlag=saveFlag, savedir=figdir,
                                            stdFlag=True, title='unitCell_repeatability_force_N')

   # Find best-fit model parameters
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
                                              k_sq=k_sq, m=m, p_lim=p_lim,
                                              loadFlag=True, hasMagnets=bool(unitData.magnets),
                                              **bilayerDict)
        plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), 'r', label="model")
        plt.fill_between(disp_plt, unitData.load-unitData.std, unitData.load+unitData.std, color='k', alpha=0.2)
        plt.legend()  
        
        p_given_exp = [unitModel.total_angle, unitModel.d/2, 'exp']
        p_guess_exp = [0.001, 5e-20, 50]
        params = model.approximate_spring(unitData.disp, -(unitData.load), p_guess_exp, p_given_exp)
        print(f"For (h, r, T) = ({1e3*unitData.h}, {unitData.r}, {unitData.T:.0f}), best-fit k = {1e3*params[0]:.2f} mNm, p_lim = [{params[1]:.2e}, {params[2]:.1f}]")

    # Initialize plot info
    colors = ['r','g','b']
    TLabels = ['RT', 'T~45$^\circ$', 'T~75$^\circ$']
    
    fig = plt.figure('unitCell_repeatability_energy', dpi=200)
    plt.xlabel("Angle $\theta$ ($^\circ$)")
    plt.ylabel("Energy (J)")
    
    fig = plt.figure('unitCell_repeatability_force_Y', dpi=200)
    plt.xlabel("Strain, $\delta/d$")
    plt.ylabel("Load (N)")
    plt.axhline(color='k',linewidth=1)

    # Analyze sample tests which include magnets
    print("Plotting with-magnet force-displacement curves with provided model parameters...")
    ucdf_Y = ucdf.loc[ucdf["magnets"] == 1]  
    for index, row in ucdf_Y.iterrows():
        unitData = row["data"]
        
        # Create the model coresponding to this row of sample test data
        iplt = unitData.T_group
        disp_plt = unitData.strain        
        unitModel = metamat.MetamaterialModel(unitData.h, unitData.r, ANGLE_NEAR_ZERO,
                                              T=unitData.T, d=unitData.d, 
                                              k_sq=k_sq, m=m, p_lim=p_lim,
                                              loadFlag=True, hasMagnets=bool(unitData.magnets),
                                              **bilayerDict)
        
        # Plot raw force-displacement data
        fig = plt.figure('unitCell_repeatability_force_Y')        
        plt.plot(disp_plt, unitData.load, colors[iplt], label="experiment, {0}".format(TLabels[iplt]))
        plt.plot(disp_plt, unitModel.model_load_disp(unitData.disp), colors[iplt]+'--',
                 label="model, {0}".format(TLabels[iplt]))
        
        # Plot modeled energy for the above
        fig = plt.figure('unitCell_repeatability_energy')
        U_total, q_range = unitModel.model_energy()
        zeroIndex = np.where(q_range > 0)[0][0]
        U_total = U_total - U_total[zeroIndex]
        plt.plot(np.degrees(q_range), U_total, colors[iplt], label=TLabels[iplt])

        # Find and plot local minima
        minima = signal.argrelmin(U_total)[0]
        maxima = signal.argrelmax(U_total)[0]
        minU = U_total[minima]
        maxU = U_total[maxima]
        plt.plot(np.degrees(q_range[minima]), minU, colors[iplt]+'v')
        plt.plot(np.degrees(q_range[maxima]), maxU, colors[iplt]+'^')
        
        # Resize y-axis
        minU.sort()
        maxU.sort()
        if maxU.size > 0:
            plt.ylim(minU[0]-0.0001, 0.003)
            
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


#%% Main
if __name__ == "__main__":   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/unitCell_properties")
    tmpdir = os.path.join(cwd,"tmp")
    
    sourcedir = os.path.join(rawdir,"unitCell_repeatability")
    #analyze_repeatability_data(sourcedir)
    