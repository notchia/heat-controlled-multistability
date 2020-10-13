# -*- coding: utf-8 -*-
'''
Fit experimental data for LCE contraction and elastic modulus. From main data
analysis script, call the following functions:
    plist, fcn = fit_LCE_contraction(fullpath)
    plist, fcn = fit_LCE_modulus(fullpath)
with
    fullpath:   str corresponding to raw data file location
    plist:      list of fitting parameters
    fcn:        function for which fitting parameters were found 
    
@author: Lucia Korpas
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import os
import re
import math

from modules import tensiletest


def model_elastic_modulus(T, dE, T0, E0):
    ''' Young's modulus as exponential + constant '''
    E = np.exp(dE*(T - T0)) + E0
    return E

def model_strain(x, ds, T0, s0):
    ''' Strain modeled as exponential + constant '''
    strain = -np.exp(ds*(x - T0)) + s0
    return strain


# Import and analyze experiments relating to this model -----------------------
def fit_LCE_strain(filepath, saveFlag=False, figdir=''):
    ''' Fit LCE contraction strain data (lengths at different temperatures, 
        measured from photos in ImageJ and tabulated) to an empirical model.
        - Input experimental data path with mm2px conversion in the name
        - Returns the fitting parameters list and the model function '''
    
    # Import data, convert to lengths/strains, and take mean and stdev
    conversion = os.path.splitext(os.path.split(filepath)[-1])[0].split('_')[-1]
    px2mm = 1/float(conversion.strip('mm2px'))
    arr = np.genfromtxt(filepath,dtype=float,delimiter=',')
    data = np.transpose(arr)
    T = data[1:,0]
    L = px2mm*data[1:,1:]
    strain = (L - L[0,:])/L[0,:]
    avg_strain = np.mean(strain, axis=1)
    std_strain = np.std(strain, axis=1)

    plt.figure(dpi=200)
    plt.errorbar(T, avg_strain, yerr=std_strain, linewidth=0.0, label="experiment",
                 marker='o', color='r', capsize=4, elinewidth=2)  
    
    # Fit averaged data to model
    maxTemp = 100 # Exponential fit is no good above this
    maxIndex = np.argwhere(T>maxTemp)[0][0]
    p_guess = [1, 0.001, 0] # dy, T0, y0
    params = opt.curve_fit(model_strain, T[:maxIndex], avg_strain[:maxIndex], p0=p_guess)[0]
    T_model = np.arange(T[0], T[maxIndex-1]+1, 1)
    L_model = model_strain(T_model, *params)

    modelstr = "LCE fit:\t strain = -exp({0:.6f}(T-{1:.2f})) + {2:.4e}".format(*params)
    print(modelstr)

    plt.xlabel("Temperature ($^{\circ}$C)")
    plt.ylabel("Strain $\epsilon$ (mm/mm)")
    plt.plot(T_model, L_model, linestyle='-', color='k', linewidth=2, zorder=3,
             label="exponential fit up to {0}$^\circ$C".format(maxTemp))
    plt.legend()
    if saveFlag:
        plt.savefig(os.path.join(figdir,'LCE_strain.svg'))
        plt.savefig(os.path.join(figdir,'LCE_strain.png'),dpi=200)

    return params, model_strain

def fit_LCE_contraction_200316(filename, saveFlag=False, figdir=''):
    ''' Fit LCE contraction strain data (lengths at different temperatures, 
        measured from photos in ImageJ and tabulated) to an empirical model.
        In this version, both strips and squares of the material were tested.
        - Takes experimental data path
        - Returns the fitting parameters list and the model function '''
    # Import data and convert to strain
    data = np.genfromtxt(filename, delimiter=',',skip_header=1)
    T = data[:,0]
    L_strip = data[:,1:4]
    L_square = data[:,4:7]
    strain_strip = (L_strip-L_strip[0,:])/L_strip[0,:]
    strain_square = (L_square-L_square[0,:])/L_square[0,:]
    strain_strip_avg = np.mean(strain_strip, axis=1)
    strain_square_avg = np.mean(strain_square, axis=1)
    strain_strip_std = np.std(strain_strip, axis=1)
    strain_square_std = np.std(strain_square, axis=1)
    errorT = 2.0 # [degrees C]
    
    # Fit model to data
    p_guess = [1, 0.001] # A, B
    params = opt.curve_fit(model_strain, T, strain_strip_avg, p0=p_guess)[0]
    
    # Plot strain-temperature data and model with error bars and fit
    plt.figure(dpi=200)
    plt.errorbar(T, strain_strip_avg, yerr=strain_strip_std, xerr=errorT,
                 fmt='', marker='o', color='r', capsize=4, elinewidth=2, label="strip")
    plt.errorbar(T, strain_square_avg, yerr=strain_square_std, xerr=errorT,
                 fmt='', marker='o', color='g', capsize=4, elinewidth=2, label="square")
    plt.plot(T, model_strain(T, params[0], params[1]),
             linestyle='-', color='k', linewidth=1, zorder=3, label="exponential fit to strip")
    plt.xlabel("Temperature ($^\circ$C)")
    plt.ylabel("Strain along print direction")
    plt.legend()
    
    modelstr = "y = -exp({0:0.6f}(x - {1:0.2f}))".format(params[0], params[1])
    print(modelstr)

    if saveFlag:
        plt.savefig(os.path.join(figdir, "LCE_strain.svg"), transparent=True)
        plt.savefig(os.path.join(figdir, "LCE_strain.png"), dpi=200)
    
    return params, model_strain
"""
def fit_LCE_modulus(filename, saveFlag=False, figdir=''):
    ''' Fit LCE elastic modulus data (from DMA) to a mostly-empirical model.
        - Takes experimental data path
        - Returns the fitting parameters list and the model function
         NOT CURRENTLY USED'''
    # Import data and convert to strain
    data = np.genfromtxt(filename, skip_header=2, delimiter='\t')
    E_s = data[:,1]
    E_l = data[:,2]
    #E_star = data[:,5] #4 for earlier data
    T = data[:,4] #7 for earlier data

    p_guess = [-0.07, 62, 0.7] # dy, T0, y0
    params = opt.curve_fit(model_elastic_modulus, T, E_s, p0=p_guess)[0]
  
    # Semilog plot shear, loss, total, and Young's moduli and fit 
    fig = plt.figure('LCE_modulus', dpi=300)
    #plt.plot(T, 1e-6*E_star, '.g', label="E*")
    plt.plot(T, 1e-6*E_s, '.r', label="E'")
    plt.plot(T, 1e-6*E_l, '.b', label="E''")
    plt.plot(T, 1e-6*model_elastic_modulus(T, *params),
             'k', linewidth=1, linestyle='-',label="Exponential fit to E'")
    plt.xlabel("Temperature ($^\circ$C)")
    plt.ylabel("Modulus (MPa)")
    ax = fig.gca()
    ax.set_yscale('log')
    plt.ylim([.001, 100])
    plt.legend(loc="lower left")

    modelstr = "LCE fit:\t modulus = exp({0:.6f}(T-{1:.2f})) + {2:.0f}".format(params[0], params[1], params[2])    
    print(modelstr)
    
    if saveFlag:
        plt.savefig(os.path.join(figdir, 'LCE_modulus.svg'), transparent=True)
        plt.savefig(os.path.join(figdir, 'LCE_modulus.png'), dpi=200)

    return params, model_elastic_modulus
"""
def fit_LCE_modulus_avg(sourcedir, saveFlag=False, figdir='', verboseFlag=False):
    ''' Fit LCE storage and loss modulus data (from DMA) to an empirical model.
        - Takes experimental data source directory
        - Smooths and interpolates E' and E'' data for each file in directory
        - Finds and plots mean and std for E' and E''
        - Finds best-fit parameters for exponential model for E''
        - Returns the fitting parameters list and the model function '''
    
    filelist = os.listdir(sourcedir)
    T_min = 20
    T_max = 150
    interpEs = []
    interpEl = []
    T_RT = []
    E_s_RT = []
    E_l_RT = []
    for filename in filelist:
        # Import data
        data = np.genfromtxt(os.path.join(sourcedir, filename), skip_header=2, delimiter='\t')
        if data.ndim == 2: # temperature ramp
            E_s = abs(data[:,1])
            E_l = abs(data[:,2])
            T = data[:,4]
            # Smooth and crop data
            windowSize = 2
            cropIndex = math.ceil(windowSize/2)
            E_s = tensiletest.running_mean_centered(E_s, windowSize)[cropIndex:-cropIndex]
            E_l = tensiletest.running_mean_centered(E_l, windowSize)[cropIndex:-cropIndex]
            T = T[cropIndex:-cropIndex]
        else: # single point
            E_s = np.array([abs(data[1])])
            E_l = np.array([abs(data[2])])
            T = np.array([data[4]])

        if len(T) != 1:
            if min(T) > T_min:
                T_min = min(T)
            if max(T) < T_max:
                T_max = max(T)
       
        # Semilog plot shear, loss, total, and Young's moduli and fit 
        if verboseFlag:
            if len(T) != 1:
                p_guess = [-0.07, 62, 0.7] # dy, T0, y0
                params = opt.curve_fit(model_elastic_modulus, T, E_s, p0=p_guess)[0]

            fig = plt.figure(dpi=200)
            plt.plot(T, 1e-6*E_s, '.r', label="E'")
            plt.plot(T, 1e-6*E_l, '.b', label="E''")
            if len(T) != 1:
                plt.plot(T, 1e-6*model_elastic_modulus(T, *params),
                         'r', linewidth=1, linestyle='-',label="Exponential fit to E'")
                modelstr = "LCE fit:\t modulus = exp({0:.6f}(T-{1:.2f})) + {2:.0f}".format(params[0], params[1], params[2])    
                print(modelstr)
        
            plt.xlabel("Temperature ($^\circ$C)")
            plt.ylabel("Modulus (MPa)")
            ax = fig.gca()
            ax.set_yscale('log')
            plt.ylim([.001, 100])
            plt.legend(loc="lower left")
    
            
        # Add interpolation functions to list
        if len(T) != 1:
            interpEs.append(interp.interp1d(T, E_s))
            interpEl.append(interp.interp1d(T, E_l))
        else:
            T_RT.append(T[0])
            E_s_RT.append(E_s[0])
            E_l_RT.append(E_l[0])    

    # Find average and standard deviation for storage and loss moduli based on
    # interpolation of smoothed data
    T_avg = np.arange(T_min, min(100, T_max)+1, 1.0)
    nFits = len(interpEs)
    Es_vals = np.zeros((len(T_avg), nFits))
    El_vals = np.zeros((len(T_avg), nFits))
    for i in range(nFits):
        f1 = interpEs[i]
        f2 = interpEl[i]
        Es_vals[:,i] = f1(T_avg)
        El_vals[:,i] = f2(T_avg)
    Es_avg = np.mean(Es_vals, axis=1)
    Es_std = np.std(Es_vals, axis=1)
    El_avg = np.mean(El_vals, axis=1)
    El_std = np.std(El_vals, axis=1)
    
    T_RT_avg = np.mean(T_RT)
    T_RT_std = np.std(T_RT)
    Es_RT_avg = np.mean(E_s_RT)
    Es_RT_std = np.std(E_s_RT)
    El_RT_avg = np.mean(E_l_RT)
    El_RT_std = np.std(E_l_RT)
    
    # Fit to averaged data
    p_guess = [-0.07, 62, 0.7] # dy, T0, y0
    params = opt.curve_fit(model_elastic_modulus, T_avg, Es_avg, p0=p_guess)[0]

    # Plot storage and loss modulus, with error band, as well as fit to E_s
    fig = plt.figure('LCE_modulus', dpi=300)
    plt.plot(T_avg, 1e-6*Es_avg, 'r', label="E'")
    plt.fill_between(T_avg, 1e-6*(Es_avg-Es_std), 1e-6*(Es_avg+Es_std), color='r', alpha=0.2)
    plt.plot(T_avg, 1e-6*El_avg, 'b', label="E''")
    plt.fill_between(T_avg, 1e-6*(El_avg-El_std), 1e-6*(El_avg+El_std), color='b', alpha=0.2)
    plt.plot(T_avg, 1e-6*model_elastic_modulus(T_avg, *params),
             'k', linewidth=1, linestyle='-',label="Exponential fit to E'")
    plt.errorbar(T_RT_avg, 1e-6*Es_RT_avg, xerr=T_RT_std, yerr=1e-6*Es_RT_std, capsize=4, fmt='.', color='r', label="E' (RT)")
    plt.errorbar(T_RT_avg, 1e-6*El_RT_avg, xerr=T_RT_std, yerr=1e-6*El_RT_std, capsize=4, fmt='.', color='b', label="E'' (RT)")
    plt.xlabel("Temperature ($^\circ$C)")
    plt.ylabel("Modulus (MPa)")
    ax = fig.gca()
    ax.set_yscale('log')
    plt.ylim([.01, 10])
    plt.legend(loc="lower left")
    
    if saveFlag:
        plt.savefig(os.path.join(figdir, 'LCE_modulus_avg.svg'), transparent=True)
        plt.savefig(os.path.join(figdir, 'LCE_modulus_avg.png'), dpi=200)

    return params, model_elastic_modulus

def fit_LCE_tensile(sourcedir, verboseFlag=False, saveFlag=True, figdir=''):
    filelist = os.listdir(sourcedir)
    T_vals = []
    Y_vals = []
    for filename in filelist:
        if os.path.splitext(filename)[-1] == '.csv':
            filepath = os.path.join(sourcedir, filename)
            speed = float(re.search("_(\d+\.\d+)mmps", filename).group(1)) #[mm/s]
            width = float(re.search("_w(\d+\.\d+)", filename).group(1)) #[mm]
            thickness = float(re.search("_t(\d+\.\d+)", filename).group(1)) #[mm]
            gaugeLength = float(re.search("_L(\d+\.\d+)", filename).group(1)) #[mm]
            temperature = float(re.search("_(\d+\.\d+)C", filename).group(1)) #[mm]
            strain, stress = tensiletest.import_tensile_experiment(filepath, speed=speed, area=width*thickness, gaugeLength=gaugeLength)
            strain, stress = tensiletest.filter_raw_data(strain, stress, 50)
            Young, strain, stress = tensiletest.fit_Young_from_tensile(strain, stress, maxVal=0.04)
            
            # Plot
            if verboseFlag:
                plt.figure(dpi=200)
                plt.title(filename)
                plt.plot(strain, stress*1e-6, 'k', label='experiment')
                plt.plot(strain, Young*strain*1e-6, 'r', label='fit')
                plt.xlabel('Strain (mm/mm)')
                plt.ylabel('Stress (MPa)')
                plt.tight_layout()
            
            T_vals.append(temperature)
            Y_vals.append(Young)
    
    # Add these values to LCE_modulus plot (created by fit_LCE_modulus)
    plt.figure('LCE_modulus')
    plt.plot(T_vals, [1e-6*Y for Y in Y_vals], '*k', markersize='12', label='Young\'s modulus (tensile)')
    plt.legend()
    if saveFlag:
        plt.savefig(os.path.join(figdir, 'LCE_modulus.svg'), transparent=True)
        plt.savefig(os.path.join(figdir, 'LCE_modulus.png'), dpi=200)

    return



if __name__ == "__main__":   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/LCE_properties")
    tmpdir = os.path.join(cwd,"tmp")
    