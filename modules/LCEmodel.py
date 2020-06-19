# -*- coding: utf-8 -*-
"""
Fit experimental data for LCE contraction and elastic modulus

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import pltformat  # module: no funcions, just standardized plot formatting 


""" Calculate LCE elastic modulus (from DMA data fit)"""  
def calculate_LCEmodulus(self, T):
    E = (10**6)*np.exp(-0.0757*(T - 62.94)) + 0.7093
    return E

""" Calculate LCE elastic modulus (from experimental data fit)"""  
def calculate_LCEstrain(self, T):
    strain = -np.exp(0.0665*(T - 151.48))
    return strain


''' Strain model as exponential. Need to update! '''
def model_strain(x, A, B):
    return -np.exp(A*(x-B))


''' Young's modulus as exponential + constant above certain temp. Need to update! '''
def model_elastic_modulus(T, T0, y0, dy):
    Y = np.exp(dy*(T - T0)) + np.full_like(T, y0)
    return Y


''' Fit LCE contraction data to model '''
def fit_LCE_contraction(filename):
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
    errorT = 2.0
    
    params = opt.curve_fit(model_strain, T, strain_strip_avg, p0=[1, 0.001])[0]
    
    # Plot strain-temperature data with error bars and fit
    fig = plt.figure()
    plt.errorbar(T, strain_strip_avg, yerr=strain_strip_std, xerr=errorT,
                 fmt='', marker='o', color='r', capsize=4, elinewidth=2, label="strip")
    plt.errorbar(T, strain_square_avg, yerr=strain_square_std, xerr=errorT,
                 fmt='', marker='o', color='g', capsize=4, elinewidth=2, label="square")
    plt.plot(T, model_strain(T, params[0], params[1]),
             linestyle='-', color='k', linewidth=2, zorder=3, label="exponential fit to strip")
    plt.xlabel("Temperature (deg C)")
    plt.ylabel("Strain along print direction")
    plt.legend()
    
    ax = fig.gca()
    modelstr = "y = -exp({0:0.4f}(x - {1:0.2f}))".format(params[0], params[1])
    ax.text(30, -0.05, modelstr, fontsize=16)
    
    return params, model_strain


''' Fit LCE elastic modulus data to model '''
def fit_LCE_elasticity(filename):
    data = np.genfromtxt(filename, skip_header=2, delimiter='\t')
    E_s = data[:,1]
    E_l = data[:,2]
    E_star = data[:,4]
    T = data[:,7]
    
    # Assumption of incompressibility for strain along director orientation
    # taken from literature; Linear elastic relation between shear and
    # elastic moduli is assumed for simplicity
    nu = 0.5 
    Young = E_s*(2*(1 + nu))

    params = opt.curve_fit(model_elastic_modulus, T, Young, p0=[30, -1, 0.001])[0]
    modelstr = "y = exp({2:0.4f}(x - {0:0.2f})) + {1:0.4f}".format(params[0], params[1], params[2])
    
    # Semilog plot shear, loss, total, and Young's moduli and fit 
    fig = plt.figure()
    plt.plot(T, (10**-6)*E_star, '.k', label="E*")
    plt.plot(T, (10**-6)*E_s, '.r', label="E'")
    plt.plot(T, (10**-6)*E_l, '.b', label="E''")
    plt.plot(T, (10**-6)*Young, '.g', label="Elastic modulus")
    plt.plot(T, model_elastic_modulus(T, params[0], params[1], params[2]), 'g', linewidth=2, linestyle='-',label="exponential fit")
    plt.xlabel("Temperature (deg C)")
    plt.ylabel("Modulus (MPa)")
    ax = fig.gca()
    ax.set_yscale('log')
    ax.text(55, 5, modelstr, fontsize=16)
    plt.ylim([.001, 100])
    plt.legend(loc="lower left")

    return params, model_elastic_modulus


if __name__ == "__main__":
    import os
    import sys
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    sys.path.append(os.path.join(cwd,"data/raw"))
    sys.path.append(os.path.join(cwd,"tmp"))
    
    fname1 = "LCE_contraction_measurements.csv"
    fname2 = "200226_LCE_DMA_1.txt"
    contractParams, contractModel = fit_LCE_contraction(fname1)
    modulusParams, modulusModel = fit_LCE_elasticity(fname2)
