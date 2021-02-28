"""
Fit experimental data for magnet strenth. From main data analysis script, call
the following function:
    plist, fcn = fit_magnet_force(fullpath)
with
    fullpath:   str corresponding to raw data file location
    plist:      list of fitting parameters
    fcn:        function for which fitting parameters were found 
    
@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import re

from modules import tensiletest


def model_force (d, m):
    """ Model force at a given distance between two identical attracting
        magnets, assuming them to be point dipoles, given the magnetic moment """
    
    mu0 = 1.256637*10**(-6) # magnetic permeability of free space
    F = (3*mu0*m**2)/(2*np.pi*d**4)
    return F


def fit_magnet_force(filename, zeroDisp=0, speed=1, filterFlag=False,
                     figFlag=False, saveFlag=False, figdir='', ):
    """ Fit cycles of force-displacement data to dipole model.  
        Check that the optional parameter values are correct for the test: 
            zeroDisp:   [mm] true distance between centers of squares at zero
                        displacement, as measured in ImageJ from photo
            speed:      [mm/s] test speed
    """
    
    # Import data; split multicycle array into tension and compression for each
    cols = [0,1,4] #time, load, cycle number
    arr = np.genfromtxt(filename,dtype=float,delimiter=',',skip_header=17,usecols=cols)
    cycle = np.split(arr, np.where(np.diff(arr[:,2]))[0]+1)
    if len(cycle) == 3:
        leg = 1
    elif len(cycle) == 1:
        leg = 0
    else:
        print("Error: can only analyze tension or single-cycle (tension-compression) data")
    disp = 1e-3*(cycle[leg][:,0]*speed - cycle[leg][0,0]*speed + zeroDisp)
    load = cycle[leg][:,1]

    if filterFlag:
        disp, load = tensiletest.filter_raw_data(disp, load, 20, cropFlag=True)

    # Fit to model and plot
    params, dm = opt.curve_fit(model_force, disp, load)
    
    if figFlag:
        plt.figure(dpi=200)
        plt.title("Cylindrical magnet, attraction")
        plt.xlabel("Displacement (mm)")
        plt.ylabel("Load (N)")
        plt.plot(1e3*disp, load, 'k')
        plt.plot(1e3*disp, model_force(disp, params[0]), 'r--', label='m = {0:.6f}'.format(params[0]))
        plt.xlim(left=0.0)
        plt.title(os.path.split(filename)[-1])
        plt.legend()
        if saveFlag:
            plt.savefig(os.path.join(figdir, "magnet_force_fit.png"),dpi=300)
    
    return params, model_force


def fit_magnet_forces(sourcedir, zeroDisp=0, saveFlag=False, figdir=''):
    """ Fit magnetic moment (assuming point dipoles) to all data in source
        directory, then find and return average of these """
    filelist = os.listdir(sourcedir)

    m_list = []
    for i, fname in enumerate(filelist):
        if os.path.splitext(fname)[-1] == '.csv':
            specs = os.path.splitext(fname)[0]
            if '200820' in specs and 'tension' in specs and 'attract' in specs:
                speed = float(re.search("_(\d+\.\d+)mmps", specs).group(1))
                if speed == 1.0:
                    fullpath = os.path.join(sourcedir, fname)
                    params, model = fit_magnet_force(fullpath, zeroDisp=zeroDisp, speed=speed, filterFlag=True)
                    m_list.append(params[0])
    moment_avg = np.mean(m_list)
    moment_std = np.std(m_list)
    print('Magnetic moment: {0:.4f} +/- {1:.4f} N/A^2'.format(moment_avg, moment_std))
    
    return moment_avg
    

def determine_temp_dependence(sourcedir, zeroDisp=0, saveFlag=False, figdir=''):
    """ Determine if magnetic moment drops degrades after exposure to high 
        temperatures. For each test, the same pair of magnets was exposed to
        the given temperatures, in increasing order """
    
    filelist = os.listdir(sourcedir)
    nT = len(filelist)
    T = np.zeros(nT)
    m = np.zeros(nT)
    
    # Fit magnet force-displacement for each test
    for i, fname in enumerate(filelist):
        fullpath = os.path.join(sourcedir, fname)
        temp = os.path.splitext(fname)[0].split('_')[-1]
        T[i] = int(temp.strip('C'))
        params, model = fit_magnet_force(fullpath, zeroDisp=zeroDisp)
        m[i] = params[0]
 
    # Plot temperature-moment relation
    plt.figure('magnet_temperature_test',dpi=200)
    plt.title("Magnetic strength degradation")
    plt.xlabel("$T$ ($^\circ$C)")
    plt.ylabel("$m$ (NA$^2$)")
    plt.plot(T, m, 'ok')
    
    if saveFlag:
        plt.savefig(os.path.join(figdir, "magnet_temp_dependence.svg"), transparent=True)
        plt.savefig(os.path.join(figdir, "magnet_temp_dependence.png"), dpi=200)
    
    return


if __name__ == '__main__':   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd, "data/raw/magnet_properties")
    tmpdir = os.path.join(cwd, "tmp")
    