# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:27:45 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pltformat  # module: no funcions, just standardized plot formatting

''' Magnetic force model to get magnetic moment '''
def model_force (d, m):
    mu0 = 1.256637*10**(-6)
    return (3*mu0*m**2)/(2*np.pi*d**4)


''' Fit cycles of force-displacement data to model '''
def fit_magnet_force(filename, fighandle=''):
    if fighandle == '':
        fig = plt.figure('magnet force')
        plt.title("Cylindrical magnet - attraction")
        plt.xlabel("Displacement (m)")
        plt.ylabel("Load (N)")
    else:
        fig = plt.figure(fighandle)

    cols = [0,1,4]
    arr = np.genfromtxt(filename,dtype=float,delimiter=',',skip_header=17,usecols=cols)#,names=["cycle","disp","load","time"]     
    
    # Split single array into individual arrays for each full cycle, then into
    # tension and compression portion of each cycle
    speed = 1 # mm/s
    zeroDisp = 7.85 # Measured from photo
    disp_t = arr[:][1]*speed - arr[0][1]*speed + zeroDisp
    load_t = arr[:][2]

    mFit, dm = curve_fit(model_force, 0.001*disp_t, load_t)
    
    plt.plot(0.001*disp_t, load_t)
    plt.plot(0.001*disp_t, model_force(0.001*disp_t, mFit), 'r--')
    print("m = {0:.4f}".format(mFit[0]))
    
    return mFit, modelMagnet


''' Determine if magnetic moment drops with temperature '''
def determine_temp_dependence(filelist):
           
    fighandle = 'magnet temp test'
    plt.figure(fighandle)
    plt.title("Cylindrical magnet - attract")
    plt.xlabel("Displacement (m)")
    plt.ylabel("Load (N)")

    fname = 'magnet_cylinder_quarterInch_40mm_1mmps_attract.csv'
    
    for fname in filelist:
        fullpath = os.path.join(sourcepath, fname) #update
        params, model = fit_magnet_force(fullpath, fighandle=fighandle)


if __name__ == '__main__':
    import os
    import sys
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    sys.path.append(os.path.join(cwd,"data/raw"))
    sys.path.append(os.path.join(cwd,"tmp"))

    fit_magnet_force(filename, fighandle='')

    determine_temp_dependence(os.listdir(sourcepath))
    #exportpath = os.path.join(basepath, 'Raw_%s'%foldername)
    #if os.path.exists(exportpath) == False:
    #    os.mkdir(exportpath)
    #filenames = os.listdir(sourcepath)