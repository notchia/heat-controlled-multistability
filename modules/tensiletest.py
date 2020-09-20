# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:18:08 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.interpolate as interp
import re
import math

def import_tensile_experiment(filepath, speed=0, area=0, gaugeLength=0, figFlag=False):
    ''' Import one-shot tensile test from our homebrew tensile tester in 414'''
    assert (speed and area and gaugeLength), 'Specify speed [mm/s], cross-sectional area [mm^2], and gauge length [mm]'
    
    cols = [0,1] #time [s], load [n]
    arr = np.genfromtxt(filepath,dtype=float,delimiter=',',skip_header=17,usecols=cols)
    disp = speed*(arr[20:10000,0] - arr[0,0]) #[mm]
    load = arr[20:10000,1] - arr[0,1] #[N]
    
    stress = load/(area*1e-6) #[Pa]
    strain = disp/gaugeLength #[mm/mm]
    
    if figFlag:
        filename = os.path.splitext(os.path.split(filepath)[-1])[0]
        plt.figure(dpi=200)
        plt.title(filename)
        plt.xlabel('Strain (mm/mm)')
        plt.ylabel('Stress (MPa)')
        plt.plot(strain, 1e-6*stress, 'k', linewidth=2, label='Experiment')
        plt.tight_layout()
    
    return strain, stress

def fit_Young_from_tensile(strain, stress, maxVal=0.01):
    ''' Fit Young's modulus from maxVal*100% strain or the whole stress-strain
        curve (if max strain < 1%) '''
    if maxVal < strain[-1]:
        maxIndex = np.argwhere(strain > maxVal)[0][0]
    else:
        maxIndex = len(strain)
    params = np.polyfit(strain[:maxIndex], stress[:maxIndex], 1)
    Young = params[0]
    return Young, strain[:maxIndex], stress[:maxIndex]


def running_mean_centered(x, N):
    ''' Convolution-based running mean with window size N '''
    window = np.ones(int(N))/float(N)
    return np.convolve(x, window, 'same')

def filter_raw_data(disp, load, N, cropFlag=False):
    ''' Take centered moving average on load with window size N and crop disp
        accordingly. Returns new disp and load of same size unless cropFlag, 
        in which case returns disp and load with ceil(N/2) points cropped from
        each end'''
    
    assert max(disp.shape) == max(load.shape), 'disp and load must be same shape'
    
    load = running_mean_centered(load, N)
    if cropFlag:
        cropIndex = math.ceil(N/2)
        disp = disp[cropIndex:-cropIndex]
        load = load[cropIndex:-cropIndex]
    
    return disp, load

'''
def import_tensile_experiment_group(sourcedir):
    filelist = os.listdir(sourcedir)
    nSamples = len(filelist)
    Yvals = np.zeros(nSamples)
    
    plt.figure(dpi=200)
    plt.xlabel('Strain (mm/mm)')
    plt.ylabel('Stress (MPa)')    
    for i, filename in enumerate(filelist):
        print(filename)
        if os.path.splitext(filename)[-1] == '.csv':
            speed = float(re.search("_(\d+\.\d+)mmps", filename).group(1)) #[mm/s]
            width = float(re.search("_w(\d+\.\d+)", filename).group(1)) #[mm]
            thickness = float(re.search("_t(\d+\.\d+)", filename).group(1)) #[mm]
            gaugeLength = float(re.search("_L(\d+\.\d+)", filename).group(1)) #[mm]
            temperature = float(re.search("_(\d+\.\d+)C", filename).group(1)) #[mm]
            area = width*thickness #[mm^2]
            strain, stress = import_tensile_experiment(os.path.join(sourcedir, filename),
                                                       speed=speed, area=area, gaugeLength=gaugeLength)
            strain, stress = filter_raw_data(strain, stress, 50)
            Y = fit_Young_from_tensile(strain, stress)
            plt.plot(strain, stress*1e-6, linewidth=2, label=filename)
            plt.plot(strain, Y*strain, label=filename+'Young')
            
            Yvals[i] = Y    
    plt.legend()
    plt.tight_layout()
    
    return 
'''