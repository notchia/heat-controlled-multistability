"""
Import, filter, and fit Young's modulus for tensile data

@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math


def import_tensile_experiment(filepath, speed=0, area=0, gaugeLength=0,
                              startPoint=20, endPoint=10000, figFlag=False):
    """ Import one-shot tensile test from our homebrew tensile tester in 414;
        Return strain, stress as numpy arrays.
        Displacement and load are each zeroed based on first data point
        (assumes no preload or prestrain). """
    
    assert (speed and area and gaugeLength), 'Specify speed [mm/s], cross-sectional area [mm^2], and gauge length [mm]'
    
    cols = [0,1] #time [s], load [n]
    arr = np.genfromtxt(filepath,dtype=float,delimiter=',',skip_header=17,usecols=cols)
    # Skip first startPoint data points due to lag between load cell recording start 
    # and displacement actuator start; takes points only up until endPoint for
    # same reason.
    disp = speed*(arr[startPoint:endPoint,0] - arr[0,0]) #[mm]
    load = arr[startPoint:endPoint,1] - arr[0,1] #[N]
    
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
    """ Fit Young's modulus from maxVal*100% strain or the whole stress-strain
        curve (if maximum strain < default or specified maxVal) """
    if maxVal < strain[-1]:
        maxIndex = np.argwhere(strain > maxVal)[0][0]
    else:
        maxIndex = len(strain)
    Young = np.polyfit(strain[:maxIndex], stress[:maxIndex], 1)[0]
    
    return Young, strain[:maxIndex], stress[:maxIndex]


def running_mean_centered(x, N):
    """ Convolution-based running mean with window size N """
    window = np.ones(int(N))/float(N)
    return np.convolve(x, window, 'same')


def filter_raw_data(disp, load, N, cropFlag=False):
    """ Take centered moving average on load with window size N and crop disp
        accordingly.
        Returns new disp and load of same size unless cropFlag, in which case
        returns disp and load with ceil(N/2) points cropped from each end. """
    
    assert max(disp.shape) == max(load.shape), 'disp and load must be same shape'
    
    load = running_mean_centered(load, N)
    if cropFlag:
        cropIndex = math.ceil(N/2)
        disp = disp[cropIndex:-cropIndex]
        load = load[cropIndex:-cropIndex]
    
    return disp, load
