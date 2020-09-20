# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:13:34 2020

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def find_temp2load_offset(sourcedir):
    filenames = os.listdir(sourcedir)
    T = []
    L = []
    plt.figure('temp-load')
    plt.xlabel('temperature ($^\circ$C)')
    plt.ylabel('load (N)')
    
    colors = ['r','g','b']
    i = 0
    for entry in filenames:  
        filename, ext = os.path.splitext(entry)
        if ext == ".csv":
            filepath = os.path.join(sourcedir, entry)
            data = np.genfromtxt(filepath,dtype=float,delimiter=',',skip_header=3)
            temperature, load, time = np.transpose(data)
            T.extend(list(temperature))
            L.extend(list(load))
            
            fig, ax = plt.subplots(dpi=200)
            plt.title(filename)
            plt.xlabel('time (min)')
            plt.ylabel('load (N)', color='r')
            plt.tick_params(axis='y', labelcolor='r')
            plt.plot(time, load, 'k', linewidth=2)
            ax.twinx()
            plt.plot(time, temperature, 'g',)
            plt.ylabel('temperature ($^\circ$C)', color='g')
            plt.tick_params(axis='y', labelcolor='g')
            plt.tight_layout()
            
            plt.figure('temp-load')
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(temperature, load, 'o'+colors[i], label=filename)
            
            pfit = np.polyfit(temperature, load, deg=1)
            label = 'y = {0:.6f}x + {1:.6f}'.format(pfit[0],pfit[1])
            plt.plot(temperature, np.polyval(pfit, temperature), colors[i], label=label)
            plt.plot(temperature, -get_temp_offset_orig(temperature), 'k')
            print('for '+filename+', '+label)
            
            i += 1
            
    plt.legend()

def get_temp_offset_orig(T):
    return -1.29*((T-24.5)/(76.9-24.5))

def get_temperature_load_offset(T):
    ''' fits found from running find_temp2load_offset on 200831 data'''
    if T < 30: # Room temperature
        loadOffset = 0
    elif T< 55: # Low fan, lowest setting
        loadOffset = 0.057113*T -1.940691
    elif T < 90: # High fan, lowest setting
        loadOffset = 0.070290*T -3.968263
    else:
        print("WARNING: don't have calibration data for higher temperature settings. Using that for high fan, lowest setting...")
        loadOffset = 0.070290*T -3.968263
    return -loadOffset       


if __name__ == "__main__":
   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw")
    tmpdir = os.path.join(cwd,"tmp")

    sourcedir = os.path.join(rawdir,'200831_temperature_calibration')
    find_temp2load_offset(sourcedir)
    