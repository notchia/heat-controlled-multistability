"""
Despite efforts to separate the load cell from the heated chamber,
heat transfer through the threaded rod still results in significant drift in
the load zero as the cell heats up. This function is an attempt to characterize
and account for this.

@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def find_temp2load_offset(sourcedir, savedir=''):
    """ Find the linear fit which best describes the load offset resulting from 
    an increase in temperature. Test conditions were as similar to the real
    setup as possible: threaded rod lowered into box, heating with heat gun """
    
    filenames = os.listdir(sourcedir)
    T = []
    L = []
    plt.figure('temp-load')
    plt.xlabel('Temperature ($^\circ$C)')
    plt.ylabel('Load (N)')
    
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
            
            fig, ax = plt.subplots(dpi=300)
            plt.title(filename)
            plt.xlabel('Time (min)')
            plt.ylabel('Load (N)', color='r')
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
            print('for '+filename+', '+label)
            
            i += 1
            
    plt.legend()
    if savedir != '':
        plt.savefig(os.path.join(savedir, 'load-temperature_calibration.png'), dpi=300)

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

    sourcedir = os.path.join(rawdir,'preprocessing/200831_temperature_calibration')
    find_temp2load_offset(sourcedir, savedir=tmpdir)
    