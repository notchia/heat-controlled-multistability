"""
Analyze PDMS stiffness, using load-displacement and DMA data. 

@author: Lucia Korpas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from modules import tensiletest


def model_elastic_modulus(T):
    """ Young's modulus as constant over timeUsed as input in bilayer model? """
    return 2.25e6


def check_PDMS_temperature_independence(filepath, saveFlag=False, figdir=''):
    """ Plot PDMS elastic modulus DMA (one test file) """
    # Import data and convert to strain
    data = np.genfromtxt(filepath, skip_header=2, delimiter='\t')
    E_s = data[:,1]
    E_l = data[:,2]
    T = data[:,4] # 7 for earlier data!
    
    # Semilog plot shear, loss, total, and Young's moduli and fit 
    fig = plt.figure('PDMS_modulus', dpi=200) # This figure handle is also used in fit_PDMS_tensile!
    plt.plot(T, 1e-6*E_s, '.r', label="E'")
    plt.plot(T, 1e-6*E_l, '.b', label="E''")
    plt.xlabel("Temperature ($^\circ$C)")
    plt.ylabel("Modulus (MPa)")
    ax = fig.gca()
    ax.set_yscale('log')
    plt.ylim([.001, 100])
    plt.legend(loc="lower left")

    if saveFlag:
        plt.savefig(os.path.join(figdir, "PDMS_modulus.svg"), transparent=True)
        plt.savefig(os.path.join(figdir, "PDMS_modulus.png"), dpi=200)

    return 


def fit_PDMS_tensile(sourcedir, verboseFlag=False, saveFlag=True, figdir=''):
    """ Process tensile data (all .csv files in sourcedir) for PDMS """
    filelist = os.listdir(sourcedir)
    T_vals = []
    Y_vals = []
    for filename in filelist:
        if os.path.splitext(filename)[-1] == '.csv':
            filepath = os.path.join(sourcedir, filename)
            
            # Get parameters from filename
            speed = float(re.search("_(\d+\.\d+)mmps", filename).group(1)) #[mm/s]
            width = float(re.search("_w(\d+\.\d+)", filename).group(1)) #[mm]
            thickness = float(re.search("_t(\d+\.\d+)", filename).group(1)) #[mm]
            gaugeLength = float(re.search("_L(\d+\.\d+)", filename).group(1)) #[mm]
            temperature = float(re.search("_(\d+\.\d+)C", filename).group(1)) #[mm]
            
            # Import, filter, and fit Young's modulus for tensile data
            strain, stress = tensiletest.import_tensile_experiment(filepath, speed=speed, area=width*thickness, gaugeLength=gaugeLength)
            strain, stress = tensiletest.filter_raw_data(strain, stress, 50)
            Young, strain, stress = tensiletest.fit_Young_from_tensile(strain, stress, maxVal=0.04)
            
            # Plot stress-strain curve and report Young's modulus
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
    
            print('PDMS Young\'s modulus @ {1:.1f}C: {0:.2f}MPa'.format(1e-6*Young, temperature))
    
    # Plot Young's modulus as a function of temperature
    plt.figure('PDMS_modulus') # This figure handle is also used in check_PDMS_temperature_independence!
    plt.plot(T_vals, [1e-6*Y for Y in Y_vals], '*k', markersize='12', label='Young\'s modulus (tensile)')
    plt.tight_layout()
    plt.legend()
    
    if saveFlag:
        plt.savefig(os.path.join(figdir, 'PDMS_modulus.svg'), transparent=True)
        plt.savefig(os.path.join(figdir, 'PDMS_modulus.png'), dpi=300)

    return


if __name__ == "__main__":   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd, "data/raw/PDMS_properties")
    tmpdir = os.path.join(cwd, "tmp")
    