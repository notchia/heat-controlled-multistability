# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:21:31 2020

@author: Lucia

UnitCellData class and associated functions for manipulating quasistatic unit 
cell data, as collected using the AML's material tester, for one-shot loading
in tension.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import os
import scipy.interpolate as interp

from modules import calibration
from modules import tensiletest
from modules import bilayermodel as bilayer
from modules import unitforcemodel as model


class UnitCellData:
    ''' Store and manipulate data for each materials-tester data file imported'''
    def __init__(self, csvpath, cropFlag=True, figFlag=False, initEmpty=False,
                 setStartLoadToZero=False, setZeroLoadFromAngle=False):
        ''' Set metadata based on filename and import load-displacement data  '''

        # Set parameter information -------------------------------------------
        csvname = os.path.splitext(os.path.split(csvpath)[-1])[0]
        self.fileid     = csvname
        self.h          = self._get_re_value(csvname,'_h(\d*\.\d+)',0.)*1e-3  # [m] total hinge thickness
        self.r          = self._get_re_value(csvname,'_r(\d*\.\d+)',0.)       # [m/m] LCE:total hinge thickness ratio
        self.T          = self._get_re_value(csvname,'_(\d*\.\d+)C',25.0)     # [degrees C] temperature
        self.d          = self._get_re_value(csvname,'_d(\d*\.\d+)',0.)*1e-3  # [m] diagonal length, in direction of tension, measured in ImageJ
        self.s          = self._get_re_value(csvname,'_s(\d*\.\d+)',0.)*1e-3  # [m] arc length, measured in ImageJ
        self.magnets    = self._get_re_value(csvname,'_([NY])_',0)            # True/False: are there (attracting) magnets?
        self.tag        = self._get_re_value(csvname,'(_tag-[.*]_)','')       # string to tag related data
        
        self.label      = 'h={0:.2f}mm, r={1:.2f}, T={2:.1f}$^\circ$C, s={3:.2f}mm, magnets={4}'.format(self.h*1e3, self.r, self.T, self.s*1e3, self.magnets) 
        self.T_group  = self._get_temperature_group()
        
        # If initializing with actual datafile, import load-displacement data 
        self.cropFlag = cropFlag
        if not initEmpty:
            speed      = self._get_re_value(csvname,'_(\d*\.\d+)mmps',0.2)        # [mm/s] test speed
            zeroDisp   = self._get_re_value(csvname,'_zeroDisp(\d*\.\d+)',0.0)    # [mm] initial distance between centers of squares, measured in ImageJ
            sampleLoad = self._get_re_value(csvname,'_(\d*\.\d+)N',0.0)           # [N] sample mass
            self.disp, self.load  = self._import_data(csvpath, speed=speed, zeroDisp=zeroDisp,
                                                      sampleLoad=sampleLoad, figFlag=figFlag)
        
            # Crop to analyzable portion of data (compression)
            if cropFlag:
                self.disp_full, self.load_full = self.disp, self.load
                self.strain_full = (self.d - self.disp_full)/self.d
                self.disp, self.load = self._crop_to_compressed()
                self.strain = (self.d - self.disp)/self.d
            else:
                self.strain = (self.d - self.disp)/self.d
                self.disp_crop, self.load_crop = self._crop_to_compressed()
                self.strain_crop = (self.d - self.crop)/self.d
            
            self.std = np.array([])
            
            if setStartLoadToZero:
                self.load = self.load - self.load[0]
            
        # Alternate use case: for averaged data, directly assign disp, strain, 
        # and load mean and standard deviation

    def get_Series(self):
        ''' Create a pandas Series to be compiled into the DataFrame for all of the data '''
        return pd.Series({"data":    self,
                          "fileid":  self.fileid,
                          "h":       self.h,
                          "r":       self.r,
                          "T_group": self.T_group,
                          "d":       self.d,
                          "s":       self.s,
                          "magnets": self.magnets})

    def set_zero_with_angle(self, bilayerDict):
        hinge = bilayer.BilayerModel(self.h, self.r, T=self.T,
                               LCE_strain_params=bilayerDict["LCE_strain_params"],
                               LCE_modulus_params=bilayerDict["LCE_modulus_params"],
                               b=bilayerDict["b"])
        nominalZero = model.rot2disp(hinge.thetaT, self.d/2)
        zeroIndex = np.where(self.disp < nominalZero)[0][0] # CHECK THIS
        self.zeroOffset = self.load[zeroIndex]
        self.load = self.load - self.zeroOffset
        if self.cropFlag:
            self.load_full = self.load_full - self.zeroOffset
        else:
            self.load_crop = self.load_crop - self.zeroOffset
        return

    def _import_data(self, filepath, speed=0.2, zeroDisp=0.0, sampleLoad=0.0,
                     figFlag=False): 
        ''' Import one-shot test in tension. 
            - speed:          [mm/s] test speed
            - zeroDisp:       [mm] initial distance between centers of squares
            - sampleLoad:     [N] sample mass
            - tempLoadOffset: [N] load offset due to shift in load zero as a result of temperature'''
       
        # Import test ---------------------------------------------------------
        cols = [0,1] #time, load
        arr = np.genfromtxt(filepath,dtype=float,delimiter=',',skip_header=17,usecols=cols)
        
        # Find head and tail beyond which no stage movement occurs (artefact of homebrew test setup)
        load = arr[:,1]
        baseNoiseLevel = 0.05 #[N]
        startTrim = np.argwhere(np.abs(load-load[0]) > baseNoiseLevel)[0][0]
        endTrim = np.argmax(load)
        
        # Trim and re-zero data according to offset values
        tempLoadOffset = calibration.get_temperature_load_offset(self.T)
        disp_exp = 1e-3*(arr[startTrim:endTrim,0]*speed - arr[startTrim,0]*speed + zeroDisp)
        load_exp = load[startTrim:endTrim] + sampleLoad + tempLoadOffset

        # Smooth data (noisy due to vibration of linear stage)
        window = 2000 # data is really dense... may need to update this later
        disp_exp, load_exp = tensiletest.filter_raw_data(disp_exp, load_exp, window, cropFlag=True)

        # Interpolate, to keep size consistent
        f1 = interp.interp1d(disp_exp, load_exp)
        nPoints = 1000
        disp = np.linspace(min(disp_exp), max(disp_exp), nPoints)
        load = f1(disp)

        # Reverse indices and load direction, for ease of interpretation
        disp = disp[::-1]
        load = -load[::-1]
        
        # Plot ----------------------------------------------------------------
        if figFlag:
            plt.figure(dpi=200)
            #filename = os.path.splitext(os.path.split(filepath)[-1])[0]
            plt.title(self.label)
            plt.xlabel('Displacement (mm)')
            plt.ylabel('$F$ (N)')
            plt.plot(1e3*disp, -load, 'k', linewidth=2, label='Experiment')
            plt.tight_layout()
    
        return disp, load
 
    def _crop_to_compressed(self):
        self.zeroIndex = np.argwhere(self.disp <= self.d)[0][0]
        return self.disp[self.zeroIndex:], self.load[self.zeroIndex:]
    
    def _get_re_value(self, strname, pattern, defaultval):
        ''' Ternary operator for assigning values using regular expressions.
            - Returns numbers as floats
            - Returns Y/N as 0/1
            - Otherwise, returns string'''
        patternmatch = re.search(pattern,strname)
        if patternmatch:
            returnval = patternmatch.group(1)
            try:
                returnval = float(returnval)
            except ValueError:
                if returnval == 'N':
                    returnval = 0
                elif returnval == 'Y':
                    returnval = 1
                else:
                    pass
            return returnval
        else:
            return defaultval    
    
    def _get_temperature_group(self):
        ''' 0, 1, 2 for room temperature, medium (~45C), and high (~75C) '''
        if self.T < 30:
            group = 0
        elif self.T < 60:
            group = 1
        else:
            group = 2
        return group
        

#%% Utilities for importing and analyzing multiple datasets
def import_all_unit_cells(sourcedir, cropFlag=True, figFlag=False, setStartLoadToZero=False,
                          bilayerDict={}):
    ''' Import all unit cell data from a directory into a DataFrame '''
    
    UCDF = pd.DataFrame()  
    count = 0
    filenames = os.listdir(sourcedir)
    for entry in filenames:  
        filename, ext = os.path.splitext(entry)
        if ext == ".csv":
            # Initialize and add data to UnitCellData class instance, then store in dataframe
            filepath = os.path.join(sourcedir, entry)
            ucd = UnitCellData(filepath, cropFlag=cropFlag, figFlag=figFlag,
                               setStartLoadToZero=setStartLoadToZero)
            if any(bilayerDict) and (ucd.magnets == 0): # update based on modeled angle!
                ucd.set_zero_with_angle(bilayerDict)
            UCDF = UCDF.append(ucd.get_Series(),ignore_index=True)
        count += 1

    if any(bilayerDict): # update based on modeled angle!
        ucdf_Y = UCDF.loc[UCDF["magnets"] == 1]
        for index, row in ucdf_Y.iterrows():
            h = row["h"]
            r = row["r"]
            T_group = row["T_group"]
            match_N = UCDF.loc[(UCDF["magnets"] == 0) & (UCDF["h"] == h)
                               & (UCDF["r"] == r) & (UCDF["T_group"] == T_group)]
            match_N = match_N.iloc[0]["data"]
            openIndex = np.where(match_N.strain > 0)[0][0]
            zeroOffset = row["data"].load[openIndex] - match_N.load[openIndex]
            print(zeroOffset)
            row["data"].load = row["data"].load - zeroOffset
            if cropFlag:
                row["data"].load_full = row["data"].load_full - zeroOffset
            else:
                row["data"].load_crop = row["data"].load_crop - zeroOffset

    # Sort and display to double check what was imported
    UCDF.sort_values(["magnets","T_group","h","r"])
    print(UCDF[["fileid"]])
    
    return UCDF

def compute_unitCell_mean_std(ucdf, nPoints=2500):
    ''' Given dataframe containing only the data to be averaged, return
        UnitCellData object corresponding to mean and standard deviation
        '''
    
    nTests = len(ucdf.index)
    load_all = np.zeros((nPoints, nTests))
    h_all = np.zeros(nTests)
    r_all = np.zeros(nTests)
    T_all = np.zeros(nTests)
    d_all = np.zeros(nTests)
    s_all = np.zeros(nTests)
    
    # Find minimum value of maximum strain reached and construct new strain vector
    maxStrain = 1
    for index, row in ucdf.iterrows():
        unit = row["data"]
        currentMax = max(unit.strain)
        if currentMax < maxStrain:
            maxStrain = currentMax
    strain = np.linspace(1e-4, maxStrain, nPoints)
    
    # Crop load to maxStrain and save interpolated version to array
    count = 0
    for index, row in ucdf.iterrows(): 
        unit = row["data"]
        load_interp = interp.interp1d(unit.strain, unit.load)
        load_unit = load_interp(strain)
        load_all[:,count] = load_unit# - unit.load[0]
        h_all[count] = unit.h
        r_all[count] = unit.r
        T_all[count] = unit.T
        d_all[count] = unit.d
        s_all[count] = unit.s
        count += 1 # not sure if index is actually from 0
    
    # Find averages
    load_avg = np.mean(load_all, axis=1)
    load_std = np.std(load_all, axis=1)
    h_avg = np.mean(h_all)
    r_avg = np.mean(r_all)
    T_avg = np.mean(T_all)
    d_avg = np.mean(d_all)
    s_avg = np.mean(s_all)
    
    ucdf_name = 'averaged_h{0:.2f}_r{1:.2f}_{2:.1f}C_d{3:.2f}_s{4:.2f}_N_1.notcsv'.format(1e3*h_avg, r_avg, T_avg, 1e3*d_avg, 1e3*s_avg)
    
    meanUnit = UnitCellData(ucdf_name, cropFlag=False, initEmpty=True)
    meanUnit.disp = d_avg - strain*d_avg
    meanUnit.strain = strain
    meanUnit.load = load_avg
    meanUnit.std = load_std
    
    return meanUnit

def plot_magnet_and_T_comparison(ucdf, stdFlag=False, legendOut=False):
    ''' Plot and compare data, grouping by magnets and temperature '''
    
    plt.figure(dpi=200)
    plt.xlabel("Strain, $\delta/d$")
    plt.ylabel("Load (N)")
    colors = ['r','g','b']
    styles = ['-','--']
    TLabels = ['RT', 'T~45$^\circ$', 'T~75$^\circ$']
    magnetLabels = ['no', 'with']
    for index, row in ucdf.iterrows():
        unit = row["data"]
        T_index = int(unit.T_group)
        m_index = int(unit.magnets)
        disp_plt = unit.strain#((unit.d - unit.disp)/unit.d)
        plt.plot(disp_plt, unit.load, color=colors[T_index], linestyle=styles[m_index],
                 label='{0}, {1} magnets'.format(TLabels[T_index], magnetLabels[m_index]))
        if stdFlag:
            plt.fill_between(disp_plt, unit.load-unit.std, unit.load+unit.std, color=colors[T_index], alpha=0.2)
    if legendOut:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()
    
    return

    
#%% Main importing function
if __name__ == "__main__":   
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw/unitCell_properties")
    tmpdir = os.path.join(cwd,"tmp")
