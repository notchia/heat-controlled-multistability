# -*- coding: utf-8 -*-
"""
Standardize plot parameters using these settings.

@author: Lucia
"""

import matplotlib.pyplot as plt

#%%

plt.rcParams['font.family']     = 'arial'
#plt.rcParams['figure.figsize']  = 9, 6      # (w=3,h=2) multiply by 3
plt.rcParams['font.size']       = 12        # Original font size is 8, multipy by above number
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12    
plt.rcParams['xtick.labelsize'] = 12    
plt.rcParams['ytick.labelsize'] = 12        
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['savefig.dpi']  = 200
plt.rcParams['patch.facecolor'] = 'white'
plt.rcParams['legend.frameon'] = False
plt.rcParams['figure.max_open_warning'] = 0
