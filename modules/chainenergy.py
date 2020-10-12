# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:33:35 2020

@author: Lucia
"""

import numpy as np
import os

import modules.pltformat # No functions, just formatting
import modules.metamaterialmodel as unit


if __name__ == "__main__":
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw")
    cleandir = os.path.join(cwd,"data/cleaned")
    tmpdir = os.path.join(cwd,"tmp")
    savedir = os.path.join(cwd,"results/figures_from_script")
    
    