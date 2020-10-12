# -*- coding: utf-8 -*-
"""
Run all analyses required to generate results for ACS-AMI paper

@author: Lucia Korpas
"""

import numpy as np
import os

import modules.pltformat # No functions, just formatting
import modules.LCEmodel as lce
import modules.PDMSmodel as pdms
import modules.magnetmodel as magnet
import modules.bilayermodel as hinge
import modules.unitforceanalysis as force
import modules.metamaterialmodel as unit

def section_header(section):
    print('\n**************************\n> Processing data on {0} properties...'.format(section))
    return None

SAVE_FLAG = True
VERBOSE_FLAG = False


#%% Run all finalized data analysis 
if __name__ == "__main__":
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    split = os.path.split(cwd)
    if split[1] == 'modules':
        cwd = split[0]
    rawdir = os.path.join(cwd,"data/raw")
    cleandir = os.path.join(cwd,"data/cleaned")
    tmpdir = os.path.join(cwd,"tmp")
    savedir = os.path.join(cwd,"results/figures_from_script")

    #%% Get PDMS modulus 
    section_header('PDMS')
    sourcedir = os.path.join(rawdir, "PDMS_properties")
    fname = "200820_PDMS_DMA.txt"
    pdms.check_PDMS_temperature_independence(os.path.join(sourcedir, fname))
    pdms.fit_PDMS_tensile(sourcedir, saveFlag=SAVE_FLAG, figdir=savedir)
    
    #%% Get LCE modulus and contraction 
    section_header('LCE')
    sourcedir = os.path.join(rawdir, "LCE_properties")
    fname_strain = "ImageJ_measurements_31.5014mm2px.csv"
    fname_modulus = "200901_LCE_DMA_2.txt"#"200820_LCE_DMA.txt"
    strainParams, strainModel = lce.fit_LCE_strain(os.path.join(sourcedir, "LCE_contraction/"+fname_strain),
                                                   saveFlag=SAVE_FLAG, figdir=savedir)
    #modulusParams, modulusModel = lce.fit_LCE_modulus(os.path.join(sourcedir, fname_modulus))
    modulusParams, modulusModel = lce.fit_LCE_modulus_avg(os.path.join(sourcedir, "LCE_DMA"),
                                                          verboseFlag=VERBOSE_FLAG, saveFlag=SAVE_FLAG, figdir=savedir)
    #lce.fit_LCE_tensile(os.path.join(sourcedir, "LCE_tensile"),
    #                    saveFlag=SAVE_FLAG, figdir=savedir)
    
    #%% Get magnetic moment and check temperature degradation 
    section_header('magnet')
    sourcedir = os.path.join(rawdir, "magnet_properties")
    moment = magnet.fit_magnet_forces(os.path.join(sourcedir, "moment"),
                                      zeroDisp=9.20,
                                      saveFlag=SAVE_FLAG, figdir=savedir)
    if VERBOSE_FLAG:
        magnet.determine_temp_dependence(os.path.join(sourcedir, "temperature_degradation"),
                                         zeroDisp=7.85, saveFlag=SAVE_FLAG, figdir=savedir)    
    
    #%% Run analysis on bilayer beams
    section_header('bilayer')
    sourcedir = os.path.join(rawdir, "bilayer_properties")
    datafile = os.path.join(sourcedir, "200819_bilayer_ImageJ_curvature.csv")
    paramfile = os.path.join(sourcedir, "200819_bilayer_parameters.csv")
    hinge.analyze_data_200819(paramfile, datafile, LCE_modulus_params=modulusParams, 
                              LCE_strain_params=strainParams, 
                              saveFlag=SAVE_FLAG, figdir=savedir, verboseFlag=VERBOSE_FLAG)  
    hinge.analyze_curvature_change_with_temp(LCE_modulus_params=modulusParams,
                                             LCE_strain_params=strainParams, 
                                             saveFlag=SAVE_FLAG, figdir=savedir, verboseFlag=VERBOSE_FLAG)

    #%% Run analysis on bending angles
    section_header('unit cell angles')
    sourcedir = os.path.join(rawdir, "unitCell_properties")
    
    print("constant r:")
    datafile = os.path.join(sourcedir, "rconst_angles_ImageJ.csv")
    paramfile = os.path.join(sourcedir, "rconst_parameters.csv")
    b_fit = hinge.analyze_bending_angles(datafile, paramfile,
                           LCE_modulus_params=modulusParams,
                           LCE_strain_params=strainParams, titlestr='fixed r',
                           saveFlag=SAVE_FLAG, figdir=savedir)
    print("b_fit: {0}".format(b_fit))
#    print("constant h:")
#    datafile = os.path.join(sourcedir, "hconst_angles_ImageJ.csv")
#    paramfile = os.path.join(sourcedir, "hconst_parameters.csv")
#    hinge.analyze_bending_angles(datafile, paramfile,
#                                 LCE_modulus_params=modulusParams,
#                                 LCE_strain_params=strainParams, titlestr='fixed h')

    #%% Run analysis on repeatability force-displacement data
    bilayerDict = {"LCE_modulus_params":modulusParams,
                   "LCE_strain_params":strainParams,
                   "b":b_fit,
                   "bFlag":'quad'}

    #%% Run analysis on r-const force-displacement data
    section_header('unit cell constant-r force')
    sourcedir = os.path.join(rawdir,"unitCell_properties/unitCell_tension_rconst")
    r_avg, ucdf = force.import_rconst_data(sourcedir, bilayerDict)
    
    ksq_fit = force.analyze_rconst_nomagnets(ucdf, bilayerDict)
    print("k_sq_fit: {0}".format(ksq_fit))

    #%%
    # CURRENTLY THIS DOES NOT WORK. run the next cell instead
    
    limFlag='exp'
    """
    p_guess = [0.18,3e-10, 18]
    weightMagnitude = 1.0# 0.5
    moment_fit, p_lim_fit = force.analyze_rconst_moment_and_collision(ucdf, ksq_fit,
                                                                      bilayerDict,
                                                                      p_guess=p_guess,
                                                                      limFlag=limFlag, 
                                                                      weightMagnitude=weightMagnitude)
    print("m_fit: {0}".format(moment_fit))
    print("p_lim_fit: {0}".format(p_lim_fit))
    """
    #%%    
    moment_fit = force.analyze_rconst_moment(ucdf, ksq_fit, bilayerDict)
    print("m_fit: {0}".format(moment_fit))
    p_lim_fit = force.analyze_rconst_collision(ucdf, ksq_fit, moment_fit, bilayerDict)
    print("p_lim_fit: {0}".format(p_lim_fit))
    
    #%% Run analysis on repeatability force-displacement data
    section_header('unit cell force repeatability')
    sourcedir = os.path.join(rawdir,"unitCell_properties/unitCell_repeatability")
    force.analyze_repeatability_data(sourcedir, bilayerDict, k_sq=ksq_fit,
                                                          m=moment_fit, #p_lim=p_lim_fit,
                                                          saveFlag=SAVE_FLAG, figdir=savedir)
    
    
    #%% extra plot
    h_repeat = 1.71e-3
    r_repeat = 0.18
    unit.analyze_h_T_relation(r_repeat, k_sq=ksq_fit, m=moment_fit, limFlag='exp', p_lim=[1e-14, 30],
                             bilayerDict=bilayerDict,
                             saveFlag=SAVE_FLAG, figdir=savedir)

    #%% Plot resulting best-fit curve
    # Commented out are the best-fit parameters found once upon a time...
    """
    ksq_fit = 0.005765489237871801
    moment_fit = 0.1897092016810536
    p_lim_fit = [1.77252756e-10, 1.91067338e+01]"""
    force.plot_final_rconst_fit(ucdf, ksq_fit, moment_fit, p_lim_fit,bilayerDict, limFlag=limFlag)

    
    #%% Helper figures: 
    h_val = 0.9e-3
    r_val = 0.25
    T_range = [25.0, 35.0, 50.0, 80.0]
    unit.plot_energy_concept(h_val, r_val, 0.0, T_range,
                             k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                             bilayerDict=bilayerDict,
                             figdir=savedir)

    #%% Misc other phase diagrams: diagonal dependence
    h_val = 1.2e-3
    ratio_val = 0.4
    thetaL_val = 0.0
    unit.analyze_diagonal_dependence(h_val, ratio_val, thetaL_val,
                                k_sq=ksq_fit, m=moment_fit, limFlag='exp', p_lim=p_lim_fit,#[1e-14, 30],
                                bilayerDict=bilayerDict,
                                saveFlag=SAVE_FLAG, figdir=savedir)

    #%% Run phase diagram generation   
    section_header('phase diagrams')    
    r_range = np.array([r_avg]) #r_const average value
    h_range = 1e-3*np.arange(0.7,2.1,0.01)
    thetaL_range = np.radians(np.arange(-15.0,15.1,0.1))
    T_range = np.array([25.0, 45.0, 78.0])#np.arange(25.0,105.0,5.0)
    unit.analyze_composites(r_range=r_range,
                            h_range=h_range,
                            thetaL_range=thetaL_range,
                            T_range=T_range,
                            k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                            bilayerDict=bilayerDict,
                            savedir=tmpdir, closeFlag=False) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    