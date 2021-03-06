"""
Run all analyses required to generate results for ACS-AMI paper,
with the exception of the Paraview-generated 3D phase diagram visualization

@author: Lucia Korpas
"""

import numpy as np
import os

import modules.pltformat # No functions, just formatting; don't delete
import modules.LCEmodel as lce
import modules.PDMSmodel as pdms
import modules.magnetmodel as magnet
import modules.bilayermodel as hinge
import modules.curvatureanalysis as curvature
import modules.angleanalysis as angle
import modules.rconstanalysis as force
import modules.metamaterialmodel as unit
import modules.repeatabilityanalysis as repeatability
import modules.parameterspaceanalysis as mapping

def section_header(section):
    print('\n**************************\n> Processing data on {0} properties...'.format(section))
    return None

def item_header(item):
    print('> {0}...'.format(item))
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
    tmpdir = os.path.join(cwd,"tmp")
    savedir = os.path.join(cwd,"results/figures_from_script")
    resdir = os.path.join(cwd,"results")
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)	

    #%% Get PDMS modulus 
    section_header('PDMS')
    sourcedir = os.path.join(rawdir, "PDMS_properties")
    
    fname_PDMS = "200820_PDMS_DMA.txt"
    pdms.check_PDMS_temperature_independence(os.path.join(sourcedir, fname_PDMS))
    pdms.fit_PDMS_tensile(sourcedir, saveFlag=SAVE_FLAG, figdir=savedir)
    
    #%% Get LCE modulus and contraction and plot DSC data
    section_header('LCE')
    sourcedir = os.path.join(rawdir, "LCE_properties")
    
    fname_strain = "ImageJ_measurements_31.5014mm2px.csv"
    fname_DSC = "LCE_DSC.txt"
    strainParams, strainModel = lce.fit_LCE_strain(os.path.join(sourcedir, "LCE_contraction/"+fname_strain),
                                                   saveFlag=SAVE_FLAG, figdir=savedir)
    modulusParams, modulusModel = lce.fit_LCE_modulus_avg(os.path.join(sourcedir, "LCE_DMA"),
                                                          verboseFlag=VERBOSE_FLAG, saveFlag=SAVE_FLAG, figdir=savedir)
    lce.plot_DSC(os.path.join(sourcedir, fname_DSC), saveFlag=SAVE_FLAG, figdir=savedir)

    #%% Get magnetic moment and check temperature degradation 
    section_header('magnet')
    sourcedir = os.path.join(rawdir, "magnet_properties")
    
    moment = magnet.fit_magnet_forces(os.path.join(sourcedir, "moment"),
                                      zeroDisp=9.20,
                                      saveFlag=SAVE_FLAG, figdir=savedir)
    if VERBOSE_FLAG:
        magnet.determine_temp_dependence(os.path.join(sourcedir, "temperature_degradation"),
                                         zeroDisp=7.85, saveFlag=SAVE_FLAG, figdir=savedir)    
    
    #%% Run analysis on free bilayer beam curvature
    section_header('bilayer')
    sourcedir = os.path.join(rawdir, "bilayer_properties")
    
    datafile = os.path.join(sourcedir, "200819_bilayer_ImageJ_curvature.csv")
    paramfile = os.path.join(sourcedir, "200819_bilayer_parameters.csv")
    
    rBestFit=False
    r_relation = curvature.analyze_bending_data(paramfile, datafile, rBestFit=rBestFit,
                                                LCE_modulus_params=modulusParams,
                                                LCE_strain_params=strainParams, 
                                                saveFlag=SAVE_FLAG, figdir=savedir,
                                                verboseFlag=VERBOSE_FLAG)  
    hinge.analyze_curvature_change_with_temp(LCE_modulus_params=modulusParams,
                                             LCE_strain_params=strainParams, 
                                             saveFlag=SAVE_FLAG, figdir=savedir,
                                             verboseFlag=VERBOSE_FLAG)

    #%% Run analysis on unit cell bending angles, without magnets
    section_header('unit cell angles')
    sourcedir = os.path.join(rawdir, "unitCell_properties")
    
    print("Using samples with nominally constant r values...")
    datafile = os.path.join(sourcedir, "rconst_angles_ImageJ.csv")
    paramfile = os.path.join(sourcedir, "rconst_parameters.csv")
    b_fit = angle.analyze_bending_angles(datafile, paramfile,
                           LCE_modulus_params=modulusParams,
                           LCE_strain_params=strainParams, titlestr='fixed r',
                           r_relation=r_relation,
                           saveFlag=SAVE_FLAG, figdir=savedir)
    print("arc length is now defined as s = b_fit[0]h^2, with b_fit = {0}".format(b_fit))

    #%% Define dictionary to easily pass the fitting parameters through the rest of the modeling
    bilayerDict = {"LCE_modulus_params":modulusParams,
                   "LCE_strain_params":strainParams,
                   "b":b_fit,
                   "bFlag":'quad'}

    #%% Run analysis on r-const force-displacement data
    section_header('unit cell constant-r force')
    sourcedir = os.path.join(rawdir,"unitCell_properties/unitCell_tension_rconst")
    
    r_avg, ucdf = force.import_rconst_data(sourcedir, bilayerDict, m=moment) #setStartLoadToZero=True

    
    #%% Find best-fit square stiffness for r-const data
    item_header("Finding best-fit k_sq")
    
    ksq_fit = force.analyze_rconst_ksq(ucdf, bilayerDict, r_relation=r_relation) #or, analyze_rconst_nomagnets
    print("k_sq_fit: {0}".format(ksq_fit))

    #%% Find best-fit magnetic moment and collision parameters for r-const data
    item_header("Finding best-fit moment and p_lim")
               
    moment_fit = force.analyze_rconst_moment(ucdf, ksq_fit, bilayerDict)
    print("m_fit: {0}".format(moment_fit))

    
    #%% Plot resulting best-fit curves
    item_header("Plotting resulting fit for r_const load-disp data")  

    limFlag='exp'   
    p_lim_fit = [1e-14, 30] # Chosen by hand!
    force.plot_final_rconst_fit(ucdf, ksq_fit, moment_fit, p_lim_fit, bilayerDict,
                                limFlag=limFlag)

    #%% Run analysis on repeatability force-displacement data, using fits from r-const data
    section_header('unit cell force repeatability')
    sourcedir = os.path.join(rawdir,"unitCell_properties/unitCell_repeatability")
    repeatability.analyze_repeatability_data(sourcedir, bilayerDict,
                                             k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                                             saveFlag=SAVE_FLAG, figdir=savedir)

    #%% Additional analysis: generate 2D phase diagram: h-T relationship for repeatability parameters
    h_repeat = 1.71e-3
    r_repeat = 0.18
    mapping.analyze_h_T_relation(r_repeat, k_sq=ksq_fit, m=moment_fit,
                                 p_lim=p_lim_fit, limFlag=limFlag,
                                 bilayerDict=bilayerDict,
                                 saveFlag=SAVE_FLAG, figdir=savedir)
        
    #%% Additional analysis: generate Fig. 1 energy concept plot
    h_val = 1.0e-3
    r_val = 0.25
    T_range = [25.0, 35.0, 50.0, 80.0]
    unit.plot_energy_concept(h_val, r_val, 0.0, T_range,
                             k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                             bilayerDict=bilayerDict,
                             figdir=savedir)

    #%% Additional analysis: plot energy landscapes corresponding to manufactured chains
    ANGLE_NEAR_ZERO = 1e-8 # to avoid numerical trouble
    unit.MetamaterialModel(0.73e-3, 0.47, ANGLE_NEAR_ZERO, T=25.0,
                           k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                           loadFlag=True, plotFlag=True,
                           **bilayerDict)
    unit.MetamaterialModel(1.50e-3, 0.40, ANGLE_NEAR_ZERO, T=25.0,
                           k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                           loadFlag=True, plotFlag=True,
                           **bilayerDict)

    #%% Generate and plot 3D phase diagrams!
    section_header('phase diagram model')    

    h_range = 1e-3*np.arange(0.5,2.005,0.01)

    T_range = np.arange(25.0,78.5,0.5) 
    T_isotherm = [25.0, 45.0, 78.0]
    i_isotherm = [np.argwhere(T_range == T_val)[0][0] for T_val in T_isotherm]
    
    r_range_orig = np.arange(0.0,1.0,0.005)
    r_range = np.polyval(r_relation, r_range_orig)
    r_index_min = np.where(r_range >= 0)[0][0]
    r_index_max = np.where(r_range < 1.0)[0][-1]
    r_range = r_range[r_index_min:r_index_max]
    print(r_range)
    r_val = np.polyval(r_relation, r_avg)
    
    thetaL_range = np.radians(np.arange(-15.0,15.05,0.1))
    thetaL_val = 0.0

	# To rerun the analysis, use an empty string: this will generate a dataset labeled with the current date, e.g., '20210412' for April 12, 2021
	# To use previous results or create new results with a custom label, supply the date string ('20210412') or custom label
    datestr = '' 
    
    item_header("h-r-T (\"main\") parameter space analysis")
    minimaMain, phasesMain, thetaTMain, theta0Main, paramDictMain, sampleModelMain = mapping.run_main_parameter_phase_boundary_analysis(thetaL_val,
                            h_range=h_range, r_range=r_range, T_range=T_range,
                            k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                            bilayerDict=bilayerDict,
                            savedir=resdir, closeFlag=False, datestr=datestr)

    boundariesMain, boundaryValsMain, boundaryDataMain = mapping.find_3D_phase_boundaries_main_params(thetaL_val,
                            h_range=h_range, r_range=r_range, T_range=T_range,
                            minima=minimaMain, phases=phasesMain, angleT_vals=thetaTMain, angle0_vals=theta0Main)

    mapping.save_boundaries(datestr, resdir, boundariesMain, boundaryDataMain, boundaryValsMain, tag='Main')
    mapping.save_isotherms(datestr, resdir, phasesMain, i_isotherm, h_range, r_range, T_range, tag='Main')    
    
    item_header("h-thetaL-T parameter space analysis")
    minima, phases, thetaT, theta0, paramDict, sampleModel = mapping.run_composite_phase_boundary_analysis(r_val,
                            h_range=h_range, thetaL_range=thetaL_range, T_range=T_range,
                            k_sq=ksq_fit, m=moment_fit, p_lim=p_lim_fit,
                            bilayerDict=bilayerDict,
                            savedir=resdir, closeFlag=False, datestr=datestr)
    
    boundaries, boundaryVals, boundaryData = mapping.find_3D_phase_boundaries(r_val,
                            h_range=h_range, thetaL_range=thetaL_range, T_range=T_range,
                            minima=minima, phases=phases, angleT_vals=thetaT, angle0_vals=theta0)
    
    mapping.save_boundaries(datestr, resdir, boundaries, boundaryData, boundaryVals)
    mapping.save_isotherms(datestr, resdir, phases, i_isotherm, h_range, thetaL_range, T_range)  
