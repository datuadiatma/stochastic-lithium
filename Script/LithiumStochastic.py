
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

from scipy.interpolate import interp1d

# Steady state mass balance equation
def calc_RLisw_steady(Fr, Rr, Fh, Rh, Fsed, Dsed, Falt, Dalt):
    """Seawater d7Li at steady state
    
    Paramaters
    ----------
    Fr : float
        Riverine input flux
    Rr : float
        Riverine d7Li ratio
    Fh : float
        Hydrothermal input flux
    Rh : float
        Hydrothermal isotope ratio
    Fsed : float
        Lithium sink due to uptake onto marine sediments (Reverse Weathering)
    Dsed : float
        Lithium isotope fractionation due to reverse weathering
    Falt : float
        Lithium sink via basaltic alteration (oceanic crust)
    Dalt :float
        Lithium isotope fractionation due to basaltic alteration

    Return
    ------
    RLisw_ss : float
        Seawater lithium isotopic ratios at steady state
    """
    RLisw_ss = (Fr*Rr + Rh*Fh + Dsed*Fsed + Falt*Dalt) / (Fsed + Falt)
    return RLisw_ss

# Function to run stochastic model
def run_sim(parameters, target_array, tolerance=1.1, mode='random',
            riverine_flux = [],
            riverine_ratio = [],
            riverine_age = [],
            hydrothermal_flux = [],
            hydrothermal_ratio = [],
            hydrothermal_age = [],
            reverseweathering_flux = [],
            rw_fractionation = [],
            rw_age = [],
            basaltalteration_flux = [],
            alt_fractionation = [],
            alt_age = []):
    """ Run stochastic modeling simulation

    Parameters
    ----------
    parameters : dict or json
        parameters to run the simulation
    target : dict or json
        d7Li values that are used as "target"
    tolerance : float, optional
        window of tolerance. Default is 1.1 permil
    mode : string
        Simulation mode. It can be:
            random : all parameters are randomized
            riverine : riverine flux are set to certain values
            riverine_ratios : riverine ratios are set to certain values
            hydrothermal :
            hydrothermal_ratios :
            reverse_weathering :
            etc
    
    Return
    ------
    results : dictionary
        Dictionary containing array of results    
    """

    # Load dictionaries or json files into variables
    if type(parameters) == dict:
        param = parameters
    else:
        with open(parameters) as f:
            param = json.load(f)
    
    if type(target_array) == dict:
        target = target_array
    else:
        with open(target_array) as f:
            target = json.load(f)
    
    # Unpack parameters into variables
    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']

    # Age in Ma
    age = np.linspace(tmin, tmax, nt)

    # Monte Carlo resampling
    s = param['sampling']

    # Resample target array to fit the size of "age" array (nt)
    resample = interp1d(target['age'], target['d7Li'], 
                        bounds_error=False, fill_value='extrapolate')
    target_d7Li = resample(age)

    # Reshape target d7Li into nt x nt matrix using zeros and transpose
    # operation.
    target_d7Li = (np.zeros((s, nt)) + target_d7Li).T

    # Initiate random number generator with seed = 614 for reproducibility
    rng = np.random.default_rng(614)

    # Simulation with 'random' as mode
    if mode == 'random':
        print('random mode')
        Fr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fr'][0], param['Fr'][1], s))
        
        Rr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rr'][0], param['Rr'][1], s))
        
        Fh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fh'][0], param['Fh'][1], s))
        
        Rh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rh'][0], param['Rh'][1], s))
        
        Fsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fsed'][0], param['Fsed'][1], s))
        
        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))
        
        Falt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Falt'][0], param['Falt'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_range, Rr_range, Fh_range, 
                                     Rh_range, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)
        
        # Filter
        print('Filtering Process')
        Fr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fr_range, 0)
        Rr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rr_range, 0)
        Fh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fh_range, 0)
        Rh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rh_range, 0)
        Fsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fsed_range, 0)
        Dsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dsed_range, 0)
        Falt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Falt_range, 0)
        Dalt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dalt_range, 0)

        sol = Fr_res[Fr_res!=0]

        print('solutions found', len(sol))

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }

    # Simulation with predetermined riverine flux
    if mode == 'riverine_flux':
        # Resample riverine flux input
        f = interp1d(riverine_age, riverine_flux)
        Fr_resampled = f(age)

        # Createa [s x nt] matrix of Fr_resampled
        Fr_input = (np.zeros((s, nt)) + Fr_resampled).T

        # Generate random values
        Rr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rr'][0], param['Rr'][1], s))
        
        Fh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fh'][0], param['Fh'][1], s))
        
        Rh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rh'][0], param['Rh'][1], s))
        
        Fsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fsed'][0], param['Fsed'][1], s))
        
        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))
        
        Falt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Falt'][0], param['Falt'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_input, Rr_range, Fh_range, 
                                     Rh_range, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)
        
        # Filter
        Fr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fr_input, 0)
        Rr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rr_range, 0)
        Fh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fh_range, 0)
        Rh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rh_range, 0)
        Fsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fsed_range, 0)
        Dsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dsed_range, 0)
        Falt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Falt_range, 0)
        Dalt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dalt_range, 0)

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }
    
    # Simulation with predetermined riverine d7Li ratios
    if mode == 'riverine_ratio':
        # Resample riverine ratio input
        f = interp1d(riverine_age, riverine_ratio)
        Rr_resampled = f(age)

        # Createa [s x nt] matrix of Rr_resampled
        Rr_input = (np.zeros((s, nt)) + Rr_resampled).T

        # Generate random values
        Fr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fr'][0], param['Fr'][1], s))
                
        Fh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fh'][0], param['Fh'][1], s))
        
        Rh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rh'][0], param['Rh'][1], s))
        
        Fsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fsed'][0], param['Fsed'][1], s))
        
        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))
        
        Falt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Falt'][0], param['Falt'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_range, Rr_input, Fh_range, 
                                     Rh_range, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)
        
        # Filter
        Fr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fr_range, 0)
        Rr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rr_input, 0)
        Fh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fh_range, 0)
        Rh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rh_range, 0)
        Fsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fsed_range, 0)
        Dsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dsed_range, 0)
        Falt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Falt_range, 0)
        Dalt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dalt_range, 0)

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }

    # Simulation with predetermined riverine d7Li ratios and fluxes
    if mode == 'riverine_flux_ratio':
        # Resample riverine flux input
        f = interp1d(riverine_age, riverine_flux)
        Fr_resampled = f(age)

        # Createa [s x nt] matrix of Rr_resampled
        Fr_input = (np.zeros((s, nt)) + Fr_resampled).T

        # Resample riverine ratio input
        f = interp1d(riverine_age, riverine_ratio)
        Rr_resampled = f(age)

        # Createa [s x nt] matrix of Rr_resampled
        Rr_input = (np.zeros((s, nt)) + Rr_resampled).T

        # Generate random values            
        Fh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fh'][0], param['Fh'][1], s))
        
        Rh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rh'][0], param['Rh'][1], s))
        
        Fsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fsed'][0], param['Fsed'][1], s))
        
        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))
        
        Falt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Falt'][0], param['Falt'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_input, Rr_input, Fh_range, 
                                     Rh_range, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)
        
        # Filter
        Fr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fr_input, 0)
        Rr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rr_input, 0)
        Fh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fh_range, 0)
        Rh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rh_range, 0)
        Fsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fsed_range, 0)
        Dsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dsed_range, 0)
        Falt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Falt_range, 0)
        Dalt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dalt_range, 0)

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }

# Simulation with predetermined hydrothermal fluxes
    if mode == 'hydrothermal_flux':
        # Resample hydrothermal flux input
        f = interp1d(hydrothermal_age, hydrothermal_flux)
        Fh_resampled = f(age)

        # Createa [s x nt] matrix of Rr_resampled
        Fh_input = (np.zeros((s, nt)) + Fh_resampled).T

        # Generate random values            
        Fr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fr'][0], param['Fr'][1], s))
        
        Rr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rr'][0], param['Rr'][1], s))
        
        Rh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rh'][0], param['Rh'][1], s))

        Fsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fsed'][0], param['Fsed'][1], s))
        
        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))
        
        Falt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Falt'][0], param['Falt'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_range, Rr_range, Fh_input, 
                                     Rh_range, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)
        
        # Filter
        Fr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fr_range, 0)
        Rr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rr_range, 0)
        Fh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fh_input, 0)
        Rh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rh_range, 0)
        Fsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fsed_range, 0)
        Dsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dsed_range, 0)
        Falt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Falt_range, 0)
        Dalt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dalt_range, 0)

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }

# Simulation with predetermined hydrothermal d7Li ratios and fluxes
    if mode == 'hydrothermal_flux_ratio':
        # Resample hydrothermal flux input
        f = interp1d(hydrothermal_age, hydrothermal_flux)
        Fh_resampled = f(age)

        # Createa [s x nt] matrix of Rr_resampled
        Fh_input = (np.zeros((s, nt)) + Fh_resampled).T

        # Resample hydrothermal ratio input
        f = interp1d(hydrothermal_age, hydrothermal_ratio)
        Rh_resampled = f(age)

        # Createa [s x nt] matrix of Rr_resampled
        Rh_input = (np.zeros((s, nt)) + Rh_resampled).T

        # Generate random values            
        Fr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fr'][0], param['Fr'][1], s))
        
        Rr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rr'][0], param['Rr'][1], s))
        
        Fsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fsed'][0], param['Fsed'][1], s))
        
        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))
        
        Falt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Falt'][0], param['Falt'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_range, Rr_range, Fh_input, 
                                     Rh_input, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)
        
        # Filter
        Fr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fr_range, 0)
        Rr_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rr_range, 0)
        Fh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fh_input, 0)
        Rh_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Rh_input, 0)
        Fsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Fsed_range, 0)
        Dsed_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dsed_range, 0)
        Falt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Falt_range, 0)
        Dalt_res = np.where(np.abs(RLisw_ss - target_d7Li)<tolerance, Dalt_range, 0)

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }
    

    return results