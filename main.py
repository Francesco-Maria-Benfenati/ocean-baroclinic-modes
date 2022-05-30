# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:10:28 2022

@author: Francesco M. Benfenati
"""
# ======================================================================
# MAIN FILE recalling the functions implemented for computing the 
# BAROCLINIC ROSSBY RADIUS MEAN VERTICAL PROFILE in a defined region.
# ======================================================================
import sys

from OBM import eos
import OBM.read_input as read
from OBM.eos import compute_BruntVaisala_freq_sq as compute_BVsq
from OBM.baroclinic_modes import compute_barocl_modes as modes

# ======================================================================
#    *  READ CONFIG FILE AND STORE VARIABLES FROM NetCDF INPUT FILE  *
# ======================================================================
# Store config file name
config_file_name = sys.argv[1]

# Store configuration parameters from JSON config file.
config_parameters = read.read_JSON_config_file(config_file_name,
                                         section_key = 'sections',
                                         title_key = 'title',
                                         items_key = 'items',
                                         name_key = 'name',
                                         type_key = 'type', 
                                         value_key = 'value'      )

# Store depth, Potential Temperature & Salinity from NetCDF input file.
[depth, pot_temperature, 
        salinity] = read.extract_data_from_NetCDF_input_file(config_parameters)

# ======================================================================
# * COMPUTE POTENTIAL DENSITY, BRUNT-VAISALA FREQUENCY & ROSSBY RADIUS *
# ======================================================================
# Compute Potential Density from Pot. Temperature & Salinity.
pot_density = eos.compute_density(depth, pot_temperature, salinity)

# Make 3D depth a 1D array.
depth_1D = depth.values[:,0,0]

# Compute Brunt-Vaisala frequency squared from Potential Density.
# Store vertical mean potential density as used within the function
# while computing Brunt-Vaisala frequency squared.
[BV_freq_sq, mean_pot_density] = compute_BVsq(depth_1D, pot_density)

# ----------------------------------------------------------------
# N° of modes of motion considered (including the barotropic one).
N_motion_modes = config_parameters['set_modes']['n_modes']
# ----------------------------------------------------------------

# Compute baroclinic Rossby radius mean vertical profile and 
# modes of motion Phi(z).
R, Phi = modes(depth_1D, BV_freq_sq, N_motion_modes)
