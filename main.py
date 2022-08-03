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
import subprocess 
import numpy as np
from netCDF4 import Dataset

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

# ======================================================================
#                  *  WRITE RESULTS ON OUTPUT FILE  *
# ======================================================================
#-----------------------------------------------------------------------
# Set output path and directory.
#-----------------------------------------------------------------------
# Store input path and experiment name.
set_paths = config_parameters['set_paths']
in_path = set_paths['indata_path']
exp_name = set_paths['experiment_name']

# Make output path coherent with input path (wether on 
# windows or linux/mac)
if in_path[-1] == '\\':
    path_end = '\\'
    option = ''
    shell_val = True
else:
    path_end = '/'
    option = '-p'
    shell_val = False

# Create directory for exp outputs within input data directory.
output_dir = exp_name + path_end
subprocess.run(['mkdir', option, in_path + output_dir], shell = shell_val)

#-----------------------------------------------------------------------
# Create output file.
#-----------------------------------------------------------------------
# Create output path and file name
output_path = in_path + output_dir + path_end
output_name = exp_name + '_output.nc' 

# Open output file for writing
out_file = Dataset(output_path + output_name, 'w', format='NETCDF4')

#-----------------------------------------------------------------------
# Create dimensions and store variables.
#-----------------------------------------------------------------------
# Create time, lat, lon, depth dimensions.
depth_len = len(depth_1D)
out_file.createDimension('time', 1)
out_file.createDimension('latitude', 1)
out_file.createDimension('longitude', 1)
out_file.createDimension('depth', depth_len)
# Create equally spaced depth grid for R, Phi.
out_file.createDimension('equally_spaced_depth_grid', len(Phi[:,0]))
del(depth_len)

# Add barotropic mode to number of modes of motion to be considered.
N_motion_modes += 1
# Create mode of motion dimension (only for R and Phi), adding the 
# barotropic one.
out_file.createDimension('mode', N_motion_modes)

# Create time variable.
time_var = out_file.createVariable('time', np.int32,'time')
time_var[:] = 1
time_var.long_name = 'time'
time_var.standard_name = 'time'

# Create latitude variable.
time_var = out_file.createVariable('latitude', np.int32,'latitude')
time_var[:] = 1
time_var.long_name = 'latitude'
time_var.standard_name = 'latitude'

# Create longitude variable.
time_var = out_file.createVariable('longitude', np.int32,'longitude')
time_var[:] = 1
time_var.long_name = 'longitude'
time_var.standard_name = 'longitude'

# Create depth variable.
depth_var = out_file.createVariable('depth', np.float32, 'depth')
depth_var[:] = depth_1D
depth_var.valid_min = min(depth_1D)
depth_var.valid_max = max(depth_1D)
depth_var.unit_long = "meters"
depth_var.units = 'm'
depth_var.long_name = "depth below sea level" 
depth_var.standard_name = "depth" 

# Create depth variable.
new_depth_var = out_file.createVariable('equally_spaced_depth_grid',
                                      np.int32, 'equally_spaced_depth_grid')
new_depth_var[:] = np.linspace(0, len(Phi[:,0])-1, len(Phi[:,0]))
new_depth_var.valid_min = 0
new_depth_var.valid_max = len(Phi[:,0])-1
new_depth_var.grid_step = '1m'
new_depth_var.unit_long = "meters"
new_depth_var.units = 'm'
new_depth_var.long_name = "depth below sea level" 
new_depth_var.standard_name = "depth" 

# Create modes variable.
mode_var = out_file.createVariable('mode', np.int32, 'mode')
mode_var[:] = [i for i in range(N_motion_modes)]
mode_var.long_name = 'modes of motion considered'
mode_var.standard_name = 'modes_of_motion'

# Create vertical mean density variable .   
dens_var = out_file.createVariable('mean_density', np.float32, 
                                    ('time','latitude','longitude','depth'))
dens_var[:] = mean_pot_density
dens_var.unit_long = mean_pot_density.attrs['unit_long']
dens_var.units = mean_pot_density.attrs['units']
dens_var.long_name = mean_pot_density.attrs['long_name']
dens_var.standard_name = mean_pot_density.attrs['standard_name']

# Create Brunt-Vaisala frequency variable.  
BVfreq_var = out_file.createVariable('BVfreq', np.float32, 
                                       ('time','latitude','longitude','depth'))
BVfreq_var[:] = np.sqrt(BV_freq_sq) * 3600
BVfreq_var.unit_long = 'cycles per hour'
BVfreq_var.units = 'cycles/hr'
BVfreq_var.long_name = 'Brunt-Vaisala frequency'
BVfreq_var.standard_name = 'BV_frequency'

# Create baroclinic Rossby radius variable. 
Ross_rad_var = out_file.createVariable('R', np.float32,
                                        ('time','latitude','longitude','mode'))
Ross_rad_var[:] = R
Ross_rad_var.unit_long = 'kilometers'
Ross_rad_var.units = 'km'
Ross_rad_var.long_name = 'Rossby deformation radius'
Ross_rad_var.standard_name = 'Rossby_radius'
    
# Create baroclinic Rossby radius variable. 
Phi_var = out_file.createVariable('phi', np.float32, 
                                  ('time','latitude','longitude',
                                   'equally_spaced_depth_grid', 'mode'))
Phi_var[:] = Phi
Phi_var.unit_long = 'dimensionless'
Phi_var.units = '1'
Phi_var.long_name = 'vertical modes of motion'
Phi_var.standard_name = 'Phi'

# Add modes attribute.
dens_var.modes = '' # no legend for density
BVfreq_var.modes = '' # no legend for BVfreq
# Modes for R & Phi.
modes_array = ['barotropic mode']
for i in range(1, N_motion_modes):
    modes_array.append('%s° baroclinic mode' %(i))
Phi_var.modes = modes_array
Ross_rad_var.modes = modes_array

# Add global attributes to file.
out_file.region = config_parameters['set_paths']['region_name'] 
out_file.period =  config_parameters['set_time']['period_name'] 
out_file.starting_time = config_parameters['set_time']['starting_time']
out_file.ending_time = config_parameters['set_time']['ending_time']
out_file.lat_min = config_parameters['set_domain']['lat_min']
out_file.lat_max = config_parameters['set_domain']['lat_max']
out_file.lon_min = config_parameters['set_domain']['lon_min']
out_file.lon_max = config_parameters['set_domain']['lon_max']

# Close output file.
out_file.close()
