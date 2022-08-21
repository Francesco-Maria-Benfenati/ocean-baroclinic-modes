# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 20:14:47 2022

@author: Francesco Maria
"""
import sys
import xarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')

# ======================================================================
#               *  SAVE PLOTS WITHIN THE OUTPUT DIRECTORY  *
# ======================================================================
# Store path to output NetCDF file, INCLUDING FILE NAME.
outdata_path = sys.argv[1]
outdata_file_name = sys.argv[2]

# Open output data file and store variables.
data_to_plot = xarray.open_dataset(outdata_path + '/' + outdata_file_name)

depth = data_to_plot.variables['depth']
new_depth = data_to_plot.variables['nondim_depth_grid']
time = data_to_plot.variables['time']
variables_to_plot = [data_to_plot.variables['mean_density'], 
                     data_to_plot.variables['BVfreq'],
                     data_to_plot.variables['R'],
                     data_to_plot.variables['phi']          ]

# -------------------------------------- 
# Functions defined on 'depth_1D' grid.
# --------------------------------------
for var in variables_to_plot[:2]:
    fig, ax = plt.subplots(figsize=(8, 8))
    for column in var[0,0,:]:
        ax.plot(column, depth, linewidth = 1.5)
    ax.grid(alpha = 0.5, color='gray', linestyle='--')
    ax.set_xlabel('%s (%s)\n %s, %s' %(var.attrs['long_name'], 
                                        var.attrs['units'],
                                        data_to_plot.attrs['region'], 
                                        data_to_plot.attrs['period']), 
                                                labelpad = 15, fontsize = 12,
                                                fontweight = 'bold'          )
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel('depth (%s)' %(depth.attrs['units']), 
                                                  fontsize = 12, labelpad = 10)
    ax.invert_yaxis()
    if var.attrs['modes'] != '':
        ax.legend(var.attrs['modes'], loc='lower right')
    fig.savefig(outdata_path + '/%s.png' %(var.attrs['standard_name']), 
                                                          bbox_inches='tight')
    
# -------------------------------------------------------
# Functions defined on 'equally_spaced_depth_grid' grid.
# -------------------------------------------------------
for var in variables_to_plot[3:]:
    fig, ax = plt.subplots(figsize=(8, 8))
    for column in var[0,0,:]:
        ax.plot(column, new_depth, linewidth = 1.5)
    ax.grid(alpha = 0.5, color='gray', linestyle='--')
    ax.axvline(x=0, color = 'k', linewidth = 0.5)
    ax.set_xlabel('%s (%s)\n %s, %s' %(var.attrs['long_name'], 
                                        var.attrs['units'],
                                        data_to_plot.attrs['region'], 
                                        data_to_plot.attrs['period']), 
                                        labelpad = 15, fontsize = 12, 
                                        fontweight = 'bold'          )
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel('depth (%s)' %(new_depth.attrs['units']), 
                                                 fontsize = 12, labelpad = 10)
    ax.invert_yaxis()
    if var.attrs['modes'] != '':
        ax.legend(var.attrs['modes'], loc='lower right')
    fig.savefig(outdata_path + '/%s.png' %(var.attrs['standard_name']), 
                                                          bbox_inches='tight')
    
# -------------------------------------- 
# Baroclinic Rossby radius (TABLE) .
# --------------------------------------
# Create figure and axis.
fig, ax = plt.subplots(figsize=(4, 4))

# Define useful attributes from output file.
long_name = data_to_plot.variables['R'].attrs['long_name']
units = data_to_plot.variables['R'].attrs['units']
rows = data_to_plot.variables['R'].attrs['modes']

# Define table colors.
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
colors = colors[::-1]

# Create text for the table.
Rossby_rad = data_to_plot.variables['R'][0,0,0,:].values
cell_text = []
for R in Rossby_rad[:]:
    cell_text.append([R])


# Add table.
ax.set_axis_off()
col_label = long_name + ' (%s)' %(units)
the_table = ax.table(cellText = cell_text,
                     rowColours=colors,
                     colLabels=[col_label],
                     rowLabels = rows,
                     colLoc='left',
                     rowLoc='center',
                     loc = 'center')

# Add title.
ax.set_title('%s (%s)\n %s, %s'
                             %(data_to_plot.variables['R'].attrs['long_name'], 
                               data_to_plot.variables['R'].attrs['units'],
                               data_to_plot.attrs['region'], 
                               data_to_plot.attrs['period']    ), 
                               fontsize = 12, 
                               fontweight = 'bold', loc = 'center'        )
# Save figure.
fig.savefig(outdata_path + '/test.png', bbox_inches='tight')

# ----------------------------------------------------------------------
# Close output data file.
data_to_plot.close()
