"""
Created on Tue Mar 29 14:59:08 2022

@author: Francesco Maria
"""
# ======================================================================
# This script is for downloading the test-case dataset.
# ======================================================================
import os
import subprocess
import datetime
from getpass import getpass

# ----------------------------------------------------------------------
#                           DEFINE TIME PERIOD (January 2021).
# ----------------------------------------------------------------------
starting_time = datetime.datetime(2021,1,1,12) # '2021-01-01 12:00:00'
ending_time = datetime.datetime(2021,1,31,12) # '2021-01-31 12:00:00'
period_name = 'Jan2021'

# ----------------------------------------------------------------------
#                               DEFINE REGIONS OF INTEREST.
# ----------------------------------------------------------------------
azores = {
  'name': 'AZORES',
  'lon_min': '-38',
  'lon_max': '-18',
  'lat_min': '30',
  'lat_max': '50'}

# Insert regions of interest within an array.
regions = [azores]

# ----------------------------------------------------------------------
#           CREATE DIRECTORY FOR EACH REGION YOU ARE INTERESTED IN.
# ----------------------------------------------------------------------
for reg in regions:
    subprocess.run(['mkdir', '-p', 'dataset_'+reg['name']])

# ----------------------------------------------------------------------
#                        DOWNLOAD DATA FROM CMEMS.
# ----------------------------------------------------------------------
# Take username and password as input from terminal. 
user = input('Username: ')
pwd = getpass()

# Download data.
for reg in regions:
    subprocess.run(['python', '-m', 'motuclient', '--motu',  
                    'https://nrt.cmems-du.eu/motu-web/Motu',
                    '--service-id', 
                    'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS',
                    '--product-id', 
                    'global-analysis-forecast-phy-001-024',
                    '--longitude-min', reg['lon_min'],
                    '--longitude-max', reg['lon_max'],
                    '--latitude-min', reg['lat_min'], 
                    '--latitude-max', reg['lat_max'],
                    '--date-min', 
                    starting_time.strftime("%Y-%m-%d %X"), 
                    '--date-max', 
                    ending_time.strftime("%Y-%m-%d %X"),
                    '--depth-min', '0.494', 
                    '--depth-max', '5727.917',
                    '--variable', 'so', '--variable', 'thetao',  
                    '--out-dir', 'dataset_'+reg['name'],
                    '--out-name', 
                    reg['name']+'_'+period_name+'.nc',
                    '--user', user, '--pwd', pwd
                    ])
            
# ----------------------------------------------------------------------
# CHECK DATA HAVE BEEN DOWNLOADED, DOWNLOAD AGAIN MISSING DATA.
# ----------------------------------------------------------------------
for reg in regions:
    for file in os.listdir('dataset_'+reg['name']+'/'):
        if file.startswith(reg['name']+'_'+period_name): 
            pass
        else: 
            print('Going to RE-DOWNLOAD missing data!')
            subprocess.run(['python', '-m', 'motuclient', '--motu',  
                            'https://nrt.cmems-du.eu/motu-web/Motu',
                            '--service-id', 
                            'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS',
                            '--product-id', 
                            'global-analysis-forecast-phy-001-024',
                            '--longitude-min', reg['lon_min'], 
                            '--longitude-max', reg['lon_max'],
                            '--latitude-min', reg['lat_min'], 
                            '--latitude-max', reg['lat_max'],
                            '--date-min', 
                            starting_time.strftime("%Y-%m-%d %X"), 
                            '--date-max', 
                            ending_time.strftime("%Y-%m-%d %X"),
                            '--depth-min', '0.494', 
                            '--depth-max', '5727.917',
                            '--variable', 'so', '--variable', 'thetao',  
                            '--out-dir', 'dataset_'+reg['name'],
                            '--out-name', 
                            reg['name']+'_'+period_name+'.nc',
                            '--user', user, '--pwd', pwd])
