# User Interface
The project is designed for the user to get the results following only a few steps:
1.  Saving the input dataset, containing Potential Temperature and Salinity variables, in a specific directory. The dataset should meet some requirements as described in the [below subsection](#dataset-requirements).
2. Editing the [JSON configuration file](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/config.json) as described in the [following subsection](#json-configuration-file).
3. Running the [main.py](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/main.py) script with the config file name as argument.
```
python main.py config.json
```

Eventually, the output products will be located in a directory created within the same one as the input dataset. The returned products are:

- a NetCDF file containing the vertical profiles of:
	- mean Potential Density
	-  mean Brunt-Vaisala frequency
	- baroclinic Rossby radius 
	- vertical structure function (both baroclinic and barotropic). 
- a plot of the vertical profile for each variable in the NetCDF output file.

>#### Note:
>The software may be run on Windows system, too. In that case, please set the input dataset path as shown within the configuration file and make sure the necessary packages are already installed on Windows.
>The mask file containing the bathymetry has to be defined on the same lat/lon grid as the input dataset!

### JSON Configuration File
The [JSON configuration file](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/config.json) is structured as a list of subdictionaries called **sections**. Each *section* consists of a **section title** and a list of **items**. Each *item* is characterized by a *name*, a *value*, a *type* and a *description* for helping the user understand the *item* function. 

>**WARNING**: 
>The only values to be edited are the one corresponding to the *item values*. Do not edit the *keys* themselves or the *item type*! 
All values within the config file has to be written as *strings*. 

The role of each section is described in the following scheme. 
```
CONFIGURATION FILE
|   "project_name": the name given to this project. It is printed while running the main script.
|   "sections"  
|       └─── "set paths":
|       |         └─── "experiment name": name to be given to the experiment. It will characterize the output products name.
|       |         └─── "region name": name of the computation domain along which Pot. Temperature and Salinity are averaged in latitude and longitude. It will appear among the output file attributes.
|       |         └─── "input file name": name of the dataset file containing Pot. Temperature and Salinity. The file must be of NetCDF format.
|       |         └─── "bathy file name": name of the NetCDF file containing the bathymetry. 
|       |         └─── "indata path": path to the dataset file named as in "input file name". It must end with '/' (if on linx or mac) or '\\' (if on windows). Please, if on windows substitute backslash with '\\'.
|       └─── "set variables": name of the variables as in the NetCDF input file. These will be the keys used for extracting variables *value* from the input file.
|       |         └─── "temperature name": name of Potential Temperature variable.
|       |         └─── "salinity name": name of Salinity variable.
|       |         └─── "lat var name": name of latitude variable (may have different key name respect to latitude dimension).
|       |         └─── "lon var name": name of longitude variable (may have different key name respect to longitude dimension).
|       |         └─── "depth var name": name of depth variable (may have different key name respect to depth dimension).
|       |         └─── "time var name": name of time variable (may have different key name respect to time dimension).
|       |         └─── "bathy var name": name of bathymetry variable within NetCDF bathymetry file.
|       └─── "set dimensions": name of the dimensions as in the NetCDF input file. These will be the keys used for extracting variables *dimension* from the input file.
|       |         └─── "lat name": latitude dimension name
|       |         └─── "lon name": longitude dimension name
|       |         └─── "depth name": depth dimension name
|       |         └─── "time name": time dimension name
|       └─── "set domain": values for defining the region of interest to the user. **NOTE: They must be strings convertible to float!**
|       |         └─── "lat min": minimum latitude value which borders the domain to the south. 
|       |         └─── "lat max": maximum latitude value which borders the domain to the north. 
|       |         └─── "lon min": minimum longitude value which borders the domain to the west. 
|       |         └─── "lon max": maximum longitude value which borders the domain to the east. 
|       |         └─── "lat step": latitude grid step within the input dataset file. It defines the precision with which the extremant values are seeked within the latitude array.
|       |         └─── "lon step": longitude grid step within the input dataset file. It defines the precision with which the extremant values are seeked within the longitude array.
|       |         NOTE: setting min value = max value corresponds to taking punctual profiles.
|       └─── "set time":
|       |         └─── "period name": name of the period during which Pot. Temperature and Salinity are averaged in time. It will appear in the output plots and among the output file attributes. 
|       |         └─── "starting time": starting value of period during which the time average take place. **NOTE:It must be a string convertible to pandas datetime format.**
|       |         └─── "ending time": ending value of period during which the time average take place. **NOTE:It must be a string convertible to pandas datetime format.**
|       |         NOTE: setting starting value = ending value corresponds to taking instantaneous profiles.
|       └─── "set modes": 
|       |         └─── "n modes": number of modes of motion for which the Rossby radius and the vertical structure function should be computed. The barotropico mode is always included as the 0th mode.
|
END
```

### Dataset Requirements
Coherently with the [configuration file](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/config.json) and the [functions](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/OBM/read_input.py) implemented for data reading, the input dataset should meet some requirements:

- It should be of **NetCDF** format.
- It should contain the following variables, although named differently:
	- Potential Temperature , Salinity, Latitude, Longitude, Depth, Time.
-  Potential Temperature and Salinity variables should depend on dimensions (although not in the following order):
	- Time, Depth, Latitude, Longitude.
- Potential Temperature and Salinity should have same sizes along each dimension.

**Note**: The input data may be just punctual instantaneous vertical profiles (i.e. depending only on depth). However the presence of time, latitude and longitude dimensions is necessary in order to correctly extract the data.