# [ocean-baroclinic-modes-1.0](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0)

This project is for computing the *baroclinic vertical structure function* and the *baroclinic Rossby radius* for  each mode of motion, in an ocean region of interest to the user.
### Table of contents
1. [Baroclinic Modes](#baroclinic-modes)
2. [Numerical Method Implemented](#numerical-method-implemented)
3. [Project Structure](#project-structure)
4. [User Interface](#user-interface)
	- [JSON configuration file](#json-configuration-file)
	- [Dataset Requirements](#dataset-requirements)
5. [Get Started!](#get-started)

## Baroclinic Modes 
[^1] Ocean vertical stratification is a complex issue which may be studied through the analysis of the **dynamic baroclinic modes of motion** and their respective **baroclinic Rossby deformation radii**.
For deriving these quantities, we start from the linearized *QuasiGeostrophic (QG)* equation of motion, describing the motion of the field geostrophic component resulting from the balance between pressure and Coriolis forces:

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big\[\frac{\partial^2p_0}{\partial&space;x^2}&plus;\frac{\partial^2p_0}{\partial&space;y^2}&plus;\frac{\partial}{\partial&space;z}\Big(\frac{1}{S}\frac{\partial&space;p_0}{\partial&space;z}\Big)\Big\]&plus;\beta\frac{\partial&space;p_0}{\partial&space;x}=0\quad,\qquad(1))

with Boundary Conditions

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big(\frac{\partial&space;p_0}{\partial&space;z}\Big)&space;=&space;0\text{&space;at&space;}z=0,1\quad;)

where $p_0$ is the pressure term expanded at the first order of the Rossby number $\epsilon$, while $\beta$ is the *beta Rossby number*. *S* is the stratification parameter, defined as

![equation](https://latex.codecogs.com/gif.image?\dpi{100}S(z)=\frac{N^2(z)H^2}{f_0^2L^2}\quad,\qquad(2))

with the Brunt-Vaisala frequency *N* obtained as

![equation](https://latex.codecogs.com/gif.image?\dpi{100}N&space;=&space;\Big\[-\frac{g}{\rho_0}\frac{\partial&space;\rho_s}{\partial&space;z}\Big]^\frac{1}{2}\quad(\rho_s\text{basic&space;mean&space;stratification),}\qquad(3)) 

where $\rho_0=1025 kg/m^3$ is the reference density value.
Here, *L* and *H* are respectively the horizontal and vertical scales of motion, assuming $L = 100 km$ and H equal to the region maximum depth ( $O(10^3 km)$ ). The *Coriolis parameter* is assumed $f_0= 10^{-4} 1/s$ while the gravitational acceleration is $g=9.806 m/s^2$.
If we consider a solution to the QG equation of type 

![equation](https://latex.codecogs.com/gif.image?\dpi{100}p_0(x,y,z,t)=f(x,y,t)\Phi(z)\quad\Big\(f=\Re\Big\[e^{i(kx+ly-\sigma{t})}\Big\]\Big\))

where $\Phi(z)$ is the _vertical structure function_, it follows

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big\[\frac{\partial^2f}{\partial&space;x^2}&plus;\frac{\partial^2f}{\partial&space;y^2}-\lambda\Big\]&plus;\beta\frac{\partial&space;f}{\partial&space;x}=0\quad\quad,\quad\lambda=-\Big\[\frac{\beta{k}}{\sigma}+k^2+l^2\Big\]\quad.)

This necessary leads to the eigenvalues/eigenvector equation

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}\qquad(4))

where the eigenvectors $\Phi_n$ are the vertical structure functions, each corresponding to a mode of motion $n=0,1,2\dots$, while the eigenvalues $\lambda_n$ may be used for computing the *baroclinic Rossby radius*

![equation](https://latex.codecogs.com/gif.image?\dpi{110}R_n&space;=&space;\frac{1}{\sqrt{\lambda_n}}\qquad(5))

for each mode *n*. As shown in Grilli, Pinardi (1999) [^1], eigenvalues are null or positive and negative values should be rejected. The trivial eigenvalue $\lambda=0$ corresponds to the **barotropic mode** $(\Phi_0(z)=1)$, while $\lambda=1,2\dots$ correspond to the **baroclinic modes** $(\Phi_{1,2\dots}(z))$. 

This project aims to compute the *baroclinic Rossby radius* and the *vertical structure function* for each mode of motion.

[^1]: F. Grilli, N. Pinardi (1999), "_Le Cause Dinamiche della Stratificazione Verticale nel Mediterraneo_".

## Numerical Method Implemented
Unless the case of particular Brunt-Vaisala frequency profiles (see LaCasce, 2012 [^2]), the above [eigenvalues/eigenvectors problem](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}) may not be analytically solved for a realistic one. Thus, it is necessary to solve it numerically  employing a change of variable:

![equation](https://latex.codecogs.com/gif.image?\dpi{110}w=\frac{1}{S}\frac{d\Phi}{dz}\\\\\Rightarrow\Phi&space;=\int_{0}^{z}Swdz+\Phi_0\quad\Big\(\Phi_0=\Phi\vert_{z=0}=const\Big\)\qquad(6))

so that

![equation](https://latex.codecogs.com/gif.image?\dpi{110}\frac{dw}{dz}=-\lambda\Phi=-\lambda\Big\(\int_{0}^{z}Swdz+\Phi_0\Big\)\\\\\Rightarrow\boxed{\frac{d^2w}{dz^2}=-\lambda{S}{w}}\quad\Big\(B.C.\quad{w=0}\quad{at}\quad{z=0,1}\Big\)\qquad(7))

obtaining a simple eigenvalues/eigenvectors problem of known resolution. 
The numerical method implemented here aims to find the eigenvalues  and eigenvectors in eq. [(4)](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}), exploiting relation [(6)](https://latex.codecogs.com/gif.image?\dpi{110}\Phi&space;=\int_{0}^{z}Swdz+\Phi_0) and numerically solving the well known problem [(7)](https://latex.codecogs.com/gif.image?\dpi{110}\frac{d^2w}{dz^2}=-\lambda{S}{w}) for $\lambda_n$ and $w_n$. This is done through few steps, starting from the Brunt-Vaisala frequency vertical profile:
1. The Brunt-Vaisala frequency is linearly interpolated on a new equally spaced  depth grid (1 m grid step).
2. Problem parameter S (depth-dependent) is computed as in eq. [(2)](https://latex.codecogs.com/gif.image?\dpi{100}S(z)=\frac{N^2(z)H^2}{f_0^2L^2}\quad,).
3. The *left* finite difference matrix  corresponding to operator $\frac{d^2}{dz^2}$ and the *right* diagonal matrix related to *S* are computed. 
The eigenvalues **discretized problem** is solved:

![equation](https://latex.codecogs.com/gif.image?\dpi{110}\frac{1}{12dz^{2}}\begin{bmatrix}0&0&\0&0&0&0&\dots&0\\\12&-24&12&0&\dots&0&\dots&0\\\\-1&16&-30&16&-1&0&\dots&0\\\0&-1&16&-30&16&\dots&\dots&0\\\\\vdots&\ddots&\ddots&\ddots&\ddots&\dots&\dots&\vdots\\\0&\dots&\dots&0&0&12&-24&12&space;\\\0&\dots&0&0&0&0&0&0\end{bmatrix}\begin{bmatrix}w_0\\\w_1\\\w_2\\\\\vdots\\\\\vdots\\\w_{n-1}\\\w_n\end{bmatrix}=-\lambda\begin{bmatrix}S_0&0&0&0&\dots&\dots&0\\\0&S_1&0&0&\dots&\dots&0\\\0&0&S_2&0&\dots&\dots&0\\\0&0&0&S_3&\dots&\dots&0\\\\\vdots&\ddots&\ddots&\ddots&\ddots&\ddots&\vdots\\\0&\dots&\dots&\dots&0&S_{n-1}&0\\\0&\dots&\dots&\dots&0&0&S_n\end{bmatrix}\begin{bmatrix}w_0\\\w_1\\\w_2\\\\\vdots\\\\\vdots\\\w_{n-1}\\\w_n\end{bmatrix})

where *n* is the number of points along depth axis, *dz* is the scaled grid step. Boundary Conditions are implemented setting the first and last lines of the finite difference matrix (L.H.S.) equal to 0.

4. Eigenvectors are found integrating eq. [(7)](https://latex.codecogs.com/gif.image?\dpi{110}\frac{d^2w}{dz^2}=-\lambda{S}{w}\quad\Big\(B.C.\quad{w=0}\quad{at}\quad{z=0,1}\Big\)) through *Numerov's* numerical method

![equation](https://latex.codecogs.com/gif.image?\dpi{110}w_{n&plus;1}=\left(\frac{2-\frac{10\Delta{t^2}}{12}\lambda{S_n}}{1+\frac{\Delta{t^2}}{12}{\lambda}S_{n&plus;1}}\right)w_n-\left(\frac{1+\frac{\Delta{t^2}}{12}{\lambda}S_{n-1}}{1+&space;\frac{\Delta{t^2}}{12}{\lambda}S_{n&plus;1}}&space;\right)w_{n-1})

where each eigenvalue is used for computing the corresponding eigenvector.
The first value of each eigenvector is computed as

![equation](https://latex.codecogs.com/gif.image?\dpi{110}w_{1}=\frac{\Delta{z}\frac{dw}{dz}\vert_{z=0}}{(1&plus;\lambda\frac{S_1\Delta&space;z^2}{6})}\quad\text{with}\quad\frac{dw}{dz}\vert_{z=0}=-\lambda\Phi_0\quad\Big(\Phi_0=\Phi\vert_{z=0}=1\Big))

where $\Phi_0$ is the surface value, equal to the modes maximum amplitude (and equal to the barotropic mode value). Here, it is set equal to 1.

5. The *baroclinic Rossby radii* $R_n$ are computed as in eq. [(5)](https://latex.codecogs.com/gif.image?\dpi{110}R_n&space;=&space;\frac{1}{\sqrt{\lambda_n}}) while the *vertical structure functions* are obtained integrating *S, w* as in [(6)](https://latex.codecogs.com/gif.image?\dpi{110}\Phi&space;=\int_{0}^{z}Swdz+\Phi_0). The integration constant is set $\Phi_0=1$, as already discussed.

[^2]: J. H. LaCasce (2012), "_Surface Quasigeostrophic Solutions and Baroclinic Modes with Exponential Stratification_".


# Project Structure
The project is divided into blocks, each one with a different function:

- A [JSON configuration file](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/config.json) is provided to the user in order to set values and parameters for carrying out the computation.
- In file [read_input.py](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/OBM/read_input.py), few functions are embedded, for reading the configuration file and the input dataset. In fact, the user is aked to supply the Salinity and Potential Temperature vertical profiles for a region (or a point), and a period (or a time instant). Through one of these functions, the input variables are extracted from the dataset and averaged (if necessary) in time, latitude and longitude. 
- In file [eos.py](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/OBM/eos.py), the Eq. Of Seawater (EOS) is implemented as in [NEMO](https://www.nemo-ocean.eu/), for computing Potential Density from Salinity and Potential Temperature. Furthermore, a function is implemented here for computing the mean Brunt-Vaisala frequency vertical profile following eq. [(3)](https://latex.codecogs.com/gif.image?\dpi{100}N&space;=&space;\Big\[-\frac{g}{\rho_s}\frac{\partial&space;\rho_s}{\partial&space;z}\Big]^\frac{1}{2}\quad(\rho_s\text{basic&space;mean&space;stratification)), after averaging Pot. Density in latitude and longitude.
- File [baroclinic_modes.py](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/OBM/baroclinic_modes.py) contains the function which implements the numerical method described in the [previous section](#numerical-method-implemented), returning the *Rossby radius* and the *vertical structure function* for a number of modes of motion set by the user (including the barotropic one).
- The directory [testings](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/tree/main/testings) contains the testing files, one for each functions block described above, and a fake configuration file only for testing purpose.
- The main file [main.py](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/main.py) recalls the so far mentioned functions, in order to compute the $R_n$ and $\Phi_n$ profiles. Furthermore, the output data are written to a NetCDF file. 
- File [run_plots.py](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/run_plots.py) is for plotting output results in NetCDF output file. Output products are saved in a directiory within the same one as the input dataset.
- The [requirements](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/pkgs_required.txt) file contains libraries and packages to be installed for running the main file. See the ["Get Started"](#get-started) section for how to install them.
- In folder ["test_case"](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/tree/main/test_case) a test case is provided to the user as a Get Started! material (see below [section](#get-started)).
- Lastly, a brief [PDF presentation](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/introducing_OBM-1.0.pdf) is provided, recapping part of the contents in the above sections. Moreover, numerical results are shown in comparison to the analytical ones obtained by LaCasce (2012) [^2] in case of constant or exponential Brunt-Vaisala frequency profiles.

## User Interface
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

# Get Started!

1. For using the software, clone the repository [ocean-baroclinic-modes-1.0](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0)  and use pip to install the required packages.

	```
	git clone https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0
	cd ocean-baroclinic-modes-1.0
	pip install -r pkgs_required.txt
	```

2. Now, a TEST CASE is provided to you for a better understanding of how to use the software. Before proceeding, you need to create a *CMEMS account*. You may find an easy way [here](https://resources.marine.copernicus.eu/registration-form).
3.  Get into the *test_case* directory and download your *test_case dataset* through the appropriate script. This may take a while. Your CMEMS username and password will be asked to you.
	```
	cd test_case
	python download_test_case_dataset.py
	```
>**IMPORTANT**: YOUR USERNAME AND PASSWORD WILL NOT BE SAVED!
4. For running the software you may just get into the software directory again and run the main script.
	```
	cd ..
	python main.py test_case/config_test_case.json
	```
5. Lastly, the **output products** will be within your dataset directory, in a sub-directory named as your experiment. Here, you will find the output file. You can esplore the output file using *ncdump* utility.
	```
	cd test_case/dataset_azores/Azores_JAN21
	ncdump -h Azores_JAN21_output.nc
	```
6. For plotting the output variables, you may just run the _run_plots.py_ script with the output products directory as first argument and the output file name as second argument.
	```
	cd ../../..
	python run_plots.py test_case/dataset_azores/Azores_JAN21 Azores_JAN21_output.nc
	```
### Enjoy the project!
