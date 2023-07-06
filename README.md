# [ocean QG baroclinic modes](https://github.com/Francesco-Maria-Benfenati/qgbaroclinic)

This project is for computing the Quasi-Geostrophic (QG) *baroclinic vertical structure function* and the *baroclinic Rossby radius* for  each mode of motion, in an ocean region of interest to the user.

## Table of contents
1. [Baroclinic Modes: theoretical background](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/doc/theoretical_background.md)
2. [Numerical Method Implemented](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/doc/numerical_method.md)
3. [Project Structure](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/doc/project_structure.md)
4. [User Interface](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/doc/user_interface.md)
	- [JSON configuration file](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/doc/user_interface.md#json-configuration-file)
	- [Dataset Requirements](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0/blob/main/doc/user_interface.md#dataset-requirements)

# Get Started!

1. For using the software, clone the repository [ocean-baroclinic-modes-1.0](https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0)  and use pip to install the required packages.

	```
	git clone https://github.com/Francesco-Maria-Benfenati/ocean-baroclinic-modes-1.0
	cd ocean-baroclinic-modes-1.0
	pip install -r requirements.txt
	```

2. Now, a TEST CASE is provided to you for a better understanding of how to use the software. Before proceeding, you need to create a *CMEMS account*. You may find an easy way [here](https://resources.marine.copernicus.eu/registration-form).
3.  Get into the *test_case* directory and download your *test_case dataset* through the appropriate script. This may take a while. Your CMEMS username and password will be asked to you.
	```
	cd data/test_case
	python download_test_case_dataset.py
	```
>**IMPORTANT**: YOUR USERNAME AND PASSWORD WILL NOT BE SAVED!
4. For running the software you may just get into the software directory again and run the main script.
	```
	cd ..
	python src/main.py data/test_case/config_test_case.json
	```
5. Lastly, the **output products** will be within your dataset directory, in a sub-directory named as your experiment. Here, you will find the output file. You can esplore the output file using *ncdump* utility.
	```
	cd data/test_case/dataset_azores/Azores_JAN21
	ncdump -h Azores_JAN21_output.nc
	```
6. For plotting the output variables, you may just run the _run_plots.py_ script with the output products directory as first argument and the output file name as second argument.
	```
	cd ../../..
	python src/plot/run_plots.py test_case/dataset_azores/Azores_JAN21 Azores_JAN21_output.nc
	```
## Enjoy the project!
