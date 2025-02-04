# Get Started!

1. Clone the repository and use pip to install the required packages.

    ```
    git clone git@github.com:Francesco-Maria-Benfenati/qgbaroclinic.git
    cd qgbaroclinic
    pip install -r requirements.txt
    ```

2. Now, modify the configuration file _config.toml_ to your needs and run the software!

    ```
    python qgbaroclinic
    ```
    If you want to specify the configuration file path and the number of processors to be used in the computation, please use the following arguments:
    ```
    python qgbaroclinic -c 'path-to-config' -p 'n'
    ```
3. You can also find useful examples about how to import the software as a package, in your python scripts. You may find examples for computing the [mean value](examples/region_mean.ipynb) or [compute a 2D map](examples/2D_map.ipynb) in an area of interest.
## Enjoy the project!
