from setuptools import setup

setup(
    name = 'qgbaroclinic',
    version = '0.1.0',
    packages = ['qgbaroclinic'],
    install_requires=[ 'plumbum',],
    entry_points = {
        'console_scripts': [
            'qgbaroclinic = __main__:QGBaroclinci',
        ]
    })

####