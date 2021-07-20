import setuptools
from setuptools import find_packages

setuptools.setup(
    name='thesis',
    version='0.1.0',
    packages=find_packages('.', exclude=('harmonic_oscillator', 'diode_clipper', 'phaser', 'thesis'))
)
