from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import scipy.special

my_integrals = [Extension('slowquant.molecularintegrals.MIcython',['slowquant/molecularintegrals/MIcython.pyx'])]

setup(ext_modules=cythonize(my_integrals), include_dirs=[numpy.get_include(), scipy.special.get_include()])
