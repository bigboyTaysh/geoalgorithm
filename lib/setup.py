from setuptools import setup
from Cython.Build import cythonize
import numpy
import math

setup(
    ext_modules = cythonize(["lib/*.pyx"]),
    include_dirs=[numpy.get_include()]
)