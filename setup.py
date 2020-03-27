"""
__author__: bishwarup
created: Wednesday, 25th March 2020 11:17:40 pm
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="utils",
    ext_modules=cythonize("./utils/*.pyx"),
    include_dirs=[np.get_include()],
)
