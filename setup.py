from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import cython_gsl

extensions = [
    Extension('lda_cython._topic_models', ['lda_cython/_topic_models.pyx'],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=[numpy.get_include(), cython_gsl.get_include()]),
]

setup(
    name="Topic Models",
    ext_modules=cythonize(extensions),
)
