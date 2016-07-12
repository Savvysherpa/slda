import os

from setuptools import setup
from setuptools.extension import Extension

try:
    import numpy as np
    import cython_gsl
except ImportError:
    print("Please install numpy and cythongsl.")

# Dealing with Cython
USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension('slda._topic_models', ['slda/_topic_models' + ext],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=[np.get_include(), cython_gsl.get_include()],),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='slda',
    version='0.1.6',
    description='''Cython implementations of Gibbs sampling for latent
                   Dirichlet allocation and its supervised variants''',
    author='Berton Earnshaw, Mimi Felicilda',
    author_email='bearnshaw@savvysherpa.com, lfelicilda@savvysherpa.com',
    url='https://github.com/Savvysherpa/slda',
    license="MIT",
    packages=['slda'],
    ext_modules=extensions,
    install_requires=[
        'Cython >= 0.20.1',
        'cythongsl',
        'numpy',
        'pypolyagamma',
        'pytest',
        'scikit-learn',
        'scipy',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',],
    keywords=['lda', 'slda', 'supervised', 'latent', 'Dirichlet', 'allocation'],
    platforms='ALL',
)
