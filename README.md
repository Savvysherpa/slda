# lda-cython
This repo contains [Cython](http://cython.org/) implementations of [Gibbs
sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) for [latent Dirichlet
allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and its
supervised variants.

## Installation

### GNU Scientific Library
This module depends on [GSL](http://www.gnu.org/software/gsl/), please install
it. For macosx users using [homebrew](http://brew.sh/), this is as simple as
```bash
$ brew install gsl
```

### pypolyagamma
The `requirements.txt` file contains the python packages this module requires.
One of these, [pypolyagamma](https://github.com/slinderman/pypolyagamma),
causes fits, for two reasons:

 1. The `pypolyagamma` module that is installed using pip does not have Python 3
 support. However, there is a
 [fork](https://github.com/marekpetrik/pypolyagamma) that does. So if you want
 Python 3 support, just clone this
 [fork](https://github.com/marekpetrik/pypolyagamma) and pip install it from the
 repo's main directory, like this:
 ```bash
 $ pip install /path/to/pypolyagamma/
 ```

 1. `pypolyagamma` requies a C/C++ compiler with [OpenMP](http://openmp.org/)
 support. Unfortunately for macosx users, Apple's native compiler, clang, does
 not ship with that support, so you need to install and use one that does. For
 macosx users using [homebrew](http://brew.sh/), this is as simple as
 ```bash
 $ brew install gcc --without-multilib
 ```
 This will install a version of gcc with OpenMP support. However, Apple makes
 things worse by aliasing gcc to point to clang! So you need to explicitly tell
 pip which gcc compiler to use. As of the writing of this README, brew installs
 major version 5 of gcc, and as a result will create a binary called gcc-5 in
 your path. So pip install `pypolyagamma` as follows:
 ```bash
 $ CC=gcc-5 CXX=gcc-5 pip install pypolyagamma
 ```
 or if you are installing directly from code for Python 3 support:
 ```bash
 $ CC=gcc-5 CXX=gcc-5 pip install /path/to/pypolyagamma/
 ```
