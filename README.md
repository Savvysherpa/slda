# slda
This repository contains [Cython](http://cython.org/) implementations of [Gibbs
sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) for [latent Dirichlet
allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and
various supervised LDAs:

- supervised LDA (linear regression)
- binary logistic supervised LDA (logistic regression)
- binary logistic hierarchical supervised LDA (trees)
- generalized relational topic models (graphs)

[![Build Status](https://travis-ci.org/Savvysherpa/slda.png)](https://travis-ci.org/Savvysherpa/slda)

## Installation

### The easy way
Use the conda-forge version [here](https://github.com/conda-forge/slda-feedstock).

### The hard way...
(Kept for posterity's sake.)

### Dependencies

#### GNU Scientific Library
This module depends on [GSL](http://www.gnu.org/software/gsl/), please install
it. For macosx users using [homebrew](http://brew.sh/), this is as simple as
```bash
$ brew install gsl
```

#### pypolyagamma-3 and gcc
This package depends on [pypolyagamma-3](https://github.com/Savvysherpa/pypolyagamma),
which is a bit of a pain because `pypolyagamma-3` requies a C/C++ compiler with
[OpenMP](http://openmp.org/) support. Unfortunately for macosx users, Apple's native
compiler, clang, does not ship with that support, so you need to install and
use one that does. For macosx users using [homebrew](http://brew.sh/),
this is as simple as:
 ```bash
 $ brew install gcc --without-multilib
 ```
This will install a version of `gcc` with OpenMP support. However, Apple makes
things worse by aliasing gcc to point to clang! So you need to explicitly tell
the shell which gcc compiler to use. As of the writing of this README, brew
installs major version 6 of gcc, and as a result will create a binary called
gcc-6 in your path. So export the following to your shell
 ```bash
 $ export CC=gcc-6 CXX=g++-6
 ```
or you can prefix the commands below with `CC=gcc-6 CXX=g++-6`.

As a result of this export, it may turn out that your shell cannot find the
libraries associated with gcc. If this is the case, specify the path to your gcc
library in the environment variable `DYLD_LIBRARY_PATH`. For example, if
you used `brew` to install gcc as above, then this is probably the right thing
to do:
```bash
$ export DYLD_LIBRARY_PATH=/usr/local/Cellar/gcc/6.1.0/lib/gcc/6/
```

### Instructions

#### Conda environment

First create the conda environment by running
 ```bash
 $ conda env create
 ```
This will install a conda environment called `slda`, defined in
`environment.yml`, that contains all the dependencies. Activate it by running
 ```bash
 $ source activate slda
 ```
Next we need to compile the C code in this repository. To do this, run
```bash
$ python setup.py build_ext --inplace
```

#### pip install slda

If you want slda installed in your environment, run:
```bash
$ pip install .
```

## Tests

To run the tests, run
```bash
$ py.test slda
```
This may take as long as 15 minutes, so be patient.

## License

This code is open source under the MIT license.

Many thanks to [Allen Riddell](https://github.com/ariddell) and his [LDA
library](https://github.com/ariddell/lda) for inspiration (and code :)
