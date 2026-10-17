# python Ocean Model (pyOM3)

Simple python-based ocean and surface wave models:


pyOM3/pe_simple:     Cartesian primitive equation model using Jax sharding 
                     or mpi4py for domain decomposition

pyOM3/pe_realistic:  version of pe_simple with pseudo-spherical coordinates 
                     for large-scale realistic applications 

pyOM3/nh_simple:     non-hydrostatic ocean model with FFT solver 
                     based on pencil decomposition and MPI message parsing

pyOM3/nh_realistic:  version of nh_simple with topography and irregular boundaries
                     but with slower CG solver

pyOM3/wim_simple:    simple spectral surface wave model using Jax sharding 
                     or mpi4py for domain decomposition

pyOM3/wim_realistic: version of wim_simple with pseudo-spherical coordinates 
                     for large-scale realistic applications

pyOM3/test:          suite of tests for pe, nh, and wim
   
based on pyOM2 and ps3D, using P3DFFT, JAX, etc.


# Installation of pyOM3

pyOM3 consists of the Python packages pe_simple, pe_realistic, nh_simple, nh_realistic,
wim_simple and wim_realistic. Required and optional Python packages are described 
in the detailed installation instructions below.

Additionally, nh_simple needs the parallel FFT `p3dfft` in `pyOM3/p3dfft`, which is 
written in Fortran. `p3dfft` needs gfortran, FFTW3 and Open MPI to be installed for 
compilation (see below). wim_simple and wim_realistic use optionally for acceleration 
the Fortran parts `Snonlin` and `Propagation` in `pyOM3/wim_simple`. The Fortran parts 
are compiled with f2py of numpy. With Python 3.12 or newer f2py needs meson and ninja, 
with Python 3.11 setuptools 65.5.0.

## Obtaining the code

```
git clone https://github.com/ceden/pyOM3.git ~/src/pyOM3
export PYTHONPATH=~/src/pyOM3:$PYTHONPATH
```

## Installing on Linux with conda

```
conda create -n pyom3 -c conda-forge python=3.11 numpy scipy netcdf4 \
      zarr mpi4py openmpi openmpi-mpifort fftw gfortran make patchelf \
      setuptools=65.5.0
conda activate pyom3
pip install jax                 # CPU only
pip install "jax[cuda12]"       # NVIDIA GPUs with CUDA 12
```

Then compile the Fortran parts (see below).
The environment must be active when `make` is called to compile.

## Installing on Linux with system packages

On Debian or Ubuntu the requirements can also be installed with apt and pip,

```
apt install gfortran libopenmpi-dev openmpi-bin libfftw3-dev patchelf
pip install numpy scipy netCDF4 zarr meson ninja
pip install --no-binary mpi4py mpi4py
pip install jax                 # CPU only
pip install "jax[cuda12]"       # NVIDIA GPUs with CUDA 12
```

Tested with Python 3.13, gfortran 13, Open MPI 4.1 and JAX on the CPU.

## Installing on macOS

```
conda create -n pyom3 -c conda-forge python=3.11 numpy scipy netcdf4 \
      zarr mpi4py openmpi openmpi-mpifort fftw gfortran make \
      setuptools=65.5.0
conda activate pyom3
pip install jax                                  # Apple processor
pip install "jax==0.4.38" "jaxlib==0.4.38"       # Intel processor
```

Then compile the Fortran parts (see below).
The environment must be active when `make` is called to compile.
For Macs with an Intel processor 0.4.38 is the last version of jaxlib.

## Choosing the backend

The backend of pe and wim is chosen when pyOM3 is imported.
With mpi4py installed the backend is numpy, and the models run in parallel with MPI.
Without mpi4py but with JAX the backend is JAX.
The environment variable `PYOM3_BACKEND` selects `numpy` or `jax`.
With mpi4py and JAX both installed, JAX is therefore used only with `PYOM3_BACKEND=jax`.
pyOM3 switches JAX to double precision itself.
nh_simple always runs with numpy and MPI.

## Compiling the Fortran parts

Each part is compiled by `make` in its directory,

```
cd ~/src/pyOM3/pyOM3/wim_simple/Snonlin;     make
cd ~/src/pyOM3/pyOM3/wim_simple/Propagation; make
cd ~/src/pyOM3/pyOM3/p3dfft;                 make
```

Each produces a file ending in `.so`.
It has to be compiled again after a change of Python, of the compiler or of the MPI library.
`make clean` removes it.
The Makefiles find FFTW and the libraries of the compiler in `$CONDA_PREFIX`.
With the packages of the system on Linux they are called with `make CONDA_PREFIX=/usr` instead.
On Linux they set the paths of these libraries in the `.so` files with `patchelf`.

## Running the model

A configuration is a Python script or Jupyter notebook, started with as many processes 
as it expects, for instance `mpirun -np 6 python3 acc1.py` in `setups/acc`.
Many examples are found in `setups/simple`.


