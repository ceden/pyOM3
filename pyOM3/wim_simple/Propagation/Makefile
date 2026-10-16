# Makefile for the Propagation f2py kernels (same recipe as Snonlin).
# -ffp-contract=off: no FMA, keeps the result bit-identical to numpy.
# Works on MacOS, Levante HPC, and generic Linux/Ubuntu

OS       := $(shell uname -s)
HOST     := $(shell hostname)
GCC_DIR   = /sw/spack-levante/gcc-11.2.0-bcn7mb/lib64
F90FLAGS  = -O3 -ffp-contract=off

.PHONY: all clean

all: Propagation.pyf Propagation.f90
ifeq ($(OS), Darwin)
	# MacOS: conda toolchain, no patchelf needed
	f2py -c Propagation.pyf Propagation.f90 --f90flags="$(F90FLAGS)"
else ifeq ($(findstring levante, $(HOST)), levante)
	# Levante: system gfortran, patchelf with system gcc path
	f2py -c Propagation.pyf Propagation.f90 --f90flags="$(F90FLAGS)"
	patchelf --set-rpath '$(GCC_DIR)' Propagation.cpython-*.so
else
	# Generic Linux/Ubuntu: conda toolchain, patchelf with conda path
	f2py -c Propagation.pyf Propagation.f90 --f90flags="$(F90FLAGS)"
	patchelf --set-rpath '$(CONDA_PREFIX)/lib' Propagation.cpython-*.so
endif

clean:
	rm -f Propagation*.so
