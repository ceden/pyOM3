# Makefile for Snonlin f2py Python wrapper
# Works on MacOS, Levante HPC, and generic Linux/Ubuntu

OS       := $(shell uname -s)
HOST     := $(shell hostname)
GCC_DIR   = /sw/spack-levante/gcc-11.2.0-bcn7mb/lib64
F90FLAGS  = -fdefault-real-8 -fdefault-double-8 -O2

.PHONY: all clean

all: Snonlin.pyf Snonlin.f90
ifeq ($(OS), Darwin)
	# MacOS: conda toolchain, no patchelf needed
	f2py -c Snonlin.pyf Snonlin.f90 --f90flags="$(F90FLAGS)"
else ifeq ($(findstring levante, $(HOST)), levante)
	# Levante: system gfortran, patchelf with system gcc path
	f2py -c Snonlin.pyf Snonlin.f90 --f90flags="$(F90FLAGS)"
	patchelf --set-rpath '$(GCC_DIR)' Snonlin.cpython-*.so
else
	# Generic Linux/Ubuntu: conda toolchain, patchelf with conda path
	f2py -c Snonlin.pyf Snonlin.f90 --f90flags="$(F90FLAGS)"
	patchelf --set-rpath '$(CONDA_PREFIX)/lib' Snonlin.cpython-*.so
endif

clean:
	rm -f Snonlin*.so
