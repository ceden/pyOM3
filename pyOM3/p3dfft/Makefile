# Makefile for p3dfft + f2py Python wrapper
# Works on MacOS, Levante HPC, marin (marin01-08) and generic Linux/Ubuntu

PWD      := $(shell pwd)
OS       := $(shell uname -s)
HOST     := $(shell hostname)
F90       = mpif90
EXT      := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Platform detection
ifeq ($(OS), Darwin)
    # MacOS: conda provides everything
    FFTW_DIR = $(CONDA_PREFIX)
else ifeq ($(findstring levante, $(HOST)), levante)
    # Levante HPC: use system modules
    FFTW_DIR = /sw/spack-levante/fftw-3.3.10-fnfhvr
    GCC_DIR  = /sw/spack-levante/gcc-11.2.0-bcn7mb/lib64
    MPI_DIR  = $(shell mpif90 -showme:libdirs | tr ' ' '\n' | head -1 | sed 's/-L//')
    RPATH    = $(FFTW_DIR)/lib:$(GCC_DIR):$(MPI_DIR)
else ifeq ($(findstring marin, $(HOST)), marin)
    # marin: module miniforge3 provides python, MPICH and fftw, system gfortran
    FFTW_DIR := $(shell python3 -c "import sys; print(sys.prefix)")
    MPI_DIR   = $(FFTW_DIR)/lib
    MPI_LIBS  = -lmpifort
    RPATH     = $(FFTW_DIR)/lib
    export MPICH_FC = gfortran
else
    # Generic Linux/Ubuntu: conda provides everything
    FFTW_DIR = $(CONDA_PREFIX)
    MPI_DIR  = $(shell mpif90 -showme:libdirs | tr ' ' '\n' | head -1 | sed 's/-L//')
    RPATH    = $(FFTW_DIR)/lib:$(MPI_DIR)
endif

# MPI flags from compiler wrapper (OpenMPI), unless set above
MPI_DIR  ?= $(shell mpif90 -showme:libdirs | tr ' ' '\n' | head -1 | sed 's/-L//')
MPI_LIBS ?= $(shell mpif90 -showme:libs | sed 's/ / -l/g' | sed 's/^/-l/')

F90FLAGS = -DFFTW -O2 -fallow-argument-mismatch -fPIC -I$(PWD) -I$(FFTW_DIR)/include
LDFLAGS  = -lp3dfft -lfftw3 -L$(PWD) -L$(FFTW_DIR)/lib -L$(MPI_DIR) $(MPI_LIBS)

.PHONY: all clean

all: libp3dfft.a
	f2py -c p3dfft_wrapper.f90 -m fortran -I$(PWD) \
	    --f90flags="$(F90FLAGS)" \
	    $(LDFLAGS)
	rm -f libp3dfft.a p3dfft.mod
ifneq ($(OS), Darwin)
	patchelf --set-rpath '$(RPATH)' fortran$(EXT)
endif

libp3dfft.a: fft_spec.o module.o fft_init.o fft_exec.o
	ar rcs $@ $^
	rm -f fft_spec.o module.o fft_init.o fft_exec.o fft_spec.mod

%.o: src/%.F90
	$(F90) $(F90FLAGS) -c $<

clean:
	rm -f *.so *.a *.o *.mod
