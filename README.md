python Ocean Model (pyOM3)

Simple python-based ocean and wave models:


pyOM3/pe_simple:     Cartesian primitive equation model using Jax sharding or mpi4py for domain decomposition

pyOM3/pe_realistic:  version of pe_simple with pseudo-spherical coordinates for large-scale realistic applications 

pyOM3/nh_simple:     non-hydrostatic ocean model with FFT solver based on pencil decomposition and MPI message parsing

pyOM3/wim_simple:    simple spectral surface wave model using Jax sharding or mpi4py for domain decomposition

pyOM3/wim_realistic: version of wim_simple with pseudo-spherical coordinates for large-scale realistic applications

pyOM3/test:          suite of tests for pe, nh, and wim
   
based on pyOM2 and ps3D
