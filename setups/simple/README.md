# Simple configurations of pyOM3

The configurations of pyOM2 (`for_config/*.f90`, `py_config/*.py`) for pyOM3,
as notebooks for the short experiments and as scripts for the long ones.
Everything is self contained in this directory. Each notebook and script
starts with

    import os, sys
    sys.path.insert(0, os.environ.get('PYOM3_ROOT', os.path.abspath('../..')))

`PYOM3_ROOT` is the directory that contains the `pyOM3` package, as for the
coupled setups. Without it the package is found two levels above the current
directory, which is this directory when a notebook is opened or a script is
started here. Set `PYOM3_ROOT` to run a script from somewhere else.

## Notebooks

Each notebook has the cells: parameters, model class, build and initial
conditions, a plot function, the time loop with a plot every `plot_every`
steps, and a final figure. The fields are plotted from the model arrays in
memory, there is no output file. The time loop is written out in the notebook
(the content of `model.loop`), so the model can be stepped in pieces.

| notebook | pyOM2 origin | what |
|---|---|---|
| `internal_wave1` | internal_wave1.f90, internal_wave1.py | internal wave beams from an oscillating source (nh_simple) |
| `kelv_helm1` | kelv_helm1.f90, kelv_helm1.py, shear1.f90 | stratified shear layer, option `sponge` (nh_simple) |
| `kelv_helm2` | kelv_helm2.py | unstratified vortex sheet (nh_simple) |
| `holmboe1` | holmboe1.py | Holmboe instability, seeded with the fastest mode from `lin_stab.py` (nh_simple) |
| `rayleigh1` | rayleigh1.f90, rayleigh.py | Rayleigh-Benard convection (nh_simple) |
| `rossby1` | rossby1.py | one layer, Rossby waves, Hovmoeller diagram |
| `eq_waves1` | eq_waves1.py | one layer, equatorial Kelvin, Yanai, Rossby and gravity waves, option `wave` |
| `enso1` | enso1.py | one layer, equatorial response to a wind patch |
| `column_tke1` | test_tke1.f90, test_tke1.py | single column, wind driven mixing with the TKE closure |
| `barbi_exp1` | barbi_exp1.f90 | warm anomaly in a closed basin on the sphere |
| `eady1` | eady0.f90, eady0.py, eady1.py, channel1.py | Eady instability with the analytical mode, growth rate check, option `restoring` |
| `salt_fountain1` | salt_fountain1.f90 | salt fountain, heat conducting wall inside the water column (nh_realistic) |
| `overflow1` | overflow1.f90 | dense water flowing over a ridge and down the slope (nh_realistic) |

`lin_stab.py` solves the viscous Taylor-Goldstein problem for U(z), B(z) on the
model levels and returns the fastest growing mode (growth rate, phase speed,
u, w, b), used by holmboe1.

## Scripts

Long runs, meant for `mpirun -np N python <name>.py` with `n_pes_x, n_pes_y`
set in the file, output to the netCDF file `<name>.cdf`.
The notebook `plot_snap` shows surface fields, zonal mean sections and the
kinetic energy time series of such a file, set `name` and the time index `it`
in its first cell.

| script | pyOM2 origin | what |
|---|---|---|
| `acc1.py` | acc1.f90, acc1.py | Southern Ocean channel, Cartesian, TKE. Option `eddies` |
| `acc2.py` | acc2.f90, acc2.py | global sector with Atlantic basin, spherical, isoneutral + GM, TKE |
| `acc3.py` | acc3.f90, acc_hi1.f90 | spherical channel with a wall. `eddies=False` is acc3, `eddies=True` is acc_hi1 |
| `jets1.py` | jets1.f90, jets1.py | zonal jets in a beta plane channel, sponge layers |
| `eady_jet1.py` | eady_jet1.f90, eady_opv1.f90 | baroclinic jet. Option `balanced` (optimal balance of eady_opv1) |
| `roullet1.py` | roullet1.f90 | Roullet et al. jet, stretched vertical grid, zonal mean restoring |
| `ml_front1.py` | ml_front.f90 | mixed layer front (3-D version without the mixed layer parameterisation) |
| `tide_eddy1.py` | ps3D tide_eddy1.f90 | internal tide beam through a balanced eddy (nh_simple, rigid lids) |
| `tide_vortical_spec1.py` | ps3D tide_vortical_spec1.f90 | internal tide beam through a spectrum of balanced eddies (nh_simple, rigid lids) |
| `lee_wave1.py` | ps3D lee_wave1.f90 | wave beam in a background current with a critical layer (nh_simple, periodic, scaled equations) |
| `overflow1_3D.py` | overflow1_3D.f90 | dense water from a shelf through a gap in a barrier down a slope, rotating tank (nh_realistic) |

The three ps3D configurations use `pyOM3/nh_simple/modes.py`: the eigenmodes
of the linear equations (geostrophic mode s = 0, gravity wave modes s = +1, -1)
on the discrete grid, the transforms to spectral space, the projection of a
field on a mode, a single mode in physical space for the wave makers, and the
balance of a state to third order in the Rossby number after Warn et al.
(1995). They run on one process.

## Conventions

The hydrostatic configurations use `pe_realistic`. It takes a scalar `dxt`
for a uniform grid and a vector for a variable grid, `coord_degree=True` for
the sphere, and it provides the coordinates:

- `self.xt, self.yt, self.zt` (and `xu, yu`) are the local coordinates of
  this processor including the halo points. They are used for the Coriolis
  parameter, the topography and the forcing fields.
- `self.xt_gl, self.yt_gl, self.zt_gl` are the global coordinates without
  halo points. They are used to build the initial conditions as full fields
  `(nz, ny, nx)` which `set_initial_conditions` distributes to the processors.

With `set_zero_at_surface=True` the depth `zt` is negative with `zt = 0` at
the surface as in pyOM2. The lowest model level is `k = 1`, `kbot = 1` means
ocean down to the bottom, `kbot = 0` is land. `eq_of_state=0` means that
`temp` is the buoyancy, `1` is the linear and `5` the TEOS-10 equation of state.
Wind stress and surface fluxes are added to the tendencies `du`, `dtemp` at
time level `tau` in `set_forcing`. The TKE closure (`enable_tke=True`) needs
`self.taux, self.tauy` and the class `tke` as a base. The zonal mean restoring
of `eady1` and `roullet1.py` needs `n_pes_x = 1`.

`nh_simple` is periodic in x and y, has rigid lids, no topography, a constant
Coriolis parameter `f0`, and buoyancy `b` as the only tracer, relative to the
background stratification `N0`. Arrays are `(x, y, z)`. The x-z slices with
`ny = 1` run on one processor only.

`nh_realistic` is `pe_realistic` with a prognostic `w` and a non-hydrostatic
pressure, so it has the same parameters, coordinates, masks and tracers, on a
Cartesian grid only. Land cells inside the water column (the wall of
`salt_fountain1`) cannot be described by `kbot`. They are set in `maskT` in
`make_topo_masks`, followed by the other masks.

## Not converted

| pyOM2 | reason |
|---|---|
| acc_small.f90, energy1.f90, energy2.f90 | small variants of acc1 and acc3 for the energy diagnostics of pyOM2 |
| eady1_unscaled.f90, eady_periodic1.f90, eady2.py | variants of eady1 that need the 3-D lin_stab module or the QG filter of pyOM2 |
| isopyc1.f90, test_adv1.f90 | need `enable_momentum_equation = False` |
| rayleigh2.f90 | rayleigh1 with a localised heat source |
| saturn1.f90 | unfinished sandbox setup |
| atmos1.f90 | the ideal gas atmosphere, see hs_atmos.py |
| test_obta1.f90 | optimal balance of a jet, see `balanced=True` in eady_jet1 |
| test_idemix1/2, test_lee1, test_rossmix1/2/3, test_bihathk1, test_obc1 | IDEMIX, lee waves, ROSSMIX, biharmonic thickness diffusion and open boundaries are not in pyOM3 |
