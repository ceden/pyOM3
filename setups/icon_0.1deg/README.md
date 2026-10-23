# icon_0.1deg in pyOM3

The 1/10 degree, 128 level global configuration of pyOM2 (`setup1.f90`,
"1/10 global with 128 levels, from Icon 10km version"), converted to pyOM3 and
forced by hourly ERA5. Ocean alone and, optionally, coupled to the wave model
on a coarsened wave grid.

Grid and topography are the pyOM2 ones, unchanged: the Icon R2B8 10 km ocean
grid interpolated onto a 1/10 degree Mercator grid. `grid_topo.nc` is read as
it stands.

    3600 x 2295 x 128, cyclic in x, 78.5 S to 69.5 N
    dxt = 1/10 deg, dyt variable 0.020 to 0.100 deg (cells stay near square)
    dzt 2.05 m at the surface to 197 m at 6100 m
    5,153,349 wet surface cells, 1058 million cells in total
    cells 11.1 km at the equator, 2.22 km at the southern edge, in BOTH
    directions

It is icon033 on a finer grid. The machinery is the shared one
(`../coupled_common`, `../forcing_common`), the setup module is
`icon033_setup.py` with four numbers changed, and everything written in
`../icon033_*/README.md` about the forcing, the coupling and the I/O holds
here. This file says what is different and what the size costs.

## Physics

Taken from `setup1.f90` where the resolution decides, from the calibrated
global4 set where the closure does.

    dt = 36 s, AB2 with eps 0.1, superbee advection
    biharmonic friction A_hbi = 2.7e10, cosPower 1.5 per Laplacian pass
    NO harmonic friction, NO isoneutral diffusion, NO GM (eddy resolving)
    TEOS-10 (eq_of_state 5)
    implicit free surface, CG epsilon 1e-9
    free_surface_tracer_flux = True
    TKE: alpha_CB 100, mxl_choice 2 (NEMO), mxl_beta 2e4, alpha_tke 2
    kappaH_min 1e-5, r_bot 1e-5

IDEMIX is on in `setup1.f90` and pe_realistic has no internal-wave module, so
the interior mixing is the TKE closure above the `kappaH_min` floor and
nothing else.

Biharmonic stability, evaluated over every wet cell with the cos^3 factor of
the two Laplacian passes included,

    S = A cos^3 dt (4/dx^2 + 4/dy^2)^2 / 2  <=  1 ,

peaks at S = 0.010 at the southern edge, where the cells are 2.2 km and cos^3
is 8e-3. icon033 runs at 0.004. The friction is not what limits the step.

## THE TIME STEP, and why it is the first thing to test

36 s is `setup1.f90`'s `dt_mom = 144.0/4`, and it decides the cost of the
whole experiment: 2400 ocean steps per model day.

It is not set by anything horizontal. POP runs the same 1/10 degree spacing
(3600 x 2163, narrowest cell 2234 m against 2220 m here) at **216 s**.

| | POP | icon01 |
|---|---|---|
| narrowest cell | 2234 m | 2220 m |
| levels | 42 | 128 |
| top cell | about 10 m | 2.05 m |
| ocean step | 216 s | 36 s |

At 3 m/s over 2.22 km the advective CFL is 0.29 at 216 s and 0.05 at 36 s.
What differs from POP is the VERTICAL grid: 2.05, 2.15, 2.25 m at the top, 25
m of water in the first ten cells. Two things scale with that, the vertical
advective CFL `w dt/dz`, which at w = 5 mm/s in the top cell is 0.53 at 216 s
and 0.09 at 36 s, and everything that divides a surface flux by the top cell
thickness. The fortran line reads `144.0/4`, i.e. the value was walked down
rather than computed.

So 36 s may be conservative by a factor of two or three. **Run the TEST three
times with `EXP_OCEAN_DT` 36, 72 and 108** and compare the canary and the
identity line before committing an allocation. Every factor on the step is
the same factor on the cost. Coupled, the wave step follows the ocean step:
at 108 s with c = 3 the default wave step is 36 s and the sub-cycle 3, so only
the ocean gets cheaper.

## Cost

By the icon033 scaling (0.47 s per step at 101,760 cells per rank) the ocean
alone is about 0.6 s per step on the 7650 ranks below:

    26 minutes per model DAY
    6 days of wall clock per model YEAR
    an 8 hour job advances about 18 model days

On 10200 ranks (80 nodes) roughly 19 minutes per model day. A ten-year spinup
is not a thing this configuration does. Plan in model months, start from an
interpolated icon033 state rather than from rest if you can, and measure one
segment before committing anything.

## Forcing

Hourly ERA5, its own set `era5_icon01_YYYY-MM.nc` in `../era5_icon01`, built
by `prep_era5_native.py`. The files stay ON THE ERA5 GRID and the model
interpolates each record to a rank's block while the run goes on
(`forcing_common/forcing_era5_native.py`): 22 GB per month there, against ten
times that on the model grid. All nine fields are in one set.

The ERA5 turbulent surface stress is the one tau_a of the experiment: the wave
u*, the ocean direct branch and, without waves, the ocean surface stress all
derive from it. Heat flux Q = sshf + slhf + ssr + str + dQ/dT (SST_ERA5 -
SST), with ssr penetrating; dQ/dT from `ecmwf_surface_forcing.nc` and the SSS
restoring target from `woa_surface_forcing.nc`, both monthly climatologies of
the pyOM2 setup.

    EXP_QNET_OFFSET   measured per period with measure_era5_offsets.py
    EXP_EMP_OFFSET    the same for P - E
    EXP_SSS_PISTON    0.17 m/day (on a 2.05 m cell that is 12 days)

ERA5 runs on a prescribed SST, so its surface energy budget is free: the mean
net flux over the wet cells is several W/m2 into the ocean where the real
ocean absorbs about 0.5, and the restoring can only remove that by holding a
warm bias. Same for P - E, which is not zero because there is no runoff.
Measure both over the period the run uses and set them. **The values in the
run script are icon033's and are NOT this grid's.**

    python3 ../forcing_common/measure_era5_offsets.py 1990-05 1991-04 \
            --grid grid_topo.nc --dir ../era5_icon01 --tag icon01 \
            --stride 7 --workers 16

## Input files

All four as the fortran setup reads them, in the run directory or wherever
`EXP_INPUT` points:

    grid_topo.nc              dxt, dyt, dzt, xu, yu, kbot
    initial_conditions.nc     temp, salt (bottom first)
    ecmwf_surface_forcing.nc  dqdt
    woa_surface_forcing.nc    sss

The initial conditions are made from the WAGHC climatology with the generic
script, which takes the grid and the bathymetry from `--grid` and needs no
change for this configuration:

    python3 ../forcing_common/make_initial_conditions.py \
        --src /pool/data/ICDC/ocean/waghc/DATA --version PYC --month 1 \
        --grid grid_topo.nc --out initial_conditions.nc \
        --hinterp bilinear

`bilinear` is the right operator here and `auto` picks it anyway: WAGHC is
0.25 degree and the model cell is 0.1 degree, so a model cell holds well under
one source centre. Budget about 40 GB of memory and half an hour to an hour;
this is the largest grid the script has been asked for. Read three things in
its report, the same three that mattered for POP: how many cells took the
level above (basins below their sill), the isolated surface water bodies with
no data at all (POP had 95 cells in 37 bodies, and those are the stability
risk -- set `kbot = 0` if one of them later shows up in the canary), and the
static-stability check (POP came out at 0.009 kg/m3 worst case; 0.1 or more
means a fill crossed a sill).

## Coupling

`EXP_WAVES=1` adds wim_realistic on a COARSENED grid, `EXP_WAVE_COARSEN = C`
(`coupler_mpi.py`): a wave cell is C x C ocean cells, its faces are ocean cell
faces, and nothing is interpolated between the two grids.

| C | wave grid | cells | narrowest | wave dt | sub-cycle | wave PEs |
|---|---|---|---|---|---|---|
| 1 | 3600 x 2295 | 5.15 M wet | 2.2 km | 12 s | 3 | not affordable |
| 2 | 1800 x 1147 | 1.29 M wet | 4.4 km | 18 s | 2 | not affordable |
| 3 | 1200 x 765 | 578 k wet | 6.7 km | 36 s | 1 | 30 x 45 |
| 4 | 900 x 573 | 326 k wet | 8.9 km | 36 s | 1 | 24 x 32 |

**C = 3 is the setting.** 3600 and 2295 both divide by 3, so no wave row or
column is ragged; the wave cells are 6.7 km, finer than the 14.8 km the
icon033 coupled test ran with; and from C = 3 the wave model steps ONCE per
ocean step, so there is no sub-cycling left to save.

The wave PE counts are estimates scaled from the icon033 C = 2 measurement
(1024 wave ranks, 0.64 s per coupling step). Read the two idle percentages in
the log (`load wave`, `load ocean`) and move ranks between the groups.

The ocean's prognostic velocity becomes the Lagrangian mean u_L, its surface
stress becomes tau_o from the wave field, and the TKE surface flux and
roughness come from Phi_oc and Hs. `EXP_DIRECT_BRANCH=1` hands the
non-wave-supported part tau_a - tau_in to the ocean as well, without which the
coupled ocean is under-forced against its ocean-alone twin.

## Running

Three directories, each with `icon01_setup.py`, `run_icon01.sh` and the input
files. The mode is the last part of the directory name:

    icon01_spinup     ocean alone, repeat year from 1990-05-01
    icon01_ocean      ocean alone, 1991-05-01 on, from the spinup restart
    icon01_coupled    the same period and the same ocean start, with waves

`TEST = 1` (the default) is 0.2 model days in two 0.1 day segments, about five
minutes, and it exercises the restart. `TEST = 0` is production.

Ocean decomposition: **90 x 85 = 7650 ranks**, blocks of exactly 40 x 27
(3600 = 2^4 3^2 5^2, 2295 = 3^3 5 17). Ragged blocks cost a factor, so stay on
divisors. Other exact ones:

    80 x  51 =  4080, blocks 45 x 45, 32 nodes
   100 x  51 =  5100, blocks 36 x 45, 40 nodes
   120 x  85 = 10200, blocks 30 x 27, 80 nodes
    72 x 135 =  9720, blocks 50 x 17, 76 nodes

The header asks for the 60 nodes the default layout holds. The coupled mode
needs its own count:

    sbatch --nodes=71 --ntasks=9088 --export=ALL,TEST=0,C=3 run_icon01.sh

With `EXP_RESTART_RAW = 1` the restart is a set of per-rank blocks and can
only be read on the layout that wrote it, so a branch must keep the spinup's.
To branch on another layout, rerun the LAST spinup segment with
`EXP_RESTART_RAW = 0` and keep it 1 in the branch.

Set

    OMP_NUM_THREADS 1, MKL_NUM_THREADS 1, OPENBLAS_NUM_THREADS 1
    srun --cpu_bind=cores --distribution=block:cyclic
    OMPI_MCA_opal_common_ucx_opal_mem_hooks 1

## Output, and how much of it there is

Everything per rank, nothing through a shared file:

* **snapshots** RAW (`snap_raw`, ocean and wave): each rank writes its own
  block as `.npy` under `raw_ocean_icon01_seg*/pe=<r>/n=<step>/`, one
  directory per segment.
* **online means** RAW as well (`RAW_SNAPSHOTS = True`), the same layout, one
  directory per window under `ocean_mean_icon01_seg*/`, with a small
  `window_n=<step>.npy` per record holding the time, the bounds, the step
  count and the dates.
* **restarts** RAW per rank, `EXP_RESTART_RAW=1`: `<name>_pe/<rank>.npz`
  beside a small header file. A raw restart BELONGS TO ITS DECOMPOSITION.
* **initial conditions** and the two monthly climatologies are read by rank 0
  one level at a time and scattered.
* the **wave snapshot** writes a CHOICE of fields, `EXP_SNAP_WAVE_FIELDS`
  (default `spectra`): 184 MB per record on the C = 3 grid against 2.2 GB for
  the full 4-D spectrum.
* the **KE budget** is OFF by default (`EXP_KE_BUDGET`), about 40 % on a step.

THE NUMBERS ARE LARGE. One 3-D field on this grid is 8.5 GB in float64.

    one raw ocean snapshot          about 70 GB
    one monthly mean record         100 GB with the standard 12 3-D fields,
                                    135 GB with AVERAGE_EDDY on (16)
    one raw ocean restart           about 170 GB
    ERA5, per month                 22 GB

At the default of a snapshot every five days that is 5 TB per model year, and
the means another 1.2 TB. **Decide what you actually want before the first
production segment**, with

    setenv EXP_AVERAGE_FIELDS temp,salt,u,v,ssh,sst,sss,qnet,taux,tauy

which keeps two 3-D fields instead of twelve, and a snapshot interval of 30
days rather than 5. `AVERAGE_EDDY` (rho, urho, vrho, wrho for the eddy fluxes,
see `../coupled_common/exp_ocean.py`) adds four 3-D fields and is worth its
space only where you mean to use it.

## Open

* the time step, see above. This is the one that matters.
* the coupling-file record is the last rank-0 gather in the coupler
  (`gather_unpadded_mpi` on `u_S` in `run_wave`). On this grid at C = 3 that
  is a 1.9 GB assembly on wave rank 0 per written record (17 GB at C = 1). It works at the
  default cadence of one record per model day, but it is the thing that will
  fail first if the cadence is raised.
* an upwind propagation option with a measured CFL limit would allow the
  ECWAM-style long wave step; wim has `f_limiter = 'upwind'` but it bypasses
  the Fortran kernels and its limit is unmeasured. At 1/10 degree this is the
  lever that matters most on the wave side.
* no sea ice, so the wave model is forced under the ice mask that the heat and
  salt fluxes respect; the wave model has no depth refraction or bottom
  dissipation; the poles are outside the domain by construction.
