# icon_0.33deg in pyOM3

The 1/3 degree, 128 level global configuration of pyOM2 (`setup1.f90`,
"1/3 global with 128 levels, from Icon 20km version"), converted to pyOM3 and
forced by hourly ERA5. Ocean alone and, optionally, coupled to the wave model.

Grid and topography are the pyOM2 ones, unchanged: R2B7 Icon ocean grid
interpolated onto a 1/3 degree Mercator grid, Gibraltar cut to a k = 27 sill,
isolated Arctic and Hudson Bay bays removed, Baltic, Black, Red Sea and
Persian Gulf removed. `grid_topo.nc` is read as it stands.

    1080 x 689 x 128, cyclic in x, 78.5 S to 69.4 N
    dxt = 1/3 deg, dyt variable 0.067 to 0.118 deg (cells stay near square)
    dzt 2.05 m at the surface to 197 m at 6100 m
    461,088 wet surface cells
    cells 37 km at the equator, 7.4 km at the southern edge

Compared by the triangle height, the R2B7 source is about 26 km
(`../README.md`, "Resolution against the Icon source grids"). This grid is
coarser than its source only equatorward of about 45 deg. Poleward of that it
resolves the flow more finely than the Icon bathymetry it carries.

## Physics

Taken from `setup1.f90` where the resolution decides, from the calibrated
global4 set where the closure does.

    dt = 216 s, AB2 with eps 0.1, superbee advection
    tracer advection outside the AB step, 2 vertical sub-steps
        (EXP_VADV_SUB_CYCLING 2, set in both batch scripts)
    biharmonic friction A_hbi = 2e11, cosPower 1.5 per Laplacian pass
    NO harmonic friction, NO isoneutral diffusion, NO GM (eddy permitting)
    TEOS-10 (eq_of_state 5)
    free surface: implicit, CG epsilon 1e-9 (90 to 100 iterations per step),
        or split-explicit, 24 sub steps, forward-backward theta 0.5
        (EXP_FREE_SURFACE)
    free_surface_tracer_flux = True          <- see FINDINGS
    TKE: alpha_CB 100, mxl_choice 2 (NEMO), mxl_beta 2e4, alpha_tke 2
    kappaH_min 1e-5, r_bot 1e-5, quadratic bottom friction r_quad_bot 2.5e-3
        <- see FINDINGS
    surface stress: relative wind with ERA5's drag coefficient
        (EXP_STRESS_MODE relwind, EXP_STRESS_UO lagrangian)

The vertical advective CFL w dt/dz in the 2.05 m top cell is 0.53 at
w = 5 mm/s. That is what the split tracer advection is for.

The split-explicit free surface replaces the CG by sub steps of the 2-D
external mode, with no global reduction. The external wave has
c = sqrt(g H) = 245 m/s at 6100 m, so c dt/dx = 7.1 on the 7.4 km cell. 24
sub steps give a sub-step CFL of 0.30. It runs for the ocean alone, and the
KE budget must be off with it. A restart of one method cannot be read by a
run with the other.

IDEMIX is on in `setup1.f90` and pe_realistic has no internal-wave module, so
the interior mixing is the TKE closure above the `kappaH_min` floor and
nothing else. The TKE numbers were scored against the de Boyer Montegut mixed
layer climatology at 4 degrees with a 10 m top cell; only the Charnock length
is grid independent.

## Forcing

Hourly ERA5, its own set `era5_icon033_YYYY-MM.nc` in `../era5_icon033`,
May 1990 to May 2000, built by `prep_era5_icon033.sh`. The files stay ON THE
ERA5 GRID and `forcing_era5_native.py` interpolates each record to a rank's
block while the run goes on: a 1/3 degree copy of a 0.28 degree field is more
storage and no more information. All nine fields are in one set, the six flux
fields, sst, and e and tp for the freshwater flux.

The 10 m wind is a second set, `era5_icon033w_YYYY-MM.nc`, beside the first
(`prep_era5_native.py --fields u10,v10 --tag icon033w`). The surface stress
is the relative-wind bulk formula (`EXP_STRESS_MODE relwind`). It keeps
ERA5's drag coefficient C_D rho_a = |tau_ERA5|/|u10|^2 and forms the stress
from the wind relative to the surface velocity u_L
(`EXP_STRESS_UO lagrangian`). In a coupled run the waves take this stress
from the ocean. `EXP_STRESS_MODE era5` takes the ERA5 stress as it is. Heat flux Q = sshf + slhf + ssr + str + dQ/dT (SST_ERA5 -
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
Measure both over the period the run uses and set them.

## I/O, and why it differs from icon15

95.2 million cells is 72 times icon15 and every serial path in the machinery
became the run.

* snapshots are RAW (`snap_raw`, ocean and wave): each rank writes its own
  block as `.npy` under `raw_ocean_icon033_seg*/pe=<r>/n=<step>/`, one
  directory per segment. `snap_nc4` is a round robin over the ranks through
  one shared file.
* restarts are RAW per rank, `EXP_RESTART_RAW=1`: `<name>_pe/<rank>.npz`
  beside a small header file that keeps the name the gathered form used.
  A raw restart BELONGS TO ITS DECOMPOSITION. Reading dispatches on the file,
  so a run that writes raw restarts can still branch from a gathered
  `EXP_INIT_OCEAN` written on another layout. `GATHERED=1` in `run.sh`
  writes the gathered form (rerun the last segment with it before changing
  the layout or moving to the GPU).
* initial conditions and the two monthly climatologies are read by rank 0 one
  level at a time and scattered (`forcing_common/block_io.py`).
* the wave snapshot writes a CHOICE of fields, `EXP_SNAP_WAVE_FIELDS`
  (default `spectra`: the moments plus E_f and E_d): 160 MB per record
  against 18 GB for all five 4-D fields.
* the KE budget is OFF by default (`EXP_KE_BUDGET`), about 40 % on a step.
  The mixed layer depth is not in the online means (a full equation of state
  per step).

## Coupling

`EXP_WAVES=1` adds wim_realistic on the same horizontal grid. The ocean's
prognostic velocity becomes the Lagrangian mean u_L, its surface stress
becomes tau_o from the wave field, and the TKE surface flux and roughness
come from Phi_oc and Hs. `EXP_DIRECT_BRANCH=1` hands the non-wave-supported
part tau_a - tau_in to the ocean as well, without which the coupled ocean is
under-forced against its ocean-alone twin.

    wave dt 36 s, sub_cycle 6, CFL 0.107 against the certified 0.125
    the narrowest cell is 7.4 km; only divisors of 216 s are available

`EXP_WAVE_COARSEN = C` puts the waves on the ocean grid coarsened by C. At
C = 2 the wave grid is 540 x 345, the wave step 72 s and the sub-cycle 3.
That is the setting of the coupled runs. The coupled mode keeps the implicit
free surface (`run.py` stops otherwise).

The exchange is BLOCK TO BLOCK over the intercommunicator
(`coupler_mpi.BlockExchange`): each rank sends its own block straight to the
ranks that own those cells. Nothing is assembled on rank 0 except the
coupling-file record, and only on its own cadence.

## The shared machinery (Oct 2026)

`icon033_setup.py` is now a configuration module of `setups/coupled_common`
and `setups/forcing_common`, as icon01 and icon15: it holds the grid source,
the parameters, the level-by-level initial conditions, the raw snapshot
form, the friction bookings and `forcing_settings()`, about 860 lines. The
driver (`run.py`), the coupler, the restarts, the online means, the energy
diagnostics, the TKE hooks and the forcing readers are the shared code
(`exp_ocean.ocean_common`, `exp_wave.wave_common`, `block_io`,
`forcing_era5_native`), reached through `cfg = Cfg(this module)`. The old
monolithic module with its own copies of all that is gone; so is the local
`run.py`.

The module runs on both layouts of the shared code without a branch of its
own: MPI (one rank per block) and the JAX backend (one process, the devices
as shards, every shard's halo inline). The only place a configuration module
touches the layout is the initial conditions, which go through
`block_io.local_block` / `pad_block` instead of a zero array and a
`[h:-h]` slice. Checked on a small synthetic input set through `run.py`,
two segments with a restart in between: numpy on one rank, MPI 2 x 2, jax
1 x 1 and jax 2 x 1 — jax against numpy on the same layout 2e-11 after 40
steps; between layouts 4e-6 in p_s on numpy as on jax, which is the CG at
epsilon 1e-9, not the layout.

## Running

Run directories beside `coupled_common` and `forcing_common`, each with
`icon033_setup.py`, the four input files and its batch script, and nothing
else (`run.py` puts the run directory LAST on the path, so a stale copy of a
shared file there cannot shadow the shared one — delete any `run.py`,
`coupler_*.py`, `forcing_*.py`, `block_io.py` left from the old layout):

    icon033_spinup    repeat year, 1990-05-01, EXP_FORCING_CYCLE_DAYS 365
                      run.sh: python3 -u ../coupled_common/run.py on 72 x 13
    icon033_gpu       the same ocean alone on the JAX backend, run_gpu.sh
    icon033_ocean     1991-05-01 to 2000-04-30, from the spinup restart
    icon033_coupled   the same period and the same ocean start, with waves

The spinup chain stops at the checkpoints 5, 6, 7, 8, 10, 15 and 20 years
(`CHECKPOINTS` in the script). There it copies the restart and its `_pe`
directory to `icon033_restart_ocean_y<N>` and resubmits toward the next one.

`run.sh` and `run_gpu.sh` run the same experiment: the same forcing, time
step, split, free surface and stress. `set FS` at the top of `run.sh`
chooses the free surface, `explicit` as on the GPU. A restart of one method
is refused by a run with the other. So a chain written with the implicit
free surface, which the CPU runs so far are, continues only with
`FS = implicit`, and a coupled branch cannot start from an explicit spinup.

`TEST=1` (`--export=ALL,TEST=1`) runs 0.2 model days in two segments to
check an installation; it is the default of `run_gpu.sh`.

### On the GPU (`run_gpu.sh`)

The ocean alone on the JAX backend, one process per GPU over N Levante GPU
nodes, `PYOM3_BACKEND=jax`, with the split-explicit free surface.
`EXP_NPX_OCEAN x EXP_NPY_OCEAN` is the shard layout and must equal the
number of GPUs. A shard is an exact division of the grid, and 689 = 13 x 53
does not divide by 2 or 4, so N nodes are **4N x 1** shards, and N must
divide 1080/4 = 270: N = 1, 2, 3, 5, 6, 9, 10 (not 4, 7, 8). The script
takes the layout from the node count.

The restart is raw PER SHARD (`EXP_RESTART_RAW 1`), one block per GPU. It
can be read on any process count with the same shard layout. An MPI restart
of the CPU runs cannot be continued on the GPU: its layout is per rank, and
its free surface is implicit. The wave model is not available on this
backend. Memory is not the issue (762 MB per 3-D field, 190 MB per shard).

Measured: **0.12 s per step on 2 nodes (8 GPUs)**, 0.8 min per model day,
4.8 h per model year, 5 model years per day. That is four times the CPU run
on 8 nodes.

Decomposition, ocean: **72 x 13 = 936 ranks**, blocks of exactly 15 x 53
(1080 = 2^3 3^3 5, 689 = 13 x 53). Ragged blocks cost a factor: 64 x 16 on
1024 ranks was 12 % slower on 10 % more cores. Set

    OMP_NUM_THREADS 1, MKL_NUM_THREADS 1, OPENBLAS_NUM_THREADS 1
    srun --cpu_bind=cores --distribution=block:cyclic
    OMPI_MCA_opal_common_ucx_opal_mem_hooks 1

Measured: ocean alone **0.47 s per step** on 936 ranks (3.2 min per model day,
10 y in about 190 h). Coupled **0.85 s per coupling step** with 4096 wave
(64 x 64) + 936 ocean on 40 nodes: 5.7 min per model day, 34 h per model
year. The ocean idles 26 % waiting for the waves; the wave leaders are
`Sinput` (182 ms, two Janssen passes per sub-step) and `apply_bc` (181 ms
over 15 halo exchanges per sub-step). With the wave grid coarsened by 2,
**0.64 s per coupling step** with 1024 wave + 936 ocean ranks, 4.2 min per
model day, 26 h per model year. The ocean step is then the limit, not the
waves. These CPU numbers are with the implicit free surface. The
split-explicit one has not been timed on the CPU.

## Findings

Worth keeping, in the order they bit.

1. **The linear free surface does not conserve the top cell's tracer.** With
   pyOM2's form the surface-face advective flux is zeroed, so the top cell
   changes by `-var dEta/dz` every step. On a 2.05 m cell with a cold-start
   SSH range of -4.2 to +2.3 m that factor is of order one: after three days
   48 % of the surface was outside 32 to 38 psu (1.9 to 61.7) and SST reached
   46.7 C. `free_surface_tracer_flux = True` fixes it. Temperature hid it
   because dQ/dT restores the top cell in 2.4 days against the salinity
   piston's 12. icon15 has the same defect, five times weaker on its 11 m
   cell.
2. **icon15 applied bottom friction twice** (its own forward-Euler one inside
   `harmonic_momentum` plus the package's implicit step, which model.py now
   calls unconditionally). Fixed there; icon033 uses the package's only.
3. **The ERA5 reader opened the monthly file per record** and broadcast with
   the pickling `bcast`: 400 ms of every ocean step, the whole machine
   waiting for rank 0. Keeping the month open and using buffer `Bcast` on one
   stacked float32 array took it to 24 ms.
4. **The coupler funnelled everything through rank 0.** A `comm.bcast` of the
   full-domain `u_S` (1.52 GB) OOM-killed the ocean at the first coupling
   step; a `Scatterv` fixed the memory but not the time, because the root's
   strided pack of 136-byte fragments is the floor — 4.6 s of a 7.5 s step.
   Block to block took the coupling step from 7.5 s to 0.85 s.
5. **Post every receive before any send.** The first block-to-block version
   deadlocked at the first coupling step: a large message goes through
   rendezvous, so an `Isend` does not complete until the matching receive
   exists. Small messages are eager and complete anyway, so a toy run cannot
   find it. `bx_test.py` checks correctness; it cannot reproduce that
   deadlock.
6. The A2d wave alarm compares a cell with its own PE's median, so it loses
   meaning as the blocks shrink; below `A2D_MIN_LIVE` cells it is disarmed.
7. **The coupled branch crashed at model day 527** with a wave CFL alarm in
   the Skagerrak (7.33E/56.61N, a 35 m column). The cause was a barotropic
   coastal jet along the North Sea coast into the Skagerrak, 4.9 m/s, with
   1.6 m SSH set-down in the Kattegat. The ocean-alone spinup has the same
   jet and survives it. The vertical CFL was not the cause (at most 0.26).
   Only the linear r_bot = 1e-5 opposed the surge, a nine-day damping on a
   35 m shelf. Quadratic bottom friction, r_quad_bot = 2.5e-3, was added. The
   restarted branch ran past it.

## Open

* `janssen_ncall = 1` would take ~90 ms off `Sinput` and balance the two
  components, at a one-step lag in the roughness. Not decided.
* the wave model's 15 `apply_bc` per sub-step.
* an upwind propagation option with a measured CFL limit would allow the
  ECWAM-style long wave step; wim has `f_limiter = 'upwind'` but it bypasses
  the Fortran kernels and its limit is unmeasured.
* the wave snapshot's `t.npy` is offset by `EXP_SPINUP_DAYS` (the spinup
  advances the wave step counter at t = 0).
* the coarser wave grid is done (`EXP_WAVE_COARSEN`, an integer coarsening
  of the ocean grid, see `../coupled_common/README.md`). It is what
  `../icon_0.1deg` uses at C = 3.
