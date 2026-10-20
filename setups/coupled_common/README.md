# coupled_common

The configuration-independent half of the coupled experiments: the driver,
the MPI plumbing between the two models, the I/O, and the two model classes
everything shares. Nothing here reads the environment for physics and nothing
here names an experiment. What a configuration IS lives in its own setup
module (see `../README.md`), what its FORCING is lives in
`../forcing_common`.

    pyOM3/setups/
      coupled_common/   this directory
      forcing_common/   the forcing sources and the input readers
      <configuration>/  setup module, batch script, input files, output

An experiment directory runs

    python3 ../coupled_common/run.py

`run.py` finds the package, `forcing_common` and the experiment module from
its own location; the current directory comes LAST on the path, so a stale
copy of a coupler file left in a run directory cannot shadow the shared one.
Set `PYOM3_ROOT` if the package does not sit one level above `setups/`.

## The files

| file | what it is |
|---|---|
| `run.py` | the driver: reads the environment, builds the two models, splits the ranks, hands over to the loop |
| `coupler_loop.py` | the three time loops (wave, coupled ocean, ocean alone), the canary, the diagnostics |
| `coupler_mpi.py` | world split, intercommunicator, gather/scatter, the block-to-block exchange and the coarse wave grid |
| `coupler_io.py` | restarts (gathered and raw), the coupling file, the snapshot helpers |
| `coupler_prof.py` | the profiler behind the tables in the log |
| `exp_common.py` | `Cfg`, the averaging-mode parser, the coarse-grid helpers, the grid reports |
| `exp_ocean.py` | `ocean_common`: forcing, coupling terms, TKE hooks, energy diagnostics, online means |
| `exp_wave.py` | `wave_common`: the wave model's mask, wind and energy diagnostics |
| `ni_diag.py` | `ni_common`: the near-inertial energy flux out of the mixed layer, online (Rimac et al. 2016) |
| `ni_report.py` | offline: the near-inertial numbers, maps and seasonal cycle from the means (`-j` parallel) |
| `heat_transport.py` | offline: meridional heat and freshwater transport and the overturning per basin from the means (`-j` parallel) |
| `transport_diff.py` | two `heat_transport.py` outputs side by side, table and figure, e.g. with and without waves |
| `bx_coarse_test.py` | MPI test of the block exchange with a coarsened wave grid |

## How a configuration plugs in

The experiment module defines two classes and a handful of module-level
names:

```python
from exp_common import Cfg
from exp_ocean  import ocean_common
from exp_wave   import wave_common

class wave(wave_common, wave_base):
    cfg = Cfg(sys.modules[__name__])

class ocean(ocean_common, tke_mixin, ocean_base):
    cfg = Cfg(sys.modules[__name__])
```

`Cfg` is the view the shared classes have of the experiment. They never read
the environment and never name a configuration: every number they need is an
attribute of the experiment module, reached as `self.cfg.NAME`, and a name
the module does not define falls back to `Cfg.DEFAULTS`, so an older module
keeps working when a switch is added here.

Names the shared classes read: `rho_0`, `cp_0`, `t_rest`, `omega_e`, `nz`,
`exp_name`, `kbot_global()`, `dzt_global()`, `zw_pyom2()`,
`forcing_settings()`, `rho_air`, and the switches below.

### Switches (`Cfg.DEFAULTS`, override in the experiment module)

| name | default | what it does |
|---|---|---|
| `KE_BUDGET` | True | re-evaluate every KE tendency each step and check the identities |
| `AVERAGE_MLD` | True | mixed layer depth in the online means (a full density call per step) |
| `AVERAGE_EDDY` | False | rho and the three density fluxes, for the eddy fluxes |
| `AVERAGE_TRACER_FLUX` | False | the three advective fluxes of temp and salt with the centred face value (uT, vT, wT, uS, vS, wS), for `heat_transport.py` |
| `AVERAGE_FIELDS` | [] | restrict the online means to these names |
| `RAW_SNAPSHOTS` | False | per-rank `.npy` snapshots AND means instead of shared netCDF |
| `SW_BOTTOM` | False | absorb the solar radiation reaching the sea floor in the deepest cell |
| `SSS_PISTON` | 0.0 | m/day; 0 = dz(top)/t_rest |
| `direct_stress_branch` | False | hand tau_a - tau_in to the ocean as well |
| `wave_coarsen` | 1 | the wave grid is the ocean grid coarsened by this factor |
| `ICE_TAU_FRAC` | 1.0 | fraction of the surface stress reaching the ocean under ice |
| `ICE_T_FREEZE` | -1.8 | the freezing point that mask uses |
| `STRESS_MODE` | 'era5' | the forced ocean's stress: 'era5' as delivered; 'relwind' keeps ERA5's drag coefficient C_D rho_a = \|tau_E\|/\|u10\|^2 and forms tau from the wind relative to the surface current (needs u10, v10 in the forcing, `wind_tag`); the waves then take the stress from the ocean (tags 9, 10, 'coupled' source) and the means get u10, v10, cd_rho, d_int |
| `STRESS_UO` | 'eulerian' | the surface current of that: u_L - u_S, or 'lagrangian' u_L |
| `STRESS_U10_MIN` | 1.0 | floor of \|u10\| in the drag coefficient [m/s] |
| `STRESS_CD_MAX` | 3e-3 | cap on the drag coefficient (hourly-mean stress against an instantaneous wind) |
| `NI_DIAG` | False | the near-inertial diagnostic of `ni_diag.py`: complex demodulation at the local f of w, p, u + iv and the stress every step; the band products go into the online means (`ni_pw`, `pw` at every W face; at the SLOPING mixed layer base the vertical part `ni_pw_mld`, `pw_mld` and the horizontal part through the slope `ni_ph_mld`, `ph_mld` = p u dh/dx + p v dh/dy, with `ni_pu_mld`, `ni_pv_mld`; `ni_ke`, `ni_wind`, `wind`, `p`, `mld_ni`). The flux out of the mixed layer is -(ni_pw_mld + ni_ph_mld), as in Rimac et al., the filter states into the raw restart. About 25 ms per step on a 100k-cell block |
| `NI_BAND` | 0.2 | relative half width of the band around f (filter time scale 1/(NI_BAND f)) |
| `NI_LEVELS` | 0 | levels from the surface the diagnostic covers, 0 = all |
| `NI_LAT_MIN` | 5.0 | f is held at this latitude towards the equator |
| `NI_MLD_EVERY` | 50 | steps between evaluations of the mixed layer depth (one density call) |
| `NI_DUMP_EVERY` | 0 | steps between float32 dumps of u, v, w, p of those levels into `<snapshots>_ni/`, for the offline analysis; 0 = none |

`register_average(name, dims, units, long_name, get)` on the ocean class adds
a field to the online means from the experiment module, pyOM2's
`register_average` in this machinery. Call it before `run.py` calls
`average_setup`, i.e. in the module's own `prepare_for_run` after `super()`.

## The environment

`run.py` reads it; the list is in its docstring and it is all named `EXP_*`
(global4 keeps its own `GLOBAL4_*` prefix for the physics, which is why the
model classes must not read the environment themselves). The ones that decide
a run rather than a setting:

    EXP_SETUP              the experiment module (no default)
    EXP_WAVES              1 coupled, 0 ocean alone
    EXP_RUNLEN_DAYS        total length; EXP_SEGMENT_DAYS this job's share
    EXP_INIT_OCEAN         restart of ANOTHER run to branch from
    EXP_INIT_CHECK         0 = only warn when that state is not on a whole
                           repeat year (the forcing jumps there)
    EXP_NPX/NPY_WAVE       the two decompositions; the wave one divides the
    EXP_NPX/NPY_OCEAN      COARSE grid when EXP_WAVE_COARSEN > 1
    EXP_RESTART_RAW        1 = per-rank restart blocks (see below)
    EXP_CANARY_EVERY       how often the state is checked (see below)
    EXP_PROFILE            0, 1 sections, 2 + model internals

## Three things worth knowing before a large run

**A raw restart belongs to its decomposition.** With `EXP_RESTART_RAW=1` the
restart is a small header file beside a directory of one `.npz` per rank, and
it can only be read on the layout that wrote it. Reading dispatches on the
file, so a run that writes raw restarts can still branch from a gathered
`EXP_INIT_OCEAN` written on another layout. To change layout mid-experiment,
rerun the last segment with `EXP_RESTART_RAW=0`.

**The canary is what stops a run before it wastes a job.** Every
`EXP_CANARY_EVERY` steps (default 100) it checks `u, v, temp, salt` against
`CANARY_BOUNDS` in `coupler_loop.py` and `p_s` for non-finite. On an alarm it
prints where each offending cell is, writes the whole step as a raw snapshot
under `<snapshots>_alarm/`, and aborts BEFORE the restart write, so nothing on
disk is poisoned. The implicit free surface is a GLOBAL solve, so one NaN
spreads over the whole domain within one step: at the default cadence the
dump is uniformly NaN and says nothing. **Set `EXP_CANARY_EVERY=1` for any run
that is failing.** The check costs a millisecond or two against a step of
order a second.

**The wave grid can be coarser than the ocean grid.** `EXP_WAVE_COARSEN = c`
makes a wave cell c x c ocean cells, with wave cell faces on ocean cell
faces, so the mapping is an index map and nothing is interpolated. The
exchange stays block to block: wave to ocean is a copy (all the exchanged
fields are per unit sea area), ocean to wave is the area mean over the wet
ocean cells of each wave cell, reduced on the ocean side and normalised on
the wave side with weights exchanged once. Measured at 1/3 degree, c = 2:
0.64 s per coupling step against 0.85 s on the same grid, on 16 nodes instead
of 40. `bx_coarse_test.py` checks the exchange on any rank count:

    srun -n 13 python3 bx_coarse_test.py 3 53 60      # c, ny, nx

## The coupling, in one paragraph

The ocean's prognostic velocity is the Lagrangian mean u_L. The wave model
sends the Stokes drift, the stress released to the ocean tau_o, the TKE flux
Phi_oc, the wave-supported stress tau_in and the significant wave height; the
ocean sends its velocity, as the area mean over each wave cell when the wave
grid is coarser. The ocean adds the vortex force -u_L x (nabla x u_S) in its
hydrostatic truncation, takes tau_o as its surface stress, drives the TKE
closure with Phi_oc and sets the surface mixing length from Hs.
`EXP_DIRECT_BRANCH=1` hands the part of the wind stress the waves did not
carry, tau_a - tau_in, to the ocean as well, so the air-sea interface
conserves momentum and the coupled run is comparable with its ocean-alone
twin.

## What is NOT here

Grid, parameters, initial conditions, friction: the experiment module.
Forcing of any kind: `../forcing_common`. The numerics: `pyOM3` itself.
