# simple setups

in `setups/simple` in number of simple idealised configuration are stored either
as jupyter notebook or python script.


# idealised coupled setups

a few more complex idealised experiments are found in:

`setups/acc` stand-alone ocean model, channel model with North Atlantic basin attached

`setups/coupled_acc` spectral wave model coupled to ocean model, channel model with North Atlantic basin attached

`setups/column` spectral wave model coupled to ocean model, single column versions

`setups/langmuir` spectral wave model coupled to ocean model, single column wave model, LES ocean model

`setups/hs_atmos`  stand-alone Held/Suarez idealised atmosphere configuration

`setups/hs_coupled`  Held/Suarez idealised atmosphere coupled to surface wave and ocean model


# realistic coupled setups

Wave-ocean coupled configurations of pyOM3: a spectral wave model
(`wim_realistic`) on the same horizontal grid as a primitive-equation ocean
(`pe_realistic`), or on an integer coarsening of it, with the ocean carrying
the Lagrangian mean velocity.

`setups/coupled_common/`  driver, coupler, the shared model classes

`setups/forcing_common/`  the forcing plug-ins and the input readers

`setups/icon_0.33deg/`   eddy-permitting surface-wave/ocean model

`setups/icon_0.1deg/`    eddy-resolving surface-wave/ocean model


On Levante the same thing is flatter: one directory per RUN beside
`coupled_common` and `forcing_common` (`icon033_spinup`, `icon033_ocean`,
`icon033_coupled`, ...), because the spinup and its two branches are separate
runs with separate restarts and output. Here one directory per CONFIGURATION
is enough, holding the setup module and the batch scripts; the run
directories are made from it on the machine.

A run directory holds its setup module, its batch script and its input files,
and nothing else. It runs

    python3 ../coupled_common/run.py

`run.py` finds the package and `forcing_common` from its own location, and it
puts the current directory LAST on the path, so a stale copy of a coupler
file left in a run directory cannot shadow the shared one. `PYOM3_ROOT`
overrides where the package is found, for a machine where `pyOM3` does not
sit one level above `setups/`.

## The configurations

| directory | setup module | grid | dt | levels | notes |
|---|---|---|---|---|---|
| `global4_era5/` | `global4_setup.py` | 90 x 40, 4 deg | 3600 s | 45 | the test bed; isoneutral + GM |
| `icon_1.5deg/` | `icon15_setup.py` | 240 x 153, 1.5 deg | 1800 s | 40 | Icon R2B4 derived |
| `icon_0.33deg/` | `icon033_setup.py` | 1080 x 689, 1/3 deg | 216 s | 128 | eddy permitting |
| `icon_0.1deg/` | `icon01_setup.py` | 3600 x 2295, 1/10 deg | 72 s | 128 | eddy resolving |
| `pop/` | `pop_era5.py` | 3600 x 2163, 1/10 deg | 216 s | 42 | ocean alone, its own driver |

The others in this directory are not ERA5-forced global configurations and do
not use the machinery described below: `acc/` and `coupled_acc/` (the
idealised ACC channel), `column/`, `langmuir/`, `hs_atmos/` and `hs_coupled/`
(the Held-Suarez atmosphere and its coupled version).

**Resolution against the Icon source grids.** icon033 and icon01 are
interpolated from the Icon-O R2B7 and R2B8 grids onto Mercator grids, so
their cells are square and shrink with cos(latitude). The Icon triangles
are nearly uniform, about 20 km (R2B7) and 10 km (R2B8) by Icon's nominal
measure. That measure is too fine for a comparison with a quadrilateral
grid. Danilov (2022, JAMES) shows that the triangle height
h = 1.32 sqrt(cell area) is the length that compares with a quadrilateral
cell side. This gives the Icon columns of the table.

| latitude | R2B7, h | icon033 | R2B8, h | icon01 |
|---|---|---|---|---|
| equator | 26 km | 37 km | 13 km | 11.1 km |
| 60 deg | 26 km | 18.6 km | 13 km | 5.6 km |
| 70 deg | 26 km | 12.7 km | 13 km | 3.8 km |
| 78.5 deg S, southern edge | 26 km | 7.4 km | 13 km | 2.2 km |

The dynamical effective resolution of the Icon-O triangular C-grid is
coarser still than h, because its spurious modes share the short scales
with the physical modes and have to be damped (Korn and Danilov 2017, JCP).
So icon033 is coarser than R2B7 only equatorward of about 45 deg. icon01 is
finer than R2B8 at all latitudes. Poleward of those latitudes both grids
resolve the flow more finely than the Icon bathymetry they carry.

Each has a `spinup` (ocean alone, repeat-year ERA5), an `ocean` branch
(interannual, ocean alone) and a `coupled` branch (the same period and the
same ocean start, with waves), so the last two differ in the wave model and
nothing else. One batch script serves all three and takes the mode from the
directory name.

## The division of labour

**What a configuration IS** lives in its setup module, about 500 to 800 lines:
the grid source, the parameters (`get_ocean_parameter`, `get_wave_parameter`),
the initial conditions, the snapshot form, the friction that differs from the
package, and `forcing_settings()`.

**Everything else is shared.** The driver, the coupler, the restarts, the
online means, the energy diagnostics and the TKE hooks are in
`coupled_common`; the forcing is in `forcing_common`. Neither names an
experiment or reads the environment for physics, so a new configuration is
one new file and the shared code is tested by all four at once.

Adding a configuration is about a day if the inputs exist: copy the nearest
setup module, change the grid source, the parameters, the initial conditions
and the friction, write `forcing_settings()`, and build the ERA5 set and the
initial conditions with the scripts in `forcing_common`.

## Getting a new configuration running

1. **`grid_topo.nc`** — dxt, dyt, dzt, xu(1), yu(1), kbot, exactly as the
   fortran setup reads them, so there is one source of truth for the grid.
2. **Initial conditions** from the WAGHC climatology:

       python3 ../forcing_common/make_initial_conditions.py \
           --src /pool/data/ICDC/ocean/waghc/DATA --version PYC --month 1 \
           --grid grid_topo.nc --out initial_conditions.nc

   Read three things in its report: how many cells took the level above
   (basins below their sill), the isolated surface bodies with no data at all
   (set `kbot = 0` if one later shows up in the canary), and the
   static-stability check. A good file comes out near 0.01 kg/m3; 0.1 or more
   means a fill crossed a sill, and that has twice been the cause of a
   1/10 degree blow-up.
3. **ERA5** with `prep_era5_native.py`, or point at an existing set: the
   native files carry nothing about the model grid and are shared between
   configurations.
4. **The two offsets** with `measure_era5_offsets.py`, per period and per
   grid. They are not optional: without them the run holds a warm bias and
   the salinity drifts.
5. **A short test** before production. The batch scripts have `TEST = 1` for
   exactly this: a model day or less in two segments, so the restart path is
   exercised.

## Choosing the time step

The hardest-won lesson of this set of experiments, so it is here rather than
in one configuration's README.

icon01 at 1/10 degree crashed after six model days at dt = 144 s. The crash
signature was a two-cell vertical sawtooth in salinity, growing downward from
15 m in one Antarctic shelf column, with temperature perfectly smooth through
the same cells (the column was isothermal, so the same instability had no
gradient to amplify) and u and v untouched. That is a vertical advection CFL
violation and nothing else.

The number: `w` reached 2.0e-2 m/s at 57 m where the cells are 4.6 m, so
`w dt/dz = 0.64`. And `w` is not an independent field, it is the integrated
horizontal divergence, which on a continental slope is `w = u . grad(H)`:
0.25 m/s across a 200 m depth change in one 2.2 km cell gives exactly the
2 cm/s observed.

**So `w` scales with 1/dx for the same topography and flow, and the vertical
CFL scales with dt/dx even though nothing in the vertical changed.** icon033
and icon01 have the SAME vertical grid; icon01's horizontal grid is 3.3 times
finer; icon033 runs at 216 s; 216/3.3 = 65 s. icon01 runs stably at 72.

Two consequences:

* the time step of a configuration follows its HORIZONTAL resolution even
  when the vertical grid is unchanged, and the fortran value is not
  necessarily the right one (`setup1.f90` reads `dt_mom = 144.0/4`, i.e. it
  was walked down by hand);
* POP at the same 2.2 km runs 216 s happily because its 42 levels make the
  cells near 60 m five times thicker.

Run the short test at two or three steps before committing an allocation.
Every factor on the step is the same factor on the cost of the experiment.

## When a run fails

1. **`EXP_CANARY_EVERY=1`** and rerun. The implicit free surface is a global
   solve, so one NaN spreads over the whole domain within one step: at the
   default cadence of 100 the alarm dump is uniformly NaN and says nothing.
   At 1 it is the step the first NaN appeared in, still local. The check
   costs a millisecond or two against a step of order a second.
2. The canary writes the whole step to `<snapshots>_alarm/` and aborts before
   the restart write, so nothing on disk is poisoned.
3. **`crash_snapshot.ipynb`** reads that dump: what went out of bounds and
   where, whether it is one cell, a strait or a patch (the count of open
   neighbours decides), the offending column's profile, the vertical and
   horizontal CFL and the surface forcing over the top cell, and the cell's
   history in the ordinary snapshots.

Three causes have accounted for every failure so far, in this order of
frequency: the initial conditions (a fill crossing a sill), the time step
(the vertical CFL above), and a one-cell puddle or strait in the topography.

## Cost, measured

| | ranks | per step | per model day | per model year |
|---|---|---|---|---|
| icon033 ocean alone | 936 | 0.47 s | 3.1 min | 19 h |
| icon033 coupled, c = 2 | 1024 + 936 | 0.64 s | 4.2 min | 26 h |
| icon01 ocean alone, dt = 72 | 7650 | 0.84 s | 17 min | 4.2 d |
| icon033 ocean alone, GPU | 8 GPUs | 0.12 s | 0.8 min | 4.8 h |

Coupling costs about 30 % at 1/3 degree with the wave grid coarsened by 2,
and the waves then cost nothing in wall time: both components sit at 0 % idle
and the wave model fits its work inside the ocean's step. On the same grid it
was 0.85 s, so the coarsening is what made the coupling affordable.

The GPU row is the JAX backend (`run_gpu.sh`) on 2 Levante GPU nodes, one
process per GPU, with the split-explicit free surface. The CPU rows use the
implicit free surface. That is 5 model years per day, four times the CPU run
on 8 nodes. The wave model does not run on the GPU yet, and icon01 has not
been measured there.

## Output

Large configurations write everything per rank and share no file:

* snapshots and, with `RAW_SNAPSHOTS`, the online means, as `.npy` under
  `pe=<rank>/n=<step>/`, one directory per segment;
* restarts as one `.npz` per rank beside a small header;
* the coupling file and the energy text file on rank 0 only, on their own
  cadence.

The shared-netCDF form is still there and is right up to a few hundred ranks;
at 936 it cost eleven minutes per mean record, and at 1/10 degree it does not
work at all.

Watch the volume before the first production segment. At 1/10 degree one 3-D
field is 8.5 GB, a snapshot about 70 GB and a monthly mean record 100 GB, so
the defaults would write several TB per model year. `EXP_AVERAGE_FIELDS`
restricts the means to what you will actually read.

## Where to look next

* `coupled_common/README.md` — the driver, the coupler, the shared classes,
  the switches, the canary, the coarse wave grid.
* `forcing_common/README.md` — the forcing interface, the two sources, the
  offsets, the block I/O.
* each run directory's own `README.md` — that configuration's grid, physics,
  layout, cost and findings.
