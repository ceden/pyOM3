# forcing_common

Surface forcing as a PLUG-IN, and the input readers the experiments share.

The ocean and the wave model do not know where their forcing comes from. They
hold one forcing object and ask it for the fields at a model time. Which data
source that object is comes from one function in the experiment module, so
changing an experiment from hourly ERA5 to a monthly climatology is a
one-line change and a new source is one new file in this directory.

## The files

| file | what it is |
|---|---|
| `forcing_base.py` | the interface, `make()`, the calendar, `MonthlyClim`, `clean_forcing` |
| `forcing_era5_flux.py` | source `era5`: hourly ERA5 fluxes plus the two climatologies used with them |
| `forcing_ecmwf_clim.py` | source `ecmwf_clim`: the monthly climatology of the pyOM2 setups |
| `forcing_coupled.py` | source `coupled`: the blocks the coupler received this step (the atmosphere's stress and heat flux, or for the waves the ocean's relative-wind stress) |
| `forcing_era5_native.py` | reader: ERA5 files on the ERA5 grid, interpolated per rank at run time; the 10 m wind may sit in a second file set (`wind_tag`) |
| `forcing_era5.py` | reader: ERA5 files already on the model grid |
| `block_io.py` | `local_block`, `BlockScatter`, `read_scatter2`: rank 0 reads, the others receive; the helpers for the two field layouts (MPI block, JAX shards) |
| `prep_era5_native.py` | build the ERA5 file set on the ERA5 grid |
| `prep_era5.py` | build it on a model grid |
| `measure_era5_offsets.py` | the two forcing offsets over a grid's wet cells |
| `make_initial_conditions.py` | initial conditions from the WAGHC climatology (see `../README.md`) |

## The interface

A forcing object serves ONE rank's block:

```python
F = make(settings, geom, fields, who)     # who = 'ocean' | 'wave'
F.get(t)        -> {name: (ny_loc, nx_loc) array} at model time t [s]
F.date(t)       -> the forcing date
F.describe()    -> lines for the log
F.close()
```

Fixed names and units, interior block, no halo, no land mask:

    taux, tauy   N/m^2   eastward / northward surface stress tau_a, T points
    qnet         W/m^2   net surface heat flux INCLUDING the solar part,
                         positive into the ocean, offsets already applied
    qsol         W/m^2   the part of qnet that penetrates
    sst_star     deg C   } restoring: Q = qnet + dqdt (sst_star - SST).
    dqdt         W/m^2/K } Leave both out and there is no restoring.
    emp          m/s     fresh water INTO the ocean (P - E), offset applied
    sss_star     g/kg    SSS restoring target
    u10, v10     m/s     10 m wind; asked for only by an ocean with the
                         relative-wind stress (coupled_common, STRESS_MODE
                         'relwind'), the `era5` source on the native grid has it

**A key that is absent switches its term off.** A source delivers only what it
has, and no switch is needed for the rest. `fields` names what the caller
wants, so the wave model asks for `taux, tauy` and nothing else is read for
it.

`geom` carries the block: `xt`, `yt` (the block's T coordinates),
`block` = (j0, i0, ny_loc, nx_loc) in the grid the FILES are on, `comm`,
`xt_gl`, `yt_gl` for verifying model-grid files, and optionally `wet`.

### What belongs on which side of the line

**In the source**, because it is knowledge about the data: the offsets, the
monthly dQ/dT and SSS climatologies, the fill-value cleaning, the time
interpolation, the file layout.

**In the model** (`../coupled_common/exp_ocean.py`), because it is physics and
the same for every source: the ice rule, the shortwave penetration profile,
the conversion to kinematic fluxes, the average of tau to U and V points, the
direct branch tau_a - tau_in, and the virtual salt flux from `emp`.

## Choosing a source

One function in the experiment module:

```python
def forcing_settings():
    src = os.environ.get('EXP_FORCING', 'era5')
    if src == 'era5':
        return {'source': 'era5', 'grid': 'native',
                'dir': FORCING_DIR, 'tag': FORCING_TAG,
                'start': FORCING_START, 'cycle_days': FORCING_CYCLE_DAYS,
                'qnet_offset': QNET_OFFSET, 'emp_offset': EMP_OFFSET,
                'freshwater': FRESHWATER,
                'clim': {'dqdt': (..., 'dqdt'), 'sss': (..., 'sss')},
                'clim_scatter': True}
    ...
```

`clim_scatter` decides how the two climatologies are read: `True` sends rank 0
through the file record by record and scatters the blocks, which is what a
large grid needs; `False` lets every rank read its own hyperslab, which is
fine up to a few hundred ranks.

### source `era5`

Hourly ERA5 fluxes. `grid` picks the reader:

* **`native`** — `era5_<tag>_YYYY-MM.nc` on the ERA5 grid (1280 x 640).
  Rank 0 reads one record and broadcasts it, every rank interpolates
  bilinearly to its own block with weights built once. A copy on a finer
  model grid is several times the storage and no more information, and it
  needs remapping weights and a coastal fill for that grid. e and tp are in
  the same files.
* **`model`** — `era5_<tag>_YYYY-MM.nc` already on the model grid
  (`prep_era5.py`); every rank reads its own hyperslab. e and tp are a
  SECOND set, `fw_tag`.

`qnet = sshf + slhf + ssr + str - qnet_offset`, `qsol = ssr`,
`sst_star = sst`, `emp = e + tp - emp_offset` (ERA5's e is negative where
water leaves).

### source `ecmwf_clim`

The monthly climatology of the pyOM2 setups: taux, tauy, qnet, swf, dqdt, sst
and sss on the model grid, interpolated on the calendar day with pyOM2's own
weights. Q = qnet + dqdt (sst - SST), swf penetrates, salinity is restored to
sss, as in `setup1.f90`.

Two differences to pyOM2. The stress is handed over on T points like every
other source and the model averages it to U and V, where pyOM2 takes the file
values at U and V as they are. And for the WAVE model a monthly mean stress
has no storms, so u* from it is far too low for realistic wave growth; a
coupled run with this source is a technical option only, and the source says
so once in the log.

## The offsets, and why they are not a model parameter

ERA5 runs on a PRESCRIBED sea surface temperature, so its surface energy
budget is free and does not close: over a model domain the annual mean of
sshf + slhf + ssr + str is several W/m2 into the ocean where the real ocean
absorbs about 0.5. The model cannot export that heat, so the SST restoring
carries it back out by holding a global warm bias of offset/(dQ/dT). The same
holds for P - E, which is not zero because there is no runoff.

Both numbers are properties of the FORCING AND THE DOMAIN, not of the model,
so they are measured before the run:

    python3 measure_era5_offsets.py 1990-05 1991-04 \
            --grid grid_topo.nc --dir ../era5_icon033 --tag icon033 \
            --stride 7 --workers 16

It handles both file layouts, reading off the first file which one it is, and
on the ERA5 grid it exploits that bilinear interpolation is linear: the
area-weighted mean over the model's wet cells of an interpolated field is a
weighted sum of the ERA5 field itself, so nothing is interpolated and every
record is a dot product.

**`--stride 7`, not 6.** The solar radiation has a diurnal cycle of 200 W/m2,
so a stride that divides 24 samples the same few phases every day and biases
the mean by more than the offset being measured. A stride coprime with 24
walks through all of them.

Measure it PER PERIOD and PER GRID: the spinup's repeat year and the branch
period are different numbers, and the two branches must use the same one or
they are not comparable. A grid change moves it too, because the wet cells
change, even when the forcing files do not.

## Sharing an ERA5 set between experiments

The `native` files carry nothing about the model grid, so one set serves any
number of configurations: point `EXP_FORCING_DIR` and `EXP_FORCING_TAG` at
it. Only the offsets have to be remeasured for the new grid.

## The 10 m wind for the relative-wind stress

ERA5's stress is the IFS bulk formula on the absolute wind. An ocean with
`STRESS_MODE 'relwind'` (coupled_common/exp_ocean.py) keeps that drag
coefficient, C_D rho_a = |tau_E|/|u10|^2, and forms the stress from the wind
relative to its surface current, so it asks the source for `u10, v10` as
well. They are not in the default file set; build them once as their own set
beside it,

    python3 prep_era5_native.py 1990-05 1991-04 --out DIR --tag <tag>w --fields u10,v10

and give the experiment the `wind_tag` (`EXP_FORCING_WIND_TAG`). The reader
opens that set next to the main one; without a `wind_tag` it looks for the
wind in the main files. The native grid only, the model-grid reader has no
wind. A set without `ssr` is written without the solar checks, as the e,tp
set is.

## Block I/O

`block_io.py` exists because the input files are contiguous float32 and a
hyperslab read by every rank is, per level and rank, dozens of strided
fragments of a few hundred bytes on one shared file. At a thousand ranks that
is latency bound and costs minutes of setup. `BlockScatter` has rank 0 read
one full 2-D record, pack the blocks in rank order and scatter them with a
single `MPI_Scatterv`. The communicator is the MODEL's, not `COMM_WORLD`: in a
coupled run the ocean sits on a sub-communicator next to the wave ranks and a
`COMM_WORLD` collective would deadlock.

## On the JAX backend

On the JAX backend one process holds the whole grid, so a block is the whole
grid (`local_block`) and every process reads the forcing for all of it. With
several processes (one per GPU) each process reads the same records. The
model's sharded arrays then take each process's piece.

The arithmetic runs on the device. Nothing in the interface changes, and
`F.get(t)` returns device arrays.

* `forcing_era5_native.py` sends each hour's records to the device as one
  float32 buffer. The bilinear interpolation to the model grid is a jitted
  gather there, and the time interpolation a jitted blend.
* `forcing_era5_flux.py` forms the interface fields (qnet, emp, the
  climatology blend) in ONE jitted kernel.
* `MonthlyClim` keeps its 12 records on the device.

On the host this arithmetic was 60 ms per step on icon033, plus 20 ms for
the hourly interpolation, a quarter of a 0.31 s GPU step. On the device it
is a few kernels. The MPI/numpy path is unchanged.

`block_io.py` also holds the helpers the shared code uses for the JAX field
layout, where every shard's halo sits inline in the array: `pad_block`,
`interior`, `local_shard`, `shard_blocks` (the per-shard raw restart) and
`jax_barrier`. They are described in `../coupled_common/README.md`.

## Adding a source

One file, one class deriving from `forcing_base.Forcing`, one `get(t)`
returning the dict above, and one line in `forcing_base.SOURCES`. The
calendar, the cycling and the monthly-climatology reader come from the base
class. Nothing in `coupled_common` changes.
