"""
SURFACE FORCING AS A PLUG-IN. The ocean and the wave model do not know where
their surface forcing comes from: they hold one forcing object and ask it for
the fields at a model time. What a data source is (hourly ERA5 on either
grid, a monthly climatology, something else later) is decided by one entry of
the experiment module, and a new source is one new file in this directory.

THE INTERFACE. A forcing object serves ONE rank's block:

    F = make(settings, geom, fields, who)
    F.get(t)   -> dict of (ny_loc, nx_loc) arrays at model time t [s],
                  INTERIOR block, no halo, no land mask, in fixed names/units
    F.date(t)  -> forcing date (datetime) of model time t
    F.describe()  -> lines for the log
    F.close()

    taux, tauy   N/m^2   eastward / northward surface stress tau_a, T points
    qnet         W/m^2   net surface heat flux INCLUDING the solar part,
                         positive into the ocean, offsets already applied
    qsol         W/m^2   the solar part of qnet, the one that penetrates
    sst_star     deg C   } restoring: Q = qnet + dqdt (sst_star - SST). Leave
    dqdt         W/m^2/K } both out and there is none.
    emp          m/s     fresh water INTO the ocean (P - E), offset applied;
                         absent = no freshwater flux
    sss_star     g/kg    SSS restoring target; absent = no restoring
    u10, v10     m/s     10 m wind, only asked for by an ocean with the
                         relative-wind stress (STRESS_MODE 'relwind')

A key that is absent switches its term off in the ocean, so a source only
delivers what it has. `fields` names what the caller wants; the wave model
asks for ('taux', 'tauy') and nothing else is read for it.

geom   dict: xt, yt   1-D coordinates [degrees] of the block's T points
             block    (j0, i0, ny_loc, nx_loc) of the block in the global
                      grid THE INPUT FILES ARE ON (the ocean grid)
             comm     the component's communicator (rank 0 reads)
             xt_gl, yt_gl  global coordinates, to verify model-grid files
             wet      optional (ny_loc, nx_loc) bool, for the cleaning
       With a coarser wave grid the wave model passes the OCEAN cells under
       its block here and averages what it gets back (exp_wave.py).

What stays in the MODEL and is the same for every source: the land mask, the
average of tau to U/V points, the ice rule, the shortwave penetration profile,
the conversion to kinematic fluxes, the direct branch tau_a - tau_in, the SSS
piston velocity, the virtual salt flux S*emp.
"""
import datetime
import importlib
import numpy as np
import pyOM3 as OM

# source name -> (module in this directory, class)
SOURCES = {'era5': ('forcing_era5_flux', 'Era5FluxForcing'),
           'ecmwf_clim': ('forcing_ecmwf_clim', 'EcmwfClimForcing'),
           'coupled': ('forcing_coupled', 'CoupledForcing')}      # the atmosphere model


def make(settings, geom, fields, who='ocean'):
    """Build the forcing object settings['source'] names."""
    name = settings['source']
    if name not in SOURCES:
        raise ValueError('unknown forcing source %r; available: %s'
                         % (name, ', '.join(sorted(SOURCES))))
    mod, cls = SOURCES[name]
    return getattr(importlib.import_module(mod), cls)(settings, geom, fields, who)


def clean_forcing(a, name, limit=None, fill=0.0, wet=None):
    """Replace non-finite entries, missing-value fills and (optionally)
    anything beyond `limit` in magnitude, and say how many cells were hit.

    This exists because the FILL VALUE IS NOT HARMLESS on the wave side. The
    ocean multiplies its forcing by a land mask and never sees it; the wave
    model turns stress into a friction velocity, u* = sqrt(|tau|/rho_air), so
    a single -1e34 cell becomes u* ~ 1e17 m/s. The WAMDI growth rate is
    linear in u* and `limit_growth` scales WITH u*, so nothing caps it: the
    measured result is ~1e15 e-folds per time step, i.e. the whole basin is
    destroyed within four steps, before the first coupling. If the
    topography and the forcing file disagree about a single coastal cell --
    and they do not have to agree, they are different files -- that is
    enough."""
    b = np.array(a, np.float64, copy=True)
    bad = ~np.isfinite(b)
    if limit is not None:
        bad |= np.abs(b) > limit
    if wet is not None:
        bad |= ~np.broadcast_to(wet.astype(bool), b.shape)
    n = int(bad.sum())
    if n:
        b = np.where(bad, fill, b)
    return b, n


def get_periodic_interval(current_time, cycle_length, rec_spacing, nbrec):
    """pyOM2's monthly-climatology interpolation weights (verbatim)."""
    loc_time = (current_time - rec_spacing*0.5
                + cycle_length*(2 - np.rint(current_time/cycle_length)))
    tmp_time = np.mod(loc_time, cycle_length)
    trec1 = 1 + int(tmp_time/rec_spacing)
    trec2 = 1 + np.mod(trec1, nbrec)
    wght2 = (tmp_time - rec_spacing*(trec1 - 1))/rec_spacing
    return trec1-1, trec2-1, 1.0-wght2, wght2


def parse_start(s):
    if isinstance(s, datetime.datetime):
        return s
    return datetime.datetime.strptime(str(s).strip(), '%Y-%m-%d')


class Forcing:
    """Base class: the calendar every source shares. Model time 0 is
    settings['start'] at 00 UTC; a cycle of C days repeats (365 = repeat
    year), C = 0 runs through the calendar."""

    def __init__(self, settings, geom, fields, who):
        self.settings, self.geom, self.who = settings, geom, who
        self.fields = tuple(fields)
        self.start = parse_start(settings['start'])
        self.cycle_days = float(settings.get('cycle_days', 365.0))
        self.comm = geom.get('comm')
        self.rank = 0 if self.comm is None else self.comm.Get_rank()

    def date(self, t):
        s = float(t)
        if self.cycle_days:
            s = np.mod(s, self.cycle_days*86400.0)
        return self.start + datetime.timedelta(seconds=s)

    def climatology_time(self, t):
        """Seconds since 1 January of the forcing year: the time axis of a
        monthly climatology."""
        d = self.date(t)
        return (d - datetime.datetime(d.year, 1, 1)).total_seconds()

    def describe(self):
        return []

    def close(self):
        return


class MonthlyClim:
    """12 monthly records of one variable on this rank's block, interpolated
    on the calendar day with pyOM2's weights (get_periodic_interval).

    spec     (path, variable name)
    scatter  True: rank 0 reads record by record and scatters (large grids);
             False: every rank reads the variable and cuts its block"""

    def __init__(self, spec, geom, scatter, label, who, limit=1e5):
        from block_io import BlockScatter
        from netCDF4 import Dataset
        path, vname = spec
        j0, i0, nyl, nxl = (int(v) for v in geom['block'])
        comm = geom.get('comm')
        rank = (OM.jax.process_index() if OM.jax else 0) if comm is None else comm.Get_rank()
        if scatter:
            sc = BlockScatter(comm, geom['block'])
            rec = np.empty((12, nyl, nxl))
            for n in range(12):
                rec[n], nbad = clean_forcing(sc.read(path, vname, (n,)), vname,
                                             limit=limit)
                if nbad and (rank == 0 or not OM.jax):   # every jax process reads the whole field; one reports
                    print('%s forcing pe %d %s record %d: %d non-finite/fill '
                          'entries zeroed' % (who, rank, vname, n, nbad), flush=True)
        else:
            with Dataset(path) as f:
                # raw values, no masked array: the fill values are wanted
                # here, clean_forcing below decides what to do with them.
                # It also stops netCDF4 warning once per rank and variable
                # that it cannot cast the file's missing_value.
                f.set_auto_mask(False)
                a = np.asarray(f.variables[vname][:], np.float64)
            rec, nbad = clean_forcing(a[:, j0:j0+nyl, i0:i0+nxl], vname, limit=limit)
            if nbad and rank == 0:
                print('%s forcing %s: %d non-finite/fill entries zeroed'
                      % (who, vname, nbad), flush=True)
        # on the jax backend the 12 records live on the device, so at() is a
        # device kernel and the result needs no host-to-device copy per step
        self.rec = OM.np.asarray(rec) if OM.jax else rec

    def at(self, t_year):
        yr = 365.0*86400.0
        n1, n2, f1, f2 = get_periodic_interval(t_year, yr, yr/12.0, 12)
        return f1*self.rec[n1] + f2*self.rec[n2]
