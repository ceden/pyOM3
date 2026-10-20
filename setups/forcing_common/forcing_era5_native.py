"""
ERA5 surface forcing on the NATIVE ERA5 grid: read the monthly files of
prep_era5_native.py (ERA5 grid, hourly) and deliver them on one rank's block
of the model grid, interpolated in space and in time.

WHY INTERPOLATE IN THE MODEL. The files stay on the ERA5 grid (1280 x 640).
A copy on a finer model grid is several times the storage and no extra
information, and it needs remapping weights and a coastal fill for that grid.
The reader turns the records into the model grid while the run goes on.

HOW IT WORKS. Rank 0 reads one record (3.3 MB, contiguous: the files are
chunked one record per chunk) and broadcasts it; every rank then interpolates
BILINEARLY to its own block with weights built once at setup. Per model hour
that is 7 fields x 2 records = 46 MB of broadcast, against ~0.5 s of compute
per time step, and the same records serve all 16.7 time steps of that hour
(dt = 216 s), so the cost is negligible. Nothing is read per rank, so the
1536-rank strided-read problem of the initial conditions does not arise.

The land points of the ERA5 files carry their nearest ocean value
(prep_era5_native.py), so a stencil that straddles a coast never picks up a land
value; the model's own land mask decides what is used.

CALENDAR AND TIME INTERPOLATION are those of the model-grid reader: model time
t = 0 is FORCING_START at 00 UTC, a cycle of C days repeats the year
(C = 365) and C = 0 runs through the calendar; the six flux fields are means
over their hour and therefore sit at the middle of it, sst sits at its stamp.
"""
import os
import calendar
import datetime
import numpy as np
from netCDF4 import Dataset

MEAN_FIELDS  = ('taux', 'tauy', 'sshf', 'slhf', 'ssr', 'str', 'e', 'tp')
INST_FIELDS  = ('sst',)
WIND_FIELDS  = ('u10', 'v10')      # analyses at their stamp too, their own file set (wind_tag)
FILE_PATTERN = 'era5_%s_%04d-%02d.nc'
DEFAULT_TAG  = 'icon033'


def parse_start(s):
    if isinstance(s, datetime.datetime):
        return s
    return datetime.datetime.strptime(str(s).strip(), '%Y-%m-%d')


class Era5NativeForcing:
    """Time- and space-interpolated ERA5 on one rank's interior block.

    directory   holds era5_<tag>_YYYY-MM.nc
    start       date of model time 0 ('YYYY-MM-DD', 00 UTC)
    cycle_days  365 = repeat year, 0 = through the calendar
    xt, yt      the rank's INTERIOR T coordinates [degrees], 1-D
    names       the fields this run needs
    tag         name part of the files, era5_<tag>_YYYY-MM.nc
    comm        MPI communicator of the run (rank 0 reads and broadcasts)
    """

    def __init__(self, directory, start, cycle_days, xt, yt, names, comm,
                 tag=DEFAULT_TAG, wind_tag=None):
        self.dir = directory
        # name part of the files, era5_<tag>_YYYY-MM.nc, as given to
        # prep_era5_native.py --tag. One experiment, one set; the 10 m wind
        # (u10, v10) may sit in a second set, era5_<wind_tag>_YYYY-MM.nc,
        # built later with --fields u10,v10 (None: the same files)
        self.tag = str(tag)
        self.wind_tag = str(wind_tag) if wind_tag else self.tag
        self.start = parse_start(start)
        self.cycle = int(round(float(cycle_days)*24.0))       # hours, 0 = calendar
        if abs(self.cycle - float(cycle_days)*24.0) > 1e-9:
            raise ValueError('forcing cycle %g days is not a whole number of hours'
                             % cycle_days)
        for n in names:
            if n not in MEAN_FIELDS + INST_FIELDS + WIND_FIELDS:
                raise ValueError('unknown ERA5 forcing field %s' % n)
        self.names = tuple(names)
        self.comm = comm
        self.xt = np.asarray(xt, np.float64)
        self.yt = np.asarray(yt, np.float64)
        self._mean = tuple(n for n in self.names if n in MEAN_FIELDS)
        self._inst = tuple(n for n in self.names if n in INST_FIELDS)
        self._wind = tuple(n for n in self.names if n in WIND_FIELDS)
        self._rec = {}          # (group, year, month, record) -> {name: block}
        self._ds = {}           # tag -> (the month's open file, (year, month)), rank 0
        self._last_t, self._last = None, None
        self._weights()

    # ------------------------------------------------------------------
    def _weights(self):
        """Bilinear stencil of every model point in the ERA5 grid. Built once.
        Longitude is cyclic, latitude is clipped at the poles (ERA5's first and
        last rows are at +-89.78 degrees)."""
        y0, m0 = self.start.year, self.start.month
        fname = os.path.join(self.dir, FILE_PATTERN % (self.tag, y0, m0))
        if not os.path.exists(fname):
            raise FileNotFoundError('ERA5 forcing file %s is missing' % fname)
        with Dataset(fname) as f:
            lat = np.asarray(f.variables['lat'][:], np.float64)
            lon = np.asarray(f.variables['lon'][:], np.float64)
        self.flip = lat[0] > lat[-1]          # ERA5 runs north -> south
        self.lat = lat[::-1] if self.flip else lat
        self.lon = lon
        self.ne_y, self.ne_x = len(lat), len(lon)

        # longitude: uniform and cyclic
        dlon = self.lon[1] - self.lon[0]
        p = np.mod(self.xt - self.lon[0], 360.0)/dlon
        i0 = np.floor(p).astype(int)
        self.wx = (p - i0)[None, :]
        self.i0 = np.mod(i0, self.ne_x)
        self.i1 = np.mod(i0 + 1, self.ne_x)
        # latitude: Gaussian, so not uniform -> searchsorted
        yq = np.clip(self.yt, self.lat[0], self.lat[-1])
        j0 = np.clip(np.searchsorted(self.lat, yq) - 1, 0, self.ne_y - 2)
        self.j0, self.j1 = j0, j0 + 1
        self.wy = ((yq - self.lat[j0])/(self.lat[j0+1] - self.lat[j0]))[:, None]

    def _interp(self, a):
        """ERA5 field (ne_y, ne_x) -> this rank's block (ny_loc, nx_loc)."""
        if self.flip:
            a = a[::-1]
        f00 = a[np.ix_(self.j0, self.i0)]
        f01 = a[np.ix_(self.j0, self.i1)]
        f10 = a[np.ix_(self.j1, self.i0)]
        f11 = a[np.ix_(self.j1, self.i1)]
        return ((1.0 - self.wy)*((1.0 - self.wx)*f00 + self.wx*f01)
                + self.wy*((1.0 - self.wx)*f10 + self.wx*f11))

    # ------------------------------------------------------------------
    def date(self, t):
        """Forcing date of model time t [s]."""
        s = float(t)
        if self.cycle:
            s = np.mod(s, self.cycle*3600.0)
        return self.start + datetime.timedelta(seconds=s)

    def climatology_time(self, t):
        """Seconds since 1 January of the forcing year, for the monthly
        climatologies (dQ/dT, SSS) that stay in use."""
        d = self.date(t)
        return (d - datetime.datetime(d.year, 1, 1)).total_seconds()

    def _hour(self, k):
        """Record k (hours since start, cycled) -> (year, month, record)."""
        if self.cycle:
            k = k % self.cycle
        d = self.start + datetime.timedelta(hours=int(k))
        m0 = datetime.datetime(d.year, d.month, 1)
        return d.year, d.month, int((d - m0).total_seconds())//3600

    def _file(self, y, m, tag=None):
        """The month's Dataset of the file set `tag`, kept OPEN between
        records (rank 0 only).

        One record is a contiguous 2 MB chunk, but an HDF5 open of the
        monthly file is not free and there is one record per field per model
        hour. Opening per record, as this reader first did, cost the whole
        job 0.4 s of every ocean step: every other rank sits in the broadcast
        below until rank 0 has finished its I/O, so one rank's opens are paid
        by all of them."""
        tag = tag or self.tag
        ds, ym = self._ds.get(tag, (None, None))
        if ym == (y, m):
            return ds
        if ds is not None:
            ds.close()
            self._ds.pop(tag, None)
        fname = os.path.join(self.dir, FILE_PATTERN % (tag, y, m))
        if not os.path.exists(fname):
            raise FileNotFoundError('ERA5 forcing file %s is missing. The run '
                                    'needs %04d-%02d.' % (fname, y, m))
        f = Dataset(fname)
        f.set_auto_mask(False)
        nt = len(f.dimensions['time'])
        if nt != calendar.monthrange(y, m)[1]*24:
            f.close()
            raise ValueError('%s has %d records, %04d-%02d has %d hours'
                             % (fname, nt, y, m, calendar.monthrange(y, m)[1]*24))
        self._ds[tag] = (f, (y, m))
        return f

    def _fetch(self, group, k, tag=None):
        """{name: interpolated block} of the fields in `group` at absolute
        record k, read once and cached.

        The whole group travels in ONE broadcast of one float32 buffer:
        mpi4py's lowercase bcast pickles the array on rank 0 and unpickles it
        on every other rank, and the shape is known everywhere, so Bcast into
        a preallocated buffer moves the same bytes without any of that. The
        file holds float32 and the interpolation promotes to float64 after
        the broadcast, so the message is half the size as well.

        COLLECTIVE: every rank must call this the same number of times. It is
        driven by the model time alone, which is identical on all of them, so
        the cache hits and misses fall the same way everywhere."""
        y, m, r = self._hour(k)
        key = (group, y, m, r)
        if key in self._rec:
            return self._rec[key]
        buf = np.empty((len(group), self.ne_y, self.ne_x), np.float32)
        if self.comm is None or self.comm.Get_rank() == 0:
            f = self._file(y, m, tag)
            for i, n in enumerate(group):
                buf[i] = f.variables[n][r]
        if self.comm is not None:
            self.comm.Bcast(buf, root=0)
        out = {n: self._interp(np.asarray(buf[i], np.float64))
               for i, n in enumerate(group)}
        if len(self._rec) > 6:            # two records of each of the three groups
            self._rec.clear()
        self._rec[key] = out
        return out

    # ------------------------------------------------------------------
    def get(self, t):
        """dict name -> (ny_loc, nx_loc) field at model time t [s]."""
        if self._last_t is not None and float(t) == self._last_t:
            return self._last
        h = float(t)/3600.0
        out = {}
        # the hourly means sit at the middle of their hour, the analysed sst
        # at its stamp, so the two groups straddle different record pairs and
        # are fetched separately -- two broadcasts each per model hour
        for group, shift, tag in ((self._mean, 0.5, self.tag), (self._inst, 0.0, self.tag),
                                  (self._wind, 0.0, self.wind_tag)):
            if not group:
                continue
            p = h - shift
            k1 = int(np.floor(p))
            w = p - k1
            a, b = self._fetch(group, k1, tag), self._fetch(group, k1 + 1, tag)
            for n in group:
                out[n] = (1.0 - w)*a[n] + w*b[n]
        self._last_t, self._last = float(t), out
        return out

    def close(self):
        for ds, ym in self._ds.values():
            ds.close()
        self._ds = {}
