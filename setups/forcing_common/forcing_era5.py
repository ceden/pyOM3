"""
ERA5 surface forcing on a model grid: read the monthly files written by
prep_era5.py and interpolate them in time on one PE's block.

FILES. One file per calendar month, era5_<tag>_YYYY-MM.nc, on the model T
grid (xt, yt of the interior, south to north), float32. The tag is the one
prep_era5.py was given, so several grids can live in one directory. Record j
of month M belongs to the hour that starts at M-01 00 UTC + j hours:
    taux, tauy          eastward / northward turbulent surface stress [N m-2]
    sshf, slhf, ssr, str  sensible, latent, net solar, net thermal heat flux
                        [W m-2], positive DOWNWARD (ECMWF convention)
        -> all six are the MEAN OVER THAT HOUR (ERA5 hourly accumulations
           divided by 3600 s)
    e, tp               evaporation and total precipitation [m s-1], both
                        DOWNWARD positive in the ECMWF convention, so e is
                        negative where water leaves the ocean and e + tp is
                        P - E. Also hourly means.
    sst                 sea surface temperature [deg C], analysis at the
                        START of that hour

CALENDAR. Model time t = 0 is the start date, 00 UTC. With a cycle of C days
the forcing date is start + mod(t, C). This is repeat-year forcing for
C = 365. With C = 0 the forcing runs through the calendar.

TIME INTERPOLATION is linear between neighbouring records. An hourly mean
sits at the middle of its hour. An instantaneous value sits at its stamp.
"""
import os
import calendar
import datetime
from collections import OrderedDict
import numpy as np
from netCDF4 import Dataset

MEAN_FIELDS  = ('taux', 'tauy', 'sshf', 'slhf', 'ssr', 'str', 'e', 'tp')
INST_FIELDS  = ('sst',)


def parse_start(s):
    """'YYYY-MM-DD' -> datetime at 00 UTC"""
    if isinstance(s, datetime.datetime):
        return s
    return datetime.datetime.strptime(str(s).strip(), '%Y-%m-%d')


class Era5Forcing:
    """Time-interpolated ERA5 fields on one PE's interior block.

    block      (j0, i0, ny_loc, nx_loc), 0-based global start and size
    names      the fields this PE needs (subset of MEAN_FIELDS + INST_FIELDS)
    tag        name part of the files, era5_<tag>_YYYY-MM.nc
    xt, yt     the model's global T coordinates; the files must carry the same
    clean      optional clean(name, array, (year, month)) -> array, applied
               once when a month is loaded
    keep       months kept in memory (2 covers every month seam)"""

    def __init__(self, directory, start, cycle_days, block, names,
                 tag='model', xt=None, yt=None, clean=None, keep=2):
        self.dir   = directory
        self.tag   = str(tag)
        self.pattern = 'era5_%s_%%04d-%%02d.nc' % self.tag
        self.start = parse_start(start)
        self.cycle = int(round(float(cycle_days)*24.0))      # hours, 0 = calendar
        if abs(self.cycle - float(cycle_days)*24.0) > 1e-9:
            raise ValueError('forcing cycle %g days is not a whole number of hours'
                             % cycle_days)
        self.j0, self.i0, self.nyl, self.nxl = (int(v) for v in block)
        for n in names:
            if n not in MEAN_FIELDS + INST_FIELDS:
                raise ValueError('unknown ERA5 forcing field %s' % n)
        self.names = tuple(names)
        self.xt = None if xt is None else np.asarray(xt, np.float64)
        self.yt = None if yt is None else np.asarray(yt, np.float64)
        self.clean = clean
        self.keep  = max(2, int(keep))
        self._months = OrderedDict()
        self._last_t, self._last = None, None

    # ------------------------------------------------------------------
    def date(self, t):
        """Forcing date of model time t [s]."""
        s = float(t)
        if self.cycle:
            s = np.mod(s, self.cycle*3600.0)
        return self.start + datetime.timedelta(seconds=s)

    def climatology_time(self, t):
        """Seconds since 1 January 00 UTC of the forcing year, the time axis
        of the monthly climatologies that stay in use (dQ/dT, SSS)."""
        d = self.date(t)
        return (d - datetime.datetime(d.year, 1, 1)).total_seconds()

    def _hour(self, k):
        """Record k (hours since start, cycled) -> (year, month, record)."""
        if self.cycle:
            k = k % self.cycle
        d = self.start + datetime.timedelta(hours=int(k))
        m0 = datetime.datetime(d.year, d.month, 1)
        return d.year, d.month, int((d - m0).total_seconds()) // 3600

    def _month(self, y, m):
        key = (y, m)
        if key in self._months:
            self._months.move_to_end(key)
            return self._months[key]
        fname = os.path.join(self.dir, self.pattern % (y, m))
        if not os.path.exists(fname):
            raise FileNotFoundError('ERA5 forcing file %s is missing. The run '
                                    'needs %04d-%02d. Build it with prep_era5.py.'
                                    % (fname, y, m))
        nt_exp = calendar.monthrange(y, m)[1]*24
        j0, i0 = self.j0, self.i0
        out = {}
        with Dataset(fname) as f:
            f.set_auto_mask(False)
            nt = len(f.dimensions['time'])
            if nt != nt_exp:
                raise ValueError('%s has %d records, %04d-%02d has %d hours'
                                 % (fname, nt, y, m, nt_exp))
            if self.xt is not None:
                for name, ref in (('xt', self.xt), ('yt', self.yt)):
                    c = np.asarray(f.variables[name][:], np.float64)
                    if c.shape != ref.shape or np.abs(c - ref).max() > 1e-6:
                        raise ValueError('%s: coordinate %s does not match the '
                                         'model grid' % (fname, name))
            for n in self.names:
                a = np.asarray(f.variables[n][:, j0:j0+self.nyl, i0:i0+self.nxl],
                               np.float64)
                if self.clean is not None:
                    a = self.clean(n, a, key)
                out[n] = a
        self._months[key] = out
        while len(self._months) > self.keep:
            self._months.popitem(last=False)
        return out

    # ------------------------------------------------------------------
    def get(self, t):
        """dict name -> (ny_loc, nx_loc) field at model time t [s]."""
        if self._last_t is not None and float(t) == self._last_t:
            return self._last
        h = float(t)/3600.0
        out = {}
        for n in self.names:
            p  = h - 0.5 if n in MEAN_FIELDS else h
            k1 = int(np.floor(p))
            w  = p - k1
            y1, m1, r1 = self._hour(k1)
            y2, m2, r2 = self._hour(k1 + 1)
            a = self._month(y1, m1)[n][r1]
            b = self._month(y2, m2)[n][r2]
            out[n] = (1.0 - w)*a + w*b
        self._last_t, self._last = float(t), out
        return out
