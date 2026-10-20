"""
Forcing source 'ecmwf_clim': the monthly climatology of the pyOM2 setups,
one file with 12 records of each variable on the MODEL grid,

    taux, tauy [N/m2]   qnet [W/m2]   swf [W/m2]   dqdt [W/m2/K]
    sst [deg C]         sss [g/kg]  (or from a second file, settings['sss'])

used as in setup1.f90: Q = qnet + dqdt (sst - SST), swf penetrates, salinity
is restored to sss. Interpolated on the calendar day with pyOM2's weights.

settings: source='ecmwf_clim', file, start, cycle_days (365),
          names {interface field: variable name} to override the defaults
          below, sss (path, var) for a separate salinity file,
          clim_scatter (bool), qnet_offset

TWO DIFFERENCES TO pyOM2. The stress is handed over on T points like every
other source and the model averages it to U and V, where pyOM2 takes the file
values at U and V as they are. And for the WAVE model a monthly mean stress
has no storms: u* from it is far too low for realistic wave growth, so a
coupled run with this source is a technical option only. That is printed once.
"""
import numpy as np
from forcing_base import Forcing, MonthlyClim

_VAR = {'taux': 'taux', 'tauy': 'tauy', 'qnet': 'qnet', 'qsol': 'swf',
        'dqdt': 'dqdt', 'sst_star': 'sst', 'sss_star': 'sss'}


class EcmwfClimForcing(Forcing):

    def __init__(self, settings, geom, fields, who):
        super().__init__(settings, geom, fields, who)
        var = dict(_VAR); var.update(settings.get('names', {}))
        sc = bool(settings.get('clim_scatter', False))
        self.qoff = float(settings.get('qnet_offset', 0.0))
        self._c = {}
        for f in self.fields:
            if f == 'emp':
                continue                     # a climatology of P - E is not in the file
            if f not in var:
                raise ValueError("forcing source 'ecmwf_clim' has no field %s" % f)
            spec = (settings['file'], var[f])
            if f == 'sss_star' and 'sss' in settings:
                spec = tuple(settings['sss'])
            self._c[f] = MonthlyClim(spec, geom, sc, f, who,
                                     limit=20.0 if f in ('taux', 'tauy') else 1e5)
        if who == 'wave' and self.rank == 0:
            print('WARNING: the wave model is forced with a MONTHLY MEAN stress '
                  '(forcing source ecmwf_clim): no storms, u* far too low for '
                  'realistic wave growth', flush=True)

    def get(self, t):
        ty = self.climatology_time(t)
        out = {f: c.at(ty) for f, c in self._c.items()}
        if 'qnet' in out and self.qoff != 0.0:
            out['qnet'] = out['qnet'] - self.qoff
        return out

    def describe(self):
        return ['forcing (%s): monthly climatology %s, fields %s'
                % (self.who, self.settings['file'], ','.join(self._c))]
