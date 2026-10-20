"""
Forcing source 'era5': hourly ERA5 fluxes, plus the two monthly climatologies
that stay in use with them (dQ/dT for the SST restoring, SSS for the salinity
restoring).

    taux, tauy   ERA5 turbulent surface stress
    qnet         sshf + slhf + ssr + str - qnet_offset
    qsol         ssr
    sst_star     ERA5 sst
    dqdt         monthly climatology, settings['clim']['dqdt'] = (path, var)
    emp          e + tp - emp_offset            (only with freshwater = True)
    sss_star     monthly climatology, settings['clim']['sss'] = (path, var)

Q = qnet + dqdt (sst_star - SST) is the linearisation of the ERA5 bulk fluxes
around the SST they were computed with.

THE FILES are read by one of two readers, settings['grid']:
    'native'  era5_<tag>_YYYY-MM.nc on the ERA5 grid (prep_era5_native.py);
              rank 0 reads a record and broadcasts it, every rank interpolates
              bilinearly to its block (forcing_era5_native.py). e and tp are
              in the same files.
    'model'   era5_<tag>_YYYY-MM.nc already on the model grid (prep_era5.py);
              every rank reads its own hyperslab (forcing_era5.py). e and tp
              are a SECOND set, tag settings['fw_tag'].

settings: source='era5', grid, dir, tag, fw_tag, start, cycle_days,
          qnet_offset [W/m2], emp_offset [m/s], freshwater (bool),
          tau_max [Pa] (cleaning limit of the model-grid reader),
          clim {'dqdt': (path, var), 'sss': (path, var)}, clim_scatter (bool)

The offsets are properties of the forcing data over the model's wet cells
(measure_era5_offsets.py), which is why they live here and not in the model.
"""
import numpy as np
from forcing_base import Forcing, MonthlyClim, clean_forcing

_HEAT = ('sshf', 'slhf', 'ssr', 'str')
# interface field -> ERA5 variables it needs
_NEEDS = {'taux': ('taux',), 'tauy': ('tauy',), 'qnet': _HEAT, 'qsol': ('ssr',),
          'sst_star': ('sst',), 'emp': ('e', 'tp'), 'dqdt': (), 'sss_star': (),
          'u10': ('u10',), 'v10': ('v10',)}
_ORDER = ('taux', 'tauy', 'sshf', 'slhf', 'ssr', 'str', 'sst', 'e', 'tp', 'u10', 'v10')


class Era5FluxForcing(Forcing):

    def __init__(self, settings, geom, fields, who):
        super().__init__(settings, geom, fields, who)
        s = settings
        self.fresh = bool(s.get('freshwater', False))
        self.qoff  = float(s.get('qnet_offset', 0.0))
        self.eoff  = float(s.get('emp_offset', 0.0))
        self.tau_max = float(s.get('tau_max', 20.0))
        want = [f for f in self.fields if f in _NEEDS
                and (f != 'emp' or self.fresh)]
        unknown = [f for f in self.fields if f not in _NEEDS]
        if unknown:
            raise ValueError("forcing source 'era5' has no field %s" % unknown)
        self.want = tuple(want)
        raw = [n for n in _ORDER if any(n in _NEEDS[f] for f in want)]
        self.grid = s.get('grid', 'native')
        self._fw = None
        if self.grid == 'native':
            from forcing_era5_native import Era5NativeForcing
            self._rd = Era5NativeForcing(s['dir'], s['start'], self.cycle_days,
                                         geom['xt'], geom['yt'], raw, self.comm,
                                         tag=s['tag'], wind_tag=s.get('wind_tag'))
        elif self.grid == 'model':
            from forcing_era5 import Era5Forcing
            if 'u10' in raw or 'v10' in raw:
                raise ValueError("forcing 'era5': the 10 m wind (u10, v10) exists for "
                                 "the native grid only (prep_era5_native.py)")
            main = [n for n in raw if n not in ('e', 'tp')]
            self._rd = Era5Forcing(s['dir'], s['start'], self.cycle_days,
                                   geom['block'], main, tag=s['tag'],
                                   xt=geom.get('xt_gl'), yt=geom.get('yt_gl'),
                                   clean=self._clean)
            if 'emp' in want:     # e and tp live in their own, small file set
                self._fw = Era5Forcing(s['dir'], s['start'], self.cycle_days,
                                       geom['block'], ('e', 'tp'),
                                       tag=s.get('fw_tag', s['tag'] + 'fw'),
                                       xt=geom.get('xt_gl'), yt=geom.get('yt_gl'),
                                       clean=self._clean)
        else:
            raise ValueError("forcing 'era5': grid must be 'native' or 'model'")
        clim = s.get('clim', {})
        sc = bool(s.get('clim_scatter', False))
        self._dqdt = (MonthlyClim(clim['dqdt'], geom, sc, 'dqdt', who)
                      if 'dqdt' in want else None)
        self._sss  = (MonthlyClim(clim['sss'], geom, sc, 'sss', who)
                      if 'sss_star' in want and 'sss' in clim else None)

    # cleaning of the model-grid files, once per loaded month. Non-finite
    # entries would reach the ocean through apply_bc and the U/V average, and
    # on the wave side a fill value that survives to sqrt(|tau|/rho_air)
    # detonates the spectrum in a few steps (see clean_forcing). With a wet
    # mask in geom (the wave model gives one) land is taken from the
    # TOPOGRAPHY and a month that is mostly rejected stops the run.
    def _clean(self, name, a, ym):
        wet = self.geom.get('wet')
        stress = name in ('taux', 'tauy')
        b, nrej = clean_forcing(a, name, limit=self.tau_max if stress else 1e5,
                                wet=wet)
        if wet is None:
            if nrej:
                print('%s forcing pe %d %04d-%02d %s: %d non-finite/out-of-range '
                      'entries zeroed' % (self.who, self.rank, ym[0], ym[1], name,
                                          nrej), flush=True)
            return b
        nbad = int((~wet).sum())*a.shape[0]               # land entries
        if nrej - nbad > 0 or self.rank == 0:
            print('%s forcing pe %d %04d-%02d %s: |%s| max %.4g, %d of %d entries '
                  'rejected (%d on land, limit %.4g)'
                  % (self.who, self.rank, ym[0], ym[1], name, name,
                     float(np.abs(b).max()), nrej, a.size, nbad,
                     self.tau_max if stress else 1e5), flush=True)
        nwet = a.size - nbad
        if nwet > 0 and nrej - nbad > 0.9*nwet:
            raise ValueError('%s forcing pe %d: %d of %d wet %s entries were '
                             'rejected. The ERA5 file is not in SI units or holds '
                             'fill values.' % (self.who, self.rank, nrej - nbad,
                                               nwet, name))
        return b

    def get(self, t):
        F = self._rd.get(t)
        out = {}
        for f in self.want:
            if f in ('taux', 'tauy', 'u10', 'v10'):
                out[f] = F[f]
            elif f == 'qnet':
                out[f] = F['sshf'] + F['slhf'] + F['ssr'] + F['str'] - self.qoff
            elif f == 'qsol':
                out[f] = F['ssr']
            elif f == 'sst_star':
                out[f] = F['sst']
            elif f == 'emp':
                G = F if self._fw is None else self._fw.get(t)
                # ERA5 e is NEGATIVE where water evaporates, tp is positive
                out[f] = G['e'] + G['tp'] - self.eoff
        if self._dqdt is not None or self._sss is not None:
            ty = self.climatology_time(t)
            if self._dqdt is not None:
                out['dqdt'] = self._dqdt.at(ty)
            if self._sss is not None:
                out['sss_star'] = self._sss.at(ty)
        return out

    def describe(self):
        s = self.settings
        lines = ['forcing (%s): hourly ERA5 on the %s grid from %s tag %s, model '
                 'time 0 = %s, cycle %g d'
                 % (self.who, 'ERA5' if self.grid == 'native' else 'model',
                    s['dir'], s['tag'], s['start'], self.cycle_days)]
        if 'qnet' in self.want and self.qoff != 0.0:
            lines.append('forcing: %.3f W/m2 subtracted from the ERA5 net heat flux'
                         % self.qoff)
        if 'emp' in self.want:
            lines.append('forcing: freshwater flux E-P from ERA5%s, offset %.3f mm/day'
                         % ('' if self._fw is None else ' (tag %s)'
                            % s.get('fw_tag', s['tag'] + 'fw'),
                            self.eoff*86400.0*1000.0))
        if 'u10' in self.want:
            lines.append('forcing: 10 m wind u10, v10 from ERA5 (tag %s), for the '
                         'relative-wind stress' % (s.get('wind_tag') or s['tag']))
        return lines

    def close(self):
        if hasattr(self._rd, 'close'):
            self._rd.close()
