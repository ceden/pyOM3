"""
THE ATMOSPHERE AS A FORCING SOURCE (source name 'coupled'). The ocean and the
wave model hold a forcing object and ask it for the fields at a model time;
here the fields are the last blocks the coupler received from the atmosphere
(through the ocean, for the waves) and `get(t)` ignores t: the fields are
piecewise constant over a coupling step, one step behind the atmosphere's
clock like every other coupling field.

    F.set(fields)   the coupler loop stores the received blocks, interior
                    (ny_loc, nx_loc) of THIS component's own grid, fixed
                    names and units of forcing_base:
                        taux, tauy  N/m^2   the atmosphere's stress tau_a
                        qnet        W/m^2   -Q_s, positive into the ocean
    F.get(t)        that dict. A key never set is absent, and the ocean
                    switches that term off (no solar part, no restoring, no
                    freshwater: the sensible heat flux is the only heat
                    exchange, energy_cycle.tex Sec. 6.4).
    F.own_grid      True: the blocks are on this component's OWN grid. With a
                    coarser wave grid exp_wave.update_forcing then skips its
                    group mean over ocean cells (the ocean already averaged).

settings: 'start' (date of model time 0, only for the calendar in the log,
default 2000-01-01) and 'cycle_days' as forcing_base.
"""
import numpy as np
from forcing_base import Forcing


class CoupledForcing(Forcing):
    own_grid = True

    def __init__(self, settings, geom, fields, who):
        st = dict(settings)
        st.setdefault('start', '2000-01-01')
        super().__init__(st, geom, fields, who)
        j0, i0, nyl, nxl = (int(v) for v in geom.get('own_block', geom['block']))
        self.shape = (nyl, nxl)
        # before the first exchange: no stress, no heat flux (the ocean and
        # the waves integrate their first step with the atmosphere's t = 0
        # state unknown, as the atmosphere integrates with SST = 288 K)
        self._fields = {'taux': np.zeros(self.shape), 'tauy': np.zeros(self.shape)}
        if who == 'ocean':
            self._fields['qnet'] = np.zeros(self.shape)

    def set(self, fields):
        for k, v in fields.items():
            a = np.asarray(v, np.float64)
            if a.shape != self.shape:
                raise ValueError('coupled forcing (%s): %s is %s, the block is %s'
                                 % (self.who, k, a.shape, self.shape))
            self._fields[k] = a
        return

    def get(self, t):
        return self._fields

    def describe(self):
        return ['forcing: coupled -- the stress (and Q_s) arrive through the coupler '
                'each step, from the atmosphere model or, for the waves, the ocean\'s '
                'relative-wind stress (%s)' % self.who]
