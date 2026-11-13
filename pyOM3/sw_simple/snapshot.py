"""
netCDF snapshots of sw_simple (class snap_nc4, add to the model class bases
in place of model): u, v, h every snapint seconds, with the coordinates of
the C grid, and the domain mean energy.

  prepare_for_run(snap_file_name='snap.cdf', show_each_time_step=True)
"""
import pyOM3                  as OM
from   pyOM3.sw_simple.model  import model
from   netCDF4                import Dataset as NF
import numpy                  as np


class snap_nc4(model):

    def prepare_for_run(self, snap_file_name='snap.cdf', show_each_time_step=True):
        super().prepare_for_run()
        self.snap_file_name, self.show_each_time_step = snap_file_name, show_each_time_step
        if not hasattr(self, 'snapint'): self.snapint = self.dt
        fid = NF(snap_file_name, mode='w')
        for d, x in (('xt', self.xt), ('yt', self.yt), ('xu', self.xu), ('yu', self.yu)):
            fid.createDimension(d, len(x))
            fid.createVariable(d, 'f8', (d,))[:] = x
            fid.variables[d].units = 'm'
        fid.createDimension('Time', None)
        fid.createVariable('Time', 'f8', ('Time',)).units = 's'
        for v, dims, units in (('u', ('Time', 'yt', 'xu'), 'm/s'), ('v', ('Time', 'yu', 'xt'), 'm/s'),
                               ('h', ('Time', 'yt', 'xt'), 'm')):
            fid.createVariable(v, 'f8', dims).units = units
        fid.createVariable('energy', 'f8', ('Time',)).long_name = 'domain mean energy per unit area/rho_0'
        for k in ('f0', 'g', 'H0', 'dt', 'Ahbi', 'Khbi'):
            setattr(fid, k, float(getattr(self, k)))
        fid.time_scheme = self.time_scheme
        fid.close()
        self._snap_n = int(round(self.snapint/self.dt))
        return

    def diagnose(self, n, tau):
        """a snapshot of the state after step n (n = 0: after the first step)"""
        super().diagnose(n, tau)
        if not hasattr(self, 'snap_file_name'): return
        t = (n + 1)*self.dt
        if self.show_each_time_step: print('t = %g s, n = %d' % (t, n))
        if (n + 1) % max(self._snap_n, 1) == 0:
            fid = NF(self.snap_file_name, mode='r+')
            i = len(fid.dimensions['Time'])
            fid.variables['Time'][i] = t
            for v in ('u', 'v', 'h'):
                fid.variables[v][i] = np.asarray(getattr(self, v))
            fid.variables['energy'][i] = self.energy()
            fid.close()
        return

    def write_snapshot(self, t=0.):
        """the current state as a snapshot at time t (e.g. the initial state)"""
        fid = NF(self.snap_file_name, mode='r+')
        i = len(fid.dimensions['Time'])
        fid.variables['Time'][i] = t
        for v in ('u', 'v', 'h'):
            fid.variables[v][i] = np.asarray(getattr(self, v))
        fid.variables['energy'][i] = self.energy()
        fid.close()
        return
