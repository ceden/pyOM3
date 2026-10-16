"""
Snapshot diagnostics for wim_idemix, based on class model.

class snap_nc4 is based on the netCDF4 module.  Written variables are
the two prognostic moments and the derived diagnostics a user of the
coupled system needs: mean frequency, mean direction, significant wave
height, and the stress on the ocean with its diagnostic partners.
"""
import numpy as np__
from   pyOM3.wim_idemix.model import model


class snap_nc4(model):


      def setup(self, snap_file_name = 'snap_idemix.cdf',
                      show_each_time_step = True):
          """
          Initialisation of the netcdf file.
          """
          if hasattr(super(), 'setup'): super().setup()
          from netCDF4 import Dataset as NF

          self.snap_file_name      = snap_file_name
          self.show_each_time_step = show_each_time_step
          self.itt = 0

          if self.my_pe == 0:
             fh = NF(self.snap_file_name, 'w', format='NETCDF4')
             fh.createDimension('x', self.nx)
             fh.createDimension('y', self.ny)
             fh.createDimension('d', self.nd)
             fh.createDimension('Time', None)
             v = fh.createVariable('x', 'f8', ('x',)); v.units = 'm'
             v[:] = (np__.arange(self.nx) + 0.5)*self.dx
             v = fh.createVariable('y', 'f8', ('y',)); v.units = 'm'
             v[:] = (np__.arange(self.ny) + 0.5)*self.dy
             v = fh.createVariable('d', 'f8', ('d',)); v.units = 'rad'
             v.long_name = 'wave direction, k_hat = (sin d, cos d)'
             v[:] = self.theta[self.dsl]
             v = fh.createVariable('Time', 'f8', ('Time',)); v.units = 's'

             def mk(name, dims, units, long_name):
                 w = fh.createVariable(name, 'f8', dims, fill_value = -1e33)
                 w.units, w.long_name = units, long_name
                 return w

             mk('Acal', ('Time','d','y','x'), 'm^2 s',
                'action per direction')
             mk('Ecal', ('Time','d','y','x'), 'm^2 s^-1',
                'energy per direction')
             mk('E',    ('Time','y','x'), 'm^2',   'total wave energy')
             mk('Hs',   ('Time','y','x'), 'm',     'significant wave height')
             mk('sigbar', ('Time','y','x'), 's^-1','mean frequency')
             mk('thetam', ('Time','y','x'), 'rad', 'mean wave direction')
             mk('tau_o',  ('Time','y','x'), 'm^2 s^-2',
                'kinematic stress on the ocean (momentum budget residual)')
             mk('tau_in', ('Time','y','x'), 'm^2 s^-2',
                'kinematic momentum uptake from the atmosphere')
             fh.close()
          self.comm.Barrier()
          return


      def diagnose(self, itt = None):
          """
          Append the current state to the snapshot file.
          """
          from netCDF4 import Dataset as NF
          itt = self.itt if itt is None else itt

          # SLICE THE DIRECTION HALO OFF BEFORE THE GATHER. gather_array
          # returns None on every PE but 0, so indexing its result
          # crashes the other PEs (it did, on 64 PEs).
          sl     = self.dsl
          Acal   = self.gather_array(self.Ad[sl])
          Ecal   = self.gather_array(self.Ed[sl])
          sigbar = self.gather_array(self.sigbar_g())
          tau_o  = self.gather_array(self.tau_o_g)
          tau_in = self.gather_array(self.tau_in_g)

          if self.my_pe == 0:
             E  = Ecal.sum(0)*self.dth
             Hs = 4.0*np__.sqrt(np__.maximum(E, 0.0))
             s  = (Ecal*self.s_x[sl]).sum(0)
             c  = (Ecal*self.s_y[sl]).sum(0)
             thm = np__.arctan2(s, c)

             fh = NF(self.snap_file_name, 'a')
             n  = len(fh.dimensions['Time'])
             fh.variables['Time'][n] = itt*self.dt
             fh.variables['Acal'][n,...] = Acal
             fh.variables['Ecal'][n,...] = Ecal
             fh.variables['E'][n,...]    = E
             fh.variables['Hs'][n,...]   = Hs
             fh.variables['sigbar'][n,...] = sigbar
             fh.variables['thetam'][n,...] = thm
             fh.variables['tau_o'][n,...]  = tau_o
             fh.variables['tau_in'][n,...] = tau_in
             fh.close()
             if self.show_each_time_step:
                print(' itt=', itt, ' t=', itt*self.dt/86400., ' d',
                      ' E=', float(E.mean()),
                      ' tau_o=', float(tau_o.mean()),
                      ' n_sub=', self.n_sub)
          self.comm.Barrier()
          return
