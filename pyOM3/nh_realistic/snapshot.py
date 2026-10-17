"""
Snapshot diagnostics: pe_realistic's snap_nc4 / snap_raw over the
non-hydrostatic model, plus the non-hydrostatic pressure p_nh (and the
iteration count of its solver in the printed line).
"""
import pyOM3           as OM
from   pyOM3.nh_realistic.model     import model
from   pyOM3.pe_realistic.snapshot  import _snap_coords
from   pyOM3.pe_simple.snapshot     import (snap_nc4 as _nc4_simple, snap_raw as _raw_simple,
                                            _snap_nc4_prepare, _snap_raw_prepare)
from   netCDF4         import Dataset as NF
import numpy           as np__


class snap_nc4(model):

      _snap_coords = _snap_coords

      def prepare_for_run(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          super().prepare_for_run()
          _snap_nc4_prepare(self, snap_file_name, show_each_time_step)
          if self.my_pe == 0:
             fid = NF(snap_file_name, mode='r+')
             fid.createVariable('p_nh', OM.prec, ('time','zt','yt','xt'))
             fid.close()
          return

      def diagnose(self, n, tau):
          """pe_simple's netCDF snapshot, then p_nh into the same record"""
          _nc4_simple.diagnose(self, n, tau)
          if n==0 or np__.mod(n,int(self.snapint/self.dt))==0.:
             for pe in range(self.n_pes):
                 self.mpi_barrier()
                 if self.my_pe == pe:
                    fid = NF(self.snap_file_name, mode='r+')
                    m = len(fid.dimensions['time']) - 1
                    fid.variables['p_nh'][(m,slice(None), slice(self.ys_pe-1,self.ye_pe),
                                             slice(self.xs_pe-1,self.xe_pe))] = self.unpad(self.p_nh)
                    fid.close()
             if self.my_pe == 0 and hasattr(self,'congr_itts_nh'):
                print(" --> p_nh written, itts_nh of this step =", self.congr_itts_nh)
          return


class snap_raw(model):

      _snap_coords = _snap_coords
      diagnose     = _raw_simple.diagnose

      def prepare_for_run(self, snap_file_name = 'raw_snap', show_each_time_step = True ):
          super().prepare_for_run()
          return _snap_raw_prepare(self, snap_file_name, show_each_time_step)
