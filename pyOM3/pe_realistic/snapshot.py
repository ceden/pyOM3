"""
Snapshot diagnostics: pe_simple's snap_nc4 / snap_raw over pe_realistic's
model. The netCDF / directory bodies and diagnose are shared (they contain
no super()); the prepare_for_run chaining line and the file coordinates are
per package.
"""
import pyOM3           as OM
from   pyOM3.pe_realistic.model import model
from   pyOM3.pe_simple.snapshot import (snap_nc4 as _nc4_simple, snap_raw as _raw_simple,
                                        _snap_nc4_prepare, _snap_raw_prepare)
import numpy           as np__


def _snap_coords(self, nx, ny, nz):
    # true grid coordinates (degrees for xt/yt/xu/yu when coord_degree,
    # metres otherwise; zt/zw per the vertical convention chosen in grid.py)
    return (np__.asarray(self.xt_gl), np__.asarray(self.yt_gl), np__.asarray(self.zt_gl),
            np__.asarray(self.xu_gl), np__.asarray(self.yu_gl), np__.asarray(self.zw_gl))


class snap_nc4(model):

      _snap_coords = _snap_coords
      diagnose     = _nc4_simple.diagnose

      def prepare_for_run(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          super().prepare_for_run()
          return _snap_nc4_prepare(self, snap_file_name, show_each_time_step)


class snap_raw(model):

      _snap_coords = _snap_coords
      diagnose     = _raw_simple.diagnose

      def prepare_for_run(self, snap_file_name = 'raw_snap', show_each_time_step = True ):
          super().prepare_for_run()
          return _snap_raw_prepare(self, snap_file_name, show_each_time_step)
