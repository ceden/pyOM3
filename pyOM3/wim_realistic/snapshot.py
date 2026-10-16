"""
Snapshot diagnostics: wim_simple's snapshot classes over wim_realistic's
model. The netCDF body and both diagnose methods are shared (_snap_setup,
_snap_raw_setup, diagnose contain no super()); the setup chaining line and
the file coordinates are per package.
"""
import pyOM3           as OM
from   pyOM3.wim_realistic.model import model
from   pyOM3.wim_simple.snapshot import snap_nc4 as _snap_simple
from   pyOM3.wim_simple.snapshot import _snap_setup
from   pyOM3.wim_simple.snapshot import _snap_raw_setup, _snap_raw_diagnose
import numpy           as np__


def _xy(self):
    # true grid coordinates (degrees longitude/latitude when coord_degree,
    # metres otherwise), from grid.calc_grid
    return np__.asarray(self.xt_gl), np__.asarray(self.yt_gl)


class snap_nc4(model):

      def setup(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          """
          Initialisation of the netcdf file (body shared with wim_simple).
          """
          if hasattr(super(),'setup'): super().setup()
          return _snap_setup(self, snap_file_name, show_each_time_step)

      diagnose = _snap_simple.diagnose

      _snap_xy = _xy


class snap_raw(model):
      """Per-PE raw snapshots (body shared with wim_simple). snap_file_name
      is a DIRECTORY; self.snap_fields chooses what is written and
      self.snapint_full gives the full 4-D spectrum its own interval."""

      def setup(self, snap_file_name = 'raw_snap', show_each_time_step = True ):
          if hasattr(super(),'setup'): super().setup()
          return _snap_raw_setup(self, snap_file_name, show_each_time_step)

      diagnose = _snap_raw_diagnose

      _snap_xy = _xy
