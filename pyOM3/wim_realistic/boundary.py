"""
 Setting halo points to boundary conditions for the WIM model on the
 variable / pseudo-spherical grid.

 The halo exchange (MPI or JAX sharding), gather and roll methods are
 wim_simple.boundary's (they carry no grid metric); only the domain
 extents Lx/Ly differ, which grid.calc_grid sets from the variable
 spacings (the scalar nx*dx of wim_simple does not exist here).
"""
import pyOM3     as OM
from   pyOM3.wim_simple.boundary import boundary as _boundary_simple


class boundary(_boundary_simple):

      def domain_decomposition(self, n_pes_x, n_pes_y, comm_in = None):
         """
         wim_simple's decomposition without the scalar Lx/Ly (set by
         grid.calc_grid from the variable spacings).
         """
         _boundary_simple.domain_decomposition(self, n_pes_x, n_pes_y, comm_in)
         self.Lx = self.Ly = None
         return
