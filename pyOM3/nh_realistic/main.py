"""
 main class of the non-hydrostatic model: pe_realistic's main (grid,
 masks, workspace) plus the fields of the w equation and the
 non-hydrostatic pressure.

 Based on class main of pe_realistic.
"""
import pyOM3                       as OM
from   pyOM3.pe_realistic.main     import main as main_pe
from   functools                   import partial


class main(main_pe):

      # the 3-D solve can be switched off: w is then diagnosed from the
      # continuity equation and the step is exactly pe_realistic's
      enable_nonhydrostatic = True

      def __init__(self, parameter = {}, n_pes_x = 1, n_pes_y = 1, halo_points = 1, comm_in = None ):
         """
         pe_realistic's initialisation; the pseudo-spherical grid is refused
         because the w equation has no spherical metric terms.
         """
         if parameter.get('coord_degree', False):
            raise ValueError('nh_realistic: coord_degree = True is not supported '
                             '(no spherical metric terms in the w equation); '
                             'use a Cartesian grid, dxt/dyt in metres')
         super().__init__(parameter, n_pes_x, n_pes_y, halo_points = halo_points, comm_in = comm_in)
         return

      def allocate_workspace(self):
          """
          pe_realistic's variables plus the w tendency dw (3 time levels),
          the non-hydrostatic pressure p_nh and the 4 coefficient fields of
          its operator (cf_nh[0..2]: east, north, top face; cf_nh[3]: diagonal).
          """
          super().allocate_workspace()

          @partial(OM.jaxjit, out_shardings=self.sharding_3D, static_argnums=(0,1,2))
          def zeros_3D(Nz,Ny,Nx):
              return OM.np.zeros((Nz,Ny,Nx), OM.prec )

          @partial(OM.jaxjit, out_shardings=self.sharding_4D, static_argnums=(0,1,2,3))
          def zeros_4D(i,Nz,Ny,Nx):
              return OM.np.zeros((i,Nz,Ny,Nx), OM.prec )

          Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
          self.dw    = zeros_4D(3,Nz,Ny,Nx)
          self.p_nh  = zeros_3D(Nz,Ny,Nx)
          self.cf_nh = zeros_4D(4,Nz,Ny,Nx)
          return

      def make_topo_masks(self):
          """
          pe_realistic's masks plus maskWc: the W mask with the surface
          face CLOSED. make_topo_masks opens maskW at the surface for the
          free-surface solver (w_surf is then nonzero). The w equation, its
          friction and the non-hydrostatic operator act on the interior
          faces only, so they use maskWc.
          """
          super().make_topo_masks()
          self.maskWc = self.apply_bc( OM.np.minimum( self.rollz(self.maskT,-1), self.maskT ) )
          return
