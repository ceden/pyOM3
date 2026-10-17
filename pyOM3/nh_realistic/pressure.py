"""
 Non-hydrostatic pressure: operator coefficients, the solve and the
 projection of (u,v,w). The surface pressure solve (rigid lid or implicit
 free surface) is pe_realistic's and is done first; the 3-D solve then
 acts on the result with the surface face closed.

 Based on class pressure of pe_realistic and the solver solver_nh.
"""
import pyOM3                           as OM
from   pyOM3.nh_realistic.main         import main
from   pyOM3.pe_realistic.pressure     import pressure as pressure_pe
from   pyOM3.nh_realistic.solver       import solver_nh
from   functools                       import partial


class pressure(main, pressure_pe, solver_nh):

      @partial(OM.jaxjit, static_argnums=0)
      def make_coeff_non_hydro(self, cf, maskU, maskV, maskWc, maskT):
          """
          Face coefficients of  V div grad  (see solver.py):
            east  face of T cell (i,j,k): transport u dyt dzt (see
                  volume_divergence), gradient (p[i+1]-p[i])/dxu
                                                  ->  maskU dyt dzt / dxu
            north face: transport v dxt dzt, gradient (p[j+1]-p[j])/dyu
                                                  ->  maskV dxt dzt / dyu
            top   face: area_t, gradient (p[k+1]-p[k])/dzw
                                                  ->  maskWc area_t / dzw
          The halo values of the east and north coefficients are set by
          apply_bc (periodic image, or zero at a closed boundary where
          maskU is zero anyway); the top coefficient is zero in the
          vertical halo rows and at the surface through maskWc.
          """
          a = slice(None)
          ae = self.apply_bc( maskU *self.dyt_y*self.dzt_z/self.dxu_x )
          an = self.apply_bc( maskV *self.dxt_x*self.dzt_z/self.dyu_y )
          at = maskWc*self.area_t/self.dzw_z
          dg = -( ae + self.rollx(ae,1) + an + self.rolly(an,1) + at + self.rollz(at,1) )
          # dry cells, and wet cells without any open face: identity row
          dg = OM.np.where( OM.np.logical_and(maskT > 0.0, dg < 0.0), dg, -1.0 )
          cf = OM.modify_array(cf, (0,a,a,a), ae, out_sharding = self.sharding_4D)
          cf = OM.modify_array(cf, (1,a,a,a), an, out_sharding = self.sharding_4D)
          cf = OM.modify_array(cf, (2,a,a,a), at, out_sharding = self.sharding_4D)
          cf = OM.modify_array(cf, (3,a,a,a), dg, out_sharding = self.sharding_4D)
          return cf

      @partial(OM.jaxjit, static_argnums=0)
      def volume_divergence(self, u, v, w, maskU, maskV, maskT):
          """
          V div (u,v,w) of each T cell from the face transports (the
          divergence of vertical_velocity times area_t dzt, Cartesian):
          utr = u dyt dzt, vtr = v dxt dzt, wtr = w area_t.
          w enters unmasked: at the surface face it is the free-surface
          w_surf (zero under the rigid lid), below the bottom it is zero.
          """
          utr = u*maskU*self.dyt_y*self.dzt_z
          vtr = v*maskV*self.dxt_x*self.dzt_z
          wtr = w*self.area_t
          return ( (utr - self.rollx(utr,1)) + (vtr - self.rolly(vtr,1))
                 + (wtr - self.rollz(wtr,1)) )*maskT

      @partial(OM.jaxjit, static_argnums=0)
      def solve_for_nh_pressure(self, u, v, w, p_nh, maskU, maskV, maskWc, maskT, cf, n_wet):
          """
          -V div grad p_nh = -V div(u,v,w)/dt, start from the previous
          p_nh, then u -= dt dp/dx, v -= dt dp/dy, w -= dt dp/dz on the
          open faces. The surface w is not changed (the top face is
          closed in the operator and in the correction), so the column
          divergence fixed by the surface solve is kept.
          """
          b = -self.volume_divergence(u, v, w, maskU, maskV, maskT)/self.dt
          p_nh, error, itts = self.cg_nh(b, p_nh, cf, maskT, n_wet)
          u = u - self.dt*( self.rollx(p_nh,-1) - p_nh )/self.dxu_x*maskU
          v = v - self.dt*( self.rolly(p_nh,-1) - p_nh )/self.dyu_y*maskV
          w = w - self.dt*( self.rollz(p_nh,-1) - p_nh )/self.dzw_z*maskWc
          return self.apply_bc(u), self.apply_bc(v), self.apply_bc(w), p_nh, error, itts
