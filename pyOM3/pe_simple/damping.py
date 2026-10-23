"""
 Damping terms like biharmonic friction.

 Based on class main.
"""
import pyOM3                    as OM
from   pyOM3.pe_simple.main     import main
from   functools                import partial

class damping(main):

          
      @partial(OM.jaxjit, static_argnums=0)    
      def biharmonic_horizontal(self,p1,mixC,mask):          
          fe = -mixC*(self.rollx(p1,-1)-p1)/self.dx*self.rollx(mask,-1)*mask
          fn = -mixC*(self.rolly(p1,-1)-p1)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)   
          
          del2 = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy )*mask
          del2 = self.apply_bc(del2)
          
          fe = (self.rollx(del2,-1)-del2)/self.dx*self.rollx(mask,-1)*mask
          fn = (self.rolly(del2,-1)-del2)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
 
          return p1 + self.dt*( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy  )*mask


    
      @partial(OM.jaxjit, static_argnums=0)    
      def biharmonic_horizontal_tendency(self,p1,mixC,mask):          
          fe = -mixC*(self.rollx(p1,-1)-p1)/self.dx*self.rollx(mask,-1)*mask
          fn = -mixC*(self.rolly(p1,-1)-p1)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)   
          
          del2 = ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy )*mask
          del2 = self.apply_bc(del2)
          
          fe = (self.rollx(del2,-1)-del2)/self.dx*self.rollx(mask,-1)*mask
          fn = (self.rolly(del2,-1)-del2)/self.dy*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
 
          return ( (fe-self.rollx(fe,1))/self.dx + (fn-self.rolly(fn,1))/self.dy  )*mask


      @partial(OM.jaxjit, static_argnums=0)    
      def harmonic_vertical_expl(self,p1,mixC,mask):          
          ft = mixC*(self.rollz(p1,-1)-p1)/self.dz*self.rollz(mask,-1)*mask 
          return p1 + self.dt* (ft-self.rollz(ft,1))/self.dz *mask

      @partial(OM.jaxjit, static_argnums=0)    
      def harmonic_vertical_expl_tendency(self,p1,mixC,mask):          
          ft = mixC*(self.rollz(p1,-1)-p1)/self.dz*self.rollz(mask,-1)*mask 
          return (ft-self.rollz(ft,1))/self.dz *mask

      # ------------------------------------------------------------------
      # bottom friction (Sep 2026). Two options, both off by default:
      #   enable_bottom_friction            linear, rate r_bot [1/s] on the
      #                                     bottom-cell velocity (pyOM2 r_bot)
      #   enable_quadratic_bottom_friction  drag C_d = r_quad_bot on the
      #                                     bottom-cell speed, rate
      #                                     r_quad_bot |u_bot| / dz_bot
      #                                     (pyOM2 r_quad_bot)
      # Applied as an IMPLICIT split step on the lowest wet cell of every
      # column (unconditionally stable in 5 m cells at 5 m/s), and, on the
      # split-explicit path, as the matching drag on the barotropic
      # transport inside the sub cycle (bottom_friction_rate_bt), which is
      # what damps the grid-scale gravity mode in shallow coastal water that
      # the implicit free surface damps by itself.
      # ------------------------------------------------------------------
      def _bottom_cell_mask(self, mask):
          """1 in the lowest wet cell of each column, 0 elsewhere (k up)"""
          below = OM.np.concatenate([OM.np.zeros_like(mask[:1]), mask[:-1]], axis=0)
          return mask*(1.0 - below)

      def bottom_friction_rate(self, u, v, maskU, maskV):
          """2-D rates r_u, r_v [1/s] of the bottom friction at U and V points,
          r = r_bot + r_quad_bot |u_bot| / dz_bot, from the bottom-cell speed"""
          r_lin  = getattr(self, 'r_bot', 0.0) if getattr(self, 'enable_bottom_friction', False) else 0.0
          r_quad = getattr(self, 'r_quad_bot', 0.0) if getattr(self, 'enable_quadratic_bottom_friction', False) else 0.0
          bu, bv = self._bottom_cell_mask(maskU), self._bottom_cell_mask(maskV)
          dzb_u = OM.np.sum(bu*self.dz, axis=0); dzb_v = OM.np.sum(bv*self.dz, axis=0)
          if r_quad == 0.0:
              return r_lin*(dzb_u > 0), r_lin*(dzb_v > 0)
          ub = OM.np.sum(bu*u, axis=0); vb = OM.np.sum(bv*v, axis=0)
          # speed at the U point: v averaged from the four surrounding V points, and vice versa
          vbu = 0.25*(vb + self.rollx(vb,-1) + self.rolly(vb,1) + self.rollx(self.rolly(vb,1),-1))
          ubv = 0.25*(ub + self.rolly(ub,-1) + self.rollx(ub,1) + self.rolly(self.rollx(ub,1),-1))
          su = OM.np.sqrt(ub**2 + vbu**2); sv = OM.np.sqrt(ubv**2 + vb**2)
          r_u = OM.np.where(dzb_u > 0, r_lin + r_quad*su/OM.np.maximum(dzb_u, 1e-12), 0.0)
          r_v = OM.np.where(dzb_v > 0, r_lin + r_quad*sv/OM.np.maximum(dzb_v, 1e-12), 0.0)
          return r_u, r_v

      def bottom_friction(self, u, v, maskU, maskV):
          """implicit split step on the bottom cell: u_bot /(1 + dt r)"""
          if not (getattr(self, 'enable_bottom_friction', False) or
                  getattr(self, 'enable_quadratic_bottom_friction', False)):
              return u, v
          return self._bottom_friction_core(u, v, maskU, maskV)

      # jitted: unjitted this was a dozen eager full-3-D kernels (the
      # bottom-cell masks, the column sums, the factors, the products), 24 ms
      # of a 220 ms step on icon033 on four GPUs; fused it is two kernels
      @partial(OM.jaxjit, static_argnums=0)
      def _bottom_friction_core(self, u, v, maskU, maskV):
          r_u, r_v = self.bottom_friction_rate(u, v, maskU, maskV)
          bu, bv = self._bottom_cell_mask(maskU), self._bottom_cell_mask(maskV)
          fu = 1.0 - bu*(1.0 - 1.0/(1.0 + self.dt*r_u[None]))
          fv = 1.0 - bv*(1.0 - 1.0/(1.0 + self.dt*r_v[None]))
          return u*fu, v*fv

      def bottom_friction_rate_bt(self, u, v, maskU, maskV, hu, hv):
          """rate [1/s] of the same bottom stress on the barotropic
          TRANSPORT: bottom stress r_cell u_bot dz_bot per unit area, with
          u_bot ~ U/h, gives dU/dt = -(r_cell dz_bot/h) U"""
          r_u, r_v = self.bottom_friction_rate(u, v, maskU, maskV)
          bu, bv = self._bottom_cell_mask(maskU), self._bottom_cell_mask(maskV)
          dzb_u = OM.np.sum(bu*self.dz, axis=0); dzb_v = OM.np.sum(bv*self.dz, axis=0)
          return r_u*dzb_u/OM.np.maximum(hu, 1e-12), r_v*dzb_v/OM.np.maximum(hv, 1e-12)
