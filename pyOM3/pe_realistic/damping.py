"""
 Damping terms like biharmonic friction, on the variable grid with
 pseudo-spherical metric factors.

 The biharmonic operator here is GENERIC (one routine for u, v and
 tracers, distinguished only by the mask), so the metric placement is that
 of a T-point field for all of them: gradients at the U/V faces
 (/(cost*dxu), /dyu), cosu-weighted meridional flux, divergence at the T
 point (/(cost*dxt), /(cost*dyt)). For the U/V rows this is metrically
 inexact (pyOM2 keeps three separate friction routines,
 harmonic_friction.f90); since the operator is a numerical damping term
 and the energetically consistent formulation is out of scope, the
 approximation is accepted here -- do NOT reuse this pattern for a
 physical operator.

 Based on class main.
"""
import pyOM3                    as OM
from   pyOM3.pe_realistic.main     import main
from   functools                import partial

class damping(main):


          
      @partial(OM.jaxjit, static_argnums=0)    
      def biharmonic_horizontal(self,p1,mixC,mask):          
          fe = -mixC*(self.rollx(p1,-1)-p1)/(self.cost_y*self.dxu_x)*self.rollx(mask,-1)*mask
          fn = -mixC*(self.rolly(p1,-1)-p1)/self.dyu_y*self.cosu_y*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)   
          
          del2 = ( (fe-self.rollx(fe,1))/(self.cost_y*self.dxt_x) 
                 + (fn-self.rolly(fn,1))/(self.cost_y*self.dyt_y) )*mask
          del2 = self.apply_bc(del2)
          
          fe = (self.rollx(del2,-1)-del2)/(self.cost_y*self.dxu_x)*self.rollx(mask,-1)*mask
          fn = (self.rolly(del2,-1)-del2)/self.dyu_y*self.cosu_y*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
 
          return p1 + self.dt*( (fe-self.rollx(fe,1))/(self.cost_y*self.dxt_x) 
                                + (fn-self.rolly(fn,1))/(self.cost_y*self.dyt_y)  )*mask


    
      @partial(OM.jaxjit, static_argnums=0)    
      def biharmonic_momentum(self,u,v,Ahbi,maskU,maskV):          
          """
          Horizontal biharmonic MOMENTUM friction with the proper U/V-row
          metric placement, ported 1:1 from pyOM2
          for_src/main/biharmonic_friction.f90 (without the dissipation
          diagnostics; energetics out of scope). This replaces the generic
          T-point biharmonic_horizontal for u and v, which is metrically
          inexact on the U/V rows -- in a Munk balance the friction term is
          leading order in the boundary layer, so the placement matters.

          u rows: x-gradient over dxt(i+1), y-flux weighted cosu(j),
                  divergence over (cost dxu, cost dyt)
          v rows: x-gradient over cosu(j) dxu(i), y-flux over dyt(j+1) with
                  weight cost(j+1), divergence over (cosu dxt, cosu dyu)
          Optional cos(y)**biha_friction_cosPower scaling as in pyOM2
          (default 0). visc = sqrt(A_hbi) is applied in each of the two
          Laplacian passes, as in pyOM2.
          """
          p    = getattr(self,'biha_friction_cosPower',0.0)
          visc = OM.np.sqrt(OM.np.abs(Ahbi))
          dxtp1_x  = OM.np.roll(self.dxt,-1)[None,None,:]
          dytp1_y  = OM.np.roll(self.dyt,-1)[None,:,None]
          costp1_y = OM.np.roll(self.cost,-1)[None,:,None]

          # ---- zonal velocity ------------------------------------------
          fxa_x, fxa_y = self.cost_y**p, self.cosu_y**p
          def _del2_u(w):
              fe = visc*fxa_x*(self.rollx(w,-1)-w)/(self.cost_y*dxtp1_x) \
                   *maskU*self.rollx(maskU,-1)
              fn = visc*fxa_y*(self.rolly(w,-1)-w)/self.dyu_y*self.cosu_y \
                   *maskU*self.rolly(maskU,-1)
              fe, fn = self.apply_bc(fe), self.apply_bc(fn)
              return (fe-self.rollx(fe,1))/(self.cost_y*self.dxu_x) \
                    +(fn-self.rolly(fn,1))/(self.cost_y*self.dyt_y)
          del2 = self.apply_bc(_del2_u(u))
          u    = u - self.dt*maskU*_del2_u(del2)

          # ---- meridional velocity -------------------------------------
          fxa_x, fxa_y = self.cosu_y**p, costp1_y**p
          def _del2_v(w):
              fe = visc*fxa_x*(self.rollx(w,-1)-w)/(self.cosu_y*self.dxu_x) \
                   *maskV*self.rollx(maskV,-1)
              fn = visc*fxa_y*(self.rolly(w,-1)-w)/dytp1_y*costp1_y \
                   *maskV*self.rolly(maskV,-1)
              fe, fn = self.apply_bc(fe), self.apply_bc(fn)
              return (fe-self.rollx(fe,1))/(self.cosu_y*self.dxt_x) \
                    +(fn-self.rolly(fn,1))/(self.dyu_y*self.cosu_y)
          del2 = self.apply_bc(_del2_v(v))
          v    = v - self.dt*maskV*_del2_v(del2)
          return self.apply_bc(u), self.apply_bc(v)

      @partial(OM.jaxjit, static_argnums=0)
      def harmonic_momentum(self,u,v,Ah,maskU,maskV):
          """
          Horizontal harmonic momentum friction, 1:1 port of pyOM2's
          harmonic_friction.f90 (metric flux placement on U/V rows;
          energetics out of scope). Same flux structure as one pass of
          biharmonic_momentum, with coefficient Ah and diffusive sign.

          Optional cos(y)**hor_friction_cosPower scaling on the fluxes as
          in pyOM2 (enable_hor_friction_cos_scaling with cosPower = p maps
          to setting self.hor_friction_cosPower = p here; default 0 = off).
          Note the effective viscosity then scales as cos**p (single pass),
          unlike the biharmonic case where cos**p per pass gives cos**(2p).
          """
          p        = getattr(self,'hor_friction_cosPower',0.0)
          dxtp1_x  = OM.np.roll(self.dxt,-1)[None,None,:]
          dytp1_y  = OM.np.roll(self.dyt,-1)[None,:,None]
          costp1_y = OM.np.roll(self.cost,-1)[None,:,None]

          # ---- zonal velocity ------------------------------------------
          fe = Ah*self.cost_y**p*(self.rollx(u,-1)-u)/(self.cost_y*dxtp1_x) \
               *maskU*self.rollx(maskU,-1)
          fn = Ah*self.cosu_y**p*(self.rolly(u,-1)-u)/self.dyu_y*self.cosu_y \
               *maskU*self.rolly(maskU,-1)
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
          u = u + self.dt*maskU*( (fe-self.rollx(fe,1))/(self.cost_y*self.dxu_x)
                                 +(fn-self.rolly(fn,1))/(self.cost_y*self.dyt_y) )

          # ---- meridional velocity -------------------------------------
          fe = Ah*self.cosu_y**p*(self.rollx(v,-1)-v)/(self.cosu_y*self.dxu_x) \
               *maskV*self.rollx(maskV,-1)
          fn = Ah*costp1_y**p*(self.rolly(v,-1)-v)/dytp1_y*costp1_y \
               *maskV*self.rolly(maskV,-1)
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
          v = v + self.dt*maskV*( (fe-self.rollx(fe,1))/(self.cosu_y*self.dxt_x)
                                 +(fn-self.rolly(fn,1))/(self.dyu_y*self.cosu_y) )
          return self.apply_bc(u), self.apply_bc(v)

      @partial(OM.jaxjit, static_argnums=0)    
      def harmonic_vertical_expl(self,p1,mixC,mask):          
          ft = mixC*(self.rollz(p1,-1)-p1)/self.dzw_z*self.rollz(mask,-1)*mask 
          return p1 + self.dt* (ft-self.rollz(ft,1))/self.dzt_z *mask

      @partial(OM.jaxjit, static_argnums=0)    
      def harmonic_vertical_expl_tendency(self,p1,mixC,mask):          
          ft = mixC*(self.rollz(p1,-1)-p1)/self.dzw_z*self.rollz(mask,-1)*mask 
          return (ft-self.rollz(ft,1))/self.dzt_z *mask

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
          dzb_u = OM.np.sum(bu*self.dzt_z, axis=0); dzb_v = OM.np.sum(bv*self.dzt_z, axis=0)
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
          dzb_u = OM.np.sum(bu*self.dzt_z, axis=0); dzb_v = OM.np.sum(bv*self.dzt_z, axis=0)
          return r_u*dzb_u/OM.np.maximum(hu, 1e-12), r_v*dzb_v/OM.np.maximum(hv, 1e-12)
