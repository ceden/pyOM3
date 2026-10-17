"""
 Vertical momentum equation. The horizontal equations are pe_realistic's.

 dw/dt = -adv(w) - d p_nh/dz + friction  on the W grid (top face of
 cell k, thickness dzw[k]). No buoyancy term: it cancels against
 -d p_h/dz of pe_realistic's hydrostatic pressure (see __init__.py).
 No metric and no Coriolis terms in the w equation (pyOM2).

 Advection in transport form, the W-cell counterpart of mom_flux_2nd_u (Cartesian, no metric factors):
 the transport through a face of the W cell k is the mean of the
 transports through the same faces of the T cells k and k+1, so the
 W-cell continuity is the mean of the two T-cell continuities and the
 scheme conserves energy with the divergence-free (u,v,w). On a uniform
 grid this is nh_simple's momentum_advection_w. As in pe_realistic the
 fluxes are not masked by cell pairs; the masked transports carry the
 boundary condition.

 Based on class momentum of pe_realistic.
"""
import pyOM3                        as OM
from   pyOM3.nh_realistic.main      import main
from   pyOM3.pe_realistic.momentum  import momentum as momentum_pe
from   functools                    import partial


class momentum(main, momentum_pe):

      @partial(OM.jaxjit, static_argnums=0)
      def mom_flux_2nd_w(self, w, u, v, maskU, maskV, maskW):
          """
          Fluxes of w through the faces of the W cell k:
            fe at (i+1/2, j, k+1/2): transport 0.5 (utr[k] + utr[k+1]),
            fn at (i, j+1/2, k+1/2): transport 0.5 (vtr[k] + vtr[k+1]),
            ft at the T point k+1:   transport 0.5 (wtr[k] + wtr[k+1]),
          utr = u dyt dzt, vtr = v dxt dzt, wtr = w area_t (masked).
          """
          utr = u*maskU*self.dzt_z*self.dyt_y
          vtr = v*maskV*self.dzt_z*self.dxt_x
          wtr = w*maskW*self.area_t
          fe = - ( utr + self.rollz(utr,-1) )*0.25*( w + self.rollx(w,-1) )
          fn = - ( vtr + self.rollz(vtr,-1) )*0.25*( w + self.rolly(w,-1) )
          ft = - ( wtr + self.rollz(wtr,-1) )*0.25*( w + self.rollz(w,-1) )
          return fe, fn, ft

      @partial(OM.jaxjit, static_argnums=0)
      def w_tendency(self, u, v, w, dw, maskU, maskV, maskWc, tau):
          """
          Advective tendency of w at time level tau, divergence over the
          W-cell volume area_t dzw.
          """
          fe, fn, ft = self.mom_flux_2nd_w(w, u, v, maskU, maskV, maskWc)
          dw_ = ( (fe-self.rollx(fe,1)) + (fn-self.rolly(fn,1)) + (ft-self.rollz(ft,1)) ) \
                *maskWc/(self.area_t*self.dzw_z)
          return OM.modify_array(dw, (tau,slice(None),slice(None),slice(None)), dw_,
                                 out_sharding = self.sharding_4D)

      @partial(OM.jaxjit, static_argnums=0)
      def update_w_AB3(self, A, B, C, tau, taum1, taum2, w, dw, maskWc):
          w += self.dt*(A*dw[tau,:,:,:] + B*dw[taum1,:,:,:] + C*dw[taum2,:,:,:])*maskWc
          return self.apply_bc(w)

      @partial(OM.jaxjit, static_argnums=0)
      def harmonic_horizontal(self, p1, mixC, mask):
          """
          Horizontal harmonic operator of a field at T points horizontally
          (w, tracers): gradients at the U/V faces, divergence at the T
          point, Cartesian. Forward step, returns the field. The placement
          is exact for w, which sits at the T point horizontally (unlike the
          generic biharmonic for u and v).
          """
          fe = mixC*(self.rollx(p1,-1)-p1)/self.dxu_x*self.rollx(mask,-1)*mask
          fn = mixC*(self.rolly(p1,-1)-p1)/self.dyu_y*self.rolly(mask,-1)*mask
          fe, fn = self.apply_bc(fe), self.apply_bc(fn)
          return p1 + self.dt*( (fe-self.rollx(fe,1))/self.dxt_x
                              + (fn-self.rolly(fn,1))/self.dyt_y )*mask

      def horizontal_friction_w(self):
          """Ahbi and/or Ah on w, the same option keys as for u and v."""
          if hasattr(self,'Ahbi'):
             self.w = self.apply_bc( self.biharmonic_horizontal(self.w, self.Ahbi, self.maskWc) )
          if hasattr(self,'Ah'):
             self.w = self.apply_bc( self.harmonic_horizontal(self.w, self.Ah, self.maskWc) )
          return
