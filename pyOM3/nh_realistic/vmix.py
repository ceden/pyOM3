"""
 Vertical friction of w, following the option hierarchy of
 pe_realistic/implicit_vmix.py (vmix_scheme explicit / implicit, TKE).

 w lives on the W grid, so the roles of the spacings swap with respect to
 implicit_vdiff: the friction flux sits at the T point k,
     F[k] = Av_T[k] (w[k] - w[k-1]) / dzt[k]    (between W cells k-1 and k),
 and the tendency of W cell k is (F[k+1] - F[k]) / dzw[k]. A flux exists
 only between two wet W cells (maskWc[k] maskWc[k-1]). Under the TKE
 closure the viscosity kappaM (W grid) is averaged to the T point,
 Av_T[k] = 0.5 (kappaM[k] + kappaM[k-1]); its dissipation of w is not
 booked into the TKE equation.
"""
import pyOM3    as OM
from   functools import partial


class vmix_w:

      @partial(OM.jaxjit, static_argnums=0)
      def explicit_vfric_w(self, w, Av_T, maskWc):
          ft = Av_T*(w - self.rollz(w,1))/self.dzt_z*maskWc*self.rollz(maskWc,1)
          return self.apply_bc( w + self.dt*(self.rollz(ft,-1) - ft)/self.dzw_z*maskWc )

      @partial(OM.jaxjit, static_argnums=0)
      def implicit_vfric_w(self, w, Av_T, maskWc):
          """
          (I - dt d_z Av d_z) w^{n+1} = w^n on the W grid, Thomas
          algorithm along axis 0 over the interior rows. With
          delta[k] = dt Av_T[k] maskWc[k] maskWc[k-1] / dzt[k]:
            lower a[k] = -delta[k]/dzw[k]   (couples k to k-1)
            upper c[k] = -delta[k+1]/dzw[k] (couples k to k+1)
            diag  b[k] = 1 - a[k] - c[k]; identity on closed faces.
          """
          h, Nz = self.halo, self.Nz
          delta = self.dt*Av_T*maskWc*self.rollz(maskWc,1)/self.dzt_z
          a = -delta/self.dzw_z
          c = -self.rollz(delta,-1)/self.dzw_z
          wet = maskWc > 0.0
          a = OM.np.where(wet, a, 0.0)
          c = OM.np.where(wet, c, 0.0)
          b = 1.0 - a - c
          d = w*1.0

          cp = OM.np.zeros_like(w)
          dp = OM.np.zeros_like(w)
          cp = OM.modify_array(cp, (h, slice(None), slice(None)), c[h]/b[h])
          dp = OM.modify_array(dp, (h, slice(None), slice(None)), d[h]/b[h])
          def _fwd(k, state):
              cp, dp = state
              m  = b[k] - a[k]*cp[k-1]
              cp = OM.modify_array(cp, (k, slice(None), slice(None)), c[k]/m)
              dp = OM.modify_array(dp, (k, slice(None), slice(None)), (d[k] - a[k]*dp[k-1])/m)
              return (cp, dp)
          cp, dp = OM.for_loop(h+1, Nz-h, _fwd, (cp, dp))
          out = w*1.0
          out = OM.modify_array(out, (Nz-h-1, slice(None), slice(None)), dp[Nz-h-1])
          def _bwd(i, out):
              k = Nz-h-2-i
              return OM.modify_array(out, (k, slice(None), slice(None)), dp[k] - cp[k]*out[k+1])
          out = OM.for_loop(0, (Nz-2*h)-1, _bwd, out)
          out = out*maskWc + w*(1.0 - maskWc)
          return self.apply_bc(out)

      def vertical_friction_w(self):
          """w friction according to the option hierarchy of implicit_vmix."""
          scheme, conv, tkeo = self._vmix_options()
          if tkeo:
             Av_T = 0.5*(self.kappaM_tke + self.rollz(self.kappaM_tke,1))
             self.w = self.implicit_vfric_w(self.w, Av_T, self.maskWc)
             return
          Av = getattr(self, 'A_v', 0.0)
          if Av <= 0.0:
             return
          Av_T = Av*OM.np.ones_like(self.w)
          if scheme == 'explicit':
             self.w = self.explicit_vfric_w(self.w, Av_T, self.maskWc)
          else:
             self.w = self.implicit_vfric_w(self.w, Av_T, self.maskWc)
          return
