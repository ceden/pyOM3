"""
 interface to ocean, delivers Stokes velocity and surface wave stress or interior force
 also calculates the Doppler shift velocity.
 
 Based on class main. 
"""

import pyOM3                 as OM
from   pyOM3.wim_simple.main        import main
import numpy                        as np__


class interface(main):

      # INTERIOR SELECTION of the exchange fields. The outputs of this class
      # are unpadded (they go to the ocean model). numpy: the y,x slices
      # [halo:-halo] are taken where they always were (_xy), and _out is
      # the identity. JAX: a slice on y,x is no interior in the inline shard
      # layout, so _xy keeps the full padded arrays (the operators are
      # pointwise in y,x and the extra work is the halo columns) and _out
      # removes every shard's halo at the end (decomposition_jax.unpad).
      # _pad_in is the inverse for the ocean fields that come in unpadded.
      def _xy(self, a):
          if OM.jax: return a
          s = slice(self.halo, -self.halo)
          return a[..., s, s]

      def _out(self, a):
          return self.unpad(a) if OM.jax else a

      def _pad_in(self, a):
          """(nf,ny,nx) interior -> padded (Nf,Ny,Nx), halo-filled"""
          if OM.jax: return self.apply_bc(self.pad(a, lead=1))
          return self.apply_bc(OM.np.pad(a, self.halo))


      def _wave_kernel(self,dz,nz):
          """
          Shared discrete vertical kernel of the wave-ocean coupling.

          The continuous kernel is Q(k,z) = 2 k e^{2kz} with int Q dz = 1. All
          coupling integrals (Stokes velocity, stress force, Doppler velocity)
          use the SAME discrete kernel so that the exchange term C of the
          energy cycle closes discretely (the Doppler projection is the adjoint
          of the profile assignment).

          Each ocean cell k with faces z_{k+-1/2} receives the cell integral
             Qdz(k,f)  = e^{2 k z_top} - e^{2 k z_bot}      ( = int_cell Q dz )
          instead of the point value Q(z_k) dz. This telescopes over the
          column, sum_k Qdz = 1 - e^{-2 k D}, so barotropic currents are
          projected exactly, the Stokes transport and the column-integrated
          stress force are exact, independent of dz.

          Returns z (cell centres), zt, zb (cell faces, top face = 0) and
             Qdz     (nz, Nf)  cell-integrated kernel
             dQdz_df (nz, Nf)  its analytic frequency derivative,
                               d/df Qdz = (2k/f) * 2 (z_t e^{2kz_t} - z_b e^{2kz_b})
          on the padded frequency axis.

          If the attribute ``closed_bottom`` is set to a true value, the
          integration of the bottom cell is extended to z -> -infinity
          (e^{2 k z_bot} -> 0), so that sum_k Qdz = 1 exactly: no momentum is
          deposited below the column floor, the bottom cell absorbs the deep
          part of the profile, and the Doppler projection implicitly assumes
          u_L constant below the floor. Recommended for shallow test columns;
          for deep columns (2 k D >> 1) it makes no difference. The tail
          antiderivatives in calc_stokes / calc_stress_work honour the same
          closure. Default is the open bottom (fraction e^{-2kD} lost).
          """
          z  = OM.np.arange(nz)*dz
          z  = z - z[-1] - dz/2
          zt, zb = z + dz/2, z - dz/2                  # top face of last cell = 0
          k2 = 2*self.omega_t[None,:]**2/self.grav     # = 2 k          (1,Nf)
          et, eb  = OM.np.exp(zt[:,None]*k2), OM.np.exp(zb[:,None]*k2)
          if getattr(self,'closed_bottom',False):
             eb = OM.modify_array(eb, (0,slice(None)), 0.0)   # z_bot -> -inf
             zb_eb = zb[:,None]*eb                            # z e^{2kz} -> 0
          else:
             zb_eb = zb[:,None]*eb
          Qdz     = et - eb
          dQdz_df = (k2/self.ft[None,:]) * (zt[:,None]*et - zb_eb) * 2
          return z, zt, zb, Qdz, dQdz_df


      def calc_stokes(self,dz,nz,E=None):
          """
          returns Stokes velocity on vertical grid given by input
          output is unpadded since it goes to ocean model

          E: optional spectrum (padded, same layout as self.E) to project
          instead of self.E. The operator (including the analytic f^-5
          tail, which is anchored linearly at bin -halo-1) is LINEAR in E,
          so calc_dt_us uses this to project the per-step increment dE.
          """
          E  = self.E if E is None else E
          sl = slice(self.halo,-self.halo)
          z, zt, zb, Qdz, _ = self._wave_kernel(dz,nz)
          # cell-integrated depth factor: replaces the point value
          # e^{2 k z_c} by its cell average Qdz/(2 k dz), so that the discrete
          # Stokes transport  sum_k u_S dz  equals the exact integral and the
          # kernel is the same as in calc_doppler_vel / calc_stress_work.
          exp = Qdz*self.grav/(2*self.omega_t[None,:]**2*dz)
          fak  = 2*self.omega_t**3/self.grav
          si = fak[:,None,None]*OM.np.sum( E[sl,:,:,:]*self.sinth[sl,None,None,None],
                                         axis=0 )*self.dd
          ci = fak[:,None,None]*OM.np.sum( E[sl,:,:,:]*self.costh[sl,None,None,None]
                                        ,axis=0 )*self.dd
          # surface amplitude at the cut-off bin (before the depth factor is
          # applied), used for the analytic high-frequency tail below.
          mc   = -self.halo-1                     # last prognostic (real) freq bin
          si_c = self._xy(si[mc])
          ci_c = self._xy(ci[mc])
          si = si[None,:,:,:]*exp[:,:,None,None] 
          ci = ci[None,:,:,:]*exp[:,:,None,None] 
          
          us = OM.np.sum(self._xy(si[:,sl])*self.dft[None,sl,None,None], axis=1)
          vs = OM.np.sum(self._xy(ci[:,sl])*self.dft[None,sl,None,None], axis=1)
 
          # contribution by the high-frequency f^-5 Phillips tail (f > f_c).
          # Exact closed form of u_S^hi(z) = (2/g) int s F(f_c,theta) omega_c^3
          #   [ exp(2 z omega_c^2/g)
          #     - omega_c sqrt(-2 pi z/g) erfc(omega_c sqrt(-2 z/g)) ] dtheta ,
          # the analytic integral over the f^-5 tail keeping the full depth
          # dependence. It reduces to the WAM single-bin estimate only for
          # |z| << g/(2 omega_c^2); at the model cut-off that decay scale is a
          # few cm, so the tail must be applied as a depth profile (all levels),
          # not lumped into the surface cell.
          # The tail profile is applied with cell-integrated weights as well:
          # with t = -z >= 0 and a = 2 omega_c^2/g the profile is
          #   B(t) = f_c [ e^{-a t} - sqrt(pi a t) erfc(sqrt(a t)) ]
          # and its antiderivative I(t) = int_0^t B dt' is (S = sqrt(t))
          #   I(t) = f_c [ (1-e^{-a t})/a - 2 sqrt(pi a) ( S^3 erfc(sqrt(a) S)/3
          #            + (1-(a S^2+1) e^{-a S^2})/(3 sqrt(pi) a^{3/2}) ) ]
          # with I(inf) = f_c/(3a) = f_c g/(6 omega_c^2), so the discrete tail
          # transport equals the exact integral (2 pi/3) f_c^2 int s F dtheta.
          # Each cell receives ( I(-z_bot) - I(-z_top) )/dz in place of B(-z_c).
          fc = self.ft[mc]
          wc = self.omega_t[mc]                   # omega_c = 2 pi f_c         
          if OM.jax is None:  from scipy.special     import erfc
          else:               from jax.scipy.special import erfc
          a = 2*wc**2/self.grav
          def _I(t):
              S = OM.np.sqrt(t)
              J = S**3*erfc(OM.np.sqrt(a)*S)/3.0 \
                  + (1.0 - (a*S**2 + 1.0)*OM.np.exp(-a*S**2))/(3.0*OM.np.sqrt(OM.np.pi)*a**1.5)
              return fc*( (1.0 - OM.np.exp(-a*t))/a - 2.0*OM.np.sqrt(OM.np.pi*a)*J )
          Ib = _I(-zb)
          if getattr(self,'closed_bottom',False):
             Ib = OM.modify_array(Ib, (0,), fc/(3.0*a))       # I(inf)
          tail = ( Ib - _I(-zt) )/dz
          us = us + tail[:,None,None]*si_c[None,:,:]
          vs = vs + tail[:,None,None]*ci_c[None,:,:]
          return self._out(OM.np.stack([us, vs]))


      def calc_dt_us(self,dz,nz,dt=None):
          """
          returns (dt_us, ust): the discrete Stokes tendency d/dt u_S
          over the last coupling interval and the current Stokes
          velocity, both (2,nz,ny_loc,nx_loc), unpadded (ocean grid).
          dt_us is the extra forcing for an ocean model whose prognostic
          velocity is the Lagrangian mean u_L = u + u_S.

          Since calc_stokes is LINEAR in E, the finite difference of its
          output between two coupling times is the exact projection of
          the discrete spectral change over the interval,
             (u_S(t2) - u_S(t1))/(t2-t1) = L[ (E(t2) - E(t1))/(t2-t1) ],
          including the AB propagation (any number of wave substeps),
          the positivity floor, the implicit source update, the land
          mask and imphftail -- everything the prognostic state
          received. Differencing AFTER the projection keeps the stored
          state small ((2,nz) per column instead of the (nd,nf)
          spectrum) and yields ust in the same call.

          Call ONCE per coupling, after the wave step(s), before the
          ocean steps the same interval; dt is the coupling interval
          (default: one wave step self.dt). Applied as a plain forward
          increment to u_L (OUTSIDE any Adams-Bashforth interpolation of
          the ocean tendencies, constant rate over the ocean substeps),
          the increments telescope so that u = u_L - u_S satisfies the
          Eulerian update exactly; the wave -> MKE exchange is
          u_L . dt_us. A side benefit of the u_L formulation: the model
          velocity goes to calc_doppler_vel directly, no u + u_S
          composition needed.

          The first call returns dt_us = 0 (one-interval error after a
          restart; add ust_last to the restart file if that matters for
          the energy budgets).
          """
          dt  = self.dt if dt is None else dt
          ust = self.calc_stokes(dz,nz)
          if getattr(self,'ust_last',None) is None:
             dt_us = OM.np.zeros_like(ust)
          else:
             dt_us = (ust - self.ust_last)/dt
          self.ust_last = ust
          return dt_us, ust
 



      def calc_stress_work(self,dz,nz):
          """
          returns depth dependent wave stress force on ocean
          output is unpadded since it goes to ocean model

          The vertical profile is discretised conservatively: each cell gets
          the CELL-INTEGRATED profile divided by dz (not the point value at
          the cell centre). The high-frequency components decay on scales
          g/(2 omega^2) (centimetres at the spectral tail), far below any
          ocean dz, so point sampling exp(2 z omega^2/g) at cell centres
          loses most of their momentum. With cell-integrated weights the
          column sum  sum_k tau(z_k) dz  equals the surface stress
          calc_stress() analytically, for any dz.
          """
          sl = slice(self.halo,-self.halo)
          z, zt, zb, Qdz, _ = self._wave_kernel(dz,nz)
          Fdiss = self.gamma_diss[None,:,:,:]*self.E
          if getattr(self, 'tail_flux', 'wamdi') == 'input' \
                  and hasattr(self, 'freq_mask') and hasattr(self, 'gamma_in'):
              Fdiss = OM.np.where(self.freq_mask[None,:,:,:],
                                  Fdiss, -self.gamma_in*self.E)     # see calc_stress
          si = OM.np.sum( Fdiss[sl,:,:,:]*self.sinth[sl,None,None,None],axis=0 )*self.dd
          ci = OM.np.sum( Fdiss[sl,:,:,:]*self.costh[sl,None,None,None],axis=0 )*self.dd

          # cell-integrated exp(2 z w^2/g) from the shared kernel (= Qdz/(2k dz))
          exp = Qdz*self.grav/(2*self.omega_t[None,:]**2*dz)
          si = -2/self.grav*si[None,:,:,:]*exp[:,:,None,None]  \
                            *self.omega_t[None,:,None,None]**3
          ci = -2/self.grav*ci[None,:,:,:]*exp[:,:,None,None]  \
                            *self.omega_t[None,:,None,None]**3

          tx = OM.np.sum(self._xy(si[:,sl])*self.dft[None,sl,None,None], axis=1)
          ty = OM.np.sum(self._xy(ci[:,sl])*self.dft[None,sl,None,None], axis=1)

          # contribution by tail. Depth profile of the f^-5 tail is
          #   ee(z) = sqrt(pi)*erfc(fc*sqrt(-b z))/(2 sqrt(-b z)),  b = 8 pi^2/g,
          # with antiderivative (t = -z >= 0)
          #   H(t) = sqrt(pi)/b * [ sqrt(b t) erfc(fc sqrt(b t))
          #                         + (1 - exp(-fc^2 b t))/(fc sqrt(pi)) ] ,
          # H(inf) = 1/(b fc). Cell weight = (H(t_bot)-H(t_top))/dz, so the
          # column sum equals the analytic integral exactly. The prefactor is
          # (2 pi)^5, which makes the depth integral of the tail identical to
          # the tail term of calc_stress (the old (pi)^5 was a factor 2^5=32
          # too small).
          if OM.jax is None:  from scipy.special     import erfc
          else:               from jax.scipy.special import erfc
          # NOTE on the tail anchor: index -halo is the frequency GHOST bin.
          # This is intentional: imphftail (model.py) operates on the full
          # padded arrays and maintains the f^-5 extension in the ghost every
          # source step, so the analytic tail continues from one bin above
          # the last physical bin without double-counting the prognostic sum.
          # CAVEAT for offline diagnostics: restart files store the interior
          # bins only -- after loading a restart, apply one imphftail call
          # before evaluating tail terms, or the anchor reads zero.
          fc, b = self.ft[-self.halo] , 8*OM.np.pi**2/self.grav

          def _H(t):
              s = OM.np.sqrt(b*t)
              return OM.np.sqrt(OM.np.pi)/b*( s*erfc(fc*s)
                       + (1 - OM.np.exp(-fc**2*b*t))/(fc*OM.np.sqrt(OM.np.pi)) )

          Hb = _H(-zb)
          if getattr(self,'closed_bottom',False):
             Hb = OM.modify_array(Hb, (0,), 1.0/(b*fc))       # H(inf)
          exp = ( Hb - _H(-zt) )/dz               # cell-integrated tail profile

          if getattr(self, 'tail_flux', 'wamdi') == 'input':
              # gamma_in-based tail, depth resolved. Per direction the tail
              # stress density above f_L = max(f_c, f_c/uu_c) is
              #   tau'(f) = C0 (a f^-2 - f^-3),  C0 = eps/4 (2pi)^2 f_c^5,
              # deposited with 2k e^{2kz}, 2k = b f^2. Cumulative from the
              # surface: V(t) = C0 [ a/fL - 1/(2 fL^2) - (a J2(t) - J3(t)) ],
              #   J2 = e^{-bL}/fL - sqrt(pi b t) erfc(sqrt(bL)),
              #   J3 = ( e^{-bL}/fL^2 - b t E1(bL) )/2,   bL = b t fL^2,
              # (V(0)=0, V(inf)= the calc_stress surface tail), so the cell
              # weights (V(t_bot)-V(t_top))/dz sum to the surface value
              # exactly. E1(x) = -expi(-x). The nonlinear cascade credit is
              # deposited in the TOP cell: its carrier waves have deposition
              # scales g/(2 omega^2) <= centimetres, far below any ocean dz.
              fac_tau, _, a, fL = self._tail_input_factors()   # (nd,Ny,Nx)
              eps0 = 1.225/1000.
              C0 = 0.25*eps0*(2*OM.np.pi)**2*fc**5
              fLc = OM.np.where(OM.np.isfinite(fL), fL, 1.0)   # dummy where no wind
              act = OM.np.where(OM.np.isfinite(fL), 1.0, 0.0)
              surf = C0*(a/fLc - 0.5/fLc**2)*act               # = fac_tau/E_c... per dir
              def _V(t):                                       # t: (nz,) -> (nz,nd,Ny,Nx)
                  bL = b*t[:,None,None,None]*fLc[None]**2
                  ex = OM.np.exp(-bL)
                  J2 = ex/fLc[None] - OM.np.sqrt(OM.np.pi*b*t[:,None,None,None])\
                                      *erfc(OM.np.sqrt(bL))
                  J3 = 0.5*(ex/fLc[None]**2
                            + b*t[:,None,None,None]*_expi_neg(OM.np.maximum(bL,1e-300)))
                  return (surf[None] - C0*(a[None]*J2 - J3))*act[None]
              Vb = _V(-zb)
              if getattr(self,'closed_bottom',False):
                  Vb = OM.modify_array(Vb, (0,), surf)        # (nd,Ny,Nx) into slot 0
              w = (Vb - _V(-zt))/dz                            # (nz,nd,Ny,Nx)
              Ec = self.E[:, -self.halo, :, :]
              tx = tx + OM.np.sum(self._xy(w[:,sl])*self._xy(Ec[None,sl])
                                *self.sinth[None,sl,None,None],axis=1)*self.dd
              ty = ty + OM.np.sum(self._xy(w[:,sl])*self._xy(Ec[None,sl])
                                *self.costh[None,sl,None,None],axis=1)*self.dd
              cx, cy, _ = self._cascade_credit()
              tx = OM.modify_array(tx, (-1,), tx[-1] + cx/dz)
              ty = OM.modify_array(ty, (-1,), ty[-1] + cy/dz)
              return self._out(OM.np.stack([tx, ty]))
          fac = 2*fc**5*2.56/self.grav**5*self.e_mean**2*self.om_mean**7
          fac = self._xy(fac)[None]*(2*OM.np.pi)**5*exp[:,None,None]

          si = OM.np.sum( self._xy(self.E[sl,-self.halo]) \
                              *self.sinth[sl,None,None],axis=0 )*self.dd
          ci = OM.np.sum( self._xy(self.E[sl,-self.halo]) \
                              *self.costh[sl,None,None],axis=0 )*self.dd 
          tx = tx + fac*si[None,:,:] 
          ty = ty + fac*ci[None,:,:]  
          return self._out(OM.np.stack([tx, ty]))


          
      def _tail_input_factors(self):
          """
          Per-direction factors of the gamma_in-based (TAUHF-like) tail flux,
          tail_flux='input'. In the diagnostic f^-5 range the equilibrium
          S_in + S_nl + S_diss = 0 holds; the flux to the ocean is -S_diss,
          booked as the analytic gamma_in integral over the tail PLUS the
          resolved-range nonlinear cascade credit (see calc_stress). With
          uu_c(theta) = 28 u* omega_c cos(theta-udir)/g (Sinput_wamdi's uu at
          the cutoff f_c) and the input active only above f_1 = f_c/uu_c:
            I_tau = eps/4 (2pi)^2 f_c^3 E_c * ( uu_c - 1/2 )        uu_c >= 1
                    eps/4 (2pi)^2 f_c^3 E_c * uu_c^2 / 2        0 < uu_c < 1
            I_phi = eps/4 (2pi)   f_c^2 g E_c * ( uu_c/2 - 1/3 )    uu_c >= 1
                    eps/4 (2pi)   f_c^2 g E_c * uu_c^3 / 6      0 < uu_c < 1
          (continuous at uu_c=1; zero for cos<=0). Integrands decay like
          f^-2, i.e. as benign as the WAMDI form -- unlike Janssen's
          dissipation, whose tail integral diverges linearly.
          Returns (fac_tau, fac_phi, a, fL): factors per (nd,Ny,Nx) to be
          multiplied by E_c(theta) and summed over directions; a = uu/f and
          fL = max(f_c, 1/a) for the depth kernel of calc_stress_work.
          """
          eps = 1.225/1000.
          fc  = self.ft[-self.halo]        # ghost-bin anchor BY DESIGN (see calc_stress_work)
          omc = 2*OM.np.pi*fc
          cos = OM.np.cos(self.theta[:,None,None] - self.udir[None,:,:])
          uu_c = OM.np.maximum(0.0, 28.*self.ustar[None,:,:]*omc*cos/self.grav)
          hi  = uu_c >= 1.0
          fac_tau = 0.25*eps*(2*OM.np.pi)**2*fc**3 * OM.np.where(
                        hi, uu_c - 0.5, 0.5*uu_c**2)
          fac_phi = 0.25*eps*(2*OM.np.pi)*fc**2*self.grav * OM.np.where(
                        hi, 0.5*uu_c - 1./3., uu_c**3/6.)
          a  = uu_c/fc                                   # uu(f) = a f
          fL = OM.np.where(uu_c > 0.0, OM.np.maximum(fc, fc/OM.np.maximum(uu_c,1e-30)), OM.np.inf)
          return fac_tau, fac_phi, a, fL

      def _cascade_credit(self):
          """
          Resolved-range nonlinear cascade flux into the tail. The exact S_nl
          conserves momentum and energy over the full spectrum, so the tail
          receives  int_tail S_nl dOmega = - int_resolved S_nl dOmega ,
          available from the stored resolved S_nl (model.time_step). Returns
          (cx, cy, ce): kinematic momentum (x,y; weight omega) and energy
          (weight g) credits, unpadded, >= 0 for a direct cascade. Note the
          discrete DIA on the truncated grid is not exactly conservative at
          the cutoff; the residual is booked here as well -- whatever the
          discrete operator removes from the resolved range is deposited in
          the tail (see wim.tex).
          """
          sl = slice(self.halo,-self.halo)
          if not hasattr(self, 'S_nl'):
              z = self._xy(OM.np.zeros((self.Ny,self.Nx), OM.prec))
              return z, z, z
          Snl = self._xy(self.S_nl[sl,sl])
          if hasattr(self, 'freq_mask'):     # prognostic bins only: the
              Snl = OM.np.where(              # diagnostic-range S_nl belongs
                  self._xy(self.freq_mask[None,sl]), Snl, 0.)  # to the tail-side
                                              # equilibrium booking
          kx_om = (self.omega_t[None,:]*self.sinth[:,None])[sl,sl]
          ky_om = (self.omega_t[None,:]*self.costh[:,None])[sl,sl]
          dft = self.dft[sl]
          cx = -OM.np.sum(OM.np.sum(Snl*kx_om[:,:,None,None],axis=0)*self.dd
                          *dft[:,None,None],axis=0)
          cy = -OM.np.sum(OM.np.sum(Snl*ky_om[:,:,None,None],axis=0)*self.dd
                          *dft[:,None,None],axis=0)
          ce = -self.grav*OM.np.sum(OM.np.sum(Snl*self.dd,axis=0)
                          *dft[:,None,None],axis=0)
          return cx, cy, ce

      def calc_stress(self):
          """
          returns surface wave stress on ocean  
          output is unpadded since it goes to ocean model
          """
          sl = slice(self.halo,-self.halo)
          S_diss = self.gamma_diss[None,:,:,:]*self.E
          if getattr(self, 'tail_flux', 'wamdi') == 'input' \
                  and hasattr(self, 'freq_mask') and hasattr(self, 'gamma_in'):
              # in-grid DIAGNOSTIC bins (imphftail-imposed, above the
              # per-column cutoff m0): the equilibrium S_in+S_nl+S_diss=0
              # holds there BY the model's own construction, so the flux to
              # the ocean is booked at gamma_in (discretely, with the actual
              # imposed spectrum); prognostic bins keep the true gamma_d
              # dissipation. The S_nl part collapses to the prognostic-masked
              # cascade credit (see _cascade_credit).
              S_diss = OM.np.where(self.freq_mask[None,:,:,:],
                                   S_diss, -self.gamma_in*self.E)
          kx_om = self.omega_t[None,:]*self.sinth[:,None]
          ky_om = self.omega_t[None,:]*self.costh[:,None]
          Sd = self._xy(S_diss[sl,sl])
          tx = -OM.np.sum(Sd*kx_om[sl,sl,None,None]*self.dd,axis=0)
          ty = -OM.np.sum(Sd*ky_om[sl,sl,None,None]*self.dd,axis=0)
          tx = OM.np.sum(tx*self.dft[sl,None,None],axis=0)
          ty = OM.np.sum(ty*self.dft[sl,None,None],axis=0)

          # contribution by tail: switch tail_flux = 'wamdi' (default) | 'input'
          if getattr(self, 'tail_flux', 'wamdi') == 'input':
              # gamma_in-based equilibrium-range flux + nonlinear cascade
              # credit (see _tail_input_factors / _cascade_credit docstrings)
              fac_tau, _, _, _ = self._tail_input_factors()
              Ec = self._xy(self.E[sl, -self.halo])
              ft_ = self._xy(fac_tau[sl])
              taux = OM.np.sum(Ec*ft_*self.sinth[sl,None,None]*self.dd, axis=0)
              tauy = OM.np.sum(Ec*ft_*self.costh[sl,None,None]*self.dd, axis=0)
              cx, cy, _ = self._cascade_credit()
              tx = tx + (taux + cx)
              ty = ty + (tauy + cy)
          else:
              # WAMDI-dissipation tail (analytic continuation of Sdiss_wamdi)
              Ec = self._xy(self.E[sl, -self.halo])
              taux = OM.np.sum(Ec*self.sinth[sl,None,None]*self.dd,axis=0)
              tauy = OM.np.sum(Ec*self.costh[sl,None,None]*self.dd,axis=0)
              fac = 2.56/self.grav**4*self.e_mean**2*self.om_mean**7*self.ft[-self.halo]**4
              fac = fac*(2*OM.np.pi)**3
              tx = tx + self._xy(fac)*taux
              ty = ty + self._xy(fac)*tauy
          return self._out(OM.np.stack([tx, ty]))



      def calc_doppler_vel(self,dz,nz,UL,VL):
          """
          calculate U(k) = int 2 k UL(z) exp(2kz) dz 
          output is padded since it goes to wave model
          """
          sl = slice(self.halo,-self.halo)
          z, zt, zb, Qdz, _ = self._wave_kernel(dz,nz)
          # U(f) = sum_k u_L(z_k) Qdz(k,f): the same cell-integrated kernel as
          # in calc_stokes/calc_stress_work (its adjoint), so the exchange term
          # C closes discretely and a barotropic current is projected exactly
          # (sum_k Qdz = 1 - e^{-2kD} -> U = u_L for depth-independent u_L).
          u = OM.np.sum( UL[:,None,:,:]*Qdz[:,sl,None,None],axis=0)
          v = OM.np.sum( VL[:,None,:,:]*Qdz[:,sl,None,None],axis=0)
          return self._pad_in(u), self._pad_in(v)


          
 
      def calc_doppler_vel_df(self,dz,nz,UL,VL):
          """
          analytic frequency derivative dU/df of the Doppler velocity
             U(f) = int 2 k UL(z) exp(2kz) dz ,   k = omega^2/g ,  omega = 2 pi f.
          Differentiating under the integral (all f-dependence is through k, with
          dk/df = 2k/f) gives
             dU/df = (2/f) int 2 k UL(z) exp(2kz) (1 + 2 k z) dz .
          Same discrete form as calc_doppler_vel with the extra factor
          (2/f)(1 + 2 k z); the (1+2kz) stays inside the z-sum, (2/f) factors out.
          Output is padded since it goes to the wave model. Replaces the numerical
          _ddf, which needed a frequency axis on U and broke for a bare (Ny,Nx)
          Doppler field.
          """
          sl = slice(self.halo,-self.halo)
          z, zt, zb, Qdz, dQdz_df = self._wave_kernel(dz,nz)
          # analytic df of the DISCRETE kernel of calc_doppler_vel, so the
          # radial refraction term R sees exactly the same Doppler projection
          # as the advection: d/df Qdz = (2k/f) 2 (z_t e^{2kz_t} - z_b e^{2kz_b}).
          du = OM.np.sum( UL[:,None,:,:]*dQdz_df[:,sl,None,None],axis=0)
          dv = OM.np.sum( VL[:,None,:,:]*dQdz_df[:,sl,None,None],axis=0)
          return self._pad_in(du), self._pad_in(dv)



      def calc_phioc(self):
          """
          Energy flux from the waves to the ocean by dissipation (WAM: PHIOC),
              Phi_oc(x,y) = - g * int S_diss df dtheta   [m^3/s^3, i.e. W/m^2
          per unit rho_w], positive INTO the ocean. This is the term
          int sigma Adot_diss dk of the wave energy budget (wim.tex ); it
          is destined to become the surface TKE flux boundary condition of the
          ocean's turbulence closure. Tail treatment follows the tail_flux
          switch: 'wamdi' (default) books the prognostic range only (no tail
          energy flux, the original behaviour); 'input' adds the analytic
          gamma_in tail integral and the resolved-range nonlinear cascade
          credit (see _tail_input_factors / _cascade_credit).
          Output is unpadded (ny_loc, nx_loc), like calc_stress.
          """
          sl = slice(self.halo,-self.halo)
          S_diss = self.gamma_diss[None,:,:,:]*self.E
          if getattr(self, 'tail_flux', 'wamdi') == 'input' \
                  and hasattr(self, 'freq_mask') and hasattr(self, 'gamma_in'):
              S_diss = OM.np.where(self.freq_mask[None,:,:,:],
                                   S_diss, -self.gamma_in*self.E)   # see calc_stress
          phi = -self.grav*OM.np.sum(
                 OM.np.sum(self._xy(S_diss[sl,sl])*self.dd,axis=0)
                 *self.dft[sl,None,None],axis=0)
          if getattr(self, 'tail_flux', 'wamdi') == 'input':
              # equilibrium-range tail energy flux (gamma_in based) plus the
              # resolved-range nonlinear cascade credit; retires the "tail
              # not included" limitation of the original diagnostic.
              _, fac_phi, _, _ = self._tail_input_factors()
              Ec = self._xy(self.E[sl, -self.halo])
              phi = phi + OM.np.sum(Ec*self._xy(fac_phi[sl])
                                    *self.dd, axis=0)
              _, _, ce = self._cascade_credit()
              phi = phi + ce
          return self._out(phi)


      def calc_phioc_work(self,dz,nz):
          """
          Depth profile of the dissipation energy flux calc_phioc [m^2/s^3],
          per ocean cell, index 0 at the bottom, unpadded (nz,ny_loc,nx_loc).
          The energy -g S_diss of each prognostic bin is deposited with the
          cell-integrated kernel of _wave_kernel, Qdz/dz with Q = 2k e^{2kz}
          (as the stress in calc_stress_work, bin by bin with sigma in place
          of k). The analytic tail and the cascade credit of tail_flux='input'
          (carrier waves with decay scales of centimetres and less) go into
          the top cell. The column sum  sum_k P_k dz  equals calc_phioc up to
          the open-bottom loss e^{-2kD} (exact with closed_bottom).
          Option phioc_kernel = 'stokes' (default, Q = 2k e^{2kz}) or
          'orbital' (Q = k e^{kz}, twice the depth scale).
          """
          sl = slice(self.halo,-self.halo)
          if getattr(self,'phioc_kernel','stokes') == 'orbital':
             z  = OM.np.arange(nz)*dz
             z  = z - z[-1] - dz/2
             zt, zb = z + dz/2, z - dz/2
             k1 = self.omega_t[None,:]**2/self.grav
             et, eb = OM.np.exp(zt[:,None]*k1), OM.np.exp(zb[:,None]*k1)
             if getattr(self,'closed_bottom',False):
                eb = OM.modify_array(eb, (0,slice(None)), 0.0)
             Qdz = et - eb
          else:
             _, _, _, Qdz, _ = self._wave_kernel(dz,nz)
          S_diss = self.gamma_diss[None,:,:,:]*self.E
          if getattr(self, 'tail_flux', 'wamdi') == 'input' \
                  and hasattr(self, 'freq_mask') and hasattr(self, 'gamma_in'):
              S_diss = OM.np.where(self.freq_mask[None,:,:,:],
                                   S_diss, -self.gamma_in*self.E)   # see calc_stress
          phi_f = -self.grav*OM.np.sum(self._xy(S_diss[sl,sl])*self.dd,axis=0) \
                  *self.dft[sl,None,None]                          # (nf,ny,nx)
          P = OM.np.sum(Qdz[:,sl,None,None]*phi_f[None,:,:,:],axis=1)/dz
          if getattr(self, 'tail_flux', 'wamdi') == 'input':
              _, fac_phi, _, _ = self._tail_input_factors()
              Ec = self._xy(self.E[sl, -self.halo])
              top = OM.np.sum(Ec*self._xy(fac_phi[sl])*self.dd, axis=0)
              _, _, ce = self._cascade_credit()
              P = OM.modify_array(P, (-1,), P[-1] + (top + ce)/dz)
          return self._out(P)



def exp1_jax(x, n_series=100, n_cf=400):
    """
    E1(x) = -expi(-x) for x > 0 under JAX, written out as in Cephes/scipy
    expn(1,x): the power series for x <= 1, the continued fraction for
    x > 1, both iterated to machine precision with the iteration frozen
    once converged (the early termination of scipy), so the result matches
    scipy.special.expi to a few 1e-15. jax.scipy.special.expi/exp1 exist
    but vectorise a scalar while_loop and take minutes on a (nz,nd,Ny,Nx)
    array (Oct 2026, jax 0.11); this runs in a second.
    """
    jnp, lax = OM.np, OM.jax.lax
    x = jnp.asarray(x)
    eps = jnp.finfo(x.dtype).eps
    one, zero = jnp.ones_like(x), jnp.zeros_like(x)
    # ---- series, x <= 1
    xs = jnp.where(x <= 1.0, x, 1.0)
    def sbody(k, c):
        yk, ans, done = c
        yk_n = yk*(-xs)/k
        ans_n = ans + yk_n/k
        t = jnp.where(ans_n != 0.0, jnp.abs(yk_n/ans_n), 1.0)
        yk  = jnp.where(done, yk, yk_n)
        ans = jnp.where(done, ans, ans_n)
        done = done | (t <= eps)
        return yk, ans, done
    _, ans_s, _ = lax.fori_loop(1, n_series, sbody, (one, zero, x > 1.0))
    e_small = -np__.euler_gamma - jnp.log(xs) - ans_s
    # ---- continued fraction, x > 1 (Cephes expn, n = 1)
    xc = jnp.where(x > 1.0, x, 2.0)
    BIG = 1.44115188075855872e17
    def cbody(k, d):
        pkm2, qkm2, pkm1, qkm1, ans, r, done = d
        odd = (k % 2) == 1
        yk = jnp.where(odd, 1.0, xc)
        xk = jnp.where(odd, 1.0 + (k - 1)/2.0, k/2.0)
        pk = pkm1*yk + pkm2*xk
        qk = qkm1*yk + qkm2*xk
        nz = qk != 0.0
        r_n = jnp.where(nz, pk/qk, r)
        t = jnp.where(nz, jnp.abs((ans - r_n)/r_n), 1.0)
        ans_n = jnp.where(nz, r_n, ans)
        big = jnp.abs(pk) > BIG
        sc = jnp.where(big, 1.0/BIG, 1.0)
        new = (pkm1*sc, qkm1*sc, pk*sc, qk*sc, ans_n, r_n)
        old = (pkm2, qkm2, pkm1, qkm1, ans, r)
        out = tuple(jnp.where(done, o, nw) for o, nw in zip(old, new))
        done = done | (t <= eps)
        return out + (done,)
    init = (one, xc, one, xc + 1.0, 1.0/(xc + 1.0), zero, x <= 1.0)
    d = lax.fori_loop(2, n_cf, cbody, init)
    e_big = d[4]*jnp.exp(-xc)
    return jnp.where(x <= 1.0, e_small, e_big)


def _expi_neg(x):
    """expi(-x) for x > 0: scipy on numpy (as before), exp1_jax under JAX"""
    if OM.jax is None:
        from scipy.special import expi
        return expi(-x)
    return -exp1_jax(x)
