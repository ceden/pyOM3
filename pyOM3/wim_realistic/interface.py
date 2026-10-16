"""
 Coupling interface for wim_realistic: wim_simple's interface with the
 vertical kernel generalised to the ocean's VARIABLE grid dzt(z).

 Subclasses wim_simple.interface and overrides exactly the three methods
 that touch the vertical spacing: _wave_kernel (face construction),
 calc_stokes and calc_stress_work (cell-average divisions /dz, the three
 analytic tail closures and the cascade-credit deposition). Everything
 else -- calc_doppler_vel/_df (pure Qdz sums), calc_dt_us, calc_stress,
 calc_phioc, the tail-factor helpers -- inherits unchanged (the tail
 factors themselves are source.py's when input_scheme = 'janssen').

 dz may be a scalar (uniform ocean, pe_simple) or a 1-D array of nz cell
 thicknesses dzt(z), bottom to top (pe_realistic convention). For scalar
 dz the original uniform arithmetic is kept verbatim, so results are
 bit-identical to wim_simple; for the array the cell faces are chained
 exactly and the cell-integrated kernel keeps all its telescoping
 identities (barotropic projection, Stokes transport, stress column sum)
 on any dzt (checked by test/wim/identities.py).

 MRO note (as for source): the nominal base chain wim_simple.main/
 boundary of the parent sits dormant behind wim_realistic's own chain in
 model(propagation, interface, source, wim_simple.model), see model.py.
"""
import pyOM3                 as OM
from   pyOM3.wim_simple.interface   import interface as _interface_uniform, _expi_neg


class interface(_interface_uniform):

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

          dz may be a SCALAR (uniform ocean grid, e.g. pe_simple) or a 1-D
          ARRAY of nz cell thicknesses dzt(z), bottom to top (pe_realistic
          convention, k ascending upward). The cell-integrated kernel is
          exact for any cell faces, so nothing downstream changes: all
          consumers divide by the per-cell thickness self._dzc set here.
          For scalar dz the face construction keeps the original uniform
          arithmetic verbatim (bit-identical results).
          """
          dz = OM.np.asarray(dz, OM.prec)
          if dz.ndim == 0:
             z  = OM.np.arange(nz)*dz
             z  = z - z[-1] - dz/2
             zt, zb = z + dz/2, z - dz/2               # top face of last cell = 0
             self._dzc = OM.np.full(nz, float(dz), OM.prec)
          else:
             if dz.shape != (nz,):
                raise ValueError('dz must be a scalar or shape (%d,), got %s'
                                 % (nz, dz.shape))
             # faces CHAINED exactly (zb[k] = zt[k-1] bitwise), so the
             # telescoping sum_k Qdz = 1 - e^{-2kD} holds to round-off of
             # the exponential alone, as on the uniform grid
             cs = OM.np.cumsum(dz)
             zt = cs - cs[-1]                          # top faces, zt[-1] = 0
             zb = OM.np.concatenate((OM.np.asarray([-cs[-1]]), zt[:-1]))
             z  = 0.5*(zt + zb)                        # cell centres
             self._dzc = dz
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
          exp = Qdz*self.grav/(2*self.omega_t[None,:]**2*self._dzc[:,None])
          fak  = 2*self.omega_t**3/self.grav
          si = fak[:,None,None]*OM.np.sum( E[sl,:,:,:]*self.sinth[sl,None,None,None],
                                         axis=0 )*self.dd
          ci = fak[:,None,None]*OM.np.sum( E[sl,:,:,:]*self.costh[sl,None,None,None]
                                        ,axis=0 )*self.dd
          # surface amplitude at the cut-off (before the depth factor is
          # applied), used for the analytic high-frequency tail below. The
          # resolved sum below covers the last real bin mc up to its upper
          # face fu[mc], so the tail starts at f_c = fu[mc], not at the bin
          # centre ft[mc] (that would count the upper half bin twice). The
          # f^-5 tail amplitude at fu[mc] follows from the bin value:
          # omega^3 E(f) = omega_t^3 E_c (f/ft)^3 (f/ft)^-5, i.e. a factor
          # (ft[mc]/fu[mc])^2.
          mc   = -self.halo-1                     # last prognostic (real) freq bin
          r_c  = (self.ft[mc]/self.fu[mc])**2
          si_c = self._xy(si[mc])*r_c
          ci_c = self._xy(ci[mc])*r_c
          si = si[None,:,:,:]*exp[:,:,None,None] 
          ci = ci[None,:,:,:]*exp[:,:,None,None] 
          
          us = OM.np.sum(self._xy(si[:,sl])*self.dft[None,sl,None,None], axis=1)
          vs = OM.np.sum(self._xy(ci[:,sl])*self.dft[None,sl,None,None], axis=1)
 
          # contribution by the high-frequency f^-5 Phillips tail (f > f_c = fu[mc]).
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
          fc = self.fu[mc]                        # upper face of the last real bin
          wc = self.omega_u[mc]                   # omega_c = 2 pi f_c
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
          tail = ( Ib - _I(-zt) )/self._dzc
          us = us + tail[:,None,None]*si_c[None,:,:]
          vs = vs + tail[:,None,None]*ci_c[None,:,:]
          return self._out(OM.np.stack([us, vs]))


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
          exp = Qdz*self.grav/(2*self.omega_t[None,:]**2*self._dzc[:,None])
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
          exp = ( Hb - _H(-zt) )/self._dzc        # cell-integrated tail profile

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
              if getattr(self, 'input_scheme', 'wamdi') == 'janssen':
                # Janssen tail (source.py): the ecwam quadrature nodes of
                # the tail stress, each deposited with 2 k_j e^{2 k_j z}
                w = self._janssen_tail_depth_weights(zt, zb)  # (nz,nd,Ny,Nx)
              else:
                fac_tau, _, a, fL = self._tail_input_factors()   # (nd,Ny,Nx)
                eps0 = self.rho_a/self.rho_w
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
                w = (Vb - _V(-zt))/self._dzc[:,None,None,None]   # (nz,nd,Ny,Nx)
              Ec = self.E[:, -self.halo, :, :]
              tx = tx + OM.np.sum(self._xy(w[:,sl])*self._xy(Ec[None,sl])
                                *self.sinth[None,sl,None,None],axis=1)*self.dd
              ty = ty + OM.np.sum(self._xy(w[:,sl])*self._xy(Ec[None,sl])
                                *self.costh[None,sl,None,None],axis=1)*self.dd
              cx, cy, _ = self._cascade_credit()
              tx = OM.modify_array(tx, (-1,), tx[-1] + cx/self._dzc[-1])
              ty = OM.modify_array(ty, (-1,), ty[-1] + cy/self._dzc[-1])
              return self._out(OM.np.stack([tx, ty]))
          fac = 2*fc**5*self.beta_o/self.grav**5*self.e_mean**2*self.om_mean**7
          fac = self._xy(fac)[None]*(2*OM.np.pi)**5*exp[:,None,None]

          si = OM.np.sum( self._xy(self.E[sl,-self.halo]) \
                              *self.sinth[sl,None,None],axis=0 )*self.dd
          ci = OM.np.sum( self._xy(self.E[sl,-self.halo]) \
                              *self.costh[sl,None,None],axis=0 )*self.dd 
          tx = tx + fac*si[None,:,:] 
          ty = ty + fac*ci[None,:,:]  
          return self._out(OM.np.stack([tx, ty]))
