"""
 progation of wave action

 Based on class main.
"""
import pyOM3                 as OM
from   pyOM3.wim_simple.main        import main
from   functools                    import partial
import numpy                        as np__


class propagation(main):


      # JIT STRUCTURE (JAX backend; OM.jaxjit is the identity on numpy).
      # The public methods mk_advection / add_propagation keep their
      # signatures and read the time-varying attributes (the Doppler
      # velocities ocean_U/V/Udf/Vdf) and the option flags; they hand them
      # to the jitted _mk_advection / _add_propagation as ARGUMENTS. With
      # self a static argument, anything read from self inside a jitted
      # method is baked into the compiled program at trace time, so a
      # current read as self.ocean_U there would stay the current of the
      # first step for the whole run. Grid tables and masks (constant for
      # the run) are read from self.

      def _current_args(self):
          """the Doppler fields as arguments of the jitted kernels, with
          static flags for the branches (None is not a valid jit argument)"""
          has_cur = hasattr(self,'ocean_U') and hasattr(self,'ocean_V')
          U = self.ocean_U if has_cur else 0.0
          V = self.ocean_V if has_cur else 0.0
          dU = getattr(self, 'ocean_Udf', None)
          dV = getattr(self, 'ocean_Vdf', None)
          has_df = has_cur and dU is not None and dV is not None \
                   and OM.np.ndim(dU) > 0 and OM.np.ndim(dV) > 0
          if not has_df: dU, dV = 0.0, 0.0
          return U, V, dU, dV, has_cur, has_df


      def mk_advection(self,xdot,ydot,fdot,ddot):
         if getattr(self,'propagation_backend','numpy') == 'fortran':
             return self.mk_advection_fortran(xdot,ydot,fdot,ddot)
         U, V, dU, dV, has_cur, has_df = self._current_args()
         return self._mk_advection(xdot,ydot,fdot,ddot,U,V,dU,dV,self.shard_consts(),
                                   has_current=has_cur, has_df=has_df)


      @partial(OM.jaxjit, static_argnums=(0,), static_argnames=('has_current','has_df'))
      def _mk_advection(self,xdot,ydot,fdot,ddot,U,V,dUdf,dVdf,consts,has_current=False,has_df=False):
         with self.bind_consts(consts):
             return self._mk_advection_body(xdot,ydot,fdot,ddot,U,V,dUdf,dVdf,has_current,has_df)

      def _mk_advection_body(self,xdot,ydot,fdot,ddot,U,V,dUdf,dVdf,has_current,has_df):
         """
         Calculates the advection speed in four dimensions

         Deep water:  omega = 2 pi f ,  omega^2 = g k ,
         intrinsic group speed  c_g = omega/k/2 = g/(2 omega) ,
         wave direction  s = (sin(theta), cos(theta)) ,
         Doppler velocity U = (U,V) = U(x,k,t), radial correction
         R = (f/2) (sin(theta) dU/df + cos(theta) dV/df) .

         Ray equations (see the refraction section of the documentation):

         xdot = c_g sin(theta) + U + R sin(theta)
         ydot = c_g cos(theta) + V + R cos(theta)

         with kdot_x/k = -(sin(theta) dU/dx + cos(theta) dV/dx)   (= kxdot)
              kdot_y/k = -(sin(theta) dU/dy + cos(theta) dV/dy)   (= kydot)

         ddot = thetadot = cos(theta) kxdot - sin(theta) kydot

         fdot = omegadot/(2 pi) = (c_g/(2 pi)) s . kdot
              = (f/2) (sin(theta) kxdot + cos(theta) kydot)

         """
         c_g = self.grav/2 /self.omega_t
         s, c  = self.sinth[:,None,None,None], self.costh[:,None,None,None]
         all_ = (slice(None),)*4
         xdot = OM.modify_array(xdot, all_, c_g[None,:,None,None]*s)   # in place on numpy
         ydot = OM.modify_array(ydot, all_, c_g[None,:,None,None]*c)

         if has_current: 
             xdot += U
             ydot += V
             half_f = 0.5 * self.ft[None,:,None,None]
             R = half_f * (s*dUdf + c*dVdf)      # radial group-velocity factor
             xdot += R * s
             ydot += R * c
         
             dUdx, dVdx = self._ddxb(U), self._avyb( self._ddxc(V))
             dUdy, dVdy = self._avxb( self._ddyc(U)), self._ddyb(V)   

             kxdot, kydot  = -(s*dUdx + c*dVdx)   , -(s*dUdy + c*dVdy)  
             # thetadot (Eq. 8) AT THE THETA FACES. The theta flux fd_k lives
             # on the face theta_{k+1/2}; sampling the rate at the bin centre
             # displaces it half a bin toward smaller theta everywhere on the
             # ring, a fixed handedness that breaks the mirror symmetry
             # (y,theta) -> (-y,pi-theta) and drove the chiral jet response
             # (jet_test: A(24h)=0.036). theta_dot depends on theta
             # analytically, so it is rebuilt exactly at the face angles
             # rather than interpolated: with s_f=sin, c_f=cos(theta+dd/2),
             #   thetadot(theta_f) = c_f kxdot(theta_f) - s_f kydot(theta_f)
             # which also zeroes the fluxes through the faces where the exact
             # rate vanishes (e.g. theta = 0, pi for a zonal shear flow).
             # kxdot/kydot above stay at bin centres for fdot, whose flux
             # lives at frequency faces and theta centres.
             su = self.sinthu[:,None,None,None]
             cu = self.costhu[:,None,None,None]
             ddot  = self.apply_bc( -su*cu*dUdx - cu*cu*dVdx
                                    +su*su*dUdy + su*cu*dVdy )
             
             # frequency refraction:  omegadot = (c/2) s . kdot
             # better self.fu??
             fdot = self.apply_bc( 0.5 * self.ft[None,:,None,None] * (s*kxdot + c*kydot) )
             
             if fdot.shape[-3] > 2*self.halo:
              # top closed face: upper face of the last real bin
              fdot = OM.modify_array(fdot, (slice(None), fdot.shape[-3]-self.halo-1,
                                            slice(None), slice(None)), 0.0)
              # bottom closed face: lower face of the first real bin (= halo-1,
              # already zeroed by apply_bc; set explicitly for symmetry/clarity)
              fdot = OM.modify_array(fdot, (slice(None), self.halo-1,
                                            slice(None), slice(None)), 0.0)
             
         else:
             fdot, ddot = OM.np.zeros_like(xdot), OM.np.zeros_like(xdot)

         
         if getattr(self, 'limit_fdot', False):    
           cfl_r = getattr(self, 'refraction_cfl', 0.5)
           fbnd  = cfl_r * self.dft[None,:,None,None] / self.dt
           dbnd  = cfl_r * self.dd / self.dt
           fdot  = OM.np.clip(fdot, -fbnd, fbnd)
           ddot  = OM.np.clip(ddot, -dbnd, dbnd)
          
         # no spectral tendency in land cells (consistent with maskTp use in
         # add_propagation / tracer_flux_superbee).
         if hasattr(self,'maskTp'):
                 ddot = ddot * self.maskTp[None,None,:,:]
                 fdot = fdot * self.maskTp[None,None,:,:]

         return (xdot,ydot,fdot,ddot)


      def _ddxc(self,a):
         """Centered d/dx"""
         return (self.rollx(a,-1) - self.rollx(a,1)) / (2*self.dx) 

      def _ddxb(self,a):
         """Backward d/dx """
         return  (a - self.rollx(a,1)) / self.dx 

      def _avxb(self,a):
         """backward average in x """
         return  (a + self.rollx(a,1)) *0.5 
        
      def _ddyc(self,a):
         """Centered d/dy"""
         return  (self.rolly(a,-1) - self.rolly(a,1)) / (2*self.dy) 

      def _ddyb(self,a):
         """backward d/dy"""
         return  (a - self.rolly(a,1)) / self.dy 

      def _avyb(self,a):
         """backward average in y"""
         return  (a + self.rolly(a,1))*0.5
   

        
      def energy_fix_flux(self,A,fx,fy,ff,fd,U=None,V=None):
          if getattr(self,'propagation_backend','numpy') == 'fortran':
              return self.energy_fix_flux_fortran(A,fx,fy,ff,fd)
          if U is None:                        # called directly: the stored current
              if not (hasattr(self,'ocean_U') and hasattr(self,'ocean_V')):
                  return ff
              U, V = self.ocean_U, self.ocean_V
          """
          Conservative energy fixer: enforce discrete conservation of the
          absolute wave energy  int Omega A dk ,  Omega = omega + U.k , by the
          propagation operator (exact receipt of the ocean work, see the
          coupling section of the documentation).

          By summation by parts the energy defect of the flux-form advection is
          a sum over faces of  F * deltaOmega. Per (y,x) water column (with the
          shared x/y face terms split half-half between neighbouring columns)
          this defect is cancelled exactly by a small corrective f-flux
              c_m = lambda(y,x) * Abar_m * deltaOmega_m ,
          with lambda chosen per column. Properties:
            - action conservation untouched (the correction is a flux)
            - solvable: the denominator sum(Abar*deltaOmega^2) is bounded below
              by the intrinsic frequency spacing wherever A > 0
            - dormant without currents (no ocean_U -> no fix; and fdot = 0
              implies ff = 0 and uniform Omega implies zero column defect)
            - local: no global reduction, only the existing halo exchange
            - magnitude ~1% of the f-fluxes in a jet-refraction test.
          Columns with (numerically) vanishing action cannot cancel the small
          defect share assigned to them by their x/y faces; the guard leaves
          them uncorrected, which is energetically negligible by construction.
          Enabled with the attribute ``energy_consistent``.
          """
          sl = slice(self.halo,-self.halo)
          s, c = self.sinth[:,None,None,None], self.costh[:,None,None,None]
          kf = self.omega_t[None,:,None,None]**2/self.grav
          Om = self.omega_t[None,:,None,None] \
               + kf*(U[None,:,:,:]*s + V[None,:,:,:]*c)

          # per-face energy-defect terms (common factor dx*dy dropped; it
          # cancels between column defect and correction)
          tx = fx*(self.rollx(Om,-1)-Om)*self.dft[None,:,None,None]*self.dd/self.dx
          ty = fy*(self.rolly(Om,-1)-Om)*self.dft[None,:,None,None]*self.dd/self.dy
          if getattr(self, 'open_coasts', False):
              # AN OPEN COASTAL FACE CARRIES AN EXPORT, NOT A DEFECT. The
              # summation by parts that defines this correction reads
              #   sum_i Om_i (F_{i-1}-F_i) = Om_1 F_0 - Om_N F_N
              #                              + sum_interior F_j (Om_{j+1}-Om_j)
              # and only the last sum is the defect. The boundary terms are
              # the absolute energy Om_wet F that genuinely leaves, booked by
              # open_coast_loss. Leaving them in would also make the
              # correction depend on Om OUTSIDE the domain, where the current
              # is whatever the land cell happens to hold (zero in practice).
              # Measured on the acc1-like configuration: they would otherwise
              # supply 58 to 91 per cent of |col| in the coastal columns, and
              # the inflated col makes the conditioning guard below reject the
              # correction in exactly those columns.
              tx = tx*(1.0 - self.maskOx[None,None,:,:])
              ty = ty*(1.0 - self.maskOy[None,None,:,:])
          dOf = self.rollf(Om,-1)-Om
          tf = ff*dOf*self.dd
          td = fd*(self.rolld(Om,-1)-Om)*self.dft[None,:,None,None]

          # column defect: sum over real (d,f); x/y faces shared half-half.
          # sum of col over all real columns equals  sum Omega dA dV / (dx dy).
          col = -( OM.np.sum((tf+td)[sl,sl,:,:],axis=(0,1))
                 + 0.5*OM.np.sum((tx+self.rollx(tx,1)
                                 +ty+self.rolly(ty,1))[sl,sl,:,:],axis=(0,1)) )

          # correction shape: Abar * deltaOmega on interior f-faces
          wgt = 0.5*(A + self.rollf(A,-1))*dOf
          wgt = OM.modify_array(wgt,(slice(None),wgt.shape[-3]-self.halo-1,
                                     slice(None),slice(None)),0.0)  # closed top
          wgt = OM.modify_array(wgt,(slice(None),self.halo-1,
                                     slice(None),slice(None)),0.0)  # closed bottom


          den = OM.np.sum((wgt*dOf*self.dd)[sl,sl,:,:],axis=(0,1))
          # apply the correction only where the denominator is well-conditioned
          # relative to the defect; elsewhere leave the (negligible) defect
          # uncorrected rather than amplifying it through a near-zero division.
          ok  = OM.np.abs(den) > 1e-3 * OM.np.abs(col) * self.dt   # dimensional guard
          lam = OM.np.where(ok, col / OM.np.where(den == 0, 1.0, den), 0.0)

          cfl = lam[None,None,:,:]*wgt
          if hasattr(self,'maskTp'): cfl = cfl*self.maskTp[None,None,:,:]
          return ff + cfl


      def open_coast_flux(self, var, fx, fy, xdot, ydot):
          if getattr(self,'propagation_backend','numpy') == 'fortran':
              # in place on coastal faces only; same values as the where()
              T = self._prop_tables()
              self._prop_f.coastflux(fx.T, fy.T, var.T, xdot.T, ydot.T,
                                     T['mOx'].T, T['mOy'].T, T['prefy'])
              return fx, fy
          """
          RADIATING (open) boundary at every coastal face, the alternative to
          a closed wall and to specular reflection. Option ``open_coasts``.

          On a face that opens onto land the limited reconstruction must not be
          used: its slope is built from both sides, so it would take information
          from outside the domain. The face is given the FIRST-ORDER UPWIND
          flux instead, which uses the wet cell alone and is the only defensible
          outflow value. In the limiter form of tracer_flux_superbee that is
          exactly cr = 0 with the UNMASKED gradient and no face mask, i.e.

              F = -( vel*(var_next + var) - |vel|*(var_next - var) )/2 .

          Waves do not exist on land (time_step masks E by maskTp, apply_bc
          zeroes the halo of a closed direction), so on an INCOMING face the
          upwind cell carries A = 0 and nothing enters -- the radiating
          condition falls out, in both orientations, with no branch on which
          side the land is.

          The masked gradient rj of the caller is deliberately left alone: it
          feeds the limiter of the NEIGHBOURING faces through rjm/rjp, and
          unmasking it there would let the coast corrupt the reconstruction of
          the adjacent interior face.

          CALL THIS AFTER apply_bc, NOT BEFORE. apply_bc writes bc_value into
          the flux at indices 0..halo-1 of a closed direction, and face halo-1
          is the west/south domain-edge face, which the divergence of the first
          real cell consumes. Overriding before apply_bc deletes the western
          and southern outflow and keeps the eastern and northern one.
          """
          vx, vy = self.rollx(var, -1), self.rolly(var, -1)
          gx = -(xdot*(vx + var) - OM.np.abs(xdot)*(vx - var))*0.5
          gy = -(ydot*(vy + var) - OM.np.abs(ydot)*(vy - var))*0.5
          return (OM.np.where(self.maskOx[None,None,:,:] > 0, gx, fx),
                  OM.np.where(self.maskOy[None,None,:,:] > 0, gy, fy))


      def open_coast_loss(self, fx, fy, tau, U=None, V=None):
          """
          Outflow rate through the open coasts, in action and in energy,
          accumulated over the faces this PE owns (main.open_tables).
          Returns the triple of rates (rA, rE, rO); add_propagation stores
          them at time level tau in _lossA_lv/_lossE_lv/_lossO_lv.

          Only the REAL direction and frequency bins are summed: the direction
          halo is a wrapped copy of a real bin and would be counted twice.
          The physical flux is -fx, and the phase-space weight of an x face is
          dy*dft*dd (dx*dft*dd in y), already folded into the table except for
          the frequency part, which varies with the bin.

          JAX: the faces are not gathered by index tables (they differ per
          shard) but weighted with the dense signed table _loss_wgt (Ny,Nx)
          of open_tables; the sum over the sharded array is global under jit.
          numpy: the sparse tables, as before (bit-identical to the goldens).
          U, V: the Doppler velocity (the stored ocean_U/V when not given).
          """
          if U is None and hasattr(self, 'ocean_U') and hasattr(self, 'ocean_V'):
              U, V = self.ocean_U, self.ocean_V
          if getattr(self,'propagation_backend','numpy') == 'fortran':
              return self.open_coast_loss_fortran(fx, fy, tau)
          h  = self.halo
          sl = slice(h, -h)                 # real bins, direction and frequency
          wA = self.dft[h:-h]
          wE = wA*self.omega_t[h:-h]
          # absolute frequency Omega = omega + k.U, the quantity energy_fix_flux
          # conserves. The export across an open face carries the UPWIND cell's
          # Omega, and the upwind cell is always the WET one: the face is pure
          # upwind and the dry side holds A = 0.
          Om = None
          if U is not None:
              sn = self.sinth[:,None,None,None]
              cs = self.costh[:,None,None,None]
              kf = self.omega_t[None,:,None,None]**2/self.grav
              Om = self.omega_t[None,:,None,None] \
                   + kf*(U[None,:,:,:]*sn + V[None,:,:,:]*cs)
          if OM.jax:
              rA = rE = rO = 0.0
              for ax, F in ((-2, fy), (-1, fx)):
                  wg = self._loss_wgt[ax][None,None,:,:]
                  Fp = -F[sl, sl]*wg                                # physical flux x weight
                  rA = rA + OM.np.sum(Fp*wA[None,:,None,None])
                  rE = rE + OM.np.sum(Fp*wE[None,:,None,None])
                  if Om is None:
                      rO = rO + OM.np.sum(Fp*wE[None,:,None,None])
                  else:
                      o2 = self.rolly(Om,-1) if ax == -2 else self.rollx(Om,-1)
                      Of = OM.np.where(wg > 0, Om, o2)[sl, sl]
                      rO = rO + OM.np.sum(Fp*Of*wA[None,:,None,None])
              return rA, rE, rO
          rA = rE = rO = 0.0
          for ax, F in ((-2, fy), (-1, fx)):
              jj, ii, wg = self._loss_idx[ax]
              if jj.size == 0:  continue
              Fp = -F[sl, sl][:, :, jj, ii]                  # physical flux
              rA += float((Fp*wA[None,:,None]*wg[None,None,:]).sum())
              rE += float((Fp*wE[None,:,None]*wg[None,None,:]).sum())
              if Om is None:
                  rO += float((Fp*wE[None,:,None]*wg[None,None,:]).sum())
              else:
                  o1 = Om[sl, sl][:, :, jj, ii]
                  o2 = OM.np.roll(Om, -1, axis=ax)[sl, sl][:, :, jj, ii]
                  Of = OM.np.where(wg[None,None,:] > 0, o1, o2)
                  rO += float((Fp*Of*wA[None,:,None]*wg[None,None,:]).sum())
          return rA, rE, rO


      def add_propagation(self,A,dA,xdot,ydot,fdot,ddot,tau):  
          """
           Calculates tendencies due to advection for wave action.
           Stores tendency in dA and overwrites at time level tau.
           With open_coasts also the outflow rates of this level.
          """
          U, V, dU, dV, has_cur, has_df = self._current_args()
          dA, rA, rE, rO = self._add_propagation(A,dA,xdot,ydot,fdot,ddot,tau,U,V,
                                                 self.shard_consts(), has_current=has_cur)
          if getattr(self, 'open_coasts', False):
              self._lossA_lv = OM.modify_array(self._lossA_lv, (tau,), rA)
              self._lossE_lv = OM.modify_array(self._lossE_lv, (tau,), rE)
              self._lossO_lv = OM.modify_array(self._lossO_lv, (tau,), rO)
          return dA


      @partial(OM.jaxjit, static_argnums=(0,), static_argnames=('has_current',))
      def _add_propagation(self,A,dA,xdot,ydot,fdot,ddot,tau,U,V,consts,has_current=False):
          with self.bind_consts(consts):
              return self._add_propagation_body(A,dA,xdot,ydot,fdot,ddot,tau,U,V,has_current)

      def _barrier(self, *a):
          """an optimisation barrier for XLA (identity on numpy): the values
          before it are computed before anything after it, no fusion across"""
          return OM.jax.lax.optimization_barrier(a) if OM.jax else a

      def _add_propagation_body(self,A,dA,xdot,ydot,fdot,ddot,tau,U,V,has_current):
          """body of add_propagation (jitted under JAX through _add_propagation)"""
          if not has_current: U = V = None
          rA = rE = rO = 0.0
          # advection flux scheme: 'superbee' (2nd-order, low diffusion, default)
          # or 'upwind' (1st-order, diffusive, robust to grid-scale noise, allows
          # a larger CFL and is ~4x cheaper per direction). Both are finite-volume
          # (face-flux) forms, so conservation and energy_fix_flux are preserved.
          if getattr(self, 'f_limiter', 'superbee') == 'upwind':
              fx,fy,ff,fd = self.tracer_flux_upwind(A,xdot,ydot,fdot,ddot,tau)
          elif getattr(self, 'propagation_backend', 'numpy') == 'fortran':
              fx,fy,ff,fd = self.tracer_flux_superbee_fortran(A,xdot,ydot,fdot,ddot,tau)
          else:
              fx,fy,ff,fd = self.tracer_flux_superbee(A,xdot,ydot,fdot,ddot,tau)
          # JAX: keep XLA from fusing the flux scheme, the energy fixer, the
          # loss sums and the divergence into one program. Each piece on its
          # own runs in milliseconds at fac=16 on an A100, the fusion of all of
          # them grinds for minutes (Oct 2026, jax 0.11); the barriers cost nothing.
          fx,fy,ff,fd = self._barrier(fx,fy,ff,fd)
          if getattr(self,'energy_consistent',False) and has_current:
              ff = self.energy_fix_flux(A,fx,fy,ff,fd,U,V)
              fx,fy,ff,fd = self._barrier(fx,fy,ff,fd)
          if getattr(self, 'open_coasts',      False):
              rA, rE, rO = self.open_coast_loss(fx,fy,tau,U,V)
              fx,fy,ff,fd,rA,rE,rO = self._barrier(fx,fy,ff,fd,rA,rE,rO)
          if getattr(self, 'propagation_backend', 'numpy') == 'fortran':
              # one pass, writes straight into dA[tau] (a Fortran-contiguous
              # view through .T, no copy); halos wrap like np.roll
              T = self._prop_tables()
              self._prop_f.divergence(fx.T, fy.T, ff.T, fd.T, T['denx'].T,
                                      T['deny'].T, self.dft, self.dd, T['mT'].T,
                                      dA[tau].T)
              return dA, rA, rE, rO
          _dA = (  (fx-self.rollx(fx,1))/self.dx 
                 + (fy-self.rolly(fy,1))/self.dy 
                 + (ff-self.rollf(ff,1))/self.dft[None,:,None,None]
                 + (fd-self.rolld(fd,1))/self.dd )
           
          # zero the tendency in land cells 
          if hasattr(self, 'maskTp'): _dA = _dA * self.maskTp[None, None, :, :]
        
          dA = OM.modify_array(dA, (tau,slice(None),slice(None),
                                        slice(None),slice(None)), _dA)
          return dA, rA, rE, rO
          
                            
      
      def _prop_metric(self):
          """
          Grid tables for the Fortran propagation kernels, in the exact form
          the numpy path divides/multiplies by (so the result is bit-identical):
          denx, deny (Ny,Nx): CFL and divergence denominators (dx, dy here;
          cost*dxt, cost*dyt in wim_realistic); prefy (Ny,): y-flux prefactor
          (1 here; cosu in wim_realistic).
          """
          Ny, Nx = self.Ny, self.Nx
          return (OM.np.full((Ny, Nx), self.dx), OM.np.full((Ny, Nx), self.dy),
                  OM.np.ones(Ny))


      def _prop_tables(self):
          """
          Static tables and workspace for the Fortran kernels, built once
          (grid and land mask do not change after setup). The four flux arrays
          and rj are persistent workspace: fresh 40 MB allocations every step
          would cost a memory pass each in page faults.
          """
          T = getattr(self, '_prop_T', None)
          if T is not None: return T
          Ny, Nx = self.Ny, self.Nx
          one = OM.np.ones((Ny, Nx))
          denx, deny, prefy = self._prop_metric()
          T = dict(denx=OM.np.ascontiguousarray(denx),
                   deny=OM.np.ascontiguousarray(deny),
                   prefy=OM.np.ascontiguousarray(prefy),
                   mU=OM.np.ascontiguousarray(self.maskUp) if hasattr(self,'maskUp') else one,
                   mV=OM.np.ascontiguousarray(self.maskVp) if hasattr(self,'maskVp') else one,
                   mT=OM.np.ascontiguousarray(self.maskTp) if hasattr(self,'maskTp') else one,
                   rj=OM.np.zeros_like(self.A), fx=OM.np.zeros_like(self.A),
                   fy=OM.np.zeros_like(self.A), ff=OM.np.zeros_like(self.A),
                   fd=OM.np.zeros_like(self.A))
          # tables and workspace of the mk_advection / energy_fix_flux /
          # open_coast kernels. The 1-D tables repeat the numpy expressions
          # exactly (same floats); efx holds the per-package metric of the
          # energy fixer (see _efix_metric), sph/tantr the spherical turning.
          T['om']  = OM.np.ascontiguousarray(self.omega_t)
          T['kf']  = OM.np.ascontiguousarray(self.omega_t**2/self.grav)
          T['cg']  = OM.np.ascontiguousarray(self.grav/2/self.omega_t)
          T['ft_'] = OM.np.ascontiguousarray(self.ft)
          T['dft_'] = OM.np.ascontiguousarray(self.dft)
          T['sth'] = OM.np.ascontiguousarray(self.sinth)
          T['cth'] = OM.np.ascontiguousarray(self.costh)
          T['sthu'] = OM.np.ascontiguousarray(self.sinthu)
          T['cthu'] = OM.np.ascontiguousarray(self.costhu)
          T['sph'], T['tantr'] = self._mk_sph()
          T['efx'] = self._efix_metric()
          T['mOx'] = OM.np.ascontiguousarray(self.maskOx) if hasattr(self,'maskOx') \
                     else OM.np.zeros((Ny, Nx))
          T['mOy'] = OM.np.ascontiguousarray(self.maskOy) if hasattr(self,'maskOy') \
                     else OM.np.zeros((Ny, Nx))
          T['zero3'] = OM.np.zeros((self.Nf, Ny, Nx))
          for k in ('efP','efQ','efD','efW','fdw','ddw'):
              T[k] = OM.np.zeros_like(self.A)
          self._prop_T = T
          return T


      def _mk_sph(self):
          """spherical-turning switch and tan(phi)/R table for the
          mk_advection kernel: off on the Cartesian grid (wim_realistic
          overrides with coord_degree and tantr_y)."""
          return 0, OM.np.zeros(self.Ny)


      def _efix_metric(self):
          """
          Metric of the energy fixer in the exact form the numpy lines use.
          wim_simple DIVIDES the x/y defect terms by dx, dy and has no area
          weight; wim_realistic overrides with multiplication by dyt, dxt and
          the cell area (divide and multiply-by-reciprocal round differently,
          hence the div flags instead of one reciprocal table).
          """
          Ny, Nx = self.Ny, self.Nx
          one = OM.np.ones((Ny, Nx))
          return dict(wx=OM.np.full((Ny, Nx), self.dx), divx=1,
                      wy=OM.np.full((Ny, Nx), self.dy), divy=1,
                      area=one, den_mult=None)


      def mk_advection_fortran(self,xdot,ydot,fdot,ddot):
          """
          mk_advection with the 4-D arithmetic in the Propagation kernels:
          one pass for (xdot,ydot), one for the raw (ddot,fdot) rates, one for
          clip+mask. The 3-D current gradients (_ddx*/_ddy*, package metric),
          apply_bc and the closed-frequency faces stay in python, unchanged.
          Value-identical to the numpy path.
          """
          P, T = self._prop_f, self._prop_tables()
          hascur = hasattr(self,'ocean_U') and hasattr(self,'ocean_V')
          z3 = T['zero3']
          if hascur:
              U, V = self.ocean_U, self.ocean_V
              du = getattr(self, 'ocean_Udf', None)
              dv = getattr(self, 'ocean_Vdf', None)
              if du is None or OM.np.ndim(du) == 0: du = z3
              if dv is None or OM.np.ndim(dv) == 0: dv = z3
              P.mkadv_vel(xdot.T, ydot.T, U.T, V.T, du.T, dv.T,
                          T['cg'], T['ft_'], T['sth'], T['cth'], 1, 1)
              dUdx, dVdx = self._ddxb(U), self._avyb(self._ddxc(V))
              dUdy, dVdy = self._avxb(self._ddyc(U)), self._ddyb(V)
              dd_, fd_ = T['ddw'], T['fdw']
              P.mkadv_rates(dd_.T, fd_.T,
                            OM.np.ascontiguousarray(dUdx).T,
                            OM.np.ascontiguousarray(dVdx).T,
                            OM.np.ascontiguousarray(dUdy).T,
                            OM.np.ascontiguousarray(dVdy).T,
                            U.T, V.T, T['cg'], T['ft_'], T['sth'], T['cth'],
                            T['sthu'], T['cthu'], T['tantr'], T['sph'])
              ddot = self.apply_bc(dd_)
              fdot = self.apply_bc(fd_)
              if fdot.shape[-3] > 2*self.halo:
                  fdot = OM.modify_array(fdot, (slice(None), fdot.shape[-3]-self.halo-1,
                                                slice(None), slice(None)), 0.0)
                  fdot = OM.modify_array(fdot, (slice(None), self.halo-1,
                                                slice(None), slice(None)), 0.0)
          else:
              P.mkadv_vel(xdot.T, ydot.T, z3.T, z3.T, z3.T, z3.T,
                          T['cg'], T['ft_'], T['sth'], T['cth'], 0, 0)
              fdot, ddot = OM.np.zeros_like(xdot), OM.np.zeros_like(xdot)
              if T['sph']:
                  # great-circle turning without currents (wim_realistic)
                  su  = self.sinthu[:,None,None,None]
                  c_g = self.grav/2 /self.omega_t
                  ddot = self.apply_bc( c_g[None,:,None,None]*su
                                        * self.tantr_y
                                        + OM.np.zeros_like(xdot) )

          do_clip = int(getattr(self, 'limit_fdot', False))
          do_mask = int(hasattr(self, 'maskTp'))
          if do_clip or do_mask:
              cfl_r = getattr(self, 'refraction_cfl', 0.5)
              fbnd  = OM.np.ascontiguousarray(cfl_r * self.dft / self.dt)
              P.mkadv_post(fdot.T, ddot.T, fbnd, cfl_r*self.dd/self.dt,
                           T['mT'].T, do_clip, do_mask)
          return (xdot,ydot,fdot,ddot)


      def energy_fix_flux_fortran(self,A,fx,fy,ff,fd):
          """
          energy_fix_flux with the 4-D fields built in one kernel pass
          (Omega recomputed pointwise) and the correction applied in another.
          The two column reductions stay np.sum on identically shaped arrays:
          numpy's pairwise summation order is part of the bit pattern, so
          they must not move into fortran. ff is updated in place.
          """
          if not (hasattr(self,'ocean_U') and hasattr(self,'ocean_V')):
              return ff
          P, T = self._prop_f, self._prop_tables()
          E  = T['efx']
          sl = slice(self.halo,-self.halo)
          P.efix_terms(A.T, fx.T, fy.T, ff.T, fd.T,
                       self.ocean_U.T, self.ocean_V.T,
                       T['om'], T['kf'], T['sth'], T['cth'], T['dft_'], self.dd,
                       E['wx'].T, E['wy'].T, E['divx'], E['divy'], E['area'].T,
                       T['mOx'].T, T['mOy'].T,
                       int(getattr(self, 'open_coasts', False)), self.halo,
                       T['efP'].T, T['efQ'].T, T['efD'].T, T['efW'].T)
          col = -( OM.np.sum(T['efP'][sl,sl,:,:],axis=(0,1))
                 + 0.5*OM.np.sum(T['efQ'][sl,sl,:,:],axis=(0,1)) )
          den = OM.np.sum(T['efD'][sl,sl,:,:],axis=(0,1))
          if E['den_mult'] is not None: den = den*E['den_mult']
          ok  = OM.np.abs(den) > 1e-3 * OM.np.abs(col) * self.dt
          lam = OM.np.where(ok, col / OM.np.where(den == 0, 1.0, den), 0.0)
          P.efix_apply(ff.T, T['efW'].T, OM.np.ascontiguousarray(lam).T, T['mT'].T)
          return ff


      def open_coast_loss_fortran(self, fx, fy, tau):
          """
          open_coast_loss without building the full 4-D Omega and rolling it:
          Omega is evaluated on the gathered coastal faces only (elementwise
          ops commute with the gather, and the reductions run over arrays of
          the same shape as before, so every float is identical). Pure numpy.
          """
          h  = self.halo
          sl = slice(h, -h)
          wA = self.dft[h:-h]
          wE = wA*self.omega_t[h:-h]
          cur = hasattr(self, 'ocean_U') and hasattr(self, 'ocean_V')
          if cur:
              s_r  = self.sinth[sl][:,None,None]
              c_r  = self.costh[sl][:,None,None]
              om_r = self.omega_t[sl][None,:,None]
              kf_r = (self.omega_t**2/self.grav)[sl][None,:,None]
              U_r, V_r = self.ocean_U[sl], self.ocean_V[sl]
          rA = rE = rO = 0.0
          for ax, F in ((-2, fy), (-1, fx)):
              jj, ii, wg = self._loss_idx[ax]
              if jj.size == 0:  continue
              Fp = -F[sl, sl][:, :, jj, ii]                  # physical flux
              rA += float((Fp*wA[None,:,None]*wg[None,None,:]).sum())
              rE += float((Fp*wE[None,:,None]*wg[None,None,:]).sum())
              if not cur:
                  rO += float((Fp*wE[None,:,None]*wg[None,None,:]).sum())
              else:
                  if ax == -1: j2, i2 = jj, (ii+1) % self.Nx
                  else:        j2, i2 = (jj+1) % self.Ny, ii
                  o1 = om_r + kf_r*(U_r[:, jj, ii]*s_r + V_r[:, jj, ii]*c_r)
                  o2 = om_r + kf_r*(U_r[:, j2, i2]*s_r + V_r[:, j2, i2]*c_r)
                  Of = OM.np.where(wg[None,None,:] > 0, o1, o2)
                  # np.sum's pairwise order follows memory layout, and the
                  # original Of (gathered from the 4-D Omega) is laid out
                  # faces-outermost. Recreate that layout so the reduction
                  # groups the same terms and the sum is bit-identical.
                  Of = OM.np.moveaxis(OM.np.ascontiguousarray(
                           OM.np.moveaxis(Of, -1, 0)), 0, -1)
                  rO += float((Fp*Of*wA[None,:,None]*wg[None,None,:]).sum())
          return rA, rE, rO


      def tracer_flux_superbee_fortran(self,var,xdot,ydot,fdot,ddot,tau):
          """
          tracer_flux_superbee with the per-axis arithmetic in the f2py kernels
          of wim_simple/Propagation (one memory pass per gradient and per flux
          instead of ~15 numpy passes). The kernels compute the interior box;
          apply_bc fills the halos exactly as in the numpy path, so the halo
          exchange, the closed frequency faces, the open coasts and the
          reflection are the same code as there. Results are bit-identical.
          Arrays are passed as .T views (Fortran-contiguous, no copy).
          """
          P, T, h, dt = self._prop_f, self._prop_tables(), self.halo, self.dt
          rj, fx, fy, ff, fd = T['rj'], T['fx'], T['fy'], T['ff'], T['fd']

          P.grad_x(var.T, T['mU'].T, rj.T, h);  rj = self.apply_bc(rj)
          P.flux_x(var.T, rj.T, xdot.T, T['denx'].T, T['mU'].T, dt, fx.T, h)

          P.grad_y(var.T, T['mV'].T, rj.T, h);  rj = self.apply_bc(rj)
          P.flux_y(var.T, rj.T, ydot.T, T['deny'].T, T['prefy'], T['mV'].T, dt, fy.T, h)

          P.grad_f(var.T, rj.T, h);  rj = self.apply_bc(rj)
          P.flux_f(var.T, rj.T, fdot.T, self.dfu, dt, ff.T, h)

          P.grad_d(var.T, rj.T, h);  rj = self.apply_bc(rj)
          P.flux_d(var.T, rj.T, ddot.T, self.dd, dt, fd.T, h)

          # close frequency flux (as in tracer_flux_superbee)
          ff = self.apply_bc(ff)
          if ff.shape[-3] > 2*self.halo:
              ff = OM.modify_array(ff, (slice(None), ff.shape[-3]-self.halo-1,
                                        slice(None), slice(None)), 0.0)
              ff = OM.modify_array(ff, (slice(None), self.halo-1,
                                        slice(None), slice(None)), 0.0)

          fx, fy = self.apply_bc(fx), self.apply_bc(fy)
          if getattr(self, 'open_coasts', False):
              fx, fy = self.open_coast_flux(var, fx, fy, xdot, ydot)
          elif getattr(self, 'enable_reflection', False):
              fx, fy = self.reflect_flux(fx, fy)
          return fx, fy, ff, self.apply_bc(fd)


      def tracer_flux_upwind(self,var,xdot,ydot,fdot,ddot,tau):
          """
          First-order upwind fluxes for wave action in all four directions.

          Structurally identical to tracer_flux_superbee but with the flux
          limiter set to zero (cr = 0), which reduces the second-order flux to
          first-order upwind. Every halo exchange (apply_bc on the masked
          gradient rj and on the fluxes), land masking, closed frequency face,
          and specular reflection is kept exactly as in the superbee version, so
          the only difference is the absence of the antidiffusive limiter term.
          More diffusive than superbee (smears sharp features) but robust to
          grid-scale noise and stable to a larger CFL.
          """
          mU = self.maskUp[None, None, :, :] if hasattr(self, 'maskUp') else 1.0
          mV = self.maskVp[None, None, :, :] if hasattr(self, 'maskVp') else 1.0

          # x
          rj = self.apply_bc( (self.rollx(var,-1) - var) * mU )
          fx = - (xdot * (self.rollx(var,-1) + var) * 0.5
                  - OM.np.abs(xdot) * rj * 0.5) * mU
          # y
          rj = self.apply_bc( (self.rolly(var,-1) - var) * mV )
          fy = - (ydot * (self.rolly(var,-1) + var) * 0.5
                  - OM.np.abs(ydot) * rj * 0.5) * mV
          # frequency
          rj = self.apply_bc( (self.rollf(var,-1) - var) )
          ff = - (fdot * (self.rollf(var,-1) + var) * 0.5
                  - OM.np.abs(fdot) * rj * 0.5)
          # direction
          rj = self.apply_bc( (self.rolld(var,-1) - var) )
          fd = - (ddot * (self.rolld(var,-1) + var) * 0.5
                  - OM.np.abs(ddot) * rj * 0.5)

          # close frequency flux (same as superbee)
          ff = self.apply_bc(ff)
          if ff.shape[-3] > 2*self.halo:
              ff = OM.modify_array(ff, (slice(None), ff.shape[-3]-self.halo-1,
                                        slice(None), slice(None)), 0.0)
              ff = OM.modify_array(ff, (slice(None), self.halo-1,
                                        slice(None), slice(None)), 0.0)

          fx, fy = self.apply_bc(fx), self.apply_bc(fy)
          if getattr(self, 'open_coasts', False):
              fx, fy = self.open_coast_flux(var, fx, fy, xdot, ydot)
          elif getattr(self, 'enable_reflection', False):
              fx, fy = self.reflect_flux(fx, fy)
          return fx, fy, ff, self.apply_bc(fd)


      def tracer_flux_superbee(self,var,xdot,ydot,fdot,ddot,tau):
          """
          from MITgcm: 2nd oder scheme with Superbee flux limiter.
          """
          def _limiter(cr):
              return OM.np.maximum(OM.np.clip(2 * cr, 0, 1), OM.np.clip(cr, 0, 2))
              
          eps = 1e-20  

          # land masks for the two physical (x,y) directions, broadcast over the
          # (d,f) axes. Applied exactly as in pe_simple.tracer_flux_superbee: the
          # gradient rj is masked before the limiter (so a land cell cannot create
          # a spurious gradient that corrupts the limiter in the adjacent ocean
          # cell), and the flux itself is masked (no flux through a dry face). The
          # spectral directions (freq, dir) have no land and are never masked.
          mU = self.maskUp[None, None, :, :] if hasattr(self, 'maskUp') else 1.0
          mV = self.maskVp[None, None, :, :] if hasattr(self, 'maskVp') else 1.0

          CFL = OM.np.abs(xdot * self.dt / self.dx)          
          rj  = self.apply_bc( (self.rollx(var,-1)-var) * mU )
          rjm = self.rollx(rj, 1) 
          rjp = self.rollx(rj,-1)                     
          cr = _limiter( OM.np.where(xdot > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fx = - (xdot * (self.rollx(var,-1) + var) * 0.5 
               - OM.np.abs(xdot) * ((1.0 - cr) + CFL * cr) * rj * 0.5) * mU

          CFL = OM.np.abs(ydot * self.dt / self.dy)          
          rj  = self.apply_bc( (self.rolly(var,-1)-var) * mV )
          rjm = self.rolly(rj, 1) 
          rjp = self.rolly(rj,-1)                     
          cr = _limiter( OM.np.where(ydot > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fy = - (ydot * (self.rolly(var,-1) + var) * 0.5 
               - OM.np.abs(ydot) * ((1.0 - cr) + CFL * cr) * rj * 0.5) * mV

          CFL = OM.np.abs(fdot * self.dt / self.dfu[None,:,None,None])          
          rj  = self.apply_bc( (self.rollf(var,-1) - var) )
          rjm = self.rollf(rj, 1) 
          rjp = self.rollf(rj,-1)                     
          cr = _limiter( OM.np.where(fdot > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          ff = - (fdot * (self.rollf(var,-1) + var) * 0.5 
               - OM.np.abs(fdot) * ((1.0 - cr) + CFL * cr) * rj * 0.5)
              
          CFL = OM.np.abs(ddot * self.dt / self.dd)          
          rj  = self.apply_bc( (self.rolld(var,-1)-var) )
          rjm = self.rolld(rj, 1) 
          rjp = self.rolld(rj,-1)                     
          cr = _limiter( OM.np.where(ddot > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fd = - (ddot * (self.rolld(var,-1) + var) * 0.5 
               - OM.np.abs(ddot) * ((1.0 - cr) + CFL * cr) * rj * 0.5)

          # close frequency flux
          ff = self.apply_bc(ff)
          if ff.shape[-3] > 2*self.halo:
              # top closed face: upper face of the last real bin
              ff = OM.modify_array(ff, (slice(None), ff.shape[-3]-self.halo-1,
                                        slice(None), slice(None)), 0.0)
              # bottom closed face: lower face of the first real bin (= halo-1,
              # already zeroed by apply_bc; set explicitly for symmetry/clarity)
              ff = OM.modify_array(ff, (slice(None), self.halo-1,
                                        slice(None), slice(None)), 0.0)

          fx, fy = self.apply_bc(fx), self.apply_bc(fy)
          if getattr(self, 'open_coasts', False):
              fx, fy = self.open_coast_flux(var, fx, fy, xdot, ydot)
          elif getattr(self, 'enable_reflection', False):
              fx, fy = self.reflect_flux(fx, fy)
          return fx, fy, ff, self.apply_bc(fd)
         
          


      def reflect_ini(self):
          """
          Precompute the directional-reflection index tables for wave reflection
          at coastlines, ported from pyOM2 idemix_spectral.f90 (reflect_ini).

          For every direction bin k, store the mirror bin kk it reflects into at
          each of the four wall types, but only at cells that actually form that
          wall (detected from maskTp land/ocean edges). A value of -1 means "no
          wall / no reflection" for that (bin, cell, wall-type).

          Mirror angles in WIM's direction convention (theta=0 -> +y north,
          theta=pi/2 -> +x east; xdot=c_g sin(theta), ydot=c_g cos(theta)):
            x-wall (west/east): reflect the x-component, sin->-sin => theta->2pi-theta
            y-wall (north/south): reflect the y-component, cos->-cos => theta->pi-theta
          Each bin is assigned to exactly one of {south,north} and one of
          {west,east} via the half-plane it travels into (the if/else in pyOM2's
          reflect_ini), translated to WIM's theta.

          Builds 3D integer arrays bc_south/north/west/east of shape (Nd,Ny,Nx).
          """
          th = np__.asarray(self.theta[self.halo:-self.halo])   # real direction bins (host)
          nd = th.shape[0]
          Ny, Nx = self.Ny, self.Nx
          two_pi = 2.0 * OM.np.pi

          def nearest(angle):
              a = angle % two_pi
              return int(np__.argmin((th - a) ** 2))

          # mirror-bin lookup (index into the real, unpadded direction axis)
          mirror_x = np__.array([nearest(two_pi - t) for t in th])   # x-walls
          mirror_y = np__.array([nearest(np__.pi - t) for t in th])  # y-walls

          bc_s = -OM.np.ones((nd, Ny, Nx), dtype='int64')
          bc_n = -OM.np.ones((nd, Ny, Nx), dtype='int64')
          bc_w = -OM.np.ones((nd, Ny, Nx), dtype='int64')
          bc_e = -OM.np.ones((nd, Ny, Nx), dtype='int64')

          m   = self.maskTp
          # wall edges (land/ocean neighbour pairs). rollx/rolly(-1) brings the
          # +1 neighbour to the current cell.
          south_edge = (m == 0.0) & (self.rolly(m, -1) == 1.0)   # land j, ocean j+1
          north_edge = (m == 1.0) & (self.rolly(m, -1) == 0.0)   # ocean j, land j+1
          west_edge  = (m == 0.0) & (self.rollx(m, -1) == 1.0)   # land i, ocean i+1
          east_edge  = (m == 1.0) & (self.rollx(m, -1) == 0.0)   # ocean i, land i+1

          for k in range(nd):
              t = th[k] % two_pi
              # north/south: cos(theta)<0 (theta in (pi/2,3pi/2]) heads south
              if (t > OM.np.pi / 2) and (t <= 3 * OM.np.pi / 2):
                  bc_s = OM.modify_array(bc_s, (k,), OM.np.where(south_edge, mirror_y[k], bc_s[k]))
              else:
                  bc_n = OM.modify_array(bc_n, (k,), OM.np.where(north_edge, mirror_y[k], bc_n[k]))
              # west/east: sin(theta)<0 (theta in (pi,2pi]) heads west
              if (t > OM.np.pi) and (t <= two_pi):
                  bc_w = OM.modify_array(bc_w, (k,), OM.np.where(west_edge, mirror_x[k], bc_w[k]))
              else:
                  bc_e = OM.modify_array(bc_e, (k,), OM.np.where(east_edge, mirror_x[k], bc_e[k]))

          self.bc_south, self.bc_north = OM.np.asarray(bc_s), OM.np.asarray(bc_n)
          self.bc_west,  self.bc_east  = OM.np.asarray(bc_w), OM.np.asarray(bc_e)
          self._mirror_x, self._mirror_y = mirror_x, mirror_y      # for _reflect_flux_vec
          return


      def reflect_flux(self, fx, fy):
          """
          Apply the wave-reflection boundary condition to the advective fluxes,
          ported verbatim from pyOM2 idemix_spectral.f90 (reflect_flux). At each
          wall the flux that would cross is removed from direction bin k and
          re-injected into the mirror bin kk, so total wave energy is conserved
          and the wave reflects specularly instead of piling up.

          fx is the x-face (east) flux, fy the y-face (north) flux, both of shape
          (Nd,Nf,Ny,Nx). The reflection index tables are per (Nd,Ny,Nx) and act
          on all frequencies. The redistribution is invariant to the overall sign
          of the flux arrays, so it applies directly to WIM's fx/fy.

          STABILITY: with reflection enabled, the superbee + AB3 advection of the
          sharp reflected feature at the wall is a CFL-limited grid instability.
          It is stable for CFL = c_g*dt/dx <~ 0.13 (the real-model threshold is
          ~0.15; use <= 0.10 for margin). This is lower than the plain-advection
          limit (~0.5). The instability is interior (the reflected packet, not the
          wall cell), so a boundary-local fix (upwind or a tapered limiter near
          the wall) does NOT raise the threshold -- only a smaller dt does. The
          storm config (dx=5km, c_g~22m/s) is at CFL=0.13 with dt=30s; dt=22s
          gives CFL=0.10.
          """
          if not hasattr(self, 'bc_east'):
              return fx, fy
          if OM.jax:
              return self._reflect_flux_vec(fx, fy)
          nd = self.theta.shape[0] - 2 * self.halo
          for k in range(nd):
              kd = self.halo + k
              # south wall: flux from the cell to the north (j+1)
              kk = self.bc_south[k]
              jj, ii = OM.np.where(kk >= 0)
              if jj.size:
                  for j, i in zip(jj, ii):
                      m = self.halo + int(kk[j, i])
                      flux = fy[kd, :, (j + 1) % self.Ny, i].copy()
                      fy[kd, :, j, i] += flux
                      fy[m,  :, j, i] -= flux
              # north wall: flux from the cell to the south (j-1)
              kk = self.bc_north[k]
              jj, ii = OM.np.where(kk >= 0)
              if jj.size:
                  for j, i in zip(jj, ii):
                      m = self.halo + int(kk[j, i])
                      flux = fy[kd, :, j - 1, i].copy()
                      fy[kd, :, j, i] += flux
                      fy[m,  :, j, i] -= flux
              # west wall: flux from the cell to the east (i+1)
              kk = self.bc_west[k]
              jj, ii = OM.np.where(kk >= 0)
              if jj.size:
                  for j, i in zip(jj, ii):
                      m = self.halo + int(kk[j, i])
                      flux = fx[kd, :, j, (i + 1) % self.Nx].copy()
                      fx[kd, :, j, i] += flux
                      fx[m,  :, j, i] -= flux
              # east wall: flux from the cell to the west (i-1)
              kk = self.bc_east[k]
              jj, ii = OM.np.where(kk >= 0)
              if jj.size:
                  for j, i in zip(jj, ii):
                      m = self.halo + int(kk[j, i])
                      flux = fx[kd, :, j, i - 1].copy()
                      fx[kd, :, j, i] += flux
                      fx[m,  :, j, i] -= flux
          return fx, fy


      def _reflect_flux_vec(self, fx, fy):
          """
          reflect_flux without the loop over the coastal cells (the JAX path):
          for every wall the flux of ALL direction bins is gathered from the
          neighbouring cell at once (the edge masks of reflect_ini select the
          cells), added to the wall cell and subtracted in the mirror bins by
          one scatter over the direction axis with the fixed mirror table
          (the mirror bin of k is the same at every cell). Same arithmetic
          as the loop; the only difference can occur in a one-cell-wide
          channel, where the loop reads a flux the opposite wall of the same
          k has already changed.
          """
          h, nd = self.halo, self.theta.shape[0] - 2*self.halo
          sd = slice(h, h+nd)
          mir_x = OM.np.asarray(self._mirror_x) + h             # real bins -> padded index
          mir_y = OM.np.asarray(self._mirror_y) + h
          def wall(F, table, src, mir):
              m = (table >= 0)[:, None, :, :]                   # (nd,1,Ny,Nx) wall cells of bin k
              flux = OM.np.where(m, src[sd], 0.0)               # (nd,Nf,Ny,Nx)
              F = F.at[sd].add(flux)
              F = F.at[mir].add(-flux)                          # scatter into the mirror bins
              return F
          fy = wall(fy, self.bc_south, self.rolly(fy, -1), mir_y)    # south wall: from j+1
          fy = wall(fy, self.bc_north, self.rolly(fy,  1), mir_y)    # north wall: from j-1
          fx = wall(fx, self.bc_west,  self.rollx(fx, -1), mir_x)    # west wall:  from i+1
          fx = wall(fx, self.bc_east,  self.rollx(fx,  1), mir_x)    # east wall:  from i-1
          return fx, fy
