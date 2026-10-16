"""
 progation of wave action

 Based on class main.
"""
import pyOM3                 as OM
from   pyOM3.wim_realistic.main        import main
from   pyOM3.wim_simple.propagation    import propagation as _wim_simple_propagation
from   functools                       import partial


class propagation(main):

      # jit structure as in wim_simple.propagation: the public methods read
      # the time-varying attributes and hand them to the jitted bodies as
      # arguments (with self static, an attribute read inside the body
      # would be a trace-time constant)
      _current_args = _wim_simple_propagation._current_args
      mk_advection  = _wim_simple_propagation.mk_advection

      _mk_advection = _wim_simple_propagation._mk_advection      # binds the constants, calls the body

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

         PSEUDO-SPHERICAL GRID (coord_degree): xdot/ydot are the local
         east/north velocities and stay as above; the spatial gradients
         d/dx, d/dy become the spherical operators (R cos phi)^-1 d/dlambda
         and R^-1 d/dphi (built into _ddx*/_ddy* through the metric grid
         spacings); and the direction and frequency rates gain the
         coordinate-turning terms of the sphere. Derived EXACTLY from the
         canonical ray equations in (lambda,phi) with conjugate momenta
         (k_e R cos phi, k_n R) and Omega = sigma(k) + k.U (deep water),
         the extra terms are

           thetadot += (c_g + s.U) sin(theta) tan(phi)/R
           sigmadot += -(sigma/2) (U.s_perp) sin(theta) tan(phi)/R ,
                        s_perp = ds/dtheta = (cos(theta), -sin(theta))

         i.e. the great-circle turning of ECMWF/WAM (IFS documentation
         Part VII, eq. 2.25c) with the intrinsic group speed replaced by
         the RADIAL advective speed c_g + s.U, plus the rotation of k.U
         under that turning in sigmadot. The ECMWF documentation keeps
         only the c_g part of the turning and no sigmadot term -- a
         truncation at O(U tan(phi)/R), i.e. ~2e-7 1/s at 60 deg for
         U = 1 m/s; the exact forms cost nothing and are used here. As in
         the Cartesian model there is no depth refraction (deep water).
         The turning rate is evaluated at the THETA FACES (sinthu), like
         the current-refraction rate below and for the same mirror-symmetry
         reason; the frequency term lives at theta centres like fdot.

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
             _dd = ( -su*cu*dUdx - cu*cu*dVdx
                     +su*su*dUdy + su*cu*dVdy )
             _fd = 0.5 * self.ft[None,:,None,None] * (s*kxdot + c*kydot)
             if self.coord_degree:
                 # exact coordinate-turning terms of the sphere (docstring):
                 # (c_g + s.U) sin(theta) tan(phi)/R at the theta faces, and
                 # the induced rotation of k.U in the frequency rate,
                 # -(f/2)(U.s_perp) sin(theta) tan(phi)/R at theta centres.
                 cg4 = c_g[None,:,None,None]
                 _dd = _dd + (cg4 + su*U + cu*V) \
                             * su * self.tantr_y
                 _fd = _fd - 0.5*self.ft[None,:,None,None] \
                             * (c*U - s*V) \
                             * s * self.tantr_y
             ddot = self.apply_bc( _dd )

             # frequency refraction:  omegadot = (c/2) s . kdot
             # better self.fu??
             fdot = self.apply_bc( _fd )
             
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
             if self.coord_degree:
                 # great-circle turning without currents:
                 # c_g sin(theta) tan(phi)/R at the theta faces (no
                 # frequency term: sigma is conserved along a great circle)
                 su = self.sinthu[:,None,None,None]
                 ddot = self.apply_bc( c_g[None,:,None,None]*su
                                       * self.tantr_y
                                       + OM.np.zeros_like(xdot) )

         
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
         """Centered d/dx on the variable/pseudo-spherical grid: the
         centre-to-centre distance xt_{i+1}-xt_{i-1} is dxu_{i-1}+dxu_i,
         times cost for the zonal metric ((R cos phi)^-1 d/dlambda).
         Reduces to /(2 dx) on a uniform Cartesian grid."""
         return ( (self.rollx(a,-1) - self.rollx(a,1))
                  / (self.cost_y*(self.dxu_x + self.dxu_xm1)) )

      def _ddxb(self,a):
         """Backward d/dx: centre-to-centre distance xt_i - xt_{i-1}
         = dxu_{i-1}, times cost."""
         return  (a - self.rollx(a,1)) / (self.cost_y*self.dxu_xm1)

      def _ddyc(self,a):
         """Centered d/dy: centre-to-centre distance dyu_{j-1}+dyu_j
         (meridian arc, no metric factor)."""
         return  ( (self.rolly(a,-1) - self.rolly(a,1))
                   / (self.dyu_y + self.dyu_ym1) )

      def _ddyb(self,a):
         """backward d/dy: distance dyu_{j-1}."""
         return  (a - self.rolly(a,1)) / self.dyu_ym1

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

          # per-face energy-defect terms in ABSOLUTE (area-integrated) units.
          # On the variable/pseudo-spherical grid the cell area varies, so no
          # common factor can be dropped: multiplying the column defect
          # Om * div * dft * dd by the cell area area_t = dxt*cost*dyt, the
          # metric of the x divergence leaves the face weight dyt (a meridian
          # arc), that of the y divergence leaves dxt (fy already carries
          # cosu), and the spectral faces keep the full area_t. Reduces to
          # the wim_simple form (common dx*dy) on a uniform Cartesian grid.
          tx = fx*(self.rollx(Om,-1)-Om)*self.dft[None,:,None,None]*self.dd*self.dyt_y
          ty = fy*(self.rolly(Om,-1)-Om)*self.dft[None,:,None,None]*self.dd*self.dxt_x
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
          tf = ff*dOf*self.dd*self.area_t
          td = fd*(self.rolld(Om,-1)-Om)*self.dft[None,:,None,None]*self.area_t

          # column defect: sum over real (d,f); x/y faces shared half-half.
          # sum of col over all real columns equals the domain integral
          # sum Omega dA/dt|adv dV (absolute units, area weights included).
          col = -( OM.np.sum((tf+td)[sl,sl,:,:],axis=(0,1))
                 + 0.5*OM.np.sum((tx+self.rollx(tx,1)
                                 +ty+self.rolly(ty,1))[sl,sl,:,:],axis=(0,1)) )

          # correction shape: Abar * deltaOmega on interior f-faces
          wgt = 0.5*(A + self.rollf(A,-1))*dOf
          wgt = OM.modify_array(wgt,(slice(None),wgt.shape[-3]-self.halo-1,
                                     slice(None),slice(None)),0.0)  # closed top
          wgt = OM.modify_array(wgt,(slice(None),self.halo-1,
                                     slice(None),slice(None)),0.0)  # closed bottom


          den = OM.np.sum((wgt*dOf*self.dd)[sl,sl,:,:],axis=(0,1))*self.area_t
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
              # in place on coastal faces only; gy carries cosu via prefy
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
          # the y replacement carries cosu like the flux it replaces
          gy = -self.cosu_y*(ydot*(vy + var) - OM.np.abs(ydot)*(vy - var))*0.5
          return (OM.np.where(self.maskOx[None,None,:,:] > 0, gx, fx),
                  OM.np.where(self.maskOy[None,None,:,:] > 0, gy, fy))


      add_propagation  = _wim_simple_propagation.add_propagation
      _add_propagation = _wim_simple_propagation._add_propagation     # binds the constants, calls the body

      def _add_propagation_body(self,A,dA,xdot,ydot,fdot,ddot,tau,U,V,has_current):
          """
           Calculates tendencies due to advection for wave action.
           Stores tendency in dA and overwrites at time level tau.
           Body of add_propagation (jitted under JAX); returns dA and the
           outflow rates of this level.
          """
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
          # flux divergence with the pseudo-spherical metric (pyOM2/
          # pe_realistic tracer convention): the y flux fy carries cosu, both
          # horizontal divergences divide by cost, the spectral directions
          # carry no metric. Reduces exactly to the wim_simple form on a
          # uniform Cartesian grid (cost=cosu=1, dxt=dx).
          if getattr(self, 'propagation_backend', 'numpy') == 'fortran':
              # one pass, writes straight into dA[tau] (a Fortran-contiguous
              # view through .T, no copy); halos wrap like np.roll
              T = self._prop_tables()
              self._prop_f.divergence(fx.T, fy.T, ff.T, fd.T, T['denx'].T,
                                      T['deny'].T, self.dft, self.dd, T['mT'].T,
                                      dA[tau].T)
              return dA, rA, rE, rO
          _dA = (  (fx-self.rollx(fx,1))/(self.cost_y*self.dxt_x)
                 + (fy-self.rolly(fy,1))/(self.cost_y*self.dyt_y)
                 + (ff-self.rollf(ff,1))/self.dft[None,:,None,None]
                 + (fd-self.rolld(fd,1))/self.dd )
           
          # zero the tendency in land cells 
          if hasattr(self, 'maskTp'): _dA = _dA * self.maskTp[None, None, :, :]
        
          dA = OM.modify_array(dA, (tau,slice(None),slice(None),
                                        slice(None),slice(None)), _dA)
          return dA, rA, rE, rO
          
                            
      
      def _prop_metric(self):
          """
          Grid tables for the Fortran kernels (see wim_simple._prop_metric),
          in exactly the products the numpy path uses: denx = cost*dxt,
          deny = cost*dyt (Ny,Nx), prefy = cosu (Ny,).
          """
          Ny, Nx = self.Ny, self.Nx
          denx = OM.np.broadcast_to(self.cost_y*self.dxt_x, (Ny, Nx))
          deny = OM.np.broadcast_to(self.cost_y*self.dyt_y, (Ny, Nx))
          return denx, deny, self.cosu_y[:, 0]

      def _mk_sph(self):
          """spherical turning for the mk_advection kernel: active under
          coord_degree, rate table tan(phi)/R of the metric grid."""
          return int(self.coord_degree), \
                 OM.np.ascontiguousarray(self.tantr_y[:, 0])

      def _efix_metric(self):
          """energy-fixer metric (see wim_simple): multiply by dyt/dxt
          instead of dividing by dx/dy, cell area on the spectral terms,
          area on the summed denominator."""
          Ny, Nx = self.Ny, self.Nx
          return dict(wx=OM.np.ascontiguousarray(
                            OM.np.broadcast_to(self.dyt_y, (Ny, Nx))), divx=0,
                      wy=OM.np.ascontiguousarray(
                            OM.np.broadcast_to(self.dxt_x, (Ny, Nx))), divy=0,
                      area=OM.np.ascontiguousarray(self.area_t),
                      den_mult=self.area_t)

      # verbatim copies of wim_simple removed in favour of assignment: these
      # bodies contain no grid information (open_coast_loss reads only the
      # face tables and the state; the reflection branches and the backward
      # averages are metric-free)
      open_coast_loss            = _wim_simple_propagation.open_coast_loss
      reflect_ini                = _wim_simple_propagation.reflect_ini
      reflect_flux               = _wim_simple_propagation.reflect_flux
      _reflect_flux_vec          = _wim_simple_propagation._reflect_flux_vec
      _avxb                      = _wim_simple_propagation._avxb
      _avyb                      = _wim_simple_propagation._avyb

      # the kernel drivers are grid-agnostic once the tables are in place:
      # reuse wim_simple's, do not copy them
      _prop_tables               = _wim_simple_propagation._prop_tables
      tracer_flux_superbee_fortran = _wim_simple_propagation.tracer_flux_superbee_fortran
      mk_advection_fortran       = _wim_simple_propagation.mk_advection_fortran
      energy_fix_flux_fortran    = _wim_simple_propagation.energy_fix_flux_fortran
      open_coast_loss_fortran       = _wim_simple_propagation.open_coast_loss_fortran


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
          # y (the flux carries cosu, see tracer_flux_superbee)
          rj = self.apply_bc( (self.rolly(var,-1) - var) * mV )
          fy = - self.cosu_y*(ydot * (self.rolly(var,-1) + var) * 0.5
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

          CFL = OM.np.abs(xdot * self.dt / (self.cost_y*self.dxt_x))
          rj  = self.apply_bc( (self.rollx(var,-1)-var) * mU )
          rjm = self.rollx(rj, 1) 
          rjp = self.rollx(rj,-1)                     
          cr = _limiter( OM.np.where(xdot > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fx = - (xdot * (self.rollx(var,-1) + var) * 0.5 
               - OM.np.abs(xdot) * ((1.0 - cr) + CFL * cr) * rj * 0.5) * mU

          # y flux carries the metric factor cosu at the face (pe_realistic
          # tracer convention); the divergence divides by cost*dyt.
          CFL = OM.np.abs(self.cosu_y*ydot * self.dt / (self.cost_y*self.dyt_y))
          rj  = self.apply_bc( (self.rolly(var,-1)-var) * mV )
          rjm = self.rolly(rj, 1) 
          rjp = self.rolly(rj,-1)                     
          cr = _limiter( OM.np.where(ydot > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj))
          fy = - self.cosu_y*(ydot * (self.rolly(var,-1) + var) * 0.5 
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
         
          

