"""
 progation of wave action

 Based on class main.
"""
import pyOM3                 as OM
from   pyOM3.wim_simple.main        import main


class propagation(main):

        
      def mk_advection(self,xdot,ydot,fdot,ddot):
        
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
         sl = slice(self.halo,-self.halo)
         c_g = self.grav/2 /self.omega_t
         s, c  = self.sinth[:,None,None,None], self.costh[:,None,None,None]
         xdot[:,:,:,:], ydot[:,:,:,:] = c_g[None,:,None,None]*s, c_g[None,:,None,None]*c

         if hasattr(self,'ocean_U') and hasattr(self,'ocean_V'): 
             xdot += self.ocean_U
             ydot += self.ocean_V
             half_f = 0.5 * self.ft[None,:,None,None]
             dUdf = getattr(self, 'ocean_Udf', 0.0)
             dVdf = getattr(self, 'ocean_Vdf', 0.0)            
             R = half_f * (s*dUdf + c*dVdf)      # radial group-velocity factor
             xdot += R * s
             ydot += R * c
         
             dUdx, dVdx = self._ddxb(self.ocean_U), self._avyb( self._ddxc(self.ocean_V))
             dUdy, dVdy = self._avxb( self._ddyc(self.ocean_U)), self._ddyb(self.ocean_V)   

             kxdot, kydot  = -(s*dUdx + c*dVdx)   , -(s*dUdy + c*dVdy)  
             ddot  = self.apply_bc( c*kxdot - s*kydot )        # thetadot  (Eq. 8)
             
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
   

        
      def energy_fix_flux(self,A,fx,fy,ff,fd):
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
          if not (hasattr(self,'ocean_U') and hasattr(self,'ocean_V')):
              return ff
          sl = slice(self.halo,-self.halo)
          s, c = self.sinth[:,None,None,None], self.costh[:,None,None,None]
          kf = self.omega_t[None,:,None,None]**2/self.grav
          Om = self.omega_t[None,:,None,None] \
               + kf*(self.ocean_U[None,:,:,:]*s + self.ocean_V[None,:,:,:]*c)

          # per-face energy-defect terms (common factor dx*dy dropped; it
          # cancels between column defect and correction)
          tx = fx*(self.rollx(Om,-1)-Om)*self.dft[None,:,None,None]*self.dd/self.dx
          ty = fy*(self.rolly(Om,-1)-Om)*self.dft[None,:,None,None]*self.dd/self.dy
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

          #lam = OM.np.where(OM.np.abs(den) > 1e-300,
          #                  col/OM.np.where(den==0,1.0,den), 0.0)
          cfl = lam[None,None,:,:]*wgt
          if hasattr(self,'maskTp'): cfl = cfl*self.maskTp[None,None,:,:]
          return ff + cfl


      def add_propagation(self,A,dA,xdot,ydot,fdot,ddot,tau):  
          """
           Calculates tendencies due to advection for wave action.
           Stores tendency in dA and overwrites at time level tau.
          """
          # advection flux scheme: 'superbee' (2nd-order, low diffusion, default)
          # or 'upwind' (1st-order, diffusive, robust to grid-scale noise, allows
          # a larger CFL and is ~4x cheaper per direction). Both are finite-volume
          # (face-flux) forms, so conservation and energy_fix_flux are preserved.
          if getattr(self, 'f_limiter', 'superbee') == 'upwind':
              fx,fy,ff,fd = self.tracer_flux_upwind(A,xdot,ydot,fdot,ddot,tau)
          else:
              fx,fy,ff,fd = self.tracer_flux_superbee(A,xdot,ydot,fdot,ddot,tau)
           
          if getattr(self,'energy_consistent',False): ff = self.energy_fix_flux(A,fx,fy,ff,fd)
          _dA = (  (fx-self.rollx(fx,1))/self.dx 
                 + (fy-self.rolly(fy,1))/self.dy 
                 + (ff-self.rollf(ff,1))/self.dft[None,:,None,None]
                 + (fd-self.rolld(fd,1))/self.dd )
           
          # zero the tendency in land cells 
          if hasattr(self, 'maskTp'): _dA = _dA * self.maskTp[None, None, :, :]
        
          return OM.modify_array(dA, (tau,slice(None),slice(None),
                                          slice(None),slice(None)), _dA)
          
                            
      
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
          if getattr(self, 'enable_reflection', False):  fx, fy = self.reflect_flux(fx, fy)
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
          if getattr(self, 'enable_reflection', False):  fx, fy = self.reflect_flux(fx, fy)
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
          th = self.theta[self.halo:-self.halo]            # real direction bins
          nd = th.shape[0]
          Ny, Nx = self.Ny, self.Nx
          two_pi = 2.0 * OM.np.pi

          def nearest(angle):
              a = angle % two_pi
              return int(OM.np.argmin((th - a) ** 2))

          # mirror-bin lookup (index into the real, unpadded direction axis)
          mirror_x = OM.np.array([nearest(two_pi - t) for t in th])   # x-walls
          mirror_y = OM.np.array([nearest(OM.np.pi - t) for t in th]) # y-walls

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
                  bc_s[k] = OM.np.where(south_edge, mirror_y[k], bc_s[k])
              else:
                  bc_n[k] = OM.np.where(north_edge, mirror_y[k], bc_n[k])
              # west/east: sin(theta)<0 (theta in (pi,2pi]) heads west
              if (t > OM.np.pi) and (t <= two_pi):
                  bc_w[k] = OM.np.where(west_edge, mirror_x[k], bc_w[k])
              else:
                  bc_e[k] = OM.np.where(east_edge, mirror_x[k], bc_e[k])

          self.bc_south, self.bc_north = bc_s, bc_n
          self.bc_west,  self.bc_east  = bc_w, bc_e
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
