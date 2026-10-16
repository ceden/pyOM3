"""
 Tracer advection outside the Adams-Bashforth step: forward in time,
 direction by direction, with the vertical direction sub-cycled.

 OPTION   self.vadv_sub_cycling = M     number of vertical sub-steps.
          Default 0 = off: nothing in this file is called and the model is
          the unsplit one, bit for bit. Needs superbee_advection = True.

 WHY      w = u.grad(H) on staircase slopes scales like 1/dx, so with thin
          levels on a fine horizontal grid the vertical CFL number w dt/dz
          limits the time step (icon01: 0.64 at dt 144 s on the Antarctic
          shelf break, a two-cell sawtooth in salinity). The superbee flux
          carries its own Lax-Wendroff dt term: stepped FORWARD it is TVD
          up to CFL 1, inside the AB step (1.6,-0.6) it is unstable above
          about 0.45, in any direction.

 SCHEME   (the multi-dimensional advection of MITgcm, gad_advection.F)
          with the FINAL velocities u, v, w of the step:
             x sweep,  y sweep,  M vertical sub-steps of dt/M
          each a forward superbee step,           w dt/dz <= M,
          then the AB update with dtemp, dsalt, which on this path hold the
          FORCING only (set_forcing) and no advection.

 Two things that are not optional:

 1. ALL of the advection leaves the AB step, not the vertical part alone.
    An oscillatory term split off next to an AB step is unstable whenever
    it cancels a term that stays inside: with x+ = E (x + dt (A f + B f-)),
    E = exp(i theta), f = i phi x/dt, the old tendency B f- is off by the
    rotation E and |lambda| = 1 + 0.6 phi sin(theta) for opposite signs.
    Horizontal and vertical advection cancel in any flow along isopycnals,
    which is the state of a slope current, so "vertical advection split,
    horizontal in AB" grows at 0.5 phi^2 per step there. The same argument
    is why the vertical MOMENTUM advection is NOT treated here: it cancels
    against terms (horizontal advection, pressure gradient) that have to
    stay in the AB step.

 2. The FINAL velocities. Forward tracer advection with the velocity of the
    start of the step, beside AB momentum, amplifies internal waves
    (|lambda| = 1 + 0.2 (omega dt)^2); with the end-of-step velocity it is
    the forward-backward pair, |lambda|^2 = 1 - 0.6 (omega dt)^2
    (MITgcm: staggerTimeStep). u, v, w at the end of a step satisfy the
    discrete continuity equation exactly, which point 3 needs.

 3. THE PSEUDO DENSITY. A flux-form sweep in one direction changes a
    uniform field, q = T (1 - dt du/dx), and the next flux-form sweep acts
    on that: left alone, T (1 - (dt D)^2)-type errors every step. So the
    sweeps carry the volume the cell would have under the divergence of the
    directions done so far (Easter 1993, Lin and Rood 1996),
         rho = 1,   rho <- rho - dts d(vel)/dx   with each sweep,
    fluxes are formed with the VALUE T = q/rho and act on the CONTENT q.
    The content is conserved to round-off, a uniform T stays uniform to
    round-off, and rho is back at 1 after the last sub-step (with the pyOM2
    closed surface face: 1 + dt w_surf/dz in the top cell, exactly what the
    unsplit scheme does there).

 THE SURFACE FACE belongs to the vertical sub-steps: with
 free_surface_tracer_flux it carries -w_surf T_surf of the CURRENT sub-step.
 The mean over the sub-steps is kept in self.vadv_fs_flux['temp'|'salt']
 (2-D, var m/s, positive into the ocean) for a configuration that books the
 tracer budget. It is applied WITHOUT AB weights.

 RESTART  no new state. dtemp, dsalt change meaning (forcing only), so the
          first step after switching the option on or off should not use
          the old tendency levels.

 Halo exchanges per tracer and step: 6, as in the unsplit path.

 Based on class main.
"""
import pyOM3                    as OM
from   pyOM3.pe_realistic.main  import main
from   functools                import partial


class vadv_split(main):


      def vadv_check(self):
          """ called from prepare_for_run """
          M = getattr(self,'vadv_sub_cycling',0)
          if M == 0: return
          if int(M) != M or M < 0:
             raise ValueError('vadv_sub_cycling must be an integer >= 0')
          if not getattr(self,'superbee_advection',False):
             raise ValueError('vadv_sub_cycling needs superbee_advection = True: forward '
                              'steps of the centred flux are unstable at any CFL number')
          self.vadv_fs_flux = {}
          return


      @partial(OM.jaxjit, static_argnums=0)
      def vadv_zero_tendency(self,dvar,tau):
          """ on the split path nothing overwrites the tendency level tau
              before set_forcing adds to it """
          return OM.modify_array(dvar, (tau,slice(None),slice(None),slice(None)), 0.*dvar[tau],
                                 out_sharding = self.sharding_4D)


      def _vadv_superbee(self,T,vel,rho,roll,dist,maskF,dts,exchange):
          """
          superbee flux (as tracer_flux_superbee) through the faces of ONE
          direction for a forward step dts. vel is the face velocity (cosu v
          for y), dist the cell width in that direction. Courant number of
          the step and of the current cell volume, |vel| dts/(rho dist).
          """
          def _limiter(cr):
              return OM.np.maximum(OM.np.clip(2 * cr, 0, 1), OM.np.clip(cr, 0, 2))
          eps  = 1e-20
          uCFL = OM.np.abs(vel * dts / (dist*rho))
          rj   = (roll(T,-1)-T)*maskF
          if exchange: rj = self.apply_bc(rj)
          else:        # vertical: no fake jump at the opened surface face
                       rj = OM.modify_array(rj, (-(self.halo+1), slice(None), slice(None)),
                                            OM.np.zeros_like(rj[-(self.halo+1)]))
          rjm  = roll(rj, 1)
          rjp  = roll(rj,-1)
          cr   = _limiter( OM.np.where(vel > 0.0, rjm, rjp) / OM.np.where(OM.np.abs(rj) < eps, eps, rj)  )
          return - (vel * (roll(T,-1) + T) * 0.5 - OM.np.abs(vel) * ((1.0 - cr) + uCFL * cr) * rj * 0.5)*maskF


      @partial(OM.jaxjit, static_argnums=0)
      def vadv_tracer(self,var,dvar,u,v,w,A,B,C,tau,taum1,taum2,maskU,maskV,maskW,maskT):
          """
          One tracer through the step: x sweep, y sweep, M vertical
          sub-steps, AB update with the forcing tendency dvar.
          Returns the tracer and the sub-step mean of the surface-face flux.
          """
          M   = int(self.vadv_sub_cycling)
          ks  = -(self.halo+1)
          q   = var
          # x
          dist = self.cost_y*self.dxt_x
          f    = self.apply_bc( self._vadv_superbee(q,u,1.0,self.rollx,dist,maskU,self.dt,True) )
          g    = -u*maskU
          q    = q   + self.dt*(f - self.rollx(f,1))/dist*maskT
          rho  = 1.0 + self.dt*(g - self.rollx(g,1))/dist*maskT
          # y
          dist = self.cost_y*self.dyt_y
          vc   = self.cosu_y*v
          T    = self.apply_bc(q/rho)
          f    = self.apply_bc( self._vadv_superbee(T,vc,rho,self.rolly,dist,maskV,self.dt,True) )
          g    = -vc*maskV
          q    = q   + self.dt*(f - self.rolly(f,1))/dist*maskT
          rho  = rho + self.dt*(g - self.rolly(g,1))/dist*maskT
          # z, M sub-steps, column local
          dts  = self.dt/M
          g    = -w*maskW
          if not getattr(self, 'free_surface_tracer_flux', False):
             g = OM.modify_array(g, (ks, slice(None), slice(None)), OM.np.zeros_like(g[ks]))
          G    = (g - self.rollz(g,1))/self.dzt_z*maskT
          fs   = OM.np.zeros_like(q[ks])
          for m in range(M):
              T = q/rho
              f = self._vadv_superbee(T,w,rho,self.rollz,self.dzt_z,maskW,dts,False)
              if getattr(self, 'free_surface_tracer_flux', False):
                 f = OM.modify_array(f, (ks, slice(None), slice(None)), -w[ks]*T[ks]*maskW[ks])
              else:
                 f = OM.modify_array(f, (ks, slice(None), slice(None)), OM.np.zeros_like(f[ks]))
              q   = q   + dts*(f - self.rollz(f,1))/self.dzt_z*maskT
              rho = rho + dts*G
              fs  = fs + f[ks]/M
          # forcing, AB weighted as in the unsplit path
          q = q + self.dt*(A*dvar[tau] + B*dvar[taum1] + C*dvar[taum2])*maskT
          return self.apply_bc(q), fs


      def vadv_update_tracers(self,A,B,C,tau,taum1,taum2):
          """ temperature and salinity through vadv_tracer, with the final
              u, v, w of the step (called in place of update_thermodynamics_AB3) """
          self.temp, self.vadv_fs_flux['temp'] = self.vadv_tracer(self.temp,self.dtemp,self.u,self.v,self.w,
                                                                  A,B,C,tau,taum1,taum2,
                                                                  self.maskU,self.maskV,self.maskW,self.maskT)
          if self.eq_of_state > 0 and self.eq_of_state != 100:
             self.salt, self.vadv_fs_flux['salt'] = self.vadv_tracer(self.salt,self.dsalt,self.u,self.v,self.w,
                                                                     A,B,C,tau,taum1,taum2,
                                                                     self.maskU,self.maskV,self.maskW,self.maskT)
          return
