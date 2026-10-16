"""
 Brings together all component for the model.
 Defines time steps for different options and a loop over them.

 Based on diverse other classes. 
"""
import pyOM3 as OM
from   pyOM3.wim_simple.propagation import propagation
from   pyOM3.wim_simple.source      import source
from   pyOM3.wim_simple.interface   import interface
import numpy                 as np__
from   functools             import partial


class model(propagation,source,interface):

          
      def time_step(self,tA,tB,tC,tau,taum1,taum2):
          """
          performs a single time step: the propagation (mk_advection and
          add_propagation), the Adams-Bashforth update of the action with
          the outflow book-keeping and the positivity floor (_ab_update),
          then the source terms (_source_update). The two bodies are jitted
          under JAX; this wrapper reads and writes the attributes, so the
          time-varying fields (wind, currents) are ARGUMENTS of the compiled
          code and never trace-time constants (see propagation.py).
          """
          self.xdot, self.ydot, self.fdot, self.ddot = \
                self.mk_advection(self.xdot, self.ydot, self.fdot, self.ddot)
          self.dA =  self.add_propagation(self.A, self.dA,
                                          self.xdot, self.ydot, self.fdot, self.ddot, tau)
          self.A, self.A_clip, self.clip_mass, self._clip_tot, dlA, dlE, dlO = \
                self._ab_update(self.A, self.dA, tA, tB, tC, tau, taum1, taum2,
                                self._lossA_lv, self._lossE_lv, self._lossO_lv, self.shard_consts())
          if getattr(self, 'open_coasts', False):
              self.loss_A, self.loss_E, self.loss_Eabs = \
                  self.loss_A + dlA, self.loss_E + dlE, self.loss_Eabs + dlO
          if getattr(self, 'floor_action', True):
              self._floor_warning()

          (self.A, self.E, self.e_mean, self.om_mean, self.gamma_in, self.gamma_diss,
           self.S_nl, freq_mask, self.E_pre_src, self.dE_src, self.dE_in,
           self.dE_ds, self.dE_nl, self.dE_tail, src_state, src_aux) = \
                self._source_update(self.A, self.ustar, self.udir, self.get_source_state(),
                                    self.shard_consts())
          self.set_source_state(src_state, src_aux)
          if freq_mask is not None: self.freq_mask = freq_mask   # exists only with mask_energy
          return


      @partial(OM.jaxjit, static_argnums=(0,))
      def _ab_update(self,A,dA,tA,tB,tC,tau,taum1,taum2,lA,lE,lO,consts):
          """
          Adams-Bashforth update of the action, the outflow book-keeping
          and the positivity floor. Returns the new A (halo-filled), the
          clipped action per bin, the clip mass and the total action (the
          floor's gauge, both over the real cells) and the three outflow
          increments of this step. consts: the sharded run constants
          (main.shard_consts), bound to self while tracing.
          """
          with self.bind_consts(consts):
              return self._ab_update_body(A,dA,tA,tB,tC,tau,taum1,taum2,lA,lE,lO)

      def _ab_update_body(self,A,dA,tA,tB,tC,tau,taum1,taum2,lA,lE,lO):
          A += self.dt*(tA*dA[tau,:] + tB*dA[taum1,:] + tC*dA[taum2,:])     # in place on numpy
          # OUTFLOW BOOK-KEEPING for open_coasts, with the SAME Adams-Bashforth
          # weights that step the state. The rate at each time level is stored
          # by propagation.open_coast_loss; combining only the tau level would
          # leave the budget open at the order of the time scheme.
          if getattr(self, 'open_coasts', False):
              dlA = self.dt*(tA*lA[tau] + tB*lA[taum1] + tC*lA[taum2])
              dlE = self.dt*(tA*lE[tau] + tB*lE[taum1] + tC*lE[taum2])
              dlO = self.dt*(tA*lO[tau] + tB*lO[taum1] + tC*lO[taum2])
          else:
              dlA = dlE = dlO = 0.0
          # POSITIVITY FLOOR (day-212.7 detonation, option floor_action,
          # default ON): the flux limiter preserves A >= 0 only under SSP
          # time stepping; AB2+eps at CFL 0.22 with wall reflection grew
          # negative-action lobes (doubling per step at a wall cell once the
          # eddy currents perturbed the flux pattern; e_mean 0.2 -> exactly 0
          # in ~10 steps, then the DIA cubic on the sign-indefinite spectrum
          # blew up x1e11/step -- gamma_diss ~ alpha^2 = 0 and the one-sided
          # limit_growth cap cannot oppose negative S). Flooring at 0 is the
          # WAM-standard guard. HONEST GAUGE: the clip mass (action the floor
          # removes, = the spurious action it injects, = the energy-
          # consistency cost) is measured BEFORE clipping, stored in
          # self.clip_mass, and warned about when above round-off scale
          # (_floor_warning). Switching floor_action=False restores the raw
          # scheme (and, at CFL > ~0.13 with reflection, the known detonation).
          if getattr(self, 'floor_action', True):
              sl_ = slice(self.halo, -self.halo)
              _Ai = self.interior_xy(A[sl_, sl_])          # real bins and cells
              A_clip = OM.np.minimum(A, 0.0)      # the action the floor removes, per bin (booking)
              clip_mass = -OM.np.sum(OM.np.minimum(_Ai, 0.0))
              clip_tot  =  OM.np.sum(OM.np.abs(_Ai))
              A = OM.np.maximum(A, 0.0)
          else:
              A_clip, clip_mass, clip_tot = None, 0.0, 0.0
          A = self.apply_bc(A)
          return A, A_clip, clip_mass, clip_tot, dlA, dlE, dlO


      def _floor_warning(self):
          """
          the warning of the positivity floor. On numpy every step (as
          before); under JAX the comparison is a device-to-host sync, so it
          is made every floor_check_steps (default 100) steps only, with the
          clip mass of that step.
          """
          if OM.jax:
              n = getattr(self, '_floor_step', 0)
              self._floor_step = n + 1
              if n % getattr(self, 'floor_check_steps', 100): return
          clip_mass, _tot = float(self.clip_mass), float(self._clip_tot)
          if clip_mass > 1e-10 * max(_tot, 1e-30):
              self._floor_warns = getattr(self, '_floor_warns', 0) + 1
              if self._floor_warns <= 20:
                  print('WARNING pe %d: action floor clipped %.3e '
                        '(%.1e of total action) -- AB2/limiter positivity '
                        'violation active%s'
                        % (getattr(self, 'my_pe', 0), clip_mass,
                           clip_mass / max(_tot, 1e-30),
                           '' if self._floor_warns < 20
                           else ' (suppressing further warnings)'),
                        flush=True)
          return


      @partial(OM.jaxjit, static_argnums=(0,))
      def _source_update(self,A,ustar,udir,src_state,consts):
          """
          The source terms: input, dissipation, DIA, the implicit update
          with the growth limiter, the land mask, the diagnostic tail and
          the halo refresh. Returns the new A and E, the booking fields and
          the source state (source.py, the hooks of a scheme with state).
          consts: the sharded run constants (main.shard_consts), bound to
          self while tracing.
          """
          with self.bind_consts(consts):
              return self._source_update_body(A,ustar,udir,src_state)

      def _source_update_body(self,A,ustar,udir,src_state):
          E =  A*self.omega_t[None,:,None,None] 

          gamma_in, src_state, aux = self.source_input(ustar,udir,E,src_state)
          e_mean, om_mean, gamma_diss, aux = self.source_diss(E,aux)  
          gamma = gamma_in + gamma_diss[None,:,:,:]
          _Snonlin, dSdE = self.Snonlin(E)  
          S_nl = _Snonlin             # stored for the tail cascade credit
                                      # (interface.calc_stress/_phioc, tail_flux='input')
          dSdE += gamma 
          S     = _Snonlin + gamma*E

          S_unlimited = S
          if self.limit_growth:
            ustar_eff = OM.np.maximum(ustar, 0.01)     # keep the cap strictly positive  
            limit = 1e-7*self.grav*ustar_eff[None,:,:] \
                         /self.ft[:,None,None]**4*om_mean[None,:,:]/(2*OM.np.pi)
            S = OM.np.minimum(S,limit[None,:,:,:]) 
              
          sl = slice(self.halo,-self.halo,None)
          # Implicit source update E += S*dt/(1 - dt*dSdE). By default the full
          # feedback dSdE (both signs) is used. Optionally, setting the attribute
          # 'implicit_dissipation_only' keeps only the dissipative (negative) part
          # in the denominator, so 1 - dt*dSdE_imp >= 1 always and the denominator
          # can never vanish/flip sign at high frequency. The growth is then
          # explicit and relies on limit_growth for its bound (see wim.tex).
          if getattr(self, 'implicit_dissipation_only', False):
              dSdE_imp = OM.np.minimum(dSdE, 0.0)
          else:
              dSdE_imp = dSdE
          # ENERGY BOOKING of the APPLIED source increments (test 14, the
          # coupler's energy budget): the implicit factor and the limiter act
          # on the sum, so the three sources share the applied increment in
          # proportion to their explicit rates; dE_in + dE_ds + dE_nl = dE_src
          # to round-off, and the sum over the spectrum of dE_src is the energy
          # the sources changed in this step. The tail overwrite below and the
          # action floor above are booked separately (dE_tail, A_clip).
          # Booked BEFORE the update: add_to_array works in place with numpy.
          dE_src = S*self.dt/(1 - self.dt*dSdE_imp)
          _fac = OM.np.where(S_unlimited != 0.0, dE_src/OM.np.where(S_unlimited != 0.0, S_unlimited, 1.0), 0.0)
          E_pre_src = OM.np.array(E)           # a copy: the explicit rates refer to it
          dE_in  = gamma_in*E*_fac
          dE_ds  = gamma_diss[None,:,:,:]*E*_fac
          dE_nl  = _Snonlin*_fac
          E = self.interior_add(E, dE_src)      # real bins and cells only
                    
          # waves do not exist on land: zero the wave energy on land cells so the
          # source terms leave nothing behind there and land cannot contaminate
          # the spectral means (e_mean / om_mean) on the next step. maskTp is 2D
          # (Ny,Nx), broadcast over the (d,f) axes. Parallels pe_simple's *maskT
          # on the tracer state and our *maskTp on the propagation tendency.
          if hasattr(self,'maskTp'):
            E = E * self.maskTp[None,None,:,:]

          dE_tail = 0.0*E
          freq_mask = None
          if self.mask_energy:
            freq_mask = self.source_cutoff(om_mean/(2*OM.np.pi),ustar,aux)
            _E_before = E
            E = imphftail(freq_mask,E,self.ft)
            dE_tail = E - _E_before                 # the tail overwrite (booking)
                
          # HALO REFRESH after the source update, SPATIAL AXES ONLY. The
          # advective update above ends with apply_bc, but the source update
          # changes the interior only, so without a refresh the x/y halo rows
          # of A lag the interior by one source increment and the next step's
          # advection computes the periodic seam fluxes from stale neighbour
          # values (edge rows drift ~1e-3 per step in a y-uniform forced run).
          # The FREQUENCY GHOST BINS must NOT be refreshed: imphftail fills
          # them with the imposed f^-5 continuation each step BY DESIGN, and
          # both the f-advection boundary flux and the tail diagnostics
          # (calc_stress anchor at bin -halo) consume exactly those values.
          # A blanket apply_bc here wiped them and collapsed the tail input
          # (tau_o/tau_a 0.66 -> 0.37 at the stress-notebook config), hence
          # keep_f_ghosts.
          A = self.apply_bc(E/self.omega_t[None,:,None,None], keep_f_ghosts=True)
          E = A*self.omega_t[None,:,None,None]
          return (A, E, e_mean, om_mean, gamma_in, gamma_diss, S_nl, freq_mask,
                  E_pre_src, dE_src, dE_in, dE_ds, dE_nl, dE_tail, src_state, aux)
    
    
      def loop(self,max_steps=1000000):         
          """
          Loop max_steps time over time steps with Adam-Bashforth 3.order.
          """
          tA,tB,tC = 1.,0.,0.        # will become Adam-Bashforth coefficients below
          tau, taum1, taum2 = 2,1,0  # pointers to time levels
          for n in range(max_steps): # time step equations 
            self.time_step(tA,tB,tC,tau,taum1,taum2)  
            self.diagnose(n,tau)   
            if n % 100 == 0: self.check_monotonicity(n)  
            tau, taum1, taum2   = np__.mod(tau+1,3), np__.mod(taum1+1,3), np__.mod(taum2+1,3)          
            if tA == 1. and tB == 0. and tC == 0:    tA, tB, tC = 1.5+0.05,  -0.5-0.05, 0. 
            #elif tA==1.5 and tB == -0.5 and tC == 0: tA, tB, tC =  23./12. , -16./12., 5./12.
          return    


          
      def check_monotonicity(self, n=0, rtol=1e-10, verbose=True):
          """
          Cheap monitor for the non-negativity of the wave action.

          The flux-limited advection guarantees A >= 0 only for an SSP time
          integration; the AB2+eps scheme used for the propagation preserves it
          in practice but without a formal guarantee (see wim.tex). Any
          violation shows up as min(A) < 0 long before a blow-up, so calling
          this every O(100) steps from the time loop catches it early at
          negligible cost.

          Checks the interior (halo-free) minimum of A against a RELATIVE
          threshold -rtol*max(A): round-off produces tiny negatives scaling
          with the field amplitude, while genuine limiter/time-scheme
          violations are many orders of magnitude larger.

          Works serial and parallel: reduces over self.comm only when more
          than one PE is present. Returns True if A is (numerically)
          non-negative, False otherwise; prints a warning on PE 0 if verbose.
          """
          sl   = slice(self.halo, -self.halo)
          if OM.jax:                       # the inline shard halos are no interior
              A4 = self.A[sl, sl]
              amin = float(OM.np.min(OM.np.where(self.maskI > 0, A4,  OM.np.inf)))
              amax = float(OM.np.max(OM.np.where(self.maskI > 0, A4, -OM.np.inf)))
          else:
              amin = float(OM.np.min(self.A[sl, sl, sl, sl]))
              amax = float(OM.np.max(self.A[sl, sl, sl, sl]))
          comm = getattr(self, 'comm', None)
          if comm is not None and comm.Get_size() > 1:
              from mpi4py import MPI
              amin = comm.allreduce(amin, op=MPI.MIN)
              amax = comm.allreduce(amax, op=MPI.MAX)
          ok = amin >= -rtol * max(amax, 1e-30)
          if not ok and verbose and getattr(self, 'my_pe', 0) == 0:
              print('WARNING: monotonicity violated at n=%d t=%.2f d: '
                    'min(A)=%.3e (%.1e of max) -- increase eps_ab or use Euler'
                    % (n, n * self.dt / 86400., amin,
                       amin / amax if amax > 0 else amin))
          return ok
    
    
      def diagnose(self,n,tau):  
          """
          Dummy routine for diagnostics.
          """
          t =  n*self.dt
          if self.my_pe == 0:  
                pass
                #print ("t=",t," n=",n)


              
      def frcutindex(self,f_mean,ustar): 
          """From WAM:
             COMPUTE LAST FREQUENCY INDEX OF PROGNOSTIC PART OF SPECTRUM.
             FREQUENCIES LE MAX(TAILFACTOR*MAX(FMNWS,FM),TAILFACTOR_PM*FPM),
             WHERE FPM IS THE PIERSON-MOSKOWITZ FREQUENCY BASED ON FRICTION
             VELOCITY. (FPM=G/(FRIC*ZPI*USTAR))

             WAM clamps the prognostic cut-off index MIJ to [1, ML], i.e. there
             is always at least one prognostic frequency bin. The unclamped
             condition ft <= f_max can be all-False for very low f_max (small
             mean frequency / friction velocity), which leaves imphftail with no
             bin to anchor the f^-5 tail to. We reproduce the clamp by forcing
             the lowest real frequency bin (index = halo) to be prognostic.
          """
          # f_pm = g/(28*2*pi*ustar). Where ustar=0 (calm ocean or land), use
          # ustar->inf so f_pm->0: the Pierson-Moskowitz term then drops out of
          # f_max (= max(2.5 f_mean, 3 f_pm) = 2.5 f_mean), which is the correct
          # no-wind limit. This avoids the divide-by-zero / f_pm=inf and leaves
          # every ustar>0 cell bit-identical.
          f_pm = self.grav/(28*2*OM.np.pi*OM.np.where(ustar > 0, ustar, OM.np.inf)) # f_pm
          f_max = OM.np.maximum(2.5*f_mean,3*f_pm) 
          mask = self.ft[:, None, None] <= f_max  # Boolean array, shape (nf, ny, nx)
          # clamp: lowest real bin (index halo) is always prognostic
          mask = OM.modify_array(mask, (self.halo, slice(None), slice(None)), True)
          return mask

      def frcutindex_old(self,f_mean,ustar): 
          """From WAM:
             COMPUTE LAST FREQUENCY INDEX OF PROGNOSTIC PART OF SPECTRUM.
             FREQUENCIES LE MAX(TAILFACTOR*MAX(FMNWS,FM),TAILFACTOR_PM*FPM),
             WHERE FPM IS THE PIERSON-MOSKOWITZ FREQUENCY BASED ON FRICTION
             VELOCITY. (FPM=G/(FRIC*ZPI*USTAR))
          """
          f_pm = self.grav/(28*2*OM.np.pi*ustar) # f_pm
          f_max = OM.np.maximum(2.5*f_mean,3*f_pm) 
          return self.ft[:, None, None] <= f_max  # Boolean array, shape (nf, ny, nx)


 
def imphftail(mask: OM.np.ndarray, E: OM.np.ndarray, ft: OM.np.ndarray) -> OM.np.ndarray:
    """
    translated from WAM fortran code to python with Claude
    
    Impose a high-frequency f^-5 tail to the spectrum where mask is False. 
    Converted from Fortran subroutine IMPHFTAIL in WAM_SOURCE_MODULE.
 
    Parameters
    ----------
    mask : np.ndarray, shape (n_fre, ny, nx), dtype bool
        True  -> prognostic bin: leave E unchanged.
        False -> diagnostic bin: overwrite E with the f^-5 tail.
        Applied uniformly across all directions.
    E : np.ndarray, shape (n_dir, n_fre, ny, nx)
        Wave spectrum.
    ft : np.ndarray, shape (n_fre,)
        Frequencies [Hz] (FR in WAM).
 
    Returns
    -------
    E : np.ndarray, shape (n_dir, n_fre, ny, nx)
        Spectrum with tail E[k,m0,...] * (ft[m0]/ft[m])**5 applied for m > m0,
        where m0 is the last prognostic frequency index per (y, x).
    """
    # Cut-off index m0: last True along the frequency axis, shape (ny, nx)
    n_fre = E.shape[1]
    m0 = (n_fre - 1) - OM.np.argmax(mask[::-1, :, :], axis=0)    # (ny, nx)
 
    # Spectrum value at the cut-off, shape (n_dir, 1, ny, nx)
    E_m0 = OM.np.take_along_axis(E, m0[OM.np.newaxis, OM.np.newaxis, :, :], axis=1)
    E_m0 = OM.np.maximum(E_m0, 0.0)
    
    # f^-5 tail ratio relative to cut-off, shape (1, n_fre, ny, nx)
    ft_ratio = (ft[m0][OM.np.newaxis, OM.np.newaxis, :, :] / 
                ft[OM.np.newaxis, :, OM.np.newaxis, OM.np.newaxis]) ** 5
 
    return OM.np.where(mask[OM.np.newaxis, :, :, :], E, E_m0 * ft_ratio)
 



          