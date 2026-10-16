"""
 main class of the wim_idemix model: constants, allocation and grid.

 wim_idemix carries the two moments of the action spectrum per direction

     Acal(theta) = int A~ dk ,   Ecal(theta) = int sigma A~ dk

 on a horizontal grid, closed by a self-similar shape ansatz (shape.py,
 composite.py).  Model arrays have shape (nd,Ny,Nx) with halo points in
 (y,x) only; sigma_bar = Ecal/Acal is diagnostic.

 Based on class boundary dealing with the domain decomposition.
"""
# provenance tag checked by the paper notebooks (pm_fit, fig_local_wim_compare)
CONSTANTS = "full-range shape constants; directional law a_theta N^b_theta = 0.20 N^5.4; beta_o = 0.876 (PM landing >= 4 d)"

import numpy as np__

tr = np__.trapezoid if hasattr(np__, 'trapezoid') else np__.trapz
from   pyOM3.wim_idemix.boundary import boundary
from   pyOM3.wim_idemix.shape    import Shape


class main(boundary):

      def setup(self):
          """
          Set the model constants, tabulate the shape-weighted source
          moments, allocate the model arrays and seed them.
          """
          self.set_constants()
          self.set_grid()
          self.tabulate_moments()
          self.allocate()
          self.set_seed()
          # horizontal land/sea mask hook, same interface as wim_simple:
          # setup calls set_topography(); the default sets no mask (all
          # ocean). An experiment class overrides it and sets self.maskTp
          # (1 ocean, 0 land, padded), which time_step applies each step.
          self.set_topography()
          return

      # ----------------------------------------------------- constants
      def set_constants(self):
          """Physical constants, closure constants and switches."""
          if not hasattr(self, 'grav'):    self.grav = 9.81
          if not hasattr(self, 'rho_a'):   self.rho_a = 1.225
          if not hasattr(self, 'rho_o'):   self.rho_o = 1024.0
          if not hasattr(self, 'dt'):      self.dt = 60.0
          if not hasattr(self, 'c_in'):    self.c_in = 1.0
          # beta_o: the WAM whitecapping constant of the paper (Eq. 27),
          # tuned by the Pierson-Moskowitz landing of the duration-limited
          # run at four days or later.  Legacy interface: passing c_out
          # instead derives beta_o = c_out*3.0e4*s_half^2/s1_c in
          # tabulate_moments (the old bare-3.0e4 parameterisation).
          if not hasattr(self, 'beta_o'):  self.beta_o = None    # 0.908, set in tabulate_moments
          if not hasattr(self, 'c_r_hat'): self.c_r_hat = 0.23   # fallback
          if not hasattr(self, 'r_c'):     self.r_c = 5.0        # fallback
          if not hasattr(self, 'use_xi_table'): self.use_xi_table = True
          # Calibrated xi-functions of the cascade, on the FIXED surface
          # x = x_win and in the FULL-RANGE convention (pair moments over
          # all x, matching the paper's definition of (Acal, Ecal)).  The
          # values below ARE the calibration: they were measured by a
          # least-squares fit of the exact S_nl flux across that surface,
          # state by state, and are carried here as a table rather than
          # read from a file.  In the over-Toba 'peak' regime they are
          # exactly state-independent (pure-family self-similarity), so
          # the flat low-xi extension is exact, not a clamp.  NOTE: the
          # table ends at xi = 14.6 and np.interp clamps beyond -- the
          # strongly eq-limited regime (xi >~ 23) runs on the edge value.
          self.xi_tab = np__.array([4.584, 6.189, 8.251, 10.314, 12.377, 14.562])
          self.cr_tab = np__.array([0.3250, 0.3250, 0.3250, 0.3188, 0.2720, 0.2423])
          self.rc_tab = np__.array([4.11, 4.11, 4.11, 4.17, 4.76, 6.45])
          # saturation-excess dissipation enhancement (designated route
          # for over-saturated states): gout *= 1 + c_sat max(0,satX-1)^2
          if not hasattr(self, 'c_sat'):   self.c_sat = 6.0      # tuned
          # N_p = phi^0_peak = 1.0383 (full-range family, paper Eq. 29);
          # default taken from the shape in tabulate_moments
          if not hasattr(self, 'B_sat'):   self.B_sat = 0.0081
          # equilibrium level of the wind-maintained range, paper eq:toba
          if not hasattr(self, 'alpha_e'): self.alpha_e = 0.062
          if not hasattr(self, 'sat_cap'): self.sat_cap = 10.0
          # the ENHANCED (over-saturation) sink acts at the FLANK, where
          # the saturation excess lives.  x_fl = 2 is fixed by the
          # flank definition, not tuned.
          if not hasattr(self, 'x_fl'):    self.x_fl = 2.0
          # physical ceiling and floor on sigbar = Ecal/Acal: the ceiling
          # guards satX ~ sigbar^5 against an action-floor blowup, the
          # floor bounds the transport speed g/(2 sigbar) (a cosmetic
          # floor permits km/s group speeds and CFL detonation)
          if not hasattr(self, 'sig_max'): self.sig_max = 2*np__.pi*1.5
          if not hasattr(self, 'sig_min'): self.sig_min = 2*np__.pi*0.04
          # directional redistribution: relaxation toward a reference
          # shape at the MEASURED narrowness rate 0.41 Nn^3.6 (extracted
          # from the exact S_nl), which makes the equilibrium spread a
          # prediction.  dyn_target is the fully-extracted companion
          # variant, REJECTED as default (unphysical growth
          # oscillations) and kept for the decision record.
          if not hasattr(self, 'c_theta'):         self.c_theta = 1.0
          if not hasattr(self, 'narrowness_rate'): self.narrowness_rate = True
          if not hasattr(self, 'Nn_cap'):          self.Nn_cap = 3.5
          if not hasattr(self, 'dyn_target'):      self.dyn_target = False
          if not hasattr(self, 'ref_spread_s'):    self.ref_spread_s = 6
          # directional rate law gamma_theta = a_theta N^b_theta gamma_nl
          # (clean-protocol extraction; see the calibration notebook)
          if not hasattr(self, 'a_theta'):         self.a_theta = 0.20
          if not hasattr(self, 'b_theta'):         self.b_theta = 5.4
          if not hasattr(self, 'implicit_dissipation_only'):
              self.implicit_dissipation_only = True
          if not hasattr(self, 'floor_action'): self.floor_action = True
          if not hasattr(self, 'Ad_floor'):     self.Ad_floor = 1e-12
          if not hasattr(self, 'seed_fp'):      self.seed_fp = 0.4
          if not hasattr(self, 'seed_E'):       self.seed_E = 1e-4
          # advection: flux form, limited second order ('superbee' as in
          # wim_simple, or 'vanleer'); theta_dot = current refraction
          # rate (None = off), supplied by the coupling and k-independent
          if not hasattr(self, 'limiter'):   self.limiter = 'superbee'
          if not hasattr(self, 'theta_dot'): self.theta_dot = None
          # distinct refraction rates from the sampled shear (paper
          # eq:Aeq, eq:Eeq), from interface.calc_refraction.  When set
          # they take precedence over theta_dot, which applies ONE rate
          # to both moments and is exact only for a depth-uniform
          # current (kept as the external override for tests).
          if not hasattr(self, 'theta_dot_A'): self.theta_dot_A = None
          if not hasattr(self, 'theta_dot_E'): self.theta_dot_E = None
          # energy exchange rate with the mean flow (paper eq:gamma-ex),
          # from interface.calc_exchange (None = off); the frequency
          # refraction of the simplified model
          if not hasattr(self, 'gamma_ex'):    self.gamma_ex = None
          # current advection velocities of the coupling (paper eq:Aeq,
          # eq:Eeq): U_A, U_E each (2, nd, Ny, Nx), component axis first,
          # from interface.calc_doppler_vel (None = off)
          if not hasattr(self, 'U_A'):       self.U_A = None
          if not hasattr(self, 'U_E'):       self.U_E = None
          # dispersion velocities V_A, V_E (same shape, from
          # interface.calc_doppler_disp): the transport gains the
          # velocity s (s . V_X) along the wave direction.  They vanish
          # for a depth-uniform current and need vertical shear.
          if not hasattr(self, 'V_A'):       self.V_A = None
          if not hasattr(self, 'V_E'):       self.V_E = None
          # adaptive source sub-stepping (see model.time_step)
          if not hasattr(self, 'adaptive_sources'): self.adaptive_sources = True
          if not hasattr(self, 'source_cfl'):   self.source_cfl = 0.2
          if not hasattr(self, 'max_substeps'): self.max_substeps = 32
          # the prognostic range: the pair is defined over x <= x_win
          if not hasattr(self, 'x_win'):     self.x_win = 8.0
          return

      # ---------------------------------------------------------- grid
      def set_grid(self):
          """Directional grid (wim/WAM convention) and local coordinates."""
          # k_hat = (sin theta, cos theta), theta clockwise from +y: the
          # same convention as wim_simple, so wind directions, theta_dot
          # and diagnostics pass between the models unconverted.
          #
          # theta carries the SAME HALO as the model arrays (wim_simple
          # pads its direction and frequency grids the same way): the
          # halo entries continue the ring past the ends, so
          # theta[halo:-halo] are the nd real bin centres and the halo
          # entries are the wrapped neighbours offset by -+2 pi.  Only
          # sin and cos of theta are ever used, and those are periodic,
          # so the offset is invisible -- but it keeps theta monotone,
          # which any face-angle arithmetic needs.
          h = self.halo
          self.theta = 2*np__.pi*(np__.arange(self.Nd) - h)/self.nd
          self.dth   = 2*np__.pi/self.nd
          # the interior slice of any haloed direction axis, used
          # wherever a moment is taken over the ring
          self.dsl   = slice(h, -h)
          # direction tables, precomputed once (wim_simple does the same):
          # sinth/costh are the (Nd,) vectors, s_x/s_y the (Nd,1,1) views
          # in which k_hat = (sin theta, cos theta) is used throughout.
          self.sinth = np__.sin(self.theta)
          self.costh = np__.cos(self.theta)
          self.s_x   = self.sinth[:, None, None]
          self.s_y   = self.costh[:, None, None]
          self.xt = (np__.arange(self.Nx) - h + self.xs_pe - 0.5)*self.dx
          self.yt = (np__.arange(self.Ny) - h + self.ys_pe - 0.5)*self.dy
          return

      # ------------------------------------------------- source moments
      def tabulate_moments(self):
          """
          Tabulate the shape-weighted source moments once.

          RANGE CONSISTENCY (paper convention): the pair (Acal, Ecal)
          is defined over the FULL range (the LHS integrals run to
          infinity); only the source integrals are windowed to x_win.
          State relations therefore use the full-range s_1/2, beta_A,
          phi^0_peak, while the source moments carry the windowed s^c_p.

          With sigma(x) = (sigbar/s_half) sqrt(x) and
          w = 28 u* sigbar cos(dtheta)/(g s_half),
            Gamma_E/Ecal = q (sigbar/s_half) I_E(w)/s_{1/2}
            Gamma_A/Acal = q (sigbar/s_half) I_A(w)
          with I_E = int x (w sqrt(x)-1)_+ phihat dx and I_A the same
          with x^{1/2}.  I_M / I_M,t are the corresponding MOMENTUM
          moments (one more factor x^{1/2}) over the peak component and
          over the tail, used by the stress bookkeeping.
          """
          sh = Shape()

          # PAPER CONVENTION: the pair (Acal, Ecal) is FULL RANGE, so all
          # state relations carry the full-range s_1/2 (paper Eq. 4) and
          # beta_A = s_1/2 s_-1/2 (Eq. 12); only the SOURCE moments are
          # windowed to (0, x_win] (the s^c_p of Eq. 28).
          self.s_half = sh.s(0.5)                       # 0.9488, full range
          self.beta_A = sh.s(0.5)*sh.s(-0.5)            # 1.0786
          if not hasattr(self, 'N_p'): self.N_p = sh.phi0   # 1.0383, Eq. 29

          xw  = np__.geomspace(1e-4, self.x_win, 4000)
          phw = sh.phihat(xw)
          sw  = lambda p: tr(xw**p*phw, xw)
          self.s_half_w = sw(0.5)              # diagnostic only
          # kernel tables for the Stokes profile (paper eq:stokes).  q_M is
          # the x^2 moment and q_A its antiderivative in zeta, which gives
          # the exact cell mean below.
          self._zeta_tab = sh.zeta.copy()
          self._qM_tab   = sh._qM.copy()
          self._qA_tab   = sh._qA.copy()
          # antiderivatives r_A, r_E for the exact cell integrals of the
          # ocean interface (interface.calc_doppler_vel); the q point
          # tables give the exact cell integrals of p_A, p_E via
          # zeta q_X at the faces (interface.calc_doppler_disp)
          self._rA_tab   = sh._rA.copy()
          self._rE_tab   = sh._rE.copy()
          self._qE_tab   = sh._qE.copy()
          # windowed to (0, x_win], paper eq:qMc.  q^c_A is the
          # antiderivative of q^c_M in zeta and gives the exact cell mean.
          xw2 = np__.geomspace(1e-4, self.x_win, 4000)
          ph2 = sh.phihat(xw2)
          ez  = np__.exp(self._zeta_tab[:,None]*xw2[None,:])
          self._qMc_tab = tr(xw2[None,:]**2*ph2[None,:]*ez, xw2, axis=1)
          self._qAc_tab = tr(xw2[None,:]   *ph2[None,:]*ez, xw2, axis=1)
          # x-grid of the tail integrals, paper eq:phitail
          self._xtail = np__.geomspace(self.x_win, 1e6, 400)
          self.s1_c     = sw(1.0)              # s^c_1 of Eq. (28)
          # dissipation downshift ratio, paper Eq. (27):
          #   d_out = s^c_{3/2} / (s_1/2 s^c_1) = 1.2116
          self.d_out  = sw(1.5)/(self.s_half*self.s1_c)
          # legacy c_out interface (old bare-3.0e4 parameterisation)
          if getattr(self, 'beta_o', None) is None:
              if hasattr(self, 'c_out'):          # legacy interface
                  self.beta_o = self.c_out*3.0e4*self.s_half**2/self.s1_c
              else:                               # tuned: PM landing (>= 4 d),
                  self.beta_o = 0.876             # clean directional law 0.20 Nn^5.4

          xg  = np__.geomspace(1e-3, self.x_win, 2000)
          ph  = sh.phihat(xg)
          wg  = np__.linspace(0.0, 12.0, 481)
          pos = np__.maximum(wg[:,None]*np__.sqrt(xg)[None,:] - 1.0, 0.0)
          self._wg = wg
          self._IE = tr(xg[None,:]*pos*ph[None,:], xg, axis=1)/self.s_half
          self._IA = tr(np__.sqrt(xg)[None,:]*pos*ph[None,:], xg, axis=1)
          self._IM = tr(xg[None,:]**1.5*pos*ph[None,:], xg, axis=1)
          return

      # ---------------------------------------------------- allocation
      def allocate(self):
          """Allocate the prognostic pair and the diagnostic stresses."""
          shp = (self.Nd, self.Ny, self.Nx)
          self.Ad = np__.zeros(shp)             # action    per direction
          self.Ed = np__.zeros(shp)             # energy    per direction
          self._satX    = np__.zeros((self.Ny, self.Nx))
          self.tau_o_g  = np__.zeros((self.Ny, self.Nx))
          self.tau_in_g = np__.zeros((self.Ny, self.Nx))
          # vector stresses (x first) and the momentum handed to the
          # mean flow by the adiabatic operators (paper eq:tauocU),
          # all kinematic m2/s2
          self.tau_o_v  = np__.zeros((2, self.Ny, self.Nx))
          self.tau_in_v = np__.zeros((2, self.Ny, self.Nx))
          self.Tmean_v  = np__.zeros((2, self.Ny, self.Nx))
          # energy handed to the mean flow per step by gamma_ex,
          # positive INTO the waves (m2/s, variance units per time)
          self.exch_g   = np__.zeros((self.Ny, self.Nx))
          self.tau_atm  = 0.0
          # open_coasts book-keeping: the action and energy that left the
          # domain through the coastal faces.  Accumulated LOCALLY every
          # step (each PE counts only faces in its own interior) and
          # summed over the PEs by coastal_loss(), so the diagnostic
          # costs no collective per time step.
          self.loss_A   = 0.0
          self.loss_E   = 0.0
          # momentum accumulators of the once-per-interval stress
          # (source.stress_couple): realized dM of the whole local cycle,
          # the input's realized share of it, and the elapsed time
          self._dMloc_x = np__.zeros((self.Ny, self.Nx))
          self._dMloc_y = np__.zeros((self.Ny, self.Nx))
          self._dMin_x  = np__.zeros((self.Ny, self.Nx))
          self._dMin_y  = np__.zeros((self.Ny, self.Nx))
          self._t_acc   = 0.0
          self.n_sub    = 1
          return

      def set_seed(self):
          """Small high-frequency sea as cold-start state."""
          sig = 2*np__.pi*self.seed_fp
          D = np__.cos(0.5*(self.theta[self.dsl] - np__.pi))**2
          D = D/(D.sum()*self.dth)
          self.Ed[self.dsl] = (self.seed_E*D)[:,None,None]
          self.Ad[self.dsl] = np__.maximum(self.Ed[self.dsl]/sig,
                                           self.Ad_floor)
          self.Ad = self.apply_bc(self.Ad)
          self.Ed = self.apply_bc(self.Ed)
          return

      # --------------------------------------------------- diagnostics
      def sigbar(self):
          """Per-direction mean frequency, floored where empty."""
          return np__.maximum(self.Ed, 1e-30) \
               / np__.maximum(self.Ad, self.Ad_floor)

      def sigbar_g(self):
          """Column mean frequency, shape (Ny,Nx)."""
          return np__.maximum(self.Ed[self.dsl].sum(0), 1e-30) \
               / np__.maximum(self.Ad[self.dsl].sum(0), self.Ad_floor)

      def e_mean(self):
          """Total wave energy per column, shape (ny_loc,nx_loc)."""
          return self.unpad(self.Ed[self.dsl].sum(0))*self.dth

      def a_mean(self):
          """Total wave action per column, shape (ny_loc,nx_loc)."""
          return self.unpad(self.Ad[self.dsl].sum(0))*self.dth

      # Depth to which the vertical integrals of the coupling are carried.
      # Every kernel decays as exp(2 kbar z), so the integrand is dead
      # below a few e-folding depths 1/(2 kbar) and the levels beneath
      # cost time without changing the answer.  z_cut is that depth in
      # METRES, positive; z_cut <= 0 integrates the full water column.
      #
      # MEASURED error of the truncation (uniform mean period, sheared
      # current, nz = 90, dz = 10 m; % change against full depth):
      #
      #        Tbar      z_cut     U_E   V_E  theta_dot  gamma  tau_work  u_S
      #        10 s      100 m    0.15  11.4     0.15     0.15    0.18    0.00
      #        10 s      200 m    0.00   0.29    0.00     0.00    0.004   0.00
      #        14 s      200 m    0.10   3.8     0.10     0.10    0.20    0.00
      #        18 s      200 m    0.60  10.5     0.60     0.60    1.34    0.00
      #
      # The Stokes drift is unaffected at any usable depth (it is carried
      # by the short waves), and so are the mean state parameters E and
      # sigbar.  V_E is the outlier by construction: its kernel p_E =
      # d(zeta q_E)/dzeta carries an extra factor zeta, peaks deeper and
      # decays more slowly, so it needs about twice the depth of the
      # others for the same accuracy.  The default 200 m holds every
      # sampled field except V_E under 1% up to a MEAN period of ~16-18 s
      # (an extreme Southern Ocean sea) and cuts 90 levels to 21.
      z_cut = 200.0

      def _z_cut_level(self, dz, nz):
          """Number of TOP ocean levels the vertical integrals need for
          the fixed cut depth z_cut. Returns nz for z_cut <= 0 or when the
          cut is deeper than the water column."""
          if self.z_cut <= 0.0:
              return nz
          return int(min(nz, np__.ceil(self.z_cut/dz) + 1))

      def calc_stokes(self, dz, nz, cell_mean = True):
          """Stokes drift profile, paper eq:stokes.

          SAME OCEAN INTERFACE AS wim_simple: name, signature and vertical
          ordering follow wim_simple.calc_stokes, so the coupling code can
          call either model identically.  The vertical index runs FROM THE
          BOTTOM (index 0 = deepest cell, index -1 = surface cell), the same
          convention as the current profiles UL/VL that both models take as
          input.  (This method replaces the removed stokes(), which returned
          the surface at index 0 -- the opposite ordering.  Any consumer of
          stokes() fails loudly here by design and needs the new name plus
          the flipped surface index.)

          u_S(z) = 2 g int dtheta  s kbar^2 Acal ( q^c_M + I_S )(2 kbar z)

          The spectrum is the shape below x_c and the wind-maintained
          spectrum of paper eq:phitail above, as for the momentum flux.
          The first part uses the windowed kernel q^c_M, the second the
          tail integral I_S over the two power laws.

          dz, nz define a uniform ocean grid of nz cells below z = 0.
          With cell_mean the cell average is returned instead of the point
          value.  The cell average of x^2 exp(2 kbar x z) is
          x [exp(zeta_t x) - exp(zeta_b x)]/(2 kbar dz), which is exact
          for both parts.  Point values lose most of the high wavenumber
          contribution on a coarse grid.

          The tail needs the wave age and the saturation ratio, so the
          model must have taken at least one time step.

          Returns (2, nz, ny_loc, nx_loc), unpadded, x component first.
          """
          # the kernel work below is the expensive part of the coupling
          # and is direction-resolved, so it runs on the REAL bins only
          sl = self.dsl
          kb = self.sigbar()[sl]**2/(self.grav*self.s_half**2)
          mc = self._z_cut_level(dz, nz)         # top levels that matter
          zt = -dz*np__.arange(mc)[:,None,None,None]            # (mc,1,1,1)
          zb = zt - dz
          if cell_mean:
              zf = -dz*np__.arange(mc + 1)[:,None,None,None]    # faces
              Q  = np__.interp(2*kb[None]*zf, self._zeta_tab,
                               self._qAc_tab, left=0.0)
              w  = (Q[:-1] - Q[1:])/(2*kb[None]*dz)
          else:
              w = np__.interp(2*kb[None]*(zt - 0.5*dz), self._zeta_tab,
                              self._qMc_tab, left=0.0)
          w = w + self._stokes_tail(kb, zt, zb, dz, cell_mean)
          amp = 2*self.grav*kb[None]*kb[None]*self.Ad[sl][None]*w
          us = np__.zeros((2, nz, self.ny_loc, self.nx_loc))
          us[0, :mc] = self.unpad((amp*self.sinth[None,sl,None,None]
                                   ).sum(1))*self.dth
          us[1, :mc] = self.unpad((amp*self.costh[None,sl,None,None]
                                   ).sum(1))*self.dth
          return us[:, ::-1]          # flip to bottom-up (wim convention)

      def set_topography(self):
          """Default: no land mask (all ocean). Override in an experiment
          class and set self.maskTp, as for wim_simple."""
          return

      def calc_stress(self):
          """Ocean stress vector, same interface as wim_simple.calc_stress.

          Returns (2, ny_loc, nx_loc), unpadded, kinematic (m2/s2), x
          component first. COMPUTES the stress for the elapsed coupling
          interval on call (source.stress_couple): the sink residual of
          the momentum accumulated in add_sources since the last call,
          plus the diagnostic-tail input eq:tautail evaluated once on the
          current state. The call cadence defines the coupling interval
          and resets the accumulators. Uses the forcing of the LAST
          time_step. Zero before the first time step.
          """
          if self._t_acc > 0.0 or getattr(self, '_last_ustar', None) is not None:
              self.stress_couple(self._last_ustar, self._last_udir)
          return self.unpad(self.tau_o_v)

      def coastal_loss(self):
          """Action and energy lost through open coasts since the start of
          the run, summed over all PEs.  Zero unless open_coasts is set.

          The local accumulators are added up here rather than every step:
          the sum is a collective, and the loss is a diagnostic that is
          read at the output cadence, not at the time step.
          """
          if self.n_pes <= 1:
              return float(self.loss_A), float(self.loss_E)
          # REDUCE OVER THIS MODEL'S OWN COMMUNICATOR, not COMM_WORLD.
          # pyOM3.global_sum uses the module-level mpi_comm = COMM_WORLD;
          # under the coupler the wave PEs are a SPLIT communicator, so a
          # COMM_WORLD reduction posted by the wave ranks alone deadlocks
          # against the ocean ranks that never join it.
          import numpy as np_
          from mpi4py import MPI as MPI_
          v = np_.array([self.loss_A, self.loss_E], dtype=float)
          out = np_.empty_like(v)
          self.comm.Allreduce(v, out, op=MPI_.SUM)
          return float(out[0]), float(out[1])

      def calc_phioc(self):
          """Energy flux from the waves to the ocean, same interface as
          wim_simple.calc_phioc: (ny_loc, nx_loc), m^3/s^3 (W/m^2 per rho_o),
          positive into the ocean. Booked as the source-cycle residual
          Phi_oc = P_in - dE/dt (add_sources), the energy sibling of the
          stress budget: the prognostic energy is full range, so the tail
          dissipation is inside the state and needs no extra term. Zero
          before the first time step.
          """
          if not hasattr(self, 'phi_g'):
              return np__.zeros((self.ny_loc, self.nx_loc))
          return self.grav*self.unpad(self.phi_g)

      def calc_wind_work(self):
          """Wind-to-wave energy input rate, (ny_loc, nx_loc), m^3/s^3,
          the P_in of the same booking as calc_phioc."""
          if not hasattr(self, 'win_g'):
              return np__.zeros((self.ny_loc, self.nx_loc))
          return self.grav*self.unpad(self.win_g)

      def calc_stress_work(self, dz, nz):
          """NOT AVAILABLE in wim_idemix: this model delivers the stress at
          the SURFACE only, through calc_stress().

          wim_simple can distribute the stress over depth because it
          resolves the spectrum: each frequency band deposits its momentum
          with its own decay scale exp(2 omega^2 z/g), so the vertical
          profile follows from the physics. wim_idemix carries a single
          kbar per direction and has no band structure to deposit with, so
          any depth profile would be an invented closure with no basis in
          the model -- and one that would silently set the depth over which
          the ocean receives wave momentum.

          Use transfer_mode = 'stress' (use_stress_work = False) in the
          experiment: the coupler then delivers the 2-D surface stress from
          calc_stress(), which the ocean applies as a surface boundary
          condition.
          """
          raise NotImplementedError(
              'wim_idemix delivers the stress at the surface only: use '
              'calc_stress() and transfer_mode = "stress". A depth '
              'distribution would need the spectral band structure that '
              'this model does not carry (see the docstring).')

      def _stokes_tail(self, kb, zt, zb, dz, cell_mean):
          """Tail part I_S of paper eq:stokes, spectrum of eq:phitail.

          Two implementations, switched by stokes_tail ('table' default,
          'quadrature' = the trapezoid over _xtail kept as the validation
          reference).  The table path uses the substitution u = zeta x in
          each branch of the two-power-law spectrum: with
          H2(t) = t int_t^inf u^-2 e^-u du (H2(0) = 1) and
          H25(t) = t^1.5 int_t^inf u^-2.5 e^-u du (H25(0) = 2/3),
          the depth integral collapses to
          S(zeta) = a [H2(zeta x_c)/x_c - H2(zeta x_s)/x_s]
                    + b H25(zeta x_s)/x_s^1.5,
          i.e. three interps of universal 1-D tables per face -- the same
          treatment every other interface kernel already gets.  The scaled
          forms are finite at zeta = 0, so the surface face is the exact
          analytic value, and the calm-cell limits (a = 0, x_s clamped
          huge) give exactly zero.  Table accuracy is measured against a
          converged 20000-node quadrature in the validation run and is
          BETTER than the 400-node trapezoid it replaces."""
          if getattr(self, 'stokes_tail', 'table') == 'quadrature':
              return self._stokes_tail_quad(kb, zt, zb, dz, cell_mean)
          if not hasattr(self, '_xi'):
              raise RuntimeError('stokes needs the wave age, take a time '
                                 'step first or set model._xi')
          if not hasattr(self, '_H2_tab'):
              self._tail_tables()
          a   = 0.5*self.alpha_e*self._xi                       # (Ny,Nx)
          b   = self.N_p/np__.maximum(self._satX, 1e-30)
          x_s = np__.maximum(np__.minimum(b/np__.maximum(a, 1e-30),
                                          1e15)**2, self.x_win)
          xc  = self.x_win
          tg  = self._Ht_grid
          def S_face(z):                                        # z <= 0 scalar
              zeta = 2.0*kb*(-z)                                # (nd,Ny,Nx)
              H2c = np__.interp(zeta*xc,  tg, self._H2_tab,  right=0.0)
              H2s = np__.interp(zeta*x_s, tg, self._H2_tab,  right=0.0)
              H25 = np__.interp(zeta*x_s, tg, self._H25_tab, right=0.0)
              return a*(H2c/xc - H2s/x_s) + b*H25/x_s**1.5
          nzl = zt.shape[0]
          out = np__.zeros((nzl,) + kb.shape)
          if cell_mean:
              Sp = S_face(0.0)
              for m_ in range(nzl):
                  Sn = S_face(-dz*(m_ + 1))
                  out[m_] = (Sp - Sn)/(2*kb*dz)
                  Sp = Sn
          else:
              for m_ in range(nzl):
                  # point value: integrand carries x^2, i.e. the u-forms
                  # H1(t) = int_t^inf u^-1 e^-u du + ln t (smooth) and
                  # H15(t) = t^0.5 int_t^inf u^-1.5 e^-u du
                  zeta = 2.0*kb*(dz*(m_ + 0.5))
                  F1c = np__.interp(zeta*xc,  self._Ht_grid, self._F1_tab,
                                    right=0.0)
                  F1s = np__.interp(zeta*x_s, self._Ht_grid, self._F1_tab,
                                    right=0.0)
                  H15 = np__.interp(zeta*x_s, self._Ht_grid, self._H15_tab,
                                    right=0.0)
                  out[m_] = a*(F1c - F1s + np__.log(x_s/xc)) \
                            + b*H15/np__.sqrt(x_s)
          return out

      def _tail_tables(self):
          """Universal 1-D tables for the table form of _stokes_tail,
          built once by fine cumulative quadrature (log grid, 2e5 points):
          H2 = t G2, H25 = t^1.5 G25 for the cell means; F1 = E1(t) + ln t
          and H15 = t^0.5 G15 for the point values.  All smooth with finite
          t -> 0 limits (1, 2/3, -gamma_E, 2) used as the left interp
          values via the t = 0 grid head."""
          u  = np__.geomspace(1e-8, 200.0, 200000)
          eu = np__.exp(-u)
          def cum_from_top(f):
              c = np__.zeros_like(f)
              df = 0.5*(f[1:] + f[:-1])*np__.diff(u)
              c[:-1] = df[::-1].cumsum()[::-1]
              return c
          G2  = cum_from_top(u**-2.0*eu)
          G25 = cum_from_top(u**-2.5*eu)
          G1  = cum_from_top(u**-1.0*eu)
          G15 = cum_from_top(u**-1.5*eu)
          gam = 0.5772156649015329
          self._Ht_grid = np__.concatenate(([0.0], u))
          self._H2_tab  = np__.concatenate(([1.0],     u**1.5*0 + u*G2))
          self._H25_tab = np__.concatenate(([2.0/3.0], u**1.5*G25))
          self._F1_tab  = np__.concatenate(([-gam],    G1 + np__.log(u)))
          self._H15_tab = np__.concatenate(([2.0],     np__.sqrt(u)*G15))
          return

      def _stokes_tail_quad(self, kb, zt, zb, dz, cell_mean):
          """Trapezoid-over-_xtail reference (the pre-table form)."""
          if not hasattr(self, '_xi'):
              raise RuntimeError('stokes needs the wave age, take a time '
                                 'step first or set model._xi')
          x   = self._xtail                                     # (nx_t,)
          lev = np__.minimum(0.5*self.alpha_e*self._xi[None,:,:]*x[:,None,None]**-3.0,
                             (self.N_p/np__.maximum(self._satX, 1e-30))[None,:,:]
                             *x[:,None,None]**-3.5)             # (nx_t,Ny,Nx)
          # Same trapezoid quadrature over self._xtail as the direct form,
          # restructured for production cost: exp(2 kbar x z) at the equally
          # spaced faces obeys the exact recursion E(z - dz) = E(z)*P with
          # P = exp(-2 kbar x dz), so ONE persistent (nx_t,nd,Ny,Nx) array
          # replaces a fresh build and two exp calls per level, and the x
          # integration is an einsum per face.  Agrees with the direct
          # evaluation to round-off (recursion drift ~ nz*eps).
          wtr = np__.zeros_like(x)                              # trapezoid
          wtr[1:-1] = 0.5*(x[2:] - x[:-2])                      # weights
          wtr[0]    = 0.5*(x[1] - x[0])
          wtr[-1]   = 0.5*(x[-1] - x[-2])
          nzl = zt.shape[0]
          out = np__.zeros((nzl,) + kb.shape)
          P   = np__.exp(-2.0*kb[None]*x[:,None,None,None]*dz)
          if cell_mean:
              C = wtr[:,None,None]*lev*x[:,None,None]           # (nx_t,Ny,Nx)
              E = np__.ones(P.shape)                            # face z = 0
              S_prev = np__.einsum('xji,xdji->dji', C, E)
              for m_ in range(nzl):
                  E *= P
                  S_next = np__.einsum('xji,xdji->dji', C, E)
                  out[m_] = (S_prev - S_next)/(2*kb*dz)
                  S_prev = S_next
          else:
              C = wtr[:,None,None]*lev*x[:,None,None]**2
              E = np__.exp(-2.0*kb[None]*x[:,None,None,None]*(0.5*dz))
              for m_ in range(nzl):
                  out[m_] = np__.einsum('xji,xdji->dji', C, E)
                  E *= P
          return out

      def mean_direction(self):
          """Energy-weighted mean direction, shape (ny_loc,nx_loc)."""
          sl = self.dsl
          s = (self.Ed[sl]*self.s_x[sl]).sum(0)
          c = (self.Ed[sl]*self.s_y[sl]).sum(0)
          return self.unpad(np__.arctan2(s, c))
