"""
TKE closure (Gaspar et al. 1990) for pe_realistic, ported from pyOM2
for_src/tke/tke.f90 (Eden 2016, Ocean Modelling 101).

Prognostic turbulent kinetic energy tke on the W grid (top faces, like
kappaH in implicit_vmix.py; pe_realistic orientation k=halo bottom .. -halo-1
surface face). Full pyOM2 dzt/dzw bookkeeping on the variable grid: the
TKE cell thickness is dzw (W grid), the spacing between W points k and k+1
is dzt(k+1), and the surface W cell is HALF-THICK (0.5*dzw(ke)), exactly
as in tke.f90.

Provides (mixin class `tke`, add to the model class bases like the other
pe_realistic components):
  init_tke()                allocate fields, set constants
  set_tke_diffusivities()   sqrttke, mixing length (mxl_choice, see below),
                            kappaM/kappaH with Ri-dependent Prandtl number,
  integrate_tke()           implicit TKE step: shear production K_diss_v,
                            buoyancy destruction P_diss_v, linearised
                            dissipation, vertical diffusion alpha_tke*kappaM,
                            surface flux BC (phi_oc from the wave model or
                            the classical (|tau|)^{3/2}), negative-surface-
                            TKE correction
  implicit_vfric()          backward-Euler vertical friction with kappaM
                            (removes the explicit A_v stability ceiling) and
                            the energetically consistent K_diss_v diagnosis

Intended call order per ocean time step (WP3 wiring):
    self.set_tke_diffusivities()
    self.implicit_vfric()          # friction with kappaM, diagnoses K_diss_v
                                   # (u_L models: set mix_eulerian=True to mix
                                   #  u_bar = u_L - u_S per paper Sec. 5)
    self.integrate_tke()           # TKE step using K_diss_v, P_diss_v
    # tracer vmix then uses self.kappaH_tke instead of convective_kappa

Energetic consistency: K_diss_v is diagnosed from the SAME discrete operator
that removes momentum (flux x shear of the updated velocities, Eden 2016
Appendix A), and P_diss_v = kappaH*N^2 is the exact transfer of the implicit
tracer mixing, so the TKE budget closes against the mean-flow budgets.
"""
from functools import partial
import pyOM3 as OM


class tke:

    # ------------------------------------------------------------------
    def init_tke(self):
        """Allocate TKE fields and set the closure constants (pyOM2 values)."""
        # all constants are configuration-settable (set the attribute before
        # init_tke); defaults are the pyOM2 tke_module defaults. NOTE: pyOM2
        # configurations of the acc family set alpha_tke = 30.0 (TKE
        # pressure-work transport) and mxl_min = 1e-8; the module default
        # alpha_tke = 1.0 confines TKE much closer to the surface.
        self.c_k        = getattr(self, 'c_k',        0.1)
        self.c_eps      = getattr(self, 'c_eps',      0.7)
        self.alpha_tke  = getattr(self, 'alpha_tke',  1.0)
        self.mxl_min    = getattr(self, 'mxl_min',    1e-12)
        self.kappaM_min = getattr(self, 'kappaM_min', 0.0)    # pyOM2 default 0
        self.kappaM_max = getattr(self, 'kappaM_max', 100.0)
        self.kappaH_min = getattr(self, 'kappaH_min', 0.0)    # pyOM2: no floor; a
                                                               # background diffusivity is a configuration choice
        self.tke_min    = getattr(self, 'tke_min',    0.0)   # pyOM2: no floor
        # wave-scale surface mixing length (Mellor & Blumberg 2004 type):
        # with mxl_choice = 1 the value at the surface W face is limited to
        # mxl_min + z0_surf + dz(top) (after the downward sweep, which uses
        # the unlimited surface value, and before the upward sweep; the
        # tke.f90 order). With mxl_choice = 2 it enters the surface length
        # l0 = max(mxl_min0, kappa z0_surf). Set z0_surf (scalar or (Ny,Nx),
        # e.g. 1.6*Hs from the wave model, updated every coupling step) when
        # injecting Craig-Banner-scale surface TKE fluxes; default 0 keeps
        # the original pyOM2 behaviour. Note the pyOM2 choice-2 sweeps grow
        # the length 1:1 with distance (no von Karman factor), so beta
        # absorbs kappa_vk.
        self.z0_surf    = getattr(self, 'z0_surf',    0.0)
        # MIXING LENGTH: which set of limits is applied to sqrt(2 tke)/N.
        #   mxl_choice = 1 (default)  pyOM2 tke.f90 tke_mxl_choice = 2, the
        #       MITgcm/OPA form: downward sweep with the UNLIMITED surface
        #       value, then the surface W point limited to
        #       mxl_min + z0_surf + dz(top), then the upward sweep. One length
        #       for both the diffusivity and the dissipation.
        #   mxl_choice = 2  NEMO 4.2 nn_mxl = 3 with ln_mxl0 = .true.: the
        #       surface value is set FIRST, so it bounds the whole
        #       near-surface profile, and the two sweeps are kept apart,
        #           l_up   limited by the distance to the surface,
        #           l_dwn  limited by the distance to the bottom,
        #           l_k   = sqrt(l_up l_dwn)   (diffusivity, Gaspar 1990)
        #           l_eps = min(l_up, l_dwn)   (dissipation)
        #       Surface length: Charnock, l0 = kappa beta |tau|/(g rho_0)
        #       (Mellor & Blumberg 2004; NEMO uses beta = 2e5 after Stacey
        #       1999, the default here is 2e4 as in the les1 note, l0 = 0.08 m
        #       at 0.1 Pa), floored
        #       at mxl_min0 = kappa z0 with z0 = 0.1 m (NEMO rn_mxl0 = 0.04 m).
        #       With z0_surf > 0 (a wave model's roughness, NEMO ln_mxhsw:
        #       z0 = 1.6 Hs) it is l0 = max(mxl_min0, kappa z0_surf) instead.
        #       The lowest wet level is seeded with mxl_min0 (NEMO's bottom
        #       condition).
        self.mxl_choice = getattr(self, 'mxl_choice', 1)
        self.mxl_kappa  = getattr(self, 'mxl_kappa',  0.4)     # von Karman
        self.mxl_beta   = getattr(self, 'mxl_beta',   2.0e4)   # Charnock (les1 note; NEMO 2e5)
        self.mxl_min0   = getattr(self, 'mxl_min0',   0.04)    # NEMO rn_mxl0
        # surface TKE flux source: wave dissipation (phi_oc) or the
        # Craig-Banner flux alpha_cb (|tau|)^{3/2} (tau kinematic);
        # alpha_cb = 1 is the pyOM2 boundary condition, ~100 Craig & Banner (1994)
        self.tke_surf_from_waves = getattr(self, 'tke_surf_from_waves', True)
        self.alpha_cb            = getattr(self, 'alpha_cb', 1.0)

        z = OM.np.zeros((self.Nz, self.Ny, self.Nx), OM.prec)
        # initial seed as the pyOM2 configurations (tke = 1d-12); the floor
        # tke_min is a separate thing and 0 by default
        self.tke        = z + getattr(self, 'tke_init', 1e-12) * self.maskW
        self.kappaM_tke = z + self.kappaM_min
        self.kappaH_tke = z + self.kappaH_min
        self.K_diss_v   = z * 1.0
        self.P_diss_v   = z * 1.0
        self.tke_diss   = z * 1.0
        self.forc_tke_surface = OM.np.zeros((self.Ny, self.Nx), OM.prec)
        self.tke_surf_corr    = OM.np.zeros((self.Ny, self.Nx), OM.prec)
        return

    # ------------------------------------------------------------------
    @partial(OM.jaxjit, static_argnums=0)
    def _mxl_choice2(self, sqrttke, nsqr):
        """Mixing length, tke_mxl_choice=2 of tke.f90: mxl = sqrt(2*tke/N^2),
        then limited so it cannot grow faster than the distance to the
        boundaries (downward and upward sweeps), floored at mxl_min."""
        h, Nz = self.halo, self.Nz
        dzt = self.dzt                                    # 1-D, dynamic index ok
        mxl = OM.np.sqrt(2.0) * sqrttke \
              / OM.np.sqrt(OM.np.maximum(1e-12, nsqr)) * self.maskW

        # downward sweep (from below the surface face):
        # mxl[k] <= mxl[k+1]+dzt[k+1]   (tke.f90 line 41; distance between
        # W points k and k+1 is the T-cell thickness dzt[k+1])
        ke = Nz - h - 1                                   # surface W face
        # ORDER AS tke.f90: downward sweep first (using the UNLIMITED surface
        # value), then the surface limit, then the upward sweep
        def _dn(i, m):
            k = ke - 1 - i                                # ke-1 .. h
            return OM.modify_array(m, (k, slice(None), slice(None)),
                                   OM.np.minimum(m[k], m[k + 1] + dzt[k + 1]))
        mxl = OM.for_loop(0, ke - h, _dn, mxl)
        mxl = OM.modify_array(mxl, (ke, slice(None), slice(None)),
                              OM.np.minimum(mxl[ke],
                                            self.mxl_min + self.z0_surf + dzt[ke]))

        # bottom condition: the lowest W point of the array (k = h, the top
        # face of cell h) is one cell above the floor, like the lowest wet W
        # point of every shallower column (where the dry point below is 0 and
        # the sweep gives <= dzt[h]). Without this the deepest columns had an
        # unlimited bottom length.
        mxl = OM.modify_array(mxl, (h, slice(None), slice(None)),
                              OM.np.minimum(mxl[h], dzt[h]))
        # upward sweep: mxl[k] <= mxl[k-1]+dzt[k]  (tke.f90 line 45)
        def _up(k, m):
            return OM.modify_array(m, (k, slice(None), slice(None)),
                                   OM.np.minimum(m[k], m[k - 1] + dzt[k]))
        mxl = OM.for_loop(h + 1, ke + 1, _up, mxl)

        return OM.np.maximum(mxl, self.mxl_min)

    # ------------------------------------------------------------------
    def surface_mxl(self):
        """Surface mixing length l0 of mxl_choice = 2 [m], on the T grid.
        Charnock (NEMO ln_mxl0): l0 = kappa beta |tau|/(g rho_0), with tau
        the KINEMATIC surface stress taux/tauy the configuration applies
        (|tau|/rho_0), averaged U/V -> T. With z0_surf > 0 the wave form
        l0 = kappa z0_surf is used instead. Always at least mxl_min0."""
        z0 = self.z0_surf
        if not OM.np.isscalar(z0) or z0 > 0.0:
            return OM.np.maximum(self.mxl_min0, self.mxl_kappa*z0) \
                   + OM.np.zeros((self.Ny, self.Nx))
        tau = getattr(self, 'tau_modulus', None)
        if tau is None and hasattr(self, 'taux'):
            # |tau| WITHOUT a U/V -> T average, exactly as _surface_forcing
            # builds the surface TKE flux: the average would read the halo of
            # a field the configuration sets without a halo exchange, which
            # made the run depend on the PE layout. A configuration that has
            # the T-point modulus can hand it over as `tau_modulus`.
            tau = OM.np.hypot(self.taux, self.tauy)
        if tau is not None:
            l0 = self.mxl_kappa*self.mxl_beta*tau/self.grav
            l0 = OM.np.maximum(self.mxl_min0, l0)
            # The HALO of l0 must be right: mxl of the halo row enters the
            # U/V -> T averages of implicit_vfric, and the configuration sets
            # taux/tauy without a halo exchange, so the run would depend on
            # the PE layout. One exchange per step is avoided by taking the
            # halo from the stress the configuration exchanged itself: set
            # tau_halo_ok = True when taux/tauy carry a valid halo (as
            # set_forcing leaves them), then no apply_bc is needed here.
            if not getattr(self, 'tau_halo_ok', False):
                l0 = self.apply_bc(l0)
            return l0
        return self.mxl_min0 + OM.np.zeros((self.Ny, self.Nx))

    @partial(OM.jaxjit, static_argnums=0)
    def _mxl_nemo(self, sqrttke, nsqr, l0):
        """Mixing length, NEMO 4.2 nn_mxl = 3 with ln_mxl0 = .true.
        Returns (l_k, l_eps): the geometric mean and the minimum of the
        surface- and bottom-limited lengths. See init_tke for the reference."""
        h, Nz = self.halo, self.Nz
        dzt = self.dzt                                    # 1-D, dynamic index ok
        ke = Nz - h - 1                                   # surface W face
        mxl = OM.np.sqrt(2.0) * sqrttke \
              / OM.np.sqrt(OM.np.maximum(1e-12, nsqr))
        # floor everywhere, DRY CELLS INCLUDED: the bottom sweep then starts
        # from mxl_min0 below the deepest wet level, which is NEMO's bottom
        # condition l(bottom) = rn_mxl0
        mxl = OM.np.maximum(mxl, self.mxl_min0)
        # SURFACE FIRST (this is the difference to mxl_choice = 1): l0 bounds
        # the whole near-surface profile through the sweep below
        mxl = OM.modify_array(mxl, (ke, slice(None), slice(None)),
                              OM.np.minimum(mxl[ke], l0))

        # l_up: limited by the distance to the surface, downward from ke
        def _dn(i, m):
            k = ke - 1 - i                                # ke-1 .. h
            return OM.modify_array(m, (k, slice(None), slice(None)),
                                   OM.np.minimum(m[k], m[k + 1] + dzt[k + 1]))
        l_up = OM.for_loop(0, ke - h, _dn, mxl * 1.0)
        # l_dwn: limited by the distance to the bottom, upward from h. Bottom
        # condition at k = h as in every shallower column (dry point below
        # = mxl_min0, so the lowest wet W point gets <= mxl_min0 + dzt[h]);
        # without this the deepest columns had an unlimited bottom length.
        def _up(k, m):
            return OM.modify_array(m, (k, slice(None), slice(None)),
                                   OM.np.minimum(m[k], m[k - 1] + dzt[k]))
        l_dwn = OM.modify_array(mxl * 1.0, (h, slice(None), slice(None)),
                                OM.np.minimum(mxl[h], self.mxl_min0 + dzt[h]))
        l_dwn = OM.for_loop(h + 1, ke + 1, _up, l_dwn)

        l_k   = OM.np.sqrt(l_up * l_dwn) * self.maskW
        l_eps = OM.np.minimum(l_up, l_dwn) * self.maskW
        fl = OM.np.maximum(self.mxl_min, 1e-30)
        return (OM.np.maximum(l_k, fl), OM.np.maximum(l_eps, fl))

    # ------------------------------------------------------------------
    def set_tke_diffusivities(self):
        """kappaM = min(kappaM_max, c_k*mxl*sqrt(tke)), kappaH = kappaM/Pr(Ri),
        kappaM_min/kappaH_min floors, zero flux through the surface face."""
        sqrttke = OM.np.sqrt(OM.np.maximum(0.0, self.tke))
        nsqr    = self.calc_nsqr(self.temp, self.salt, self.rho, self.maskW)
        if self.mxl_choice == 2:
            mxl, mxl_eps = self._mxl_nemo(sqrttke, nsqr, self.surface_mxl())
        elif self.mxl_choice == 1:
            mxl = self._mxl_choice2(sqrttke, nsqr)
            mxl_eps = mxl
        else:
            raise ValueError('mxl_choice must be 1 (pyOM2) or 2 (NEMO), got %r'
                             % (self.mxl_choice,))

        kappaM = OM.np.minimum(self.kappaM_max, self.c_k * mxl * sqrttke)
        Ri     = nsqr / OM.np.maximum(1e-12,
                    self.K_diss_v / OM.np.maximum(1e-12, kappaM))
        Pr     = OM.np.maximum(1.0, OM.np.minimum(10.0, 6.6 * Ri))
        kappaH = kappaM / Pr
        kappaM = OM.np.maximum(self.kappaM_min, kappaM)
        kappaH = OM.np.maximum(self.kappaH_min, kappaH)
        # NO extra convective branch: with the TKE closure on, unstable
        # columns get Pr = 1 (Ri < 0), kappaH = kappaM, exactly as pyOM2's
        # tke.f90. A kappaH = 1 override where N^2 < 0 was here until
        # Sep 8 2026 and was a bug (convective adjustment on top of TKE).
        # no diffusive flux through the surface face (cf. convective_kappa)
        ks = -self.halo - 1
        kappaH = OM.modify_array(kappaH, (ks, slice(None), slice(None)),
                                 OM.np.zeros_like(kappaH[ks]))
        self.kappaM_tke = kappaM * self.maskW
        self.kappaH_tke = kappaH * self.maskW
        # mxl is the length of the DIFFUSIVITY (l_k), mxl_eps that of the
        # DISSIPATION (l_eps); they differ only for mxl_choice = 2
        self.mxl, self.mxl_eps = mxl, mxl_eps
        self.sqrttke, self.Nsqr = sqrttke, nsqr
        return

    # ------------------------------------------------------------------
    def implicit_vfric(self):
        """Backward-Euler vertical friction with viscosity kappaM: removes
        the explicit A_v stability limit. Follows pyOM2
        vertical_friction.f90 (implicit_vert_friction) and the discrete
        dissipation identity of the pyOM2 documentation:
          - kappaM is averaged T->U (T->V) before building the friction
            operator (pyOM2: fxa = 0.5*(kappaM(i,j,k)+kappaM(i+1,j,k))),
          - K_diss_v is flux x shear of the UPDATED (taup1) velocities at
            the W faces, horizontally averaged U/V -> T,
        so the mean-flow KE the operator removes reappears in the TKE
        forcing without discretisation residual.

        u_L formulation (attribute mix_eulerian + u_S present): the
        operator acts on the EULERIAN velocity u_bar = u_L - u_S (paper
        Sec. 5: the closure parameterises Eulerian u'u'), and the booking
        becomes flux x LAGRANGIAN shear, split into K_diss_E (Eulerian
        shear production, >= 0) and K_diss_L (Stokes/Langmuir production,
        sign-indefinite); K_diss_v = K_diss_E + K_diss_L."""
        kmu = 0.5 * (self.kappaM_tke + self.rollx(self.kappaM_tke, -1))  # T->U
        kmv = 0.5 * (self.kappaM_tke + self.rolly(self.kappaM_tke, -1))  # T->V
        # a momentum flux exists only across a face BETWEEN TWO WET U (V)
        # cells (pyOM2 vertical_friction.f90: flux_top * maskU(k+1)*maskU(k)).
        # Without this, a U column shallower than its deeper T neighbour had
        # a nonzero kmu at its bottom face: the solver dragged the lowest wet
        # cell against the dry cell below (spurious bottom friction on every
        # topographic step) and the dissipation booked at that face fell half
        # into a dry T cell and was lost (5-7% of K_diss_v in global4).
        kmu = kmu * self.maskU * self.rollz(self.maskU, -1)
        kmv = kmv * self.maskV * self.rollz(self.maskV, -1)
        # NO momentum flux through the surface face: pe_realistic's maskW is 1
        # there, so without this the implicit solve couples the top cell to
        # the zero-velocity z-halo above -- a phantom no-slip lid with drag
        # rate kappaM_surf/dz^2 (25x worse at dz=10 than dz=50). The wind
        # stress enters as an explicit surface flux in the driver instead.
        # This also removes the spurious K_diss shear production against
        # that ghost. (kappaH is zeroed the same way in
        # set_tke_diffusivities.)
        ks = -self.halo - 1
        kmu = OM.modify_array(kmu, (ks, slice(None), slice(None)),
                              OM.np.zeros_like(kmu[ks]))
        kmv = OM.modify_array(kmv, (ks, slice(None), slice(None)),
                              OM.np.zeros_like(kmv[ks]))
        # --- mix the EULERIAN mean velocity (paper Sec. 5) ----------------
        # The down-gradient closure parameterises u'u' with EULERIAN
        # fluctuations, so the friction term is dz(c_u K dz u_bar), NOT
        # dz(c_u K dz u_L): mixing u_L would spuriously diffuse the Stokes
        # shear. With mix_eulerian set and u_S present (u_L formulation),
        # subtract u_S (piecewise constant over the step) before the
        # implicit solve and add it back after -- this treats exactly
        # dz K dz u_bar backward-Euler, applied as an increment to u_L.
        if getattr(self, 'mix_eulerian', False) and hasattr(self, 'u_S'):
            uS_U = 0.5 * (self.u_S + self.rollx(self.u_S, -1)) * self.maskU
            vS_V = 0.5 * (self.v_S + self.rolly(self.v_S, -1)) * self.maskV
        else:
            uS_U = vS_V = None
        ue = self.u if uS_U is None else self.u - uS_U
        ve = self.v if vS_V is None else self.v - vS_V
        ue_new = self.implicit_vdiff(ue, kmu, self.maskU)
        ve_new = self.implicit_vdiff(ve, kmv, self.maskV)
        u_new = ue_new if uS_U is None else ue_new + uS_U
        v_new = ve_new if vS_V is None else ve_new + vS_V
        # --- energetically consistent K_diss_v (paper Eqs. 21/22) ---------
        # flux is K dz(u_bar_new); the KE it removes is against the MODEL
        # velocity u_L, so the booking is flux x LAGRANGIAN shear:
        #   K dz(u_bar) . dz(u_L) = P_E + P_L,
        #   P_E = K |dz u_bar|^2 >= 0        (Eulerian shear production)
        #   P_L = K dz(u_bar).dz(u_S)        (Stokes/Langmuir production,
        #                                     sign-indefinite)
        # kept separately in K_diss_E / K_diss_L for the TKE budget
        # diagnostics; K_diss_v = P_E + P_L feeds integrate_tke unchanged.
        # Without mix_eulerian this reduces bit-identically to the old
        # booking (P_L = 0, dz u_bar = dz u_model).
        shu = (self.rollz(ue_new, -1) - ue_new) / self.dzw_z  # Eulerian du/dz at W
        shv = (self.rollz(ve_new, -1) - ve_new) / self.dzw_z
        # U/V -> T with the METRIC WEIGHTS (pyOM2 ugrid_to_tgrid /
        # vgrid_to_tgrid): the dissipation is a volume density on U/V cells, so
        # the T-cell value is the area-weighted mean of its two neighbours,
        # sum_T K_diss*area_t = sum_U pu*area_u + sum_V pv*area_v exactly. A
        # plain 0.5*(p[i]+p[i-1]) (here until Sep 8 2026) is off by cosu/cost
        # in y on the sphere, 6% of K_diss_v at 60 deg latitude.
        def u2t(p): return 0.5 * (p * self.area_u + self.rollx(p * self.area_u, 1)) / self.area_t
        def v2t(p): return 0.5 * (p * self.area_v + self.rolly(p * self.area_v, 1)) / self.area_t
        pu  = kmu * shu * shu * self.maskW                 # flux x Eul. shear
        pv  = kmv * shv * shv * self.maskW
        KdE = u2t(pu) + v2t(pv)
        self.K_diss_E = self.apply_bc(KdE * self.maskW)
        if uS_U is not None:
            ssu = (self.rollz(uS_U, -1) - uS_U) / self.dzw_z  # Stokes shear at W
            ssv = (self.rollz(vS_V, -1) - vS_V) / self.dzw_z
            pu  = kmu * shu * ssu * self.maskW             # flux x Stokes shear
            pv  = kmv * shv * ssv * self.maskW
            KdL = u2t(pu) + v2t(pv)
            self.K_diss_L = self.apply_bc(KdL * self.maskW)
        else:
            self.K_diss_L = OM.np.zeros_like(self.K_diss_E)
        self.K_diss_v = self.K_diss_E + self.K_diss_L
        # halo refresh: implicit_vdiff returns ue_new with fresh halos, but
        # the u_S part added back (uS_U from rollx of the haloed u_S) is not
        # valid in the outermost halo column, and the next step's advection,
        # w and tendencies read u there. Found by the coupled global4 restart
        # test: identical A/B state except this halo -> 6e-5 in u after one
        # step (the restart's apply_bc repaired what the run had wrong).
        self.u, self.v = self.apply_bc(u_new), self.apply_bc(v_new)
        return

    # ------------------------------------------------------------------
    def _surface_forcing(self):
        """Surface TKE flux [m^3/s^3]: wave dissipation flux phi_oc from wim
        (WP2), or the Craig-Banner flux alpha_cb (|tau|)^{3/2} (tau kinematic)."""
        if self.tke_surf_from_waves and hasattr(self, 'phi_oc'):
            return OM.np.maximum(0.0, self.phi_oc)
        if hasattr(self, 'taux'):
            return self.alpha_cb * OM.np.hypot(self.taux, self.tauy) ** 1.5
        return self.forc_tke_surface * 0.0

    # ------------------------------------------------------------------
    @partial(OM.jaxjit, static_argnums=0)
    def _tke_solve(self, tke, forc, forc_surf, sqrttke, mxl_eps, kappaM):
        """One backward-Euler TKE step: general Thomas solve, coefficients
        exactly as tke.f90 (uniform dz; surface W cell 0.5*dz thick)."""
        h, Nz, dt = self.halo, self.Nz, self.dt
        ke = Nz - h - 1                                    # surface W face

        # TKE diffusivity on the faces BETWEEN W cells (T levels):
        # delta[k] couples W cells k and k+1 across T level k+1
        # (tke.f90: delta(k) = dt/dzt(k+1)*alpha_tke*0.5*(kappaM_k+kappaM_k+1))
        dztp1 = OM.np.roll(self.dzt, -1)[:, None, None]    # dzt[k+1] at k
        delta = dt / dztp1 * self.alpha_tke * 0.5 * (kappaM + self.rollz(kappaM, -1))
        delta = OM.modify_array(delta, (ke, slice(None), slice(None)),
                                OM.np.zeros_like(delta[ke]))
        # TKE diffuses only between two WET W levels (pyOM2: a_tri(ks) = 0,
        # delta defined for k = ks..ke-1). Without this the lowest wet level
        # exchanged TKE with the dry level below it (held at zero): a leak of
        # TKE into the ground at every column bottom (found Sep 9 2026 by the
        # TKE budget identity).
        delta = delta * self.maskW * self.rollz(self.maskW, -1)

        diss = dt * self.c_eps * sqrttke / mxl_eps         # linearised sink
        # TKE cell thickness is dzw (W grid; tke.f90: a_tri = -delta(k-1)/dzw(k))
        a = -self.rollz(delta, 1) / self.dzw_z             # couples k, k-1
        c = -delta / self.dzw_z                            # couples k, k+1
        b = 1.0 + delta / self.dzw_z + self.rollz(delta, 1) / self.dzw_z + diss
        # surface W cell: half thickness 0.5*dzw(ke) (tke.f90 ke rows)
        a = OM.modify_array(a, (ke, slice(None), slice(None)),
                            -self.rollz(delta, 1)[ke] / (0.5 * self.dzw[ke]))
        b = OM.modify_array(b, (ke, slice(None), slice(None)),
                            1.0 + self.rollz(delta, 1)[ke] / (0.5 * self.dzw[ke]) + diss[ke])
        d = tke + dt * forc
        d = OM.modify_array(d, (ke, slice(None), slice(None)),
                            d[ke] + dt * forc_surf / (0.5 * self.dzw[ke]))

        m = self.maskW
        a = a * m
        c = c * m
        b = OM.np.where(m > 0.0, b, 1.0)

        # Thomas along axis 0 (same structure as implicit_vdiff)
        cp = OM.np.zeros_like(tke)
        dp = OM.np.zeros_like(tke)
        cp = OM.modify_array(cp, (h, slice(None), slice(None)), c[h] / b[h])
        dp = OM.modify_array(dp, (h, slice(None), slice(None)), d[h] / b[h])
        def _fwd(k, st):
            cp, dp = st
            mm = b[k] - a[k] * cp[k - 1]
            cp = OM.modify_array(cp, (k, slice(None), slice(None)), c[k] / mm)
            dp = OM.modify_array(dp, (k, slice(None), slice(None)),
                                 (d[k] - a[k] * dp[k - 1]) / mm)
            return (cp, dp)
        cp, dp = OM.for_loop(h + 1, Nz - h, _fwd, (cp, dp))
        out = tke * 1.0
        out = OM.modify_array(out, (Nz - h - 1, slice(None), slice(None)),
                              dp[Nz - h - 1])
        def _bwd(i, o):
            k = Nz - h - 2 - i
            return OM.modify_array(o, (k, slice(None), slice(None)),
                                   dp[k] - cp[k] * o[k + 1])
        out = OM.for_loop(0, (Nz - 2 * h) - 1, _bwd, out)
        return out * m + tke * (1.0 - m)

    # ------------------------------------------------------------------
    def integrate_tke(self):
        """TKE step: forc = K_diss_v - P_diss_v, implicit dissipation and
        diffusion, surface flux BC, negative-surface-TKE correction."""
        # buoyancy destruction: exact transfer of the implicit tracer mixing.
        # N^2 must be evaluated from the UPDATED buoyancy (the tracer solve
        # runs before this in the time step): backward-Euler diffusion does
        # its PE work against the relaxing gradients, and the exact discrete
        # transfer is kappaH * N^2(b^{n+1}) -- the same taup1 rule as K_diss_v.
        # Using the pre-mixing N^2 overdrains the TKE whenever
        # kappaH*dt/dz^2 is not small (alpha_tke=30 regimes).
        # self.rho is the density of the UPDATED tracers: model.py evaluates it
        # once per step, after the vertical mixing and before this call
        # (pyOM2: calc_eq_of_state(taup1) before the P_diss_v block).
        from pyOM3.density import gsw_drhodTS, \
            linear_eq_of_state_drhodT, linear_eq_of_state_drhodS
        nsqr_new = self.calc_nsqr(self.temp, self.salt, self.rho, self.maskW)
        P = self.kappaH_tke * nsqr_new * self.maskW
        # SURFACE BUOYANCY FLUX (pyOM2 thermodynamics.f90, both branches):
        #   P_diss_v(nz) = -g/rho_0 * forc_rho_surface,
        #   forc_rho_surface = drho/dT * forc_temp_surface + drho/dS * forc_salt_surface
        # with the surface tracer FLUXES (K m/s, psu m/s, positive into the
        # ocean) the configuration provides as forc_temp_surface /
        # forc_salt_surface. Surface cooling (buoyancy loss) is thus a TKE
        # source: this is the convective TKE production. Without the two
        # attributes the term is zero (as it was here before Sep 8 2026).
        if hasattr(self, 'forc_temp_surface'):
            ks = -self.halo - 1
            eos = self.eq_of_state
            if eos == 0:                       # temp IS buoyancy b, rho = -b*rho_0/g
                drT, drS = -1024.0/9.81, 0.0
            elif eos == 1:
                drT, drS = linear_eq_of_state_drhodT(), linear_eq_of_state_drhodS()
            elif eos == 5:
                p_s = None if self.p0 is None else self.p0[ks]
                drT, drS = gsw_drhodTS(self.salt[ks], self.temp[ks], p_s)
            else:
                drT, drS = 0.0, 0.0
            frho = drT * self.forc_temp_surface
            if hasattr(self, 'forc_salt_surface'):
                frho = frho + drS * self.forc_salt_surface
            P = OM.modify_array(P, (ks, slice(None), slice(None)),
                                -self.grav / self.rho_0 * frho * self.maskT[ks])
        self.P_diss_v = P
        forc = self.K_diss_v - self.P_diss_v
        self.forc_tke_surface = self._surface_forcing()

        tke_new = self._tke_solve(self.tke, forc, self.forc_tke_surface,
                                  self.sqrttke, self.mxl_eps, self.kappaM_tke)

        # dissipation diagnostic (with the UPDATED tke, as in tke.f90)
        self.tke_diss = self.c_eps / self.mxl_eps * self.sqrttke * tke_new * self.maskW

        # surface correction: if the surface cell went negative (strong
        # buoyancy drain), add the missing TKE and book it (tke.f90)
        ke = -self.halo - 1
        neg = OM.np.minimum(tke_new[ke], 0.0)
        self.tke_surf_corr = -neg * (0.5 * self.dzw[ke]) / self.dt
        tke_new = OM.modify_array(tke_new, (ke, slice(None), slice(None)),
                                  OM.np.maximum(tke_new[ke], 0.0))

        # tke.f90 has NO floor (only the negative surface value is zeroed
        # above); a floor is applied only if tke_min > 0 is asked for
        if self.tke_min > 0.0:
            tke_new = OM.np.maximum(tke_new, self.tke_min * self.maskW)
        self.tke = self.apply_bc(tke_new)
        return
