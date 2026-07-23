"""
Implicit vertical diffusion of tracers with simple convective adjustment for
pe_simple, after pyOM2 (tke.f90: `where (Nsqr<0) kappaH=1`; thermodynamics.f90:
backward-Euler tridiagonal vmix solve).

Implicit is required because convective adjustment sets kappaH = O(1) m^2/s,
which violates the explicit limit kappaH*dt/dz^2 < 1/2 (on the acc1 grid the
ceiling is ~0.52 m^2/s). Backward-Euler is unconditionally stable.

pe_simple conventions: z is axis 0 of (Nz,Ny,Nx); bottom k=halo, surface
k=-halo-1; dz is a uniform scalar. Uses only OM.np / OM.for_loop /
OM.modify_array, so it stays JAX- and MPI-compatible. Mix into the class
hierarchy (e.g. let `damping` also inherit from `implicit_vmix`).
"""

import pyOM3 as OM
from pyOM3.density import get_rho
from functools import partial


class implicit_vmix:

    # default: explicit vertical tracer mixing (as in pe_simple / pyOM2).
    # set enable_implicit_vmix = True in the config to use the backward-Euler
    # solve below, which is what makes convective adjustment (large kappa_conv)
    # stable.
    enable_implicit_vmix = False

    # ------------------------------------------------------------------
    # N^2 on the W grid (top face of cell k).
    #   Nsqr[k] = -grav/rho_0/dz * ( rho[k+1] - rho[k] )
    #
    # This raw density difference is only a valid stability measure when the
    # equation of state is pressure-INDEPENDENT, i.e. rho = rho(S,T) alone:
    #   * eq_of_state = 0   (temp == buoyancy b, rho = -b*rho_0/grav)  -> Nsqr = db/dz
    #   * eq_of_state = 1   (linear rho(S,T))
    # For a pressure-dependent / in-situ EOS the difference rho[k+1]-rho[k] mixes
    # the true stratification with the compressibility signal, so the sign of
    # Nsqr is meaningless for convection detection. The correct treatment
    # references both parcels to a common pressure (adiabatic displacement),
    # which is NOT done here. We therefore refuse to run for such states rather
    # than silently produce a wrong Nsqr. Only 0 and 1 are supported.
    # (eq_of_state = 100 is the ideal-gas atmosphere: pressure dependent AND
    #  upside down, also unsupported by this simple convective test.)
    # ------------------------------------------------------------------
    _NSQR_SUPPORTED_EOS = (0, 1)

    @partial(OM.jaxjit, static_argnums=0)
    def calc_nsqr(self, temp, salt, rho, maskW):
        if self.eq_of_state not in self._NSQR_SUPPORTED_EOS:
            raise NotImplementedError(
                "convective adjustment / Nsqr only implemented for a "
                "pressure-independent equation of state (eq_of_state in "
                f"{self._NSQR_SUPPORTED_EOS}); got eq_of_state={self.eq_of_state}. "
                "A pressure-dependent EOS needs an adiabatic-displacement Nsqr, "
                "which is not implemented here.")
        rho_up = get_rho(self.rollz(salt, -1), self.rollz(temp, -1), None, self.eq_of_state)
        return -self.grav / self.rho_0 / self.dz * (rho_up - rho) * maskW

    # ------------------------------------------------------------------
    # convective diffusivity on the W grid:
    #   kappaH = kappa_background  everywhere,
    #          = kappa_conv        where statically unstable (Nsqr < 0)
    # Returned on the W grid (top face of cell k), consistent with Nsqr.
    # ------------------------------------------------------------------
    @partial(OM.jaxjit, static_argnums=0)
    def convective_kappa(self, temp, salt, rho, maskW):
        nsqr   = self.calc_nsqr(temp, salt, rho, maskW)
        kap_bg = getattr(self, 'kappa_T', 0.0)
        kap_cv = getattr(self, 'kappa_conv', 1.0)          # pyOM2 uses 1.0 m^2/s
        kappaH = OM.np.where(nsqr < 0.0, kap_cv, kap_bg) * maskW

        # The free-surface top face must be CLOSED to tracer vertical mixing.
        # make_topo_masks OPENS maskW at the surface (maskW[-halo-1] =
        # maskW[-halo-2]) for the pressure/momentum (w) solver, but there is no
        # physical diffusive tracer flux through the sea surface. Leaving it
        # open, combined with the wrap-around in calc_nsqr's rollz(temp,-1) at
        # the top cell (which brings the bottom value to the surface and can
        # give a spurious Nsqr<0), lets kappa_conv act on the surface face and
        # drains the surface buoyancy ~1000x faster than the surface restoring
        # can build it. Force the surface top face closed here.
        kappaH = OM.modify_array(kappaH, (-self.halo - 1, slice(None), slice(None)),
                                 OM.np.zeros_like(kappaH[-self.halo - 1]))
        return kappaH

    # ------------------------------------------------------------------
    # implicit (backward-Euler) vertical diffusion of a tracer var:
    #     (I - dt d_z kappaH d_z) var^{n+1} = var^n
    # solved column-by-column with the Thomas algorithm along axis 0.
    #
    # kappaH is on the W grid (top face of cell k). With uniform dz and the
    # pe_simple orientation (k=halo bottom ... k=-halo-1 surface), the flux on
    # the top face of cell k is  F[k] = kappaH[k]*(var[k+1]-var[k])/dz , and the
    # tendency is  (F[k]-F[k-1])/dz. maskW[k]=0 on the top face closes the flux
    # at the surface and below the bottom automatically.
    # ------------------------------------------------------------------
    @partial(OM.jaxjit, static_argnums=0)
    def implicit_vdiff(self, var, kappaH, maskT):
        dt, dz = self.dt, self.dz
        h  = self.halo
        Nz = self.Nz

        # delta[k] = dt/dz * kappaH[k]  (kappaH on top face of cell k)
        delta = dt / dz * kappaH

        # Sub/super/diagonal of (I - dt d_z kappaH d_z), interior cells only.
        # lower a[k]  couples k to k-1  : -delta[k-1]/dz
        # upper c[k]  couples k to k+1  : -delta[k]  /dz
        # diag  b[k]  = 1 - a[k] - c[k]
        a = -self.rollz(delta, 1) / dz          # a[k] = -delta[k-1]/dz
        c = -delta / dz                          # c[k] = -delta[k]  /dz

        # restrict to wet interior column; land/halo rows become identity (b=1)
        a = a * maskT
        c = c * maskT
        b = 1.0 - a - c
        # on dry cells force identity so the solve is well-posed
        b = OM.np.where(maskT > 0.0, b, 1.0)
        a = OM.np.where(maskT > 0.0, a, 0.0)
        c = OM.np.where(maskT > 0.0, c, 0.0)

        d = var * 1.0                            # rhs = var^n (copy)

        # ---- Thomas algorithm along axis 0 (k = h .. Nz-h-1) --------------
        # forward sweep
        cp = OM.np.zeros_like(var)
        dp = OM.np.zeros_like(var)
        cp = OM.modify_array(cp, (h,   slice(None), slice(None)), c[h] / b[h])
        dp = OM.modify_array(dp, (h,   slice(None), slice(None)), d[h] / b[h])

        def _fwd(k, state):
            cp, dp = state
            m   = b[k] - a[k] * cp[k-1]
            cp  = OM.modify_array(cp, (k, slice(None), slice(None)), c[k] / m)
            dp  = OM.modify_array(dp, (k, slice(None), slice(None)),
                                  (d[k] - a[k] * dp[k-1]) / m)
            return (cp, dp)

        cp, dp = OM.for_loop(h + 1, Nz - h, _fwd, (cp, dp))

        # back substitution
        out = var * 1.0
        out = OM.modify_array(out, (Nz - h - 1, slice(None), slice(None)), dp[Nz - h - 1])

        def _bwd(i, out):
            k = Nz - h - 2 - i                   # descend from Nz-h-2 down to h
            return OM.modify_array(out, (k, slice(None), slice(None)),
                                   dp[k] - cp[k] * out[k + 1])

        out = OM.for_loop(0, (Nz - 2 * h) - 1, _bwd, out)
        out = out * maskT + var * (1.0 - maskT)  # keep dry cells untouched
        return self.apply_bc(out)

    # ------------------------------------------------------------------
    # convenience: one call that builds kappaH (with convection) and applies
    # the implicit vertical diffusion to temp (and salt if active).
    # Call this INSTEAD of the explicit harmonic_vertical_expl in model.py.
    # ------------------------------------------------------------------
    def vertical_mixing_implicit(self):
        # kappaH from the current density field, so convection is detected with
        # the active equation of state; temp and salt share the same kappaH.
        kappaH = self.convective_kappa(self.temp, self.salt, self.rho, self.maskW)
        self.temp = self.implicit_vdiff(self.temp, kappaH, self.maskT)
        if self.eq_of_state > 0 and self.eq_of_state != 100:
            self.salt = self.implicit_vdiff(self.salt, kappaH, self.maskT)
        return


    # ==================================================================
    # Vertical mixing / friction OPTION HIERARCHY (pyOM2-style)
    #
    #   vmix_scheme = 'explicit':
    #       fixed coefficients kappa_T (buoyancy) and A_v (momentum) only;
    #       the coefficient size is CHECKED against the explicit stability
    #       limit  4*max(coef)*dt/dz^2 <= 0.5  at prepare_for_run (see
    #       check_vmix_stability); convective adjustment and TKE are
    #       refused under the explicit scheme.
    #   vmix_scheme = 'implicit'  (backward-Euler, unconditionally stable):
    #       - constant (possibly large) kappa_T and A_v, or
    #       - enable_conv_adjust: fixed background kappa_T for buoyancy,
    #         but kappa_conv (large) where statically unstable
    #         ("convective adjustment"); momentum stays at fixed A_v, or
    #       - enable_tke: the TKE closure (tke.py) OVERRIDES all constant
    #         coefficients for both momentum and buoyancy
    #         (kappaM_tke / kappaH_tke, floors kappaM_min / kappaH_min).
    #
    # Defaults preserve the previous behaviour: enable_implicit_vmix=True
    # maps to vmix_scheme='implicit' with enable_conv_adjust=True.
    # ==================================================================
    def _vmix_options(self):
        scheme = getattr(self, 'vmix_scheme', None)
        if scheme is None:                     # legacy flag mapping
            scheme = 'implicit' if getattr(self, 'enable_implicit_vmix', True) \
                     else 'explicit'
        conv = getattr(self, 'enable_conv_adjust', scheme == 'implicit')
        tkeo = getattr(self, 'enable_tke', False)
        if scheme == 'explicit' and (conv or tkeo):
            raise RuntimeError('convective adjustment / TKE require '
                               "vmix_scheme='implicit'")
        return scheme, conv, tkeo

    def check_vmix_stability(self):
        """Explicit-scheme guard: 4*coef*dt/dz^2 <= 0.5 (diffusive stability
        of the forward-Euler vertical operator). Called from prepare_for_run;
        a no-op under the implicit scheme."""
        scheme, conv, tkeo = self._vmix_options()
        if scheme != 'explicit':
            return
        coef = max(getattr(self, 'kappa_T', 0.0), getattr(self, 'A_v', 0.0))
        cn = 4.0 * coef * self.dt / self.dz ** 2
        if cn > 0.5:
            raise RuntimeError(
                'explicit vertical mixing unstable: 4*coef*dt/dz^2 = %.3f > 0.5 '
                '(coef=%.3e, dt=%.0f, dz=%.1f). Reduce the coefficient or use '
                "vmix_scheme='implicit'." % (cn, coef, self.dt, self.dz))
        return

    def vertical_mixing(self):
        """Tracer vertical mixing according to the option hierarchy."""
        scheme, conv, tkeo = self._vmix_options()
        if scheme == 'explicit':
            if getattr(self, 'kappa_T', 0.0) > 0.0:
                self.temp = self.harmonic_vertical_expl(self.temp, self.kappa_T, self.maskT)
                if self.eq_of_state > 0 and self.eq_of_state != 100:
                    self.salt = self.harmonic_vertical_expl(self.salt, self.kappa_T, self.maskT)
            return
        if tkeo:
            kappaH = self.kappaH_tke               # set_tke_diffusivities ran
        elif conv:
            kappaH = self.convective_kappa(self.temp, self.salt, self.rho, self.maskW)
        else:
            kappaH = getattr(self, 'kappa_T', 0.0) * self.maskW
        self.temp = self.implicit_vdiff(self.temp, kappaH, self.maskT)
        if self.eq_of_state > 0 and self.eq_of_state != 100:
            self.salt = self.implicit_vdiff(self.salt, kappaH, self.maskT)
        return

    def vertical_friction(self):
        """Momentum vertical friction according to the option hierarchy.
        Under TKE this is implicit_vfric (tke.py), which also diagnoses
        K_diss_v with the exact discrete pairing."""
        scheme, conv, tkeo = self._vmix_options()
        if tkeo:
            self.implicit_vfric()
            return
        Av = getattr(self, 'A_v', 0.0)
        if Av <= 0.0:
            return
        if scheme == 'explicit':
            dz = self.dz
            ftu = Av * (self.rollz(self.u, -1) - self.u) / dz \
                  * self.rollz(self.maskU, -1) * self.maskU
            ftv = Av * (self.rollz(self.v, -1) - self.v) / dz \
                  * self.rollz(self.maskV, -1) * self.maskV
            self.u = self.u + self.dt * (ftu - self.rollz(ftu, 1)) / dz * self.maskU
            self.v = self.v + self.dt * (ftv - self.rollz(ftv, 1)) / dz * self.maskV
            self.u, self.v = self.apply_bc(self.u), self.apply_bc(self.v)
            return
        kav = Av * self.maskW
        self.u = self.implicit_vdiff(self.u, kav, self.maskU)
        self.v = self.implicit_vdiff(self.v, kav, self.maskV)
        return
