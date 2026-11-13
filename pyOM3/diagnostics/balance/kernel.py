"""

 Truncated kernel approximation of the flow decomposition of warn.py.

 The zero order balanced state is the projection on the geostrophic mode.
 Per vertical mode n it is the inversion of the linear potential vorticity
   Pi_n = dv_n/dx - du_n/dy - (f/c_n^2) avg(p_n)
 at the vorticity points with the operator  L_n = lap5 - (f^2/c_n^2) avg avg.
 The inverse of L_n is a convolution with a kernel that decays on the
 deformation radius L_n = c_n/f. The kernel is truncated with a Gaussian
 window at the radius R = max(R_over_L L_n, floor_cells dx) and the
 inversion is solved by conjugate gradients on the exact local operator
 L_n with the truncated kernel as preconditioner, each iteration one
 application of the kernel and of the local stencil, until the residual
 is below pcg_tol. The Gaussian window keeps the preconditioner positive
 definite, which a cosine taper does not. The plain defect correction
 (the kernel applied to the residual) used before diverges on the C grid
 for the vertical modes with L_n below the grid spacing: there the
 Coriolis averaging makes the operator nearly singular along the Nyquist
 lines, the exact kernel has a long checkerboard tail that no truncation
 keeps, and the correction factor leaves the unit circle. Conjugate
 gradients converge for any truncation radius, in more iterations where
 the preconditioner is poor (about 25 at floor 16, 40 at floor 8, 80 at
 floor 4 for 20 levels at L_1 = 1.6 dx). Each mode stops iterating when
 its own residual is below pcg_tol. The barotropic mode is never iterated.
 The barotropic mode under the rigid lid is a Poisson inversion without
 decaying kernel and is kept exact. The buoyancy component that produces
 no pressure (one profile per column) is added as a local column
 operation. With untruncated kernels the recipe reproduces the projection
 of warn.py to machine precision.

 The first order correction of Warn et al. is a linear map of the
 quadratic model tendencies of the zero order state. Per mode the map is
 a convolution with nine kernels with the symbols
   S_Xj = i ( q+_X p+_j / omega_+ + q-_X p-_j / omega_- )
 which decay on L_n since omega >= f, truncated with a cosine taper at the
 same radius. The taper is flat only to R/2, so R_over_L = 8 is needed
 for the first order to reproduce the exact kernels (R_over_L = 3 leaves
 a ten times larger residual at second order in the resolved modes).
 The second order applies the same nine kernels to the cross tendencies
 between zero and first order minus the time derivative and the damping
 of the first order. The time derivative is computed from nsteps model
 steps started from the first order balanced state, as in warn.py
 with pTf1_numeric=True, with the kernel decomposition at both times.
 With untruncated kernels both orders match warn.py to machine
 precision.

 With a variable Coriolis parameter the kernels are those of the reference
 f0 of init_balance, and the difference (f - f0) u enters with the
 nonlinear tendencies in the first and second order, as in warn.py.
 The zero order is the geostrophic state (with f0) whose linear potential
 vorticity with the local f, dv_n/dx - du_n/dy - (f/c_n^2) avg(p_n), is
 that of the fields, found by fixed point iterations of the inversion with
 f0 (as _geostrophic_amp of warn.py).

 Needs init_balance of warn.py first, then init_kernel. One process.

 Moved here from pe_simple/diag_kernel_balance.py (Nov 2026), the code
 unchanged.
"""
import numpy                 as np
import pyOM3                 as OM
from pyOM3.diagnostics.balance.warn import warn_pe_simple, hat_ksqr, hat_epm


class kernel_pe_simple(warn_pe_simple):

    # ------------------------------------------------------------------
    def init_kernel(self, R_over_L=8., nref=100, floor_cells=8, trunc=True, verbose=True,
                    pcg_tol=1e-8):
        """
        R_over_L     truncation radius in deformation radii of each mode
        nref         maximal conjugate gradient iterations of the zero order inversion
        floor_cells  minimal truncation radius in grid cells
        trunc        False keeps the kernels untruncated (for validation)
        pcg_tol      relative residual at which the iterations stop (per mode)
        """
        if not hasattr(self, 'bal_phi'):
            raise RuntimeError('kernel: call init_balance first')
        if self.implicit_free_surface and verbose:
            # the kernels and the modes are those of the rigid lid; with the
            # implicit free surface of the model they are an approximation
            # (the external gravity waves are fast and leave with the
            # divergent barotropic flow, the balanced flow has none)
            print('kernel: the model runs with the implicit free surface, the kernels are those of the rigid lid')
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        f0 = self.bal_f0
        self.ker_nref, self.ker_trunc, self.ker_tol = nref, trunc, pcg_tol

        # full grid wavenumbers, arrays are (mode, y, x)
        kxf = 2*np.pi*np.fft.fftfreq(nx, dx)
        kyf = 2*np.pi*np.fft.fftfreq(ny, dy)
        KXf, KYf = np.meshgrid(kxf, kyf)                    # (y, x)
        # periodic distance from the origin
        xw = np.minimum(np.arange(nx)*dx, nx*dx - np.arange(nx)*dx)
        yw = np.minimum(np.arange(ny)*dy, ny*dy - np.arange(ny)*dy)
        Rd = np.sqrt(xw[None, :]**2 + yw[:, None]**2)

        # truncation radius, cosine taper (first order) and Gaussian window
        # (zero order preconditioner) per mode
        c = np.sqrt(self.bal_c2)
        R = np.maximum(R_over_L*c/f0, floor_cells*dx)
        W = np.zeros((nz, ny, nx)); Wg = np.zeros((nz, ny, nx))
        for n in range(nz):
            W[n] = 0.5*(1 + np.cos(np.pi*np.clip((Rd - R[n]/2)/(R[n]/2), 0, 1)))
            Wg[n] = np.exp(-(Rd/(R[n]/2))**2)

        # zero order kernels, mode 0 is the exact Poisson inversion
        IC2 = self.bal_inv_c2[:, None, None]
        Lhat = -(hat_ksqr(KXf, dx) + hat_ksqr(KYf, dy))[None, :, :] \
               - f0**2*IC2*(hat_epm(KXf, dx)*hat_epm(KYf, dy))[None, :, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            iL = np.where(np.abs(Lhat) > 0, 1./Lhat, 0.)
        KF = iL.astype(complex)
        if trunc:
            Kd = np.real(np.fft.ifft2(iL[1:], axes=(1, 2)))
            KF[1:] = np.fft.fft2(Kd*Wg[1:], axes=(1, 2))
        self.ker_KF = KF

        # the nine first order kernels, rfft layout as the symbols
        qp, qm = self.bal_q[1], self.bal_q[-1]
        pp, pm = self.bal_p[1], self.bal_p[-1]
        iomp, iomm = self.bal_iom[1], self.bal_iom[-1]
        dlt = np.zeros((ny, nx)); dlt[0, 0] = 1.
        Fd = np.fft.fft(np.fft.rfft(dlt, axis=1), axis=0)*dx*dy
        S = np.zeros((3, 3, nz, ny, nx//2 + 1), dtype=complex)
        for iX in range(3):
            for jD in range(3):
                s = 1j*(qp[iX]*pp[jD]*iomp + qm[iX]*pm[jD]*iomm)
                if trunc:
                    ker = np.fft.irfft(np.fft.ifft(s[1:]*Fd, axis=1),
                                       n=nx, axis=2)/(dx*dy)
                    s[1:] = np.fft.fft(np.fft.rfft(ker*W[1:], axis=2),
                                       axis=1)*dx*dy/Fd
                S[iX, jD] = s
        self.ker_S = S
        if verbose and self.my_pe == 0:
            print('kernel: R = %s km, up to %d cg iterations%s' %
                  (np.array2string(np.minimum(R, R[1])[:4]/1e3, precision=1),
                   nref, '' if trunc else ', untruncated'))
        return

    # ------------------------------------------------------------------
    # zero order by truncated kernel inversion
    def _ker_stencils(self, a, which):
        """C grid stencil operations on (mode, y, x) arrays, x is the last axis"""
        if which == 'avg_h2c':
            return 0.25*(a + np.roll(a, -1, 2) + np.roll(a, -1, 1)
                         + np.roll(np.roll(a, -1, 2), -1, 1))
        if which == 'avg_c2h':
            return 0.25*(a + np.roll(a, 1, 2) + np.roll(a, 1, 1)
                         + np.roll(np.roll(a, 1, 2), 1, 1))
        if which == 'lap5':
            return ((np.roll(a, -1, 2) - 2*a + np.roll(a, 1, 2))/self.dx**2
                    + (np.roll(a, -1, 1) - 2*a + np.roll(a, 1, 1))/self.dy**2)

    def kernel_zero(self, u, v, b, p):
        """
        zero order balanced state from the fields (with halo points) by the
        truncated kernel inversion, returns u, v, w, b, p, p_s as project(0)
        """
        dx, dy, dz = self.dx, self.dy, self.dz
        f0, IC2 = self.bal_f0, self.bal_inv_c2[:, None, None]
        phi, psi = self.bal_phi, self.bal_psi
        to_m = lambda a: np.einsum('nk,kji->nji', self.bal_phi_inv,
                                   self._interior(a))*dz
        Un, Vn, Pn = to_m(u), to_m(v), to_m(p)
        # linear potential vorticity at the vorticity points per mode
        Pi = ((np.roll(Vn, -1, 2) - Vn)/dx - (np.roll(Un, -1, 1) - Un)/dy
              - f0*IC2*self._ker_stencils(Pn, 'avg_h2c'))
        # inversion by conjugate gradients on -L psi = -Pi, preconditioned
        # with the truncated kernel (-K), each mode its own system
        K = lambda r: -np.real(np.fft.ifft2(np.fft.fft2(r, axes=(1, 2))*self.ker_KF, axes=(1, 2)))
        A = lambda x: -(self._ker_stencils(x, 'lap5')
                        - f0**2*IC2*self._ker_stencils(self._ker_stencils(x, 'avg_c2h'), 'avg_h2c'))
        dot = lambda a, c: np.sum(a*c, axis=(1, 2))[:, None, None]
        def inv(rhs, ps):
            ps = K(rhs) if ps is None else ps + K(rhs - A(ps))
            r = rhs - A(ps)
            r0 = np.sqrt(dot(rhs, rhs)) + 1e-300
            self.ker_its = 0
            if self.ker_trunc:
                # each mode iterates only until its own residual is below the
                # tolerance and is frozen afterwards. Mode 0 is inverted exactly
                # and never iterated: conjugate gradients on its round-off
                # residual are unstable (the operator has the constant null space)
                act = np.sqrt(dot(r, r))/r0 >= self.ker_tol
                act[0] = False
                z = K(r); p = z*1; rz = dot(r, z)
                for it in range(self.ker_nref):
                    if not act.any(): break
                    Ap = A(p); pAp = dot(p, Ap)
                    a = np.where(act & (pAp > 0), rz/np.where(pAp > 0, pAp, 1.), 0.)
                    ps = ps + a*p; r = r - a*Ap
                    self.ker_its = it + 1
                    act = act & (np.sqrt(dot(r, r))/r0 >= self.ker_tol)
                    if not act.any(): break
                    z = K(r); rz_new = dot(r, z)
                    beta = np.where(act & (rz > 0), rz_new/np.where(rz > 0, rz, 1.), 0.)
                    p = np.where(act, z + beta*p, p); rz = np.where(act, rz_new, rz)
            return ps
        ps = inv(-Pi, None)
        its = self.ker_its
        if self.bal_df is not None:
            # variable f: the geostrophic state has the potential vorticity with the
            # local f of the fields, Pi - (f - f0)/c^2 avg(p_n - p0_n), with p0 the
            # pressure of the geostrophic state itself (see _geostrophic_amp of
            # warn.py), by fixed point iterations
            h = self.halo
            dfc = self._ker_stencils(np.asarray(self.bal_df)[None, h:-h, h:-h], 'avg_h2c')
            n0 = np.sqrt(np.sum(ps**2)) + 1e-300
            for it in range(30):
                P0 = f0*self._ker_stencils(ps, 'avg_c2h')
                rhs = -(Pi - dfc*IC2*self._ker_stencils(Pn - P0, 'avg_h2c'))
                ps_new = inv(rhs, ps)
                its += self.ker_its
                d = np.sqrt(np.sum((ps_new - ps)**2))/n0
                ps = ps_new
                if d < max(self.ker_tol, 1e-12): break
            self.ker_pv_its = it + 1
        self.ker_its = its
        # balanced mode amplitudes from the streamfunction
        Ug = -(ps - np.roll(ps, 1, 1))/dy
        Vg = (ps - np.roll(ps, 1, 2))/dx
        Pg = f0*self._ker_stencils(ps, 'avg_c2h')
        # the buoyancy component without pressure, local in each column
        b_null = self._interior(b) - np.einsum('kn,nji->kji', psi, Pn)/dz
        uo = self._pad(np.einsum('kn,nji->kji', phi, Ug)/dz)*self.maskU
        vo = self._pad(np.einsum('kn,nji->kji', phi, Vg)/dz)*self.maskV
        po = self._pad(np.einsum('kn,nji->kji', phi, Pg)/dz)
        bm = self._pad(np.einsum('kn,nji->kji', psi, Pg)/dz)
        bo = bm + self._pad(b_null)
        wo = self.vertical_velocity(uo, vo, OM.np.zeros_like(uo), self.maskW)
        ks = -(self.halo + 1)
        p_s = po[ks] + dz/2*bm[ks]
        # the higher orders act on the mode part of b, without the null space,
        # exactly as the orders of warn.py
        self.ker_b_mode = bm
        return uo, vo, wo, bo, po, p_s

    def kernel_project(self):
        """zero order balanced state of the model state, as project(0)"""
        u, v, b, p = self.model_state()
        return self.kernel_zero(u, v, b, p)

    # ------------------------------------------------------------------
    # first and second order by the nine truncated kernels
    def _nonlin_beta(self, u, v, b):
        """advection tendencies plus the (f - f0) Coriolis term (variable f)"""
        F = self._nonlin(u, v, b)
        B = self._beta(u, v)
        return F if B is None else [a + c for a, c in zip(F, B)]

    def _K9(self, F):
        """the nine kernels applied to the amplitude triple F = (U, V, P)"""
        return [sum(self.ker_S[iX, jD]*F[jD] for jD in range(3))
                for iX in range(3)]

    def kernel_first(self, X0):
        """
        first order correction of the zero order state X0 = (u, v, b, ...),
        returns the fields (u, v, w, b, p, p_s) and the amplitude triple
        """
        A1 = self._K9(self._nonlin_beta(*X0[:3]))
        return self.backward(*A1), A1

    def kernel_second(self, X0, X1, A1, p_s0, p_s1):
        """
        second order correction from the zero order X0 = (u, v, b), the first
        order X1 = (u, v, b) with its amplitudes A1 and the surface pressures,
        time derivative of the first order from bal_nsteps model steps,
        returns the fields (u, v, w, b, p, p_s)
        """
        return self.backward(*self._second_amps(X0, X1, A1, p_s0, p_s1))

    def _second_amps(self, X0, X1, A1, p_s0, p_s1):
        """amplitude triple of the second order correction"""
        F0 = self._nonlin(*X0[:3])
        F1 = self._nonlin(*X1[:3])
        F01 = self._nonlin(*[a + c for a, c in zip(X0[:3], X1[:3])])
        dF = [a - c - d for a, c, d in zip(F01, F0, F1)]
        B1 = self._beta(*X1[:2])
        if B1 is not None:
            dF = [a + c for a, c in zip(dF, B1)]
        # time derivative of the first order from model steps
        Xn = self._run(X0[0] + X1[0], X0[1] + X1[1], X0[2] + X1[2],
                       p_s0 + p_s1, self.bal_nsteps)
        Z = self.kernel_zero(Xn[0], Xn[1], Xn[2], Xn[3])
        Fn = self._nonlin_beta(Z[0], Z[1], self.ker_b_mode)
        A1n = self._K9(Fn)
        dt = self.bal_nsteps*self.dt
        G = [d - (a1n - a1)/dt - self.bal_AhK4*a1
             for d, a1n, a1 in zip(dF, A1n, A1)]
        return self._K9(G)

    def _mode_rms(self, A):
        """rms per vertical mode of the spectral amplitude array"""
        a = np.abs(np.asarray(A))**2
        a[:, :, 1:-1] *= 2.                                # rfft layout
        return np.sqrt(np.sum(a, axis=(1, 2)))

    def kernel_balanced(self, order=2, per_mode=True):
        """
        balanced state of the model state up to the order (0, 1 or 2) with the
        truncated kernels, returns the fields (u, v, w, b, p, p_s).
        per_mode carries each vertical mode only up to the order where its own
        increment (the rms of the velocity amplitude of the order per mode)
        still decreases, the optimal truncation of the asymptotic series,
        which converges per mode with the factor Ro/Bu_n. The first order is
        always kept: it is forced by the quadratic terms of the zero order of
        all modes and lands in other modes than its forcing (mode 1 alone
        forces modes 0 and 2), so it cannot be compared with the zero order of
        the same mode. Only the second order is compared with the first order
        of the same mode, except where the first order of the mode vanishes to
        round-off. The comparison is made only in the modes where the series
        is expected to diverge, n >= m_c with m_c = N0 H/(pi u_rms) (u_rms of
        the zero order), the criterion of slave_modes='auto' of warn.py.
        Below m_c the second order is always kept: there the increments of
        the two orders can come from different sources (the eddies and the
        waves in other modes, or with a variable Coriolis parameter the term
        (f - f0) u, which forces the forcing mode itself), and the comparison
        would drop a convergent second order.
        """
        Z = self.kernel_project()
        out = list(Z)
        if order >= 1:
            X0 = (Z[0], Z[1], self.ker_b_mode)
            A1 = self._K9(self._nonlin_beta(*X0))
            if per_mode:
                inc1 = self._mode_rms(A1[0]) + self._mode_rms(A1[1])
            F1 = self.backward(*A1)
            out = [a + c for a, c in zip(out, F1)]
            if order >= 2:
                X1 = (F1[0], F1[1], F1[3])
                A2 = self._second_amps(X0, X1, A1, Z[5], F1[5])
                if per_mode:
                    inc2 = self._mode_rms(A2[0]) + self._mode_rms(A2[1])
                    # where the first order vanishes to round-off (the selection
                    # rule of the cosine modes under constant N gives exact zeros)
                    # the comparison means nothing and the second order is kept
                    absent = inc1 <= 1e-10*inc1.max()
                    # below the 'auto' cut m_c the series converges, always keep
                    u_rms = float(np.sqrt(np.mean(np.asarray(self._interior(Z[0]))**2
                                                  + np.asarray(self._interior(Z[1]))**2)))
                    m_c = self.bal_N0*self.nz*self.dz/(np.pi*u_rms) if u_rms > 0 else np.inf
                    low = np.arange(self.nz) < m_c
                    keep2 = ((inc2 < inc1) | absent | low)[:, None, None]
                    self.ker_keep2 = keep2[:, 0, 0]
                    A2 = [a*keep2 for a in A2]
                F2 = self.backward(*A2)
                out = [a + c for a, c in zip(out, F2)]
        return out

    # ------------------------------------------------------------------
    def OB_kernel_project(self, last_tau):
        """
        projection on the geostrophic mode for optimal.py, replacing
        the time averaging of the linear model. The model temp is the
        buoyancy perturbation b (the level mean is removed in balance_it),
        last_tau is the time level of p_s of the state.
        Returns u, v, w, b, p_s of the projected state.
        """
        b = self.temp*1
        p = self.p_s[last_tau][None, :, :] + self.hydrostatic(b)
        u, v, w, b, pf, p_s = self.kernel_zero(self.u, self.v, b, p)
        return u, v, w, b, p_s

    def OB_fft_project(self, last_tau):
        """
        exact spectral projection on the geostrophic mode for
        optimal.py, the same conventions as OB_kernel_project.
        Returns u, v, w, b, p_s of the projected state.
        """
        b = self.temp*1
        p = self.p_s[last_tau][None, :, :] + self.hydrostatic(b)
        F = self.forward(self.u, self.v, p)
        g = self._geostrophic_amp(self.u, self.v, p)
        uo, vo, wo, bo, po, p_s = self.backward(*[g*qq for qq in self.bal_q[0]])
        # keep the buoyancy component without pressure, as the kernel projection
        R = self.backward(*F)
        bo = bo + (b - R[3])
        return uo, vo, wo, bo, p_s
