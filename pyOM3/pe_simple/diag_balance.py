"""
 Balance of the model state after Warn et al. (1995) to fourth order in the
 Rossby number, the method of Eden, Chouksey and Olbers (2019, JPO 49, 2393).
 The balanced state is the geostrophic mode g0 plus the slaved gravity wave
 modes f1 .. f4. Port of the grav3D code (py_src/balance2.py,
 eigenvectors3.py, my_fft_reordered.py) to pe_simple.

 Only for pe_simple with buoyancy as tracer (eq_of_state=0), constant
 Coriolis parameter f0, constant background stratification N0 with
 temp = N0**2 z + b and z = 0 at the surface, 2nd order advection, a
 periodic domain without topography, rigid lid or implicit free surface
 (pressure method), one process.

 The linear equations of pe_simple are projected on the vertical modes
 phi_n(k), the eigenvectors of the discrete operator M (Eden et al. 2019,
 their Eqs. A5 and A6) for the free surface or the cosine modes for the
 rigid lid, and Fourier transformed horizontally. Every (kx, ky, n) has
 three eigenmodes of the state vector (u, v, p) with p the full pressure,
 the geostrophic mode s = 0 and the gravity wave modes s = +1, -1 with
 frequency omega. vec_q(s) is the eigenvector, vec_p(s) its dual with
 p(s).q(s') = delta(s,s'), so that g = p.F is the amplitude of mode s in
 F = (u, v, p) and g q its projection. The eigenvectors are those of the
 discrete equations on the C grid. Buoyancy follows from p by the discrete
 hydrostatic relation, w from (u, v) by the discrete continuity equation.

 Spectral arrays are (nz, ny, nx/2 + 1): vertical mode index, FFT in y,
 real FFT in x. Transforms are done with numpy on the whole field.

 Provides (class balance, add to the model class bases):
   init_balance(N0, ...)      modes, wavenumbers, eigenvectors
   forward(u,v,p)             physical (with halo) -> spectral
   backward(U,V,P)            spectral -> physical u,v,w,b,p,p_s (with halo)
   project(s)                 projection of the model state on mode s
   decompose(order)           the orders 0..order of the balanced state,
                              the model state is not changed
   balance(order)             the model state balanced to order 0..4

 The biharmonic damping Ahbi (= Khbi) is accounted for as in Eq. (27) of
 Eden et al. (2019): as an order Ro term for all but the largest wavenumbers,
 where the complex frequency omega + i Ahbi k^4 is used instead.
"""
import pyOM3                     as OM
from   pyOM3.pe_simple.model     import model
from   pyOM3.density             import get_rho
import numpy                     as np


def hat_ksqr(k, d): return 2.*(1. - np.cos(k*d))/d**2
def hat_epm(k, d):  return 0.5*(1. + np.cos(k*d))
def hat_kp(k, d):   return -1j*(np.exp(1j*k*d) - 1.)/d
def hat_ep(k, d):   return (np.exp(1j*k*d) + 1.)*0.5
def hat_em(k, d):   return (np.exp(-1j*k*d) + 1.)*0.5


class balance(model):

    # ------------------------------------------------------------------
    def init_balance(self, N0, nsteps=10, damping=True, pTf1_numeric=False, verbose=True):
        """
        N0            background stratification, temp = N0**2 z + b with z = 0 at the surface
        nsteps        model steps for the time derivatives of f2, f3 (Baer and Tribbia 1977)
        damping       account for the biharmonic damping Ahbi (needs Khbi = Ahbi)
        pTf1_numeric  time derivative of f1 from a model run instead of the analytic Eq. (25)
        """
        if self.n_pes > 1:
            raise RuntimeError('balance: one process only')
        if self.explicit_free_surface:
            raise RuntimeError('balance: needs the pressure method (rigid lid or implicit free surface)')
        if self.eq_of_state != 0:
            raise RuntimeError('balance: needs buoyancy as tracer (eq_of_state=0)')
        if not (self.periodic_in_x and self.periodic_in_y):
            raise RuntimeError('balance: needs a periodic domain')
        if getattr(self, 'superbee_advection', False) or getattr(self, 'weno_advection', False):
            raise RuntimeError('balance: needs 2nd order advection')
        self.bal_N0, self.bal_nsteps, self.bal_pTf1_numeric = N0, nsteps, pTf1_numeric
        self.bal_verbose = verbose
        self.bal_tau = 0                                   # time level of p_s of the model state
        h = self.halo
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        f0 = float(np.asarray(self.coriolis_t)[h, h])
        if abs(np.asarray(self.coriolis_t)[h:-h, h:-h] - f0).max() > 0:
            raise RuntimeError('balance: needs a constant Coriolis parameter')
        self.bal_f0 = f0

        # background buoyancy, z = 0 at the surface
        zt = (np.arange(self.Nz) - h - nz + 0.5)*dz
        self.bal_b_back = OM.np.asarray(N0**2*zt[:, None, None] + np.zeros((self.Nz, self.Ny, self.Nx)))

        # horizontal wavenumbers
        self.bal_kx = 2*np.pi*np.fft.rfftfreq(nx, dx)
        self.bal_ky = 2*np.pi*np.fft.fftfreq(ny, dy)
        KX, KY = self.bal_kx[None, None, :], self.bal_ky[None, :, None]

        # vertical modes phi[k, n], k = 0 at the bottom, with c^2_n
        if self.implicit_free_surface:
            g = self.grav
            k = np.arange(nz)
            M = dz*(g - N0**2*dz/2) + N0**2*dz**2*(nz - np.maximum(k[:, None], k[None, :])) \
                - N0**2*dz**2/4*np.eye(nz)
            c2, phi = np.linalg.eigh(M)
            c2, phi = c2[::-1], phi[:, ::-1]                # n = 0 is the barotropic mode
            inv_c2 = 1./c2
        else:
            lam = np.arange(nz)*np.pi/(nz*dz)
            k = np.arange(nz)
            phi = np.cos(lam[None, :]*(k[:, None] + 0.5)*dz)*np.sqrt(2./nz)
            phi[:, 0] = np.sqrt(1./nz)
            with np.errstate(divide='ignore'):
                c2 = np.where(lam > 0, N0**2*hat_epm(lam, dz)/hat_ksqr(lam, dz), self.grav*nz*dz)
            inv_c2 = np.where(lam > 0, 1./c2, 0.)          # the barotropic mode has no waves
        # buoyancy from the pressure amplitudes, b_k = sum_n P_n psi[k, n]
        psi = -N0**2*inv_c2[None, :]*dz*(np.cumsum(phi, axis=0) - phi/2)
        self.bal_phi, self.bal_psi, self.bal_c2, self.bal_inv_c2 = phi, psi, c2, inv_c2
        C2, IC2 = c2[:, None, None], inv_c2[:, None, None]

        # eigenvectors and their duals
        k2 = hat_ksqr(KX, dx) + hat_ksqr(KY, dy)
        f2 = f0**2*hat_epm(KX, dx)*hat_epm(KY, dy)
        kh2 = KX**2 + KY**2
        om2 = C2*k2 + f2
        om = np.sqrt(om2)
        self.bal_om = om
        q, p = {}, {}
        for s in (0, 1, -1):
            with np.errstate(invalid='ignore', divide='ignore'):
                q1 = np.where(kh2 > 0, (-1j*f0*hat_ep(KX, dx)*hat_em(KY, dy)*hat_kp(KY, dy)
                                        + s*om*hat_kp(KX, dx))/(f2 - s**2*om2), -1j*s)
                q2 = np.where(kh2 > 0, (+1j*f0*hat_em(KX, dx)*hat_ep(KY, dy)*hat_kp(KX, dx)
                                        + s*om*hat_kp(KY, dy))/(f2 - s**2*om2), s**2)
                q3 = np.where(kh2 > 0, 1., 1. - s**2) + 0*om
                pc = abs(s**2*om2 - f2)/((1 + s**2)*(k2 + f2*IC2))
                p1 = np.where(kh2 > 0, pc*np.conj(q1), 1j*s/2.)
                p2 = np.where(kh2 > 0, pc*np.conj(q2), s**2/2.)
                p3 = np.where(kh2 > 0, pc*IC2, 1. - s**2) + 0*om
            if s == 0:
                # on the Nyquist lines kx = pi/dx or ky = pi/dy the averaging
                # of the Coriolis term vanishes (f2 = 0), the geostrophic mode
                # becomes the non-divergent mode without pressure
                kapx, kapy = 2*np.sin(KX*dx/2)/dx, 2*np.sin(KY*dy/2)/dy
                nyq = (f2 == 0) + 0*om
                q1 = np.where(nyq, np.exp(1j*KX*dx/2)*kapy + 0*om, q1)
                q2 = np.where(nyq, -np.exp(1j*KY*dy/2)*kapx + 0*om, q2)
                q3 = np.where(nyq, 0., q3)
                p1 = np.where(nyq, np.conj(q1)/k2, p1)
                p2 = np.where(nyq, np.conj(q2)/k2, p2)
                p3 = np.where(nyq, 0., p3)
            if s != 0 and not self.implicit_free_surface:   # no waves in the rigid lid barotropic mode
                for a in (q1, q2, q3, p1, p2, p3): a[0] = 0.
            q[s] = [np.nan_to_num(a) for a in (q1, q2, q3)]
            p[s] = [np.nan_to_num(a) for a in (p1, p2, p3)]
        self.bal_q, self.bal_p = q, p

        # inverse frequencies and the damping
        Ahbi = getattr(self, 'Ahbi', 0.) if damping else 0.
        if damping and abs(getattr(self, 'Khbi', 0.) - Ahbi) > 0:
            raise RuntimeError('balance: needs Khbi = Ahbi')
        Ah = Ahbi*(hat_ksqr(KX, dx) + hat_ksqr(KY, dy))**2 + 0*om
        kmax2 = (self.bal_kx[-1] - 2*self.bal_kx[1])**2      # the largest wavenumbers
        large = kh2 > kmax2
        self.bal_Ah = Ah                                     # damping of the geostrophic mode
        self.bal_AhK4 = np.where(large, 0., Ah)              # damping as order Ro term
        self.bal_iom = {}
        for s in (1, -1):
            iom = np.where(large, 1./(s*om + 1j*Ah), 1./(s*om))
            if not self.implicit_free_surface: iom[0] = 0.
            self.bal_iom[s] = iom
        if verbose and self.my_pe == 0:
            print('balance: %s, %d vertical modes, c = %s m/s' %
                  ('free surface' if self.implicit_free_surface else 'rigid lid', nz,
                   np.array2string(np.sqrt(c2[:4]), precision=3)))
        return

    def diagnose(self, n, tau):
        """keep the time level of p_s of the model state"""
        self.bal_tau = tau
        return super().diagnose(n, tau)

    # ------------------------------------------------------------------
    # transforms
    def _interior(self, a):
        h = self.halo
        return np.asarray(a)[h:-h, h:-h, h:-h]

    def _fft(self, a, mode):
        A = np.fft.fft(np.fft.rfft(a, axis=2), axis=1)*self.dx*self.dy
        return np.einsum('kn,kji->nji', mode, A)*self.dz

    def _ifft(self, A, mode):
        a = np.einsum('kn,nji->kji', mode, A)/self.dz
        return np.fft.irfft(np.fft.ifft(a, axis=1), n=self.nx, axis=2)/(self.dx*self.dy)

    def _pad(self, a):
        h = self.halo
        out = np.zeros((self.Nz, self.Ny, self.Nx))
        out[h:-h, h:-h, h:-h] = a
        return self.apply_bc(OM.np.asarray(out))

    def forward(self, u, v, p):
        """spectral amplitudes (U, V, P) of the fields with halo points"""
        phi = self.bal_phi
        return [self._fft(self._interior(a), phi) for a in (u, v, p)]

    def backward(self, U, V, P):
        """physical fields u, v, w, b, p, p_s with halo points from the amplitudes"""
        u = self._pad(self._ifft(U, self.bal_phi))*self.maskU
        v = self._pad(self._ifft(V, self.bal_phi))*self.maskV
        p = self._pad(self._ifft(P, self.bal_phi))
        b = self._pad(self._ifft(P, self.bal_psi))
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        ks = -(self.halo + 1)
        p_s = p[ks] + self.dz/2*b[ks]
        return u, v, w, b, p, p_s

    def hydrostatic(self, b):
        """pressure from buoyancy, referenced to the surface, the discrete hydrostatic relation of pe_simple"""
        rho = -b*self.rho_0/self.grav
        return self.hydrostatic_pressure(rho, OM.np.zeros_like(b), self.maskT)

    def model_state(self):
        """u, v, b, p of the model state, b the buoyancy perturbation"""
        b = self.temp - self.bal_b_back
        p = self.p_s[self.bal_tau][None, :, :] + self.hydrostatic(b)
        return self.u, self.v, b, p

    def _amp(self, F, s):
        return sum(pp*FF for pp, FF in zip(self.bal_p[s], F))

    def _I(self, F, s):
        return 1j*self._amp(F, s)

    def project(self, s):
        """projection of the model state on mode s, returns u, v, w, b, p, p_s"""
        u, v, b, p = self.model_state()
        g = self._amp(self.forward(u, v, p), s)
        return self.backward(*[g*qq for qq in self.bal_q[s]])

    # ------------------------------------------------------------------
    # the model
    def _nonlin(self, u, v, b):
        """spectral amplitudes of the advection tendencies of u, v and of p from b"""
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        fe, fn, ft = self.mom_flux_2nd_u(u, u, v, w, self.maskU, self.maskV, self.maskW)
        du = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskU
        fe, fn, ft = self.mom_flux_2nd_v(v, u, v, w, self.maskU, self.maskV, self.maskW)
        dv = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskV
        fe, fn, ft = self.tracer_flux_2nd(b, u, v, w, self.maskU, self.maskV, self.maskW)
        db = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskT
        return self.forward(self.apply_bc(du), self.apply_bc(dv), self.hydrostatic(self.apply_bc(db)))

    def _run(self, u, v, b, p_s, nsteps):
        """nsteps of the model from (u, v, b, p_s) without forcing, the state is restored afterwards"""
        names = ('u', 'v', 'w', 'temp', 'salt', 'rho', 'p_s', 'p_h', 'du', 'dv', 'dtemp', 'dsalt')
        save = {a: getattr(self, a)*1 for a in names}
        save_tau = self.bal_tau
        self.u, self.v = u*1, v*1
        self.temp = b + self.bal_b_back
        self.rho = get_rho(self.salt, self.temp, self.p0, self.eq_of_state)
        self.w = self.vertical_velocity(self.u, self.v, self.w, self.maskW)
        for i in range(3):
            self.p_s = OM.modify_array(self.p_s, (i, slice(None), slice(None)), p_s)
        self.set_forcing = lambda tau: None               # shadow the forcing of the configuration
        A, B, C = 1., 0., 0.
        tau, taum1, taum2 = 2, 1, 0
        for n in range(nsteps):
            self.time_step_implicit_pressure(A, B, C, tau, taum1, taum2)
            last = tau
            tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
            A, B, C = (1.5, -0.5, 0.) if n == 0 else (23./12, -16./12, 5./12)
        del self.set_forcing
        b = self.temp - self.bal_b_back
        out = (self.u, self.v, b, self.p_s[last][None, :, :] + self.hydrostatic(b))
        for a in names: setattr(self, a, save[a])
        self.bal_tau = save_tau
        return out

    # ------------------------------------------------------------------
    # the orders of the balanced state in spectral space
    def _modes(self, fp, fm):
        """amplitudes (U, V, P) of the wave modes with amplitudes fp, fm"""
        return [fp*a + fm*c for a, c in zip(self.bal_q[1], self.bal_q[-1])]

    def _fields(self, F):
        """u, v, b, p in physical space from the amplitudes (U, V, P)"""
        u, v, w, b, p, p_s = self.backward(*F)
        return u, v, b, p

    def _first_order(self, X0):
        F = self._nonlin(*X0[:3])
        fp = self._I(F, 1)*self.bal_iom[1]
        fm = self._I(F, -1)*self.bal_iom[-1]
        return fp, fm

    def _second_order(self, X0, X1, fp1, fm1):
        iom, AhK4 = self.bal_iom, self.bal_AhK4
        F = self._nonlin(*X0[:3])
        I_00, I_0p, I_0m = self._I(F, 0), self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*X1[:3])
        I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c for a, c in zip(X0[:3], X1[:3])])
        I_0f1p, I_0f1m = self._I(F, 1), self._I(F, -1)
        if not self.bal_pTf1_numeric:
            # the time derivative of f1 from Eq. (25)
            g0 = self._amp(self.forward(X0[0], X0[1], X0[3]), 0)
            pTg0 = -1j*I_00 - self.bal_Ah*g0
            XT0 = self._fields([pTg0*qq for qq in self.bal_q[0]])
            F = self._nonlin(*XT0[:3])
            I_pT0p, I_pT0m = self._I(F, 1), self._I(F, -1)
            F = self._nonlin(*[a + c for a, c in zip(X0[:3], XT0[:3])])
            I_0pT0p, I_0pT0m = self._I(F, 1), self._I(F, -1)
            pTfp = (I_0pT0p - I_0p - I_pT0p)*iom[1]
            pTfm = (I_0pT0m - I_0m - I_pT0m)*iom[-1]
        else:
            # the time derivative of f1 from nsteps model steps
            Xn = self._run(*[a + c for a, c in zip(X0[:2], X1[:2])],
                           X0[2] + X1[2], self._p_s(X0, X1), self.bal_nsteps)
            X0n, X1n, fp1n, fm1n = self._orders_01(Xn)
            pTfp = (fp1n - fp1)/(self.bal_nsteps*self.dt)
            pTfm = (fm1n - fm1)/(self.bal_nsteps*self.dt)
        fp = (I_0f1p - I_0p - I_f1p - 1j*(pTfp + AhK4*fp1))*iom[1]
        fm = (I_0f1m - I_0m - I_f1m - 1j*(pTfm + AhK4*fm1))*iom[-1]
        return fp, fm

    def _third_order(self, X0, X1, X2, fp1, fm1, fp2, fm2):
        iom, AhK4 = self.bal_iom, self.bal_AhK4
        F = self._nonlin(*X0[:3])
        I_0p, I_0m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*X2[:3])
        I_f2p, I_f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c for a, c in zip(X0[:3], X2[:3])])
        I_0f2p, I_0f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*X1[:3])
        I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
        # the time derivative of f2 from nsteps model steps
        Xn = self._run(*[a + c + d for a, c, d in zip(X0[:2], X1[:2], X2[:2])],
                       X0[2] + X1[2] + X2[2], self._p_s(X0, X1, X2), self.bal_nsteps)
        X0n, X1n, fp1n, fm1n = self._orders_01(Xn)
        fp2n, fm2n = self._second_order(X0n, X1n, fp1n, fm1n)
        pTfp = (fp2n - fp2)/(self.bal_nsteps*self.dt)
        pTfm = (fm2n - fm2)/(self.bal_nsteps*self.dt)
        fp = (I_0f2p + I_f1p - I_f2p - I_0p - 1j*(pTfp + AhK4*fp2))*iom[1]
        fm = (I_0f2m + I_f1m - I_f2m - I_0m - 1j*(pTfm + AhK4*fm2))*iom[-1]
        return fp, fm

    def _fourth_order(self, X0, X1, X2, X3, fp1, fm1, fp2, fm2, fp3, fm3):
        iom, AhK4 = self.bal_iom, self.bal_AhK4
        F = self._nonlin(*X0[:3])
        I_0p, I_0m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*X1[:3])
        I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*X3[:3])
        I_f3p, I_f3m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c for a, c in zip(X0[:3], X3[:3])])
        I_0f3p, I_0f3m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*X2[:3])
        I_f2p, I_f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c for a, c in zip(X1[:3], X2[:3])])
        I_f1f2p, I_f1f2m = self._I(F, 1), self._I(F, -1)
        L_p = I_f1f2p - I_f1p - I_f2p
        L_m = I_f1f2m - I_f1m - I_f2m
        # the time derivative of f3 from nsteps model steps
        Xn = self._run(*[a + c + d + e for a, c, d, e in zip(X0[:2], X1[:2], X2[:2], X3[:2])],
                       X0[2] + X1[2] + X2[2] + X3[2], self._p_s(X0, X1, X2, X3), self.bal_nsteps)
        X0n, X1n, fp1n, fm1n = self._orders_01(Xn)
        fp2n, fm2n = self._second_order(X0n, X1n, fp1n, fm1n)
        X2n = self._fields(self._modes(fp2n, fm2n))
        fp3n, fm3n = self._third_order(X0n, X1n, X2n, fp1n, fm1n, fp2n, fm2n)
        pTfp = (fp3n - fp3)/(self.bal_nsteps*self.dt)
        pTfm = (fm3n - fm3)/(self.bal_nsteps*self.dt)
        fp = (I_0f3p - I_f3p - I_0p + L_p - 1j*(pTfp + AhK4*fp3))*iom[1]
        fm = (I_0f3m - I_f3m - I_0m + L_m - 1j*(pTfm + AhK4*fm3))*iom[-1]
        return fp, fm

    def _p_s(self, *X):
        """surface pressure of the sum of the states X = (u, v, b, p)"""
        ks = -(self.halo + 1)
        return sum(x[3][ks] + self.dz/2*x[2][ks] for x in X)

    def _orders_01(self, X):
        """geostrophic mode and first order of the state X = (u, v, b, p)"""
        g0 = self._amp(self.forward(X[0], X[1], X[3]), 0)
        X0 = self._fields([g0*qq for qq in self.bal_q[0]])
        fp1, fm1 = self._first_order(X0)
        X1 = self._fields(self._modes(fp1, fm1))
        return X0, X1, fp1, fm1

    # ------------------------------------------------------------------
    def decompose(self, order=4):
        """
        The orders 0..order of the balanced model state after Warn et al.:
        the geostrophic mode of order Ro^0 and the slaved gravity wave modes
        of order Ro^1 to Ro^order. Returns the list of the orders, each a
        tuple of the fields (u, v, w, b, p_s) with halo points, b the
        buoyancy perturbation. The model state is not changed.
        """
        if order < 0 or order > 4: raise ValueError('balance: order 0..4')
        def pr(s):
            if self.bal_verbose and self.my_pe == 0: print(s)
        pr('balance: zero order')
        X0, X1, fp1, fm1 = self._orders_01(self.model_state())
        X, f = [X0, X1], [(fp1, fm1)]
        if order >= 2:
            pr('balance: second order')
            f.append(self._second_order(X0, X1, fp1, fm1))
            X.append(self._fields(self._modes(*f[-1])))
        if order >= 3:
            pr('balance: third order')
            f.append(self._third_order(X[0], X[1], X[2], *f[0], *f[1]))
            X.append(self._fields(self._modes(*f[-1])))
        if order >= 4:
            pr('balance: fourth order')
            f.append(self._fourth_order(X[0], X[1], X[2], X[3], *f[0], *f[1], *f[2]))
            X.append(self._fields(self._modes(*f[-1])))
        out = []
        for x in X[:order + 1]:
            w = self.vertical_velocity(x[0], x[1], OM.np.zeros_like(x[0]), self.maskW)
            out.append((x[0], x[1], w, x[2], self._p_s(x)))
        # The vertical checkerboard b_k = (-1)^k beta with p_s = dz beta/2 has
        # no pressure p and is invisible in the modes of (u, v, p), it is the
        # null space of the discrete hydrostatic relation. This part of the
        # model state is not dynamical, neither balanced nor wave. It is added
        # to the zero order here, so that the balanced state keeps the full
        # pressure and the residual contains only dynamical modes.
        u, v, b, p = self.model_state()
        R = self.backward(*self.forward(u, v, p))
        ks = -(self.halo + 1)
        b_null = b - R[3]
        p_s_null = self.p_s[self.bal_tau] - R[5]
        self.bal_null = (b_null, p_s_null)
        x = out[0]
        out[0] = (x[0], x[1], x[2], x[3] + b_null, x[4] + p_s_null)
        return out

    def balance(self, order=4):
        """
        Balance the model state to the given order, sets u, v, w, temp, rho, p_s.
        The orders are kept in self.bal_orders, see decompose.
        """
        self.bal_orders = self.decompose(order)
        u = sum(x[0] for x in self.bal_orders)
        v = sum(x[1] for x in self.bal_orders)
        b = sum(x[3] for x in self.bal_orders)
        p_s = sum(x[4] for x in self.bal_orders)
        self.u, self.v = self.apply_bc(u)*self.maskU, self.apply_bc(v)*self.maskV
        self.w = self.vertical_velocity(self.u, self.v, self.w, self.maskW)
        self.temp = self.apply_bc(b + self.bal_b_back)
        self.rho = get_rho(self.salt, self.temp, self.p0, self.eq_of_state)
        self.p_h = self.hydrostatic_pressure(self.rho, self.p_h, self.maskT)
        for i in range(3):
            self.p_s = OM.modify_array(self.p_s, (i, slice(None), slice(None)), self.apply_bc(p_s))
        return

    def norm(self, X, Y):
        """relative norm of the difference of two states (u, v, w, b), 2 |X - Y|^2 / (|X|^2 + |Y|^2)"""
        d = sum(float(np.sum(self._interior(x - y)**2)) for x, y in zip(X[:4], Y[:4]))
        n = sum(float(np.sum(self._interior(x)**2)) + float(np.sum(self._interior(y)**2))
                for x, y in zip(X[:4], Y[:4]))
        return 2*d/n
