"""
 Balance of the model state after Warn et al. (1995) to fourth order in the
 Rossby number for nh_simple, the counterpart of pe_simple/diag_balance.py,
 ported from ps3D (eigenvectors.f90, diag_balance.f90, my_fft.f90).

 In spectral space every wavenumber (kx, ky, kz) has three eigenmodes,
 the geostrophic (vortical) mode s = 0 and the two gravity wave modes
 s = +1, -1 with frequency omega. vec_q(s) is the eigenvector, vec_p(s)
 the dual vector with p(s) . q(s') = delta(s, s'), so that g = p . F is the
 amplitude of mode s in the field F = (u, v, w, b) and g q its projection.
 The eigenvectors are those of the discrete equations on the C grid.

 Spectral space is (nx/2 + 1, ny, nz): a real FFT in x, FFTs in y and z for
 periodic_in_z, and the vertical modes of the rigid lids otherwise
 (kz = k pi/Lz, mode matrices phi for u, v, psi for w and psib for b).
 Transforms are done with numpy on the whole field, one process only.

 Provides (class balance, add to the model class bases):
   init_balance()             wavenumbers, broadcast arrays, mode matrices
   omega2(), vec_q(s), vec_p(s)
   forward(u,v,w,b)           physical (with halo) -> spectral
   backward(U,V,W,B)          spectral -> physical (with halo)
   project(u,v,w,b,s)         projection on mode s in physical space
   mode_fields(i,j,k,s)       one eigenmode in physical space, for wave makers
   decompose(order)           the orders 0..order of the balanced state,
                              the model state is not changed
   balance(order)             the model state (u,v,w,b) balanced to
                              order 0..4 in the Rossby number Ro

 The balance follows Eden, Chouksey and Olbers (2019, JPO 49, 2393) and is
 the same as in pe_simple/diag_balance.py: the slaved wave modes f1 .. f4
 from the interaction integrals of the model's own advection terms, the time
 derivative of f1 from their Eq. (25), those of f2, f3 from nsteps model
 steps, the biharmonic damping Ahbi (= Khbi) as in their Eq. (27). The part
 of the state that no mode represents (the horizontal mean of b, the vertical
 checkerboard of b with the rigid lids, the Nyquist lines) has no dynamics
 and is added to the zero order.
"""
import numpy as np
import pyOM3 as OM
from pyOM3.nh_simple.model import model


def hat_ksqr(k, d): return 2.*(1. - np.cos(k*d))/d**2
def hat_epm(k, d):  return 0.5*(1. + np.cos(k*d))
def hat_kp(k, d):   return -1j*(np.exp(1j*k*d) - 1.)/d
def hat_ep(k, d):   return (np.exp(1j*k*d) + 1.)*0.5
def hat_em(k, d):   return (np.exp(-1j*k*d) + 1.)*0.5


class balance(model):

    # ------------------------------------------------------------------
    def init_balance(self, nsteps=10, damping=True, verbose=True):
        """
        nsteps     model steps for the time derivatives of f2, f3 (Baer and Tribbia 1977)
        damping    account for the biharmonic damping Ahbi (needs Khbi = Ahbi, Kvbi = Avbi)
        """
        if self.n_pes > 1:
            raise RuntimeError('balance: one process only')
        self.bal_nsteps, self.bal_damping, self.bal_verbose = nsteps, damping, verbose
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        self.kx = 2*np.pi*np.fft.rfftfreq(nx, dx)
        self.ky = 2*np.pi*np.fft.fftfreq(ny, dy)
        if self.periodic_in_z:
            self.kz = 2*np.pi*np.fft.fftfreq(nz, dz)
        else:
            self.kz = np.arange(nz)*np.pi/self.Lz
            # vertical modes: psi on W levels (z = k dz), phi and psib on T levels
            k  = np.arange(1, nz + 1)[:, None]
            k2 = np.arange(nz)[None, :]
            psi = np.sqrt(2./self.Lz)*np.sin(k2*k*np.pi/nz)
            psim = np.vstack([np.zeros(nz), psi[:-1]])            # psi(k-1), zero at the bottom
            with np.errstate(divide='ignore', invalid='ignore'):
                phi  = (psi - psim)/(2*np.sin(k2*np.pi/(2*nz)))
                psib = (psi + psim)/(2*np.cos(k2*np.pi/(2*nz)))
            phi[:, 0] = np.sqrt(1./self.Lz); psi[:, 0] = 0.; psib[:, 0] = 0.
            self.phi, self.psi, self.psib = phi, psi, psib
        self.KX = self.kx[:, None, None]
        self.KY = self.ky[None, :, None]
        self.KZ = self.kz[None, None, :]
        return

    # ------------------------------------------------------------------
    def _coef(self):
        """k2, f2_hat, N2_hat, vertical factor m2 (and lam, arg for the lids)"""
        dx, dy, dz = self.dx, self.dy, self.dz
        k2 = hat_ksqr(self.KX, dx) + hat_ksqr(self.KY, dy)
        f2 = self.f0**2*hat_epm(self.KX, dx)*hat_epm(self.KY, dy)
        if self.periodic_in_z:
            N2 = self.N0**2*hat_epm(self.KZ, dz)
            m2 = hat_ksqr(self.KZ, dz)
        else:
            arg = self.KZ*dz/2
            N2 = self.N0**2*np.cos(arg)**2
            m2 = 4*np.sin(arg)**2/dz**2
        return k2, f2, N2, m2

    def omega2(self):
        k2, f2, N2, m2 = self._coef()
        Ksqr = self.dsqr*k2 + m2
        with np.errstate(divide='ignore', invalid='ignore'):
            om2 = np.where(Ksqr > 0, (N2*k2 + f2*m2)/Ksqr, f2)   # the mean flow is an inertial oscillation
        return om2

    def coeff_p(self, s):
        k2, f2, N2, m2 = self._coef()
        om2 = self.omega2()
        d = k2*N2 + f2*m2
        with np.errstate(divide='ignore', invalid='ignore'):
            if s == 0:
                p = np.where(d > 0, f2*N2/d, 0.)
            else:
                p = np.where(d > 0, (N2 - self.dsqr*om2)*(om2 - f2)/(2*d), 0.)
        return p

    def vec_q(self, s):
        dx, dy, dz = self.dx, self.dy, self.dz
        KX, KY, KZ = self.KX, self.KY, self.KZ
        k2, f2, N2, m2 = self._coef()
        om2 = self.omega2(); om = np.sqrt(om2)
        kh2 = KX**2 + KY**2 + 0*KZ
        with np.errstate(divide='ignore', invalid='ignore'):
            fxa = f2 - s**2*om2
            qx = (-1j*self.f0*hat_ep(KX, dx)*hat_em(KY, dy)*hat_kp(KY, dy) + s*om*hat_kp(KX, dx))/fxa
            qy = (+1j*self.f0*hat_em(KX, dx)*hat_ep(KY, dy)*hat_kp(KX, dx) + s*om*hat_kp(KY, dy))/fxa
            fxb = N2 - self.dsqr*s**2*om2
            if self.periodic_in_z:
                qz = hat_kp(KZ, dz)*s*om/fxb
                qb = 1j*hat_em(KZ, dz)*hat_kp(KZ, dz)*self.N0**2/fxb
            else:
                lam = 2*np.sin(KZ*dz/2)/dz
                qz = 1j*lam*s*om/fxb
                qb = -lam*np.cos(KZ*dz/2)*self.N0**2/fxb
        qz = np.where((kh2 > 0) & (KZ**2 > 0), qz, 0.); qb = np.where((kh2 > 0) & (KZ**2 > 0), qb, 0.)
        qx = np.where(kh2 > 0, qx, -1j*s); qy = np.where(kh2 > 0, qy, s**2)
        zero = (s == 0) & ((f2 == 0) | (N2 == 0)) | (s != 0) & (om2 == 0)
        qx, qy, qz, qb = (np.where(zero, 0., q) for q in (qx, qy, qz, qb))
        if s == 0:
            # on the Nyquist lines kx = pi/dx or ky = pi/dy the averaging of
            # the Coriolis term vanishes (f2 = 0) and the vortical mode is the
            # non-divergent horizontal flow without b
            nyq = (f2 == 0) & (kh2 > 0)
            qx = np.where(nyq, np.exp(1j*KX*dx/2)*2*np.sin(KY*dy/2)/dy + 0*KZ, qx)
            qy = np.where(nyq, -np.exp(1j*KY*dy/2)*2*np.sin(KX*dx/2)/dx + 0*KZ, qy)
        if not self.periodic_in_z and s != 0:               # no waves without vertical structure between the lids
            qx, qy, qz, qb = (np.where((KZ**2 == 0) & (kh2 > 0), 0., q) for q in (qx, qy, qz, qb))
        if self.periodic_in_z and s != 0:                   # the horizontal gravity wave kz = 0
            hz = (kh2 > 0) & (KZ**2 == 0)
            qx = np.where(hz, 0., qx); qy = np.where(hz, 0., qy)
            qz = np.where(hz, 1j*s, qz); qb = np.where(hz, s**2*self.N0*np.sqrt(self.dsqr), qb)
        return [np.nan_to_num(q) + 0*KX*KY*KZ for q in (qx, qy, qz, qb)]

    def vec_p(self, s):
        p = self.coeff_p(s)
        qx, qy, qz, qb = self.vec_q(s)
        px, py, pz, pb = p*np.conj(qx), p*np.conj(qy), p*np.conj(qz)*self.dsqr, p*np.conj(qb)/self.N0**2
        KX, KY, KZ = self.KX, self.KY, self.KZ
        kh2 = KX**2 + KY**2 + 0*KZ
        if s == 0:
            k2, f2, N2, m2 = self._coef()
            nyq = (f2 == 0) & (kh2 > 0)
            px = np.where(nyq, np.conj(qx)/k2, px); py = np.where(nyq, np.conj(qy)/k2, py)
        # the horizontal mean flow, an inertial oscillation (the formula gives p = 0)
        v = (kh2 == 0) + 0*KZ
        px = np.where(v, 1j*s/2, px); py = np.where(v, s**2/2, py)
        if self.periodic_in_z:
            if s != 0:
                hz = (kh2 > 0) & (KZ**2 == 0)
                px = np.where(hz, 0., px); py = np.where(hz, 0., py)
                pz = np.where(hz, -1j*s/2, pz); pb = np.where(hz, s**2/(2*self.N0*np.sqrt(self.dsqr)), pb)
        return [px, py, pz, pb]

    # ------------------------------------------------------------------
    def _interior(self, a):
        h = self.halo
        return np.asarray(a)[h:-h, h:-h, h:-h]

    def _fft(self, a, mode=None):
        A = np.fft.fft(np.fft.rfft(a, axis=0), axis=1)*self.dx*self.dy
        if self.periodic_in_z:
            return np.fft.fft(A, axis=2)*self.dz
        return (A @ mode)*self.dz

    def _ifft(self, A, mode=None):
        if self.periodic_in_z:
            A = np.fft.ifft(A, axis=2)/self.dz
        else:
            A = A @ mode.T
        return np.fft.irfft(np.fft.ifft(A, axis=1), n=self.nx, axis=0)/(self.dx*self.dy)

    def forward(self, u, v, w, b):
        m = (None,)*4 if self.periodic_in_z else (self.phi, self.phi, self.psi, self.psib)
        return [self._fft(self._interior(a), mo) for a, mo in zip((u, v, w, b), m)]

    def backward(self, U, V, W, B):
        m = (None,)*4 if self.periodic_in_z else (self.phi, self.phi, self.psi, self.psib)
        out = []
        for A, mo in zip((U, V, W, B), m):
            a = np.zeros((self.Nx, self.Ny, self.Nz))
            h = self.halo
            a[h:-h, h:-h, h:-h] = self._ifft(A, mo)
            out.append(self.apply_bc(OM.np.asarray(a)))
        out[2] = self.apply_vertical_flux_bc(out[2])
        return out

    def project(self, u, v, w, b, s):
        F = self.forward(u, v, w, b)
        g = self._amp(F, s)
        q = self.vec_q(s)
        return self.backward(*[g*qq for qq in q])

    def mode_fields(self, i, j, k, s):
        """
        the single eigenmode (kx[i], ky[j], kz[k], s) with unit amplitude in
        physical space: the fields (u,v,w,b) of its real part and of its
        imaginary part, so that a*real + b*imag is the mode with complex
        amplitude a + i b and exp(-i omega t)*(real + i imag) its propagation
        """
        q = self.vec_q(s)
        shape = (len(self.kx), self.ny, self.nz)
        out = []
        for fac in (1., -1j):
            F = []
            for qq in q:
                A = np.zeros(shape, complex); A[i, j, k] = fac*qq[i, j, k]; F.append(A)
            out.append(self.backward(*F))
        return out

    def _amp(self, F, s):
        p = self.vec_p(s)
        return sum(pp*FF for pp, FF in zip(p, F))

    # ------------------------------------------------------------------
    # the model
    def _nonlin(self, u, v, w, b):
        """spectral amplitudes of the advection tendencies of (u,v,w,b) divided by Ro"""
        save = self.u, self.v, self.w, self.b
        self.u, self.v, self.w, self.b = u, v, w, b
        du = self.momentum_advection_u()/self.Ro
        dv = self.momentum_advection_v()/self.Ro
        dw = self.momentum_advection_w()/self.Ro/self.dsqr
        db = self.buoyancy_advection_2nd()/self.Ro
        self.u, self.v, self.w, self.b = save
        return self.forward(du, dv, dw, db)

    def _run(self, u, v, w, b, nsteps):
        """nsteps of the model from (u,v,w,b) without forcing, state restored afterwards"""
        save = (self.u*1, self.v*1, self.w*1, self.b*1, self.p*1, self.du*1, self.dv*1, self.dw*1, self.db*1)
        self.u, self.v, self.w, self.b = u*1, v*1, w*1, b*1
        self.set_forcing = lambda tau: None               # shadow the forcing of the configuration
        A, B, C = 1., 0., 0.
        tau, taum1, taum2 = 2, 1, 0
        for n in range(nsteps):
            self.time_step(A, B, C, tau, taum1, taum2)
            tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
            A, B, C = (1.5, -0.5, 0.) if n == 0 else (23./12, -16./12, 5./12)
        del self.set_forcing
        out = self.u, self.v, self.w, self.b
        self.u, self.v, self.w, self.b, self.p, self.du, self.dv, self.dw, self.db = save
        return out

    def _I(self, F, s):
        return 1j*self._amp(F, s)

    def _init_damping(self):
        """inverse frequencies and the damping, Eq. (27) of Eden et al. (2019)"""
        om = np.sqrt(self.omega2())
        k2, f2, N2, m2 = self._coef()
        D = 0*om
        if self.bal_damping and hasattr(self, 'Ahbi'):
            if abs(getattr(self, 'Khbi', 0.) - self.Ahbi) > 0 or \
               abs(getattr(self, 'Kvbi', 0.) - getattr(self, 'Avbi', 0.)) > 0:
                raise RuntimeError('balance: the damping needs Khbi = Ahbi and Kvbi = Avbi')
            if hasattr(self, 'Avbi'): D = (k2 + m2)*(self.Ahbi*k2 + self.Avbi*m2) + 0*om   # biharmonic
            else:                     D = self.Ahbi*k2**2 + 0*om                              # biharmonic_h_only
        kmax2 = (self.kx[-1] - 2*self.kx[1])**2             # the largest wavenumbers
        large = (self.KX**2 + self.KY**2 > kmax2) + 0*om
        self.bal_Ah = D                                      # damping of the geostrophic mode
        self.bal_AhK4 = np.where(large, 0., D)               # damping as order Ro term
        self.bal_iom = {}
        for s in (1, -1):
            with np.errstate(divide='ignore', invalid='ignore'):
                iom = np.where(large, 1./(s*om + 1j*D), 1./(s*om))
            self.bal_iom[s] = np.where(om > 0, np.nan_to_num(iom), 0.)
        return

    # ------------------------------------------------------------------
    # the orders of the balanced state, X = (u, v, w, b) in physical space
    def _modes(self, fp, fm):
        return [fp*a + fm*c for a, c in zip(self.vec_q(1), self.vec_q(-1))]

    def _fields(self, F):
        return self.backward(*F)

    def _orders_01(self, X):
        """geostrophic mode and first order of the state X"""
        g0 = self._amp(self.forward(*X), 0)
        X0 = self._fields([g0*qq for qq in self.vec_q(0)])
        fp1, fm1 = self._first_order(X0)
        X1 = self._fields([self.Ro*a for a in self._modes(fp1, fm1)])
        return X0, X1, fp1, fm1

    def _first_order(self, X0):
        F = self._nonlin(*X0)
        return self._I(F, 1)*self.bal_iom[1], self._I(F, -1)*self.bal_iom[-1]

    def _second_order(self, X0, X1, fp1, fm1):
        Ro, iom, AhK4 = self.Ro, self.bal_iom, self.bal_AhK4
        F = self._nonlin(*X0)
        I_00, I_0p, I_0m = self._I(F, 0), self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro for a in X1])
        I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c/Ro for a, c in zip(X0, X1)])
        I_0f1p, I_0f1m = self._I(F, 1), self._I(F, -1)
        # the time derivative of f1 from Eq. (25)
        g0 = self._amp(self.forward(*X0), 0)
        pTg0 = -1j*I_00 - self.bal_Ah*g0/Ro
        XT0 = self._fields([pTg0*qq for qq in self.vec_q(0)])
        F = self._nonlin(*XT0)
        I_pT0p, I_pT0m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c for a, c in zip(X0, XT0)])
        I_0pT0p, I_0pT0m = self._I(F, 1), self._I(F, -1)
        pTfp = (I_0pT0p - I_0p - I_pT0p)*iom[1]
        pTfm = (I_0pT0m - I_0m - I_pT0m)*iom[-1]
        fp = (I_0f1p - I_0p - I_f1p - 1j*(pTfp + AhK4/Ro*fp1))*iom[1]
        fm = (I_0f1m - I_0m - I_f1m - 1j*(pTfm + AhK4/Ro*fm1))*iom[-1]
        return fp, fm

    def _third_order(self, X0, X1, X2, fp1, fm1, fp2, fm2):
        Ro, iom, AhK4 = self.Ro, self.bal_iom, self.bal_AhK4
        F = self._nonlin(*X0)
        I_0p, I_0m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro**2 for a in X2])
        I_f2p, I_f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c/Ro**2 for a, c in zip(X0, X2)])
        I_0f2p, I_0f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro for a in X1])
        I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
        # the time derivative of f2 from nsteps model steps
        Xn = self._run(*[a + c + d for a, c, d in zip(X0, X1, X2)], self.bal_nsteps)
        X0n, X1n, fp1n, fm1n = self._orders_01(Xn)
        fp2n, fm2n = self._second_order(X0n, X1n, fp1n, fm1n)
        pTfp = (fp2n - fp2)/(self.bal_nsteps*self.dt)/Ro
        pTfm = (fm2n - fm2)/(self.bal_nsteps*self.dt)/Ro
        fp = (I_0f2p + I_f1p - I_f2p - I_0p - 1j*(pTfp + AhK4/Ro*fp2))*iom[1]
        fm = (I_0f2m + I_f1m - I_f2m - I_0m - 1j*(pTfm + AhK4/Ro*fm2))*iom[-1]
        return fp, fm

    def _fourth_order(self, X0, X1, X2, X3, fp1, fm1, fp2, fm2, fp3, fm3):
        Ro, iom, AhK4 = self.Ro, self.bal_iom, self.bal_AhK4
        F = self._nonlin(*X0)
        I_0p, I_0m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro for a in X1])
        I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro**3 for a in X3])
        I_f3p, I_f3m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c/Ro**3 for a, c in zip(X0, X3)])
        I_0f3p, I_0f3m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro**2 for a in X2])
        I_f2p, I_f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a/Ro + c/Ro**2 for a, c in zip(X1, X2)])
        I_f1f2p, I_f1f2m = self._I(F, 1), self._I(F, -1)
        L_p = I_f1f2p - I_f1p - I_f2p
        L_m = I_f1f2m - I_f1m - I_f2m
        # the time derivative of f3 from nsteps model steps
        Xn = self._run(*[a + c + d + e for a, c, d, e in zip(X0, X1, X2, X3)], self.bal_nsteps)
        X0n, X1n, fp1n, fm1n = self._orders_01(Xn)
        fp2n, fm2n = self._second_order(X0n, X1n, fp1n, fm1n)
        X2n = self._fields([Ro**2*a for a in self._modes(fp2n, fm2n)])
        fp3n, fm3n = self._third_order(X0n, X1n, X2n, fp1n, fm1n, fp2n, fm2n)
        pTfp = (fp3n - fp3)/(self.bal_nsteps*self.dt)/Ro
        pTfm = (fm3n - fm3)/(self.bal_nsteps*self.dt)/Ro
        fp = (I_0f3p - I_f3p - I_0p + L_p - 1j*(pTfp + AhK4/Ro*fp3))*iom[1]
        fm = (I_0f3m - I_f3m - I_0m + L_m - 1j*(pTfm + AhK4/Ro*fm3))*iom[-1]
        return fp, fm

    # ------------------------------------------------------------------
    def decompose(self, order=4):
        """
        The orders 0..order of the balanced model state after Warn et al.:
        the geostrophic mode of order Ro^0 and the slaved gravity wave modes
        of order Ro^1 to Ro^order. Returns the list of the orders, each a
        tuple of the fields (u, v, w, b) with halo points. The model state
        is not changed.
        """
        if order < 0 or order > 4: raise ValueError('balance: order 0..4')
        if not hasattr(self, 'bal_iom'): self._init_damping()
        def pr(s):
            if self.bal_verbose and self.my_pe == 0: print(s)
        Ro = self.Ro
        pr('balance: zero order')
        state = (self.u, self.v, self.w, self.b)
        X0, X1, fp1, fm1 = self._orders_01(state)
        X, f = [X0, X1], [(fp1, fm1)]
        if order >= 2:
            pr('balance: second order')
            f.append(self._second_order(X0, X1, fp1, fm1))
            X.append(self._fields([Ro**2*a for a in self._modes(*f[-1])]))
        if order >= 3:
            pr('balance: third order')
            f.append(self._third_order(X[0], X[1], X[2], *f[0], *f[1]))
            X.append(self._fields([Ro**3*a for a in self._modes(*f[-1])]))
        if order >= 4:
            pr('balance: fourth order')
            f.append(self._fourth_order(X[0], X[1], X[2], X[3], *f[0], *f[1], *f[2]))
            X.append(self._fields([Ro**4*a for a in self._modes(*f[-1])]))
        X = X[:order + 1]
        # the part of b that no mode represents (the horizontal mean of b,
        # the vertical checkerboard of b with the rigid lids, b on the
        # Nyquist plane kz = pi/dz) has no dynamics, neither balanced nor
        # wave; it is added to the zero order so that the residual contains
        # only dynamical modes
        F = self.forward(*state)
        g = {s: self._amp(F, s) for s in (0, 1, -1)}
        R = self.backward(*[sum(g[s]*self.vec_q(s)[c] for s in (0, 1, -1)) for c in range(4)])
        # only b: what is missing in u, v, w can only be divergent flow, which
        # the model removes with the next pressure solve anyway
        self.bal_null = state[3] - R[3]
        X[0] = (X[0][0], X[0][1], X[0][2], X[0][3] + self.bal_null)
        return X

    def balance(self, order=3):
        """
        Balance the model state (u,v,w,b) after Warn et al. to the given
        order. Sets u, v, w, b and keeps the orders as u0..u4, v0..
        """
        X = self.decompose(order)
        for m, x in enumerate(X):
            for name, a in zip('uvwb', x): setattr(self, name + str(m), a)
        self.u, self.v, self.w, self.b = [sum(x[i] for x in X) for i in range(4)]
        self.w = self.apply_vertical_flux_bc(self.w)
        return

    def norm(self, X, Y):
        """relative norm of the difference of two states (u, v, w, b), 2 |X - Y|^2 / (|X|^2 + |Y|^2)"""
        d = sum(float(np.sum(self._interior(x - y)**2)) for x, y in zip(X[:4], Y[:4]))
        n = sum(float(np.sum(self._interior(x)**2)) + float(np.sum(self._interior(y)**2))
                for x, y in zip(X[:4], Y[:4]))
        return 2*d/n
