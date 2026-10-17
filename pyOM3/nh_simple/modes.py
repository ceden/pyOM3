"""
 Eigenmodes of the linear equations of nh_simple and the higher order
 balance after Warn et al. (1995), ported from ps3D (eigenvectors.f90,
 diag_balance.f90, my_fft.f90).

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

 Provides (mixin class `modes`, add to the model class bases):
   init_modes()               wavenumbers, broadcast arrays, mode matrices
   omega2(), vec_q(s), vec_p(s)
   forward(u,v,w,b)           physical (with halo) -> spectral
   backward(U,V,W,B)          spectral -> physical (with halo)
   project(u,v,w,b,s)         projection on mode s in physical space
   mode_fields(i,j,k,s)       one eigenmode in physical space, for wave makers
   balance(order)             the model state (u,v,w,b) balanced to
                              order 0..3 in the Rossby number Ro
"""
import numpy as np
import pyOM3 as OM


def hat_ksqr(k, d): return 2.*(1. - np.cos(k*d))/d**2
def hat_epm(k, d):  return 0.5*(1. + np.cos(k*d))
def hat_kp(k, d):   return -1j*(np.exp(1j*k*d) - 1.)/d
def hat_ep(k, d):   return (np.exp(1j*k*d) + 1.)*0.5
def hat_em(k, d):   return (np.exp(-1j*k*d) + 1.)*0.5


class modes:

    # ------------------------------------------------------------------
    def init_modes(self):
        if self.n_pes > 1:
            raise RuntimeError('modes: one process only')
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
            om2 = np.where(Ksqr > 0, (N2*k2 + f2*m2)/Ksqr, 0.)
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
        if self.periodic_in_z and s != 0:                   # the horizontal gravity wave kz = 0
            hz = (kh2 > 0) & (KZ**2 == 0)
            qx = np.where(hz, 0., qx); qy = np.where(hz, 0., qy)
            qz = np.where(hz, 1j*s, qz); qb = np.where(hz, s**2*self.N0*np.sqrt(self.dsqr), qb)
        return [np.nan_to_num(q) + 0*KX*KY*KZ for q in (qx, qy, qz, qb)]

    def vec_p(self, s):
        p = self.coeff_p(s)
        qx, qy, qz, qb = self.vec_q(s)
        px, py, pz, pb = p*np.conj(qx), p*np.conj(qy), p*np.conj(qz)*self.dsqr, p*np.conj(qb)/self.N0**2
        if self.periodic_in_z:
            KX, KY, KZ = self.KX, self.KY, self.KZ
            kh2 = KX**2 + KY**2 + 0*KZ
            v = (kh2 == 0) & (KZ**2 > 0)
            px = np.where(v, 1j*s/2, px); py = np.where(v, s**2/2, py)
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
    def _nonlin(self, u, v, w, b):
        """spectral tendencies of the advection terms of (u,v,w,b) divided by Ro"""
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

    def balance(self, order=3):
        """
        Balance the model state (u,v,w,b) after Warn et al.: the geostrophic
        mode of order Ro^0 plus the slaved gravity wave modes of order Ro^1
        to Ro^3. Sets u, v, w, b and keeps the orders as u0..u3, v0..
        """
        Ro = self.Ro
        om = np.sqrt(self.omega2())
        inv_om = np.where(om > 0, 1./np.where(om > 0, om, 1.), 0.)
        qp, qm, q0 = self.vec_q(1), self.vec_q(-1), self.vec_q(0)

        def zero_order():
            u0 = self.project(self.u, self.v, self.w, self.b, 0)
            return u0

        def first_order(u0):
            F = self._nonlin(*u0)
            I_00, I_0p, I_0m = self._I(F, 0), self._I(F, 1), self._I(F, -1)
            f1p, f1m = I_0p*inv_om, -I_0m*inv_om
            u1 = self.backward(*[Ro*(f1p*a + f1m*c) for a, c in zip(qp, qm)])
            return u1, (I_00, I_0p, I_0m, f1p, f1m)

        def second_order(u0, u1, I):
            I_00, I_0p, I_0m, f1p, f1m = I
            F = self._nonlin(*[a/Ro for a in u1])
            I_f1p, I_f1m = self._I(F, 1), self._I(F, -1)
            F = self._nonlin(*[a + c/Ro for a, c in zip(u0, u1)])
            I_0f1p, I_0f1m = self._I(F, 1), self._I(F, -1)
            pT0 = self.backward(*[-1j*I_00*a for a in q0])
            F = self._nonlin(*pT0)
            I_pT0p, I_pT0m = self._I(F, 1), self._I(F, -1)
            F = self._nonlin(*[a + c for a, c in zip(u0, pT0)])
            I_0pT0p, I_0pT0m = self._I(F, 1), self._I(F, -1)
            pTf1p = (I_0pT0p - I_0p - I_pT0p)*inv_om
            pTf1m = -(I_0pT0m - I_0m - I_pT0m)*inv_om
            f2p = (I_0f1p - I_0p - I_f1p - 1j*pTf1p)*inv_om
            f2m = -(I_0f1m - I_0m - I_f1m - 1j*pTf1m)*inv_om
            u2 = self.backward(*[Ro**2*(f2p*a + f2m*c) for a, c in zip(qp, qm)])
            return u2, (I_f1p, I_f1m, f2p, f2m)

        u0 = zero_order()
        if order == 0:
            self.u, self.v, self.w, self.b = u0
            return
        u1, I1 = first_order(u0)
        if order == 1:
            self.u, self.v, self.w, self.b = [a + c for a, c in zip(u0, u1)]
            return
        u2, I2 = second_order(u0, u1, I1)
        if order == 2:
            self.u, self.v, self.w, self.b = [a + c + d for a, c, d in zip(u0, u1, u2)]
            return
        # third order
        I_00, I_0p, I_0m, f1p, f1m = I1
        I_f1p, I_f1m, f2p, f2m = I2
        F = self._nonlin(*[a/Ro**2 for a in u2])
        I_f2p, I_f2m = self._I(F, 1), self._I(F, -1)
        F = self._nonlin(*[a + c/Ro**2 for a, c in zip(u0, u2)])
        I_0f2p, I_0f2m = self._I(F, 1), self._I(F, -1)
        # time derivative of f2 from 10 model steps
        nsteps = 10
        u3 = self._run(*[a + c + d for a, c, d in zip(u0, u1, u2)], nsteps)
        u0n = self.project(*u3, 0)
        u1n, I1n = first_order(u0n)
        u2n, I2n = second_order(u0n, u1n, I1n)
        pTf2p = (I2n[2] - f2p)/(nsteps*self.dt)/Ro
        pTf2m = (I2n[3] - f2m)/(nsteps*self.dt)/Ro
        f3p = (I_0f2p + I_f1p - I_f2p - I_0p - 1j*pTf2p)*inv_om
        f3m = -(I_0f2m + I_f1m - I_f2m - I_0m - 1j*pTf2m)*inv_om
        u3 = self.backward(*[Ro**3*(f3p*a + f3m*c) for a, c in zip(qp, qm)])
        self.u0, self.v0, self.w0, self.b0 = u0
        self.u1, self.v1, self.w1, self.b1 = u1
        self.u2, self.v2, self.w2, self.b2 = u2
        self.u3, self.v3, self.w3, self.b3 = u3
        self.u, self.v, self.w, self.b = [a + c + d + e for a, c, d, e in zip(u0, u1, u2, u3)]
        return
