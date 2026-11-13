"""
Shared parts of the balance methods (Nov 2026, step 3 of the move into
pyOM3/diagnostics/balance):

  hat_*         symbols of the discrete differences and averages on the C grid
  _grid         interior arrays, padding with halo, the hydrostatic pressure of
                a buoyancy field (pe_simple and pe_realistic)
  _stratified   pe_simple: the background stratification and Coriolis
                parameter, the vertical modes, the model state (u, v, b, p)
                and a model run without forcing, used by warn_pe_simple
                (kernel_pe_simple) and band_pe_simple
"""
import numpy                 as np
import pyOM3                 as OM
from   pyOM3.density         import get_rho


def hat_ksqr(k, d): return 2.*(1. - np.cos(k*d))/d**2

def hat_epm(k, d):  return 0.5*(1. + np.cos(k*d))

def hat_kp(k, d):   return -1j*(np.exp(1j*k*d) - 1.)/d

def hat_ep(k, d):   return (np.exp(1j*k*d) + 1.)*0.5

def hat_em(k, d):   return (np.exp(-1j*k*d) + 1.)*0.5

def hat_km(k, d):   return -1j*(1. - np.exp(-1j*k*d))/d


class _grid:
    """interior arrays, padding with halo, the hydrostatic pressure"""

    def _interior(self, a):
        h = self.halo
        return np.asarray(a)[h:-h, h:-h, h:-h]

    def _pad(self, a):
        h = self.halo
        out = np.zeros((self.Nz, self.Ny, self.Nx))
        out[h:-h, h:-h, h:-h] = a
        return self.apply_bc(OM.np.asarray(out))

    def hydrostatic(self, b):
        """pressure from buoyancy, referenced to the surface, the discrete hydrostatic relation of the model"""
        rho = -b*self.rho_0/self.grav
        return self.hydrostatic_pressure(rho, OM.np.zeros_like(b), self.maskT)


class _stratified(_grid):
    """pe_simple: background stratification, vertical modes, model state and model run"""

    def _init_stratification(self, N0, nsteps=10, verbose=True, f0=None):
        """
        the background of the pe_simple balance methods: the checks of the
        model, the background buoyancy bal_b_back from N0 (a number, or N^2 at
        the w levels, rigid lid), the reference Coriolis parameter bal_f0 and
        its deviation bal_df, the horizontal wavenumbers, and the vertical
        modes bal_phi (pressure), bal_psi (buoyancy) with c_n^2 bal_c2.
        Arguments as init_balance of warn_pe_simple.
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
        N2w = None
        if not np.isscalar(N0):
            N2w = np.asarray(N0, dtype='f8')
            if self.implicit_free_surface:
                raise RuntimeError('balance: variable N^2 needs the rigid lid')
            N0 = float(np.sqrt(np.mean(N2w)))
        self.bal_N0, self.bal_nsteps = N0, nsteps
        self.bal_N2w = N2w
        self.bal_verbose = verbose
        self.bal_tau = 0                                   # time level of p_s of the model state
        h = self.halo
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        fT = np.asarray(self.coriolis_t, dtype='f8')
        fi = fT[h:-h, h:-h]
        if f0 is None:
            f0 = float(fi[0, 0]) if fi.max() == fi.min() else float(fi.mean())
        f0 = float(f0)
        self.bal_f0 = f0
        self.bal_df = None
        if abs(fi - f0).max() > 0:
            # variable Coriolis parameter, (f - f0) u is a nonlinear term
            self.bal_df = OM.np.asarray(fT - f0)
            if verbose and self.my_pe == 0:
                print('balance: variable f, f0 = %.4g 1/s, max |f - f0|/f0 = %.3g'
                      % (f0, abs(fi - f0).max()/abs(f0)))

        # background buoyancy, z = 0 at the surface, from the discrete
        # integral of N^2 at the w levels (equal to N0**2 zt for constant N^2)
        N2col = N2w if N2w is not None else N0**2 + np.zeros(nz)
        Bcol = np.zeros(nz)
        for k in range(nz-2, -1, -1):
            Bcol[k] = Bcol[k+1] - N2col[k]*dz
        Bcol -= Bcol[-1] + N2col[-1]*dz/2.
        self.bal_B = Bcol
        Bfull = np.zeros(self.Nz)
        Bfull[h:-h] = Bcol
        Bfull[:h] = Bcol[0]; Bfull[-h:] = Bcol[-1]
        self.bal_b_back = OM.np.asarray(Bfull[:, None, None] + np.zeros((self.Nz, self.Ny, self.Nx)))

        # horizontal wavenumbers
        self.bal_kx = 2*np.pi*np.fft.rfftfreq(nx, dx)
        self.bal_ky = 2*np.pi*np.fft.fftfreq(ny, dy)
        KX, KY = self.bal_kx[None, None, :], self.bal_ky[None, :, None]

        # vertical modes phi[k, n], k = 0 at the bottom, with c^2_n
        if N2w is not None:
            self.bal_f0 = f0
            c2b, phi_b, psi_b = self._modes_from_probe(2, 3, Bcol)
            c2c, phi_c, _ = self._modes_from_probe(4, 7, Bcol)
            if len(c2b) != nz-1 or len(c2c) != nz-1:
                raise RuntimeError('balance: %d gravity modes, expected %d'
                                   % (len(c2b), nz-1))
            dev = np.max(np.abs(c2c - c2b)/c2b)
            devp = 0.
            for n_ in range(nz-1):
                a_, b_ = phi_b[:, n_], phi_c[:, n_]
                devp = max(devp, abs(abs(a_@b_)/(np.linalg.norm(a_)*np.linalg.norm(b_)) - 1.))
            if dev > 1e-7 or devp > 1e-7:
                raise RuntimeError('balance: vertical structures depend on the '
                                   'wavenumber (c2 %.1e, phi %.1e)' % (dev, devp))
            phi = np.zeros((nz, nz)); psi = np.zeros((nz, nz))
            phi[:, 0] = np.sqrt(1./nz)
            phi[:, 1:] = phi_b
            psi[:, 1:] = psi_b
            c2 = np.zeros(nz)
            c2[0] = self.grav*nz*dz
            c2[1:] = c2b
            inv_c2 = np.zeros(nz)
            inv_c2[1:] = 1./c2[1:]
            if verbose and self.my_pe == 0:
                print('balance: probe consistency c2 %.1e, structures %.1e' % (dev, devp))
        elif self.implicit_free_surface:
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
        if N2w is None:
            psi = -N0**2*inv_c2[None, :]*dz*(np.cumsum(phi, axis=0) - phi/2)
        self.bal_phi, self.bal_psi, self.bal_c2, self.bal_inv_c2 = phi, psi, c2, inv_c2
        self.bal_phi_inv = np.linalg.inv(phi)
        return

    def diagnose(self, n, tau):
        """keep the time level of p_s of the model state"""
        self.bal_tau = tau
        return super().diagnose(n, tau)

    def model_state(self):
        """u, v, b, p of the model state, b the buoyancy perturbation"""
        b = self.temp - self.bal_b_back
        p = self.p_s[self.bal_tau][None, :, :] + self.hydrostatic(b)
        return self.u, self.v, b, p

    def _run(self, u, v, b, p_s, nsteps):
        """nsteps of the model from (u, v, b, p_s) without forcing, the state is restored afterwards"""
        names = ('u', 'v', 'w', 'temp', 'salt', 'rho', 'p_s', 'p_h', 'du', 'dv', 'dtemp', 'dsalt')
        save = {a: getattr(self, a)*1 for a in names}
        save_tau = self.bal_tau
        # a cold start: the old tendency levels enter the first steps with
        # weight zero, which does not remove a NaN left by an earlier run
        for a in ('du', 'dv', 'dtemp', 'dsalt'):
            setattr(self, a, OM.np.zeros_like(getattr(self, a)))
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

    def _hyd_column(self, Bc):
        """discrete hydrostatic pressure column from a buoyancy column,
        p = 0 at the top t level, the surface pressure is separate"""
        nz, dz = self.nz, self.dz
        P = np.zeros(nz, dtype=complex)
        for k in range(nz-2, -1, -1):
            P[k] = P[k+1] - dz*(Bc[k] + Bc[k+1])/2.
        return P

    def _column_operator(self, kx, ky, B):
        """the linear operator of the columns (U, V, B) at wavenumber
        (kx, ky), assembled by probing, as a 3 nz x 3 nz complex matrix"""
        nz, dz, f0 = self.nz, self.dz, self.bal_f0
        dx, dy = self.dx, self.dy
        # the pressure gradient at the u and v points is the forward
        # difference (symbol i kp), the divergence at the t points the
        # backward difference (symbol i km)
        ipx, ipy = 1j*hat_kp(kx, dx), 1j*hat_kp(ky, dy)
        imx, imy = 1j*hat_km(kx, dx), 1j*hat_km(ky, dy)
        fex = hat_ep(kx, dx)*hat_em(ky, dy)
        fem = hat_em(kx, dx)*hat_ep(ky, dy)
        k2 = hat_ksqr(kx, dx) + hat_ksqr(ky, dy)
        def tend(U, V, Bc):
            div = imx*U + imy*V
            w = -np.cumsum(div)*dz                    # w at the w levels
            F = np.zeros(nz, dtype=complex)           # flux w Bbar^z
            F[:nz-1] = w[:nz-1]*(B[:nz-1] + B[1:])/2.
            dB = np.empty(nz, dtype=complex)
            dB[0] = -F[0]/dz
            dB[1:] = -(F[1:] - F[:nz-1])/dz
            dB -= B*div                       # horizontal advection of B(z)
            P = self._hyd_column(Bc)
            dU = f0*fex*V - ipx*P
            dV = -f0*fem*U - ipy*P
            # rigid lid: P_s such that the mean divergence tendency vanishes,
            # the divergence of -i kp P_s has the symbol +ksqr (im ip = -ksqr)
            Ps = -np.mean(imx*dU + imy*dV)/k2 if k2 > 0 else 0.
            return dU - ipx*Ps, dV - ipy*Ps, dB
        A = np.zeros((3*nz, 3*nz), dtype=complex)
        for i in range(3*nz):
            x = np.zeros(3*nz, dtype=complex); x[i] = 1.
            dU, dV, dB = tend(x[:nz], x[nz:2*nz], x[2*nz:])
            A[:nz, i], A[nz:2*nz, i], A[2*nz:, i] = dU, dV, dB
        return A

    def _modes_from_probe(self, ix, iy, B):
        """c_n^2, pressure structures phi_n and buoyancy structures psi_n
        of the gravity wave modes at the probe wavenumber indices (ix, iy)"""
        nz, f0 = self.nz, self.bal_f0
        dx, dy = self.dx, self.dy
        kx = 2*np.pi*np.fft.rfftfreq(self.nx, dx)[ix]
        ky = 2*np.pi*np.fft.fftfreq(self.ny, dy)[iy]
        A = self._column_operator(kx, ky, B)
        lam, vec = np.linalg.eig(A)
        om = np.imag(lam)
        k2 = hat_ksqr(kx, dx) + hat_ksqr(ky, dy)
        f2 = f0**2*hat_epm(kx, dx)*hat_epm(ky, dy)
        iw = np.where(om > np.sqrt(f2)*(1. + 1e-8))[0]
        iw = iw[np.argsort(-om[iw])]                  # fast to slow
        c2, phi, psi = [], [], []
        ikx = 1j*hat_kp(kx, dx)
        fex = hat_ep(kx, dx)*hat_em(ky, dy)
        for i in iw:
            U, V, Bc = vec[:nz, i], vec[nz:2*nz, i], vec[2*nz:, i]
            # full pressure of the mode from its U momentum balance
            Pfull = (f0*fex*V - 1j*om[i]*U)/ikx
            s = Pfull[np.argmax(np.abs(Pfull))]       # make the columns real
            Pfull, Bc = Pfull/s, Bc/s
            if np.max(np.abs(np.imag(Pfull))) > 1e-8*np.max(np.abs(Pfull)):
                raise RuntimeError('balance_n: mode pressure not real')
            c2.append((om[i]**2 - f2)/k2)
            phi.append(np.real(Pfull))
            psi.append(np.real(Bc))
        return np.array(c2), np.array(phi).T, np.array(psi).T
