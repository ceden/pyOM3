"""
Balance of the model state after Warn et al. (1995) to fourth order in the
Rossby number: warn_pe_simple for pe_simple and warn_nh_simple for nh_simple
(add the class to the model class bases). Moved here from
pe_simple/diag_balance.py and nh_simple/diag_balance.py (Nov 2026), the code
unchanged.

warn_pe_simple
--------------

 Balance of the model state after Warn et al. (1995) to fourth order in the
 Rossby number, the method of Eden, Chouksey and Olbers (2019, JPO 49, 2393).
 The balanced state is the geostrophic mode g0 plus the slaved gravity wave
 modes f1 .. f4. Port of the grav3D code (py_src/balance2.py,
 eigenvectors3.py, my_fft_reordered.py) to pe_simple.

 Only for pe_simple with buoyancy as tracer (eq_of_state=0), Coriolis
 parameter f0 (a variable coriolis_t enters as the nonlinear term
 (f - f0) u, see init_balance), 2nd order advection, a periodic domain without
 topography, rigid lid or implicit free surface (pressure method), one
 process. The background stratification is constant N0 with
 temp = N0**2 z + b and z = 0 at the surface when N0 is a scalar. When N0
 is an array of nz values of N^2 at the w levels (bottom to top), the
 background stratification is variable and the analytic vertical modes are
 replaced by the numerical eigenvectors of the discretized linear column
 dynamics, assembled with the exact model stencils and verified to be
 independent of the wavenumber at two probes (rigid lid only). The modes
 are then not orthogonal and the forward transform uses the matrix inverse
 of phi.

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

 Provides (class warn_pe_simple, add to the model class bases):
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

 The series diverges in the high vertical modes. As in warn_nh_simple (warn.py),
 with slave_modes the wave modes are slaved only for the vertical modes
 n < m_c (vertical wavenumber kz = n pi/H with H the depth), at all orders.
 Above the cut no wave mode is slaved and the wave part there stays in the
 residual. The zero order is not affected. slave_modes = 'auto' sets the cut
 at u_rms kz/N0 = 1, with u_rms the rms horizontal velocity of the geostrophic
 mode of the state, evaluated at each call of decompose.


warn_nh_simple
--------------

 Balance of the model state after Warn et al. (1995) to fourth order in the
 Rossby number for nh_simple, the counterpart of warn_pe_simple (warn.py),
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

 Provides (class warn_nh_simple, add to the model class bases):
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
 the same as in warn_pe_simple (warn.py): the slaved wave modes f1 .. f4
 from the interaction integrals of the model's own advection terms, the time
 derivative of f1 from their Eq. (25), those of f2, f3 from nsteps model
 steps, the biharmonic damping Ahbi (= Khbi) as in their Eq. (27). The part
 of the state that no mode represents (the horizontal mean of b, the vertical
 checkerboard of b with the rigid lids, the Nyquist lines) has no dynamics
 and is added to the zero order.

 The series diverges in the high vertical modes. A wave mode with vertical
 wavenumber kz has the frequency N0 kh/kz (for large Burger number) and the
 balanced flow advects with Ro u kh, so the mode can only be slaved if the
 vertical Froude number Ro u kz/N0 is small. With slave_modes the wave
 modes are slaved only for the vertical wavenumbers below a cut, at all
 orders. Above the cut no wave mode is slaved and the wave part there stays
 in the residual. The zero order is not affected. slave_modes = 'auto'
 sets the cut at Ro u_rms |kz|/N0 = 1, with u_rms the rms horizontal
 velocity of the geostrophic mode of the state, evaluated at each call of
 decompose.

"""
import pyOM3                     as OM
from   pyOM3.pe_simple.model     import model as _pe_simple
from   pyOM3.density             import get_rho
import numpy                     as np
try:
    from pyOM3.nh_simple.model import model as _nh_simple
except ImportError as _e:              # nh_simple needs the p3dfft build
    _nh_error = _e
    class _nh_simple:
        def __init__(self, *a, **k):
            raise ImportError('warn_nh_simple needs nh_simple, which needs the p3dfft build: %s' % _nh_error)


def hat_ksqr(k, d): return 2.*(1. - np.cos(k*d))/d**2
def hat_epm(k, d):  return 0.5*(1. + np.cos(k*d))
def hat_kp(k, d):   return -1j*(np.exp(1j*k*d) - 1.)/d
def hat_ep(k, d):   return (np.exp(1j*k*d) + 1.)*0.5
def hat_em(k, d):   return (np.exp(-1j*k*d) + 1.)*0.5
def hat_km(k, d):   return -1j*(1. - np.exp(-1j*k*d))/d


class warn_pe_simple(_pe_simple):

    # ------------------------------------------------------------------
    def init_balance(self, N0, nsteps=10, damping=True, pTf1_numeric=False, verbose=True,
                     slave_modes=None, f0=None):
        """
        N0            background stratification, temp = N0**2 z + b with z = 0 at the surface
        nsteps        model steps for the time derivatives of f2, f3 (Baer and Tribbia 1977)
        damping       account for the biharmonic damping Ahbi (needs Khbi = Ahbi)
        pTf1_numeric  time derivative of f1 from a model run instead of the analytic Eq. (25)
        slave_modes   None: wave modes slaved in all vertical modes
                      m_c (integer): wave modes slaved only in the vertical modes n < m_c
                      'auto': wave modes slaved only for u_rms kz/N0 < 1, kz = n pi/H
        f0            reference Coriolis parameter of the modes (default: the constant
                      coriolis_t, or its mean over the domain if it varies). With a
                      variable coriolis_t the difference (f - f0) times the velocity
                      (the Coriolis term of the model) is treated as a nonlinear
                      term, of order Ro, in all orders of the expansion, and the
                      geostrophic mode is that of the linear potential vorticity
                      with the local f (see _geostrophic_amp)
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
        self.bal_N0, self.bal_nsteps, self.bal_pTf1_numeric = N0, nsteps, pTf1_numeric
        self.bal_N2w = N2w
        self.bal_verbose = verbose
        self.bal_slave_modes = slave_modes
        self.bal_slave = None
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
        if mode is self.bal_phi:
            return np.einsum('nk,kji->nji', self.bal_phi_inv, A)*self.dz
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
        I = 1j*self._amp(F, s)
        if s != 0 and getattr(self, 'bal_slave', None) is not None:
            I = I*self.bal_slave                          # wave modes above the cut are not slaved
        return I

    def _set_slave(self, state):
        """the vertical modes in which wave modes are slaved, see slave_modes"""
        sm = getattr(self, 'bal_slave_modes', None)
        if sm is None:
            self.bal_slave = None
            return
        H = self.nz*self.dz
        if sm == 'auto':
            u, v, b, p = state
            g0 = self._amp(self.forward(u, v, p), 0)
            u0, v0 = self._fields([g0*qq for qq in self.bal_q[0]])[:2]
            u_rms = np.sqrt(np.mean(self._interior(u0)**2 + self._interior(v0)**2))
            m_c = self.bal_N0*H/(np.pi*u_rms) if u_rms > 0 else np.inf
        else:
            m_c = sm
        n = np.arange(self.nz)
        self.bal_slave = (n < m_c)[:, None, None]
        if self.bal_verbose and self.my_pe == 0:
            print('balance: wave modes slaved for n < %g, %d of %d vertical modes'
                  % (m_c, int(np.sum(n < m_c)), self.nz))
        return

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

    def _beta(self, u, v):
        """
        spectral amplitudes of the tendencies of u and v by the Coriolis term
        with f - f0, the discretization of the model, None for constant f
        """
        df = self.bal_df
        if df is None:
            return None
        du = self.maskU*(df[None, :, :]*(v + self.rolly(v, 1))
                         + self.rollx(df, -1)[None, :, :]*self.rollx(v + self.rolly(v, 1), -1))*0.25
        dv = -self.maskV*(df[None, :, :]*(u + self.rollx(u, 1))
                          + self.rolly(df, -1)[None, :, :]*self.rolly(u + self.rollx(u, 1), -1))*0.25
        return self.forward(self.apply_bc(du), self.apply_bc(dv), OM.np.zeros_like(u))

    def _local_pv_state(self, p):
        """
        variable f: the geostrophic state (u, v, p with halo points) whose
        linear potential vorticity with f0 is -(f - f0)/c_n^2 avg(p_n), the
        difference between the linear potential vorticity with the local f,
        dv/dx - du/dy - (f/c_n^2) avg(p_n), and that with f0, for the
        pressure p. None for constant f. See _geostrophic_amp.
        """
        if self.bal_df is None:
            return None
        h, dx, dy, dz, f0 = self.halo, self.dx, self.dy, self.dz, self.bal_f0
        IC2 = self.bal_inv_c2[:, None, None]
        r = np.roll
        Pn = np.einsum('nk,kji->nji', self.bal_phi_inv, self._interior(p))*dz
        df = np.asarray(self.bal_df)[h:-h, h:-h]
        dfc = 0.25*(df + r(df, -1, 1) + r(df, -1, 0) + r(r(df, -1, 1), -1, 0))
        Pc = 0.25*(Pn + r(Pn, -1, 2) + r(Pn, -1, 1) + r(r(Pn, -1, 2), -1, 1))
        dPi = -dfc[None, :, :]*IC2*Pc
        # exact inversion with f0 at the vorticity points
        KX = 2*np.pi*np.fft.fftfreq(self.nx, dx)[None, None, :]
        KY = 2*np.pi*np.fft.fftfreq(self.ny, dy)[None, :, None]
        Lhat = -(hat_ksqr(KX, dx) + hat_ksqr(KY, dy)) - f0**2*IC2*hat_epm(KX, dx)*hat_epm(KY, dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            iL = np.where(np.abs(Lhat) > 0, 1./Lhat, 0.)
        ps = np.real(np.fft.ifft2(np.fft.fft2(dPi, axes=(1, 2))*iL, axes=(1, 2)))
        Ug = -(ps - r(ps, 1, 1))/dy
        Vg = (ps - r(ps, 1, 2))/dx
        Pg = f0*0.25*(ps + r(ps, 1, 2) + r(ps, 1, 1) + r(r(ps, 1, 2), 1, 1))
        phi = self.bal_phi
        u = self._pad(np.einsum('kn,nji->kji', phi, Ug)/dz)*self.maskU
        v = self._pad(np.einsum('kn,nji->kji', phi, Vg)/dz)*self.maskV
        p = self._pad(np.einsum('kn,nji->kji', phi, Pg)/dz)
        return u, v, p

    def _geostrophic_amp(self, u, v, p, tol=1e-12, maxit=30):
        """
        amplitude g0 of the geostrophic mode of the fields u, v, p (with halo).
        For constant f the projection on the mode s = 0. For variable f the
        geostrophic state X0 is the one whose linear potential vorticity with
        the local f equals that of the fields. The waves at the local f do
        not carry this potential vorticity, but they do carry potential
        vorticity with f0, (f - f0)/c_n^2 avg(p_n) of their pressure, so that
        the plain projection would put part of the waves into g0. X0 follows
        from the projection of the fields plus the state of _local_pv_state
        for the pressure p - p0 that is not that of X0, by fixed point
        iterations (contraction about max |f - f0|/f0).
        """
        F = self.forward(u, v, p)
        g0 = self._amp(F, 0)
        if self.bal_df is None:
            return g0
        n0 = np.sqrt(np.sum(np.abs(g0)**2)) + 1e-300
        for it in range(maxit):
            p0 = self._fields([g0*qq for qq in self.bal_q[0]])[3]
            C = self._local_pv_state(p - p0)
            g = self._amp([a + c for a, c in zip(F, self.forward(*C))], 0)
            d = np.sqrt(np.sum(np.abs(g - g0)**2))/n0
            g0 = g
            if d < tol: break
        self.bal_pv_its = it + 1
        return g0

    def _Ib(self, X, s):
        """projection of the f - f0 tendency of the state X on mode s, 0 for constant f"""
        F = self._beta(X[0], X[1])
        return 0. if F is None else self._I(F, s)

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
        fp = (self._I(F, 1) + self._Ib(X0, 1))*self.bal_iom[1]
        fm = (self._I(F, -1) + self._Ib(X0, -1))*self.bal_iom[-1]
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
            pTg0 = -1j*(I_00 + self._Ib(X0, 0)) - self.bal_Ah*g0
            XT0 = self._fields([pTg0*qq for qq in self.bal_q[0]])
            F = self._nonlin(*XT0[:3])
            I_pT0p, I_pT0m = self._I(F, 1), self._I(F, -1)
            F = self._nonlin(*[a + c for a, c in zip(X0[:3], XT0[:3])])
            I_0pT0p, I_0pT0m = self._I(F, 1), self._I(F, -1)
            pTfp = (I_0pT0p - I_0p - I_pT0p + self._Ib(XT0, 1))*iom[1]
            pTfm = (I_0pT0m - I_0m - I_pT0m + self._Ib(XT0, -1))*iom[-1]
        else:
            # the time derivative of f1 from nsteps model steps
            Xn = self._run(*[a + c for a, c in zip(X0[:2], X1[:2])],
                           X0[2] + X1[2], self._p_s(X0, X1), self.bal_nsteps)
            X0n, X1n, fp1n, fm1n = self._orders_01(Xn)
            pTfp = (fp1n - fp1)/(self.bal_nsteps*self.dt)
            pTfm = (fm1n - fm1)/(self.bal_nsteps*self.dt)
        fp = (I_0f1p - I_0p - I_f1p + self._Ib(X1, 1) - 1j*(pTfp + AhK4*fp1))*iom[1]
        fm = (I_0f1m - I_0m - I_f1m + self._Ib(X1, -1) - 1j*(pTfm + AhK4*fm1))*iom[-1]
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
        fp = (I_0f2p + I_f1p - I_f2p - I_0p + self._Ib(X2, 1) - 1j*(pTfp + AhK4*fp2))*iom[1]
        fm = (I_0f2m + I_f1m - I_f2m - I_0m + self._Ib(X2, -1) - 1j*(pTfm + AhK4*fm2))*iom[-1]
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
        fp = (I_0f3p - I_f3p - I_0p + L_p + self._Ib(X3, 1) - 1j*(pTfp + AhK4*fp3))*iom[1]
        fm = (I_0f3m - I_f3m - I_0m + L_m + self._Ib(X3, -1) - 1j*(pTfm + AhK4*fm3))*iom[-1]
        return fp, fm

    def _p_s(self, *X):
        """surface pressure of the sum of the states X = (u, v, b, p)"""
        ks = -(self.halo + 1)
        return sum(x[3][ks] + self.dz/2*x[2][ks] for x in X)

    def _orders_01(self, X):
        """geostrophic mode and first order of the state X = (u, v, b, p)"""
        g0 = self._geostrophic_amp(X[0], X[1], X[3])
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
        state = self.model_state()
        self._set_slave(state)
        pr('balance: zero order')
        X0, X1, fp1, fm1 = self._orders_01(state)
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

    # ------------------------------------------------------------------
    # variable N^2: vertical modes from the discretized linear dynamics
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


class warn_nh_simple(_nh_simple):

    # ------------------------------------------------------------------
    def init_balance(self, nsteps=10, damping=True, verbose=True, slave_modes=None):
        """
        nsteps       model steps for the time derivatives of f2, f3 (Baer and Tribbia 1977)
        damping      account for the biharmonic damping Ahbi (needs Khbi = Ahbi, Kvbi = Avbi)
        slave_modes  None: wave modes slaved at all vertical wavenumbers
                     m_c (integer): wave modes slaved only for |kz| < m_c pi/Lz
                     (rigid lids) or |kz| < 2 m_c pi/Lz (periodic in z)
                     'auto': wave modes slaved only for Ro u_rms |kz|/N0 < 1
        """
        if self.n_pes > 1:
            raise RuntimeError('balance: one process only')
        self.bal_nsteps, self.bal_damping, self.bal_verbose = nsteps, damping, verbose
        self.bal_slave_modes = slave_modes
        self.bal_slave = None
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
        I = 1j*self._amp(F, s)
        if s != 0 and getattr(self, 'bal_slave', None) is not None:
            I = np.where(self.bal_slave, I, 0.)          # wave modes above the cut are not slaved
        return I

    def _set_slave(self, state):
        """the vertical wavenumbers at which wave modes are slaved, see slave_modes"""
        sm = getattr(self, 'bal_slave_modes', None)
        kz = np.abs(self.KZ) + 0*self.KX*self.KY
        if sm is None:
            self.bal_slave = None
            return
        if sm == 'auto':
            g0 = self._amp(self.forward(*state), 0)
            u0, v0 = self._fields([g0*qq for qq in self.vec_q(0)])[:2]
            u_rms = np.sqrt(np.mean(self._interior(u0)**2 + self._interior(v0)**2))
            kz_c = self.N0/(self.Ro*u_rms) if u_rms > 0 else np.inf
        else:
            kz_c = sm*(2 if self.periodic_in_z else 1)*np.pi/self.Lz
        self.bal_slave = kz < kz_c
        if self.bal_verbose and self.my_pe == 0:
            print('balance: wave modes slaved for |kz| < %g, %d of %d vertical wavenumbers'
                  % (kz_c, int(np.sum(np.abs(self.kz) < kz_c)), self.nz))
        return

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
        self._set_slave(state)
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
