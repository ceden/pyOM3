"""
References of the 2-D shallow-water code of the Lagrangian-filtering draft
(sw_model.py, balance.py, eigenvectors.py, my_fft.py, opv_balance.py,
sw_particles.py and the kernel recipes of run_pv_table.py and
run_pv_first.py) for the port to pyOM3 as sw_simple (Nov 2026). The port
must reproduce these to round-off (its test comes with sw_simple).

    python3 sw_reference.py SRC            write sw_reference.npz from the
                                           original code in directory SRC
    python3 sw_reference.py SRC --check    recompute and compare with the
                                           stored file (reproducibility)

The case: 31 x 31 points (odd, no Nyquist line) on 2 pi x 2 pi, f0 = 1,
c^2 = Bu = 1, Ro = 0.1, dt = 0.01, nondimensional as in the draft. The
initial state is the balanced random field of the draft (spectrum with
d = 7, k0 = 3 at this resolution, seed 12345, fourth-order balanced, the
same recipe as make_bal of run_hires.py) plus one wave mode (3, 2). All
arrays are interior arrays in the x-first layout of the original code,
(Nx, Ny) without the halo.

Stored (keys of the npz):
  init/{u,v,h}                    the initial state
  step50_<scheme>/{u,v,h}         50 model steps, schemes AB, AB3, AB4, RK3
  state/{u,v,h}                   200 AB3 steps: the state of the diagnostics
  warn<n>/{u,v,h}                 balance_<n>_order, n = 0..4 (no damping)
  warn_damped<n>/{u,v,h}          the same with Ahbi = Khbi = 1e-5 (n = 0..4)
  ob_<variant>/{u,v,h}            opv.diag: fft_cos (T = 1, 3 iterations),
                                  fft_exp, average (time averaging T_ave = 2 pi,
                                  2 iterations)
  kernel0_exact/{u,v,h}           PV inversion with the exact inverse (run_pv_table)
  kernel0_R3/{u,v,h}              the tapered kernel R = 3 (run_pv_table)
  kernel1_R3/{u,v,h}              the first order with the nine kernels truncated
                                  at R = 3 (run_pv_first)
  particles_<scheme>/{x,y,u,v,h}  144 particles (12 x 12 regular), 100 steps
                                  of the AB3 model from the state, positions
                                  every 10 steps (x, y: (11, 144)) and the
                                  interpolated u, v, h at the end; velocities
                                  times Ro as in run_hires.py; schemes Euler,
                                  RK3, AB2, AB3, AB4
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sw_reference.npz')

N = 31
LX = LY = 2*np.pi
DX, DY = LX/N, LY/N
DT, F0, CSQR, RO = 0.01, 1., 1., 0.1


def params(**kw):
    pa = {'Lx': LX, 'Ly': LY, 'Nx': N + 2, 'Ny': N + 2, 'dx': DX, 'dy': DY, 'dt': DT, 'snapint': 1e9,
          'f0': F0, 'csqr': CSQR, 'Ro': RO, 'time_scheme': 'AB3', 'AB_eps': 0.0, 'Ahbi': 0., 'Khbi': 0.}
    pa.update(kw)
    return pa


def compute(src):
    sys.path.insert(0, os.path.abspath(src))
    from sw_model import sw_model
    from sw_particles import sw_particles
    from balance import balance
    from opv_balance import opv
    from eigenvectors import vec_q, coeff_p, hat_ksqr, hat_epm
    from my_fft import my_fft, my_ifft
    out = {}
    put = lambda tag, u, v, h: out.update({tag + '/u': np.array(u), tag + '/v': np.array(v), tag + '/h': np.array(h)})

    pa = params()
    B = balance(pa)
    kx = np.fft.fftshift(2*np.pi*np.fft.fftfreq(N, DY))
    ky = np.fft.fftshift(2*np.pi*np.fft.fftfreq(N, DX))
    Ky, Kx = np.meshgrid(ky, kx)

    # the balanced random field (make_bal of run_hires.py, k0 = 3) and a wave
    K = np.sqrt(Kx**2 + Ky**2)
    d, k0 = 7., 3.
    b = (7. + d)/4.; a = (4./7.)*b - 1
    rng = np.random.default_rng(12345)
    r = rng.standard_normal(Kx.shape) + 1j*rng.standard_normal(Kx.shape)
    H = np.sqrt(K**6/(K**2 + a*k0**2)**(2*b))*r
    kx_max = 2./3.*np.amax(np.abs(kx))
    H = np.where(np.sqrt((Kx*LX)**2 + (Ky*LY)**2) >= kx_max*LX, 0, H)
    u_, v_, h_ = B.inverse_trans(-1j*Ky*H/F0, 1j*Kx*H/F0, H)
    u_, v_, h_ = B.balance_zero_order(u_, v_, h_)
    h_ -= np.mean(h_[:])
    scal = 0.2/np.amax(np.abs(h_[:]))
    ub, vb, hb = B.balance_fourth_order(u_*scal, v_*scal, h_*scal)
    E = np.zeros(Kx.shape, 'f')
    E[N//2 + 2, N//2 + 3] = 0.5
    qxp, qyp, qzp = vec_q(Kx, Ky, +1., DX, DY, F0, CSQR)
    pp = coeff_p(Kx, Ky, +1., DX, DY, F0, CSQR)
    uw, vw, hw = [np.real(my_ifft(np.sqrt(E*2*pp)*q, DX, DY))*2*np.pi for q in (qxp, qyp, qzp)]
    ui, vi, hi = ub + uw, vb + vw, hb + hw
    put('init', ui, vi, hi)

    def model(pa_, u, v, h, nsteps, cls=sw_model):
        M = cls(pa_); M.allocate_memory()
        M.u[1:-1, 1:-1], M.v[1:-1, 1:-1], M.h[1:-1, 1:-1] = u, v, h
        M.boundary_conditions()
        M.loop(nsteps)
        return M

    for sch in ('AB', 'AB3', 'AB4', 'RK3'):
        M = model(params(time_scheme=sch), ui, vi, hi, 50)
        put('step50_%s' % sch, M.u[1:-1, 1:-1], M.v[1:-1, 1:-1], M.h[1:-1, 1:-1])
    M = model(pa, ui, vi, hi, 200)
    us, vs, hs = M.u[1:-1, 1:-1].copy(), M.v[1:-1, 1:-1].copy(), M.h[1:-1, 1:-1].copy()
    put('state', us, vs, hs)

    # Warn et al. to fourth order, without and with damping
    for tag, B_ in (('warn', B), ('warn_damped', balance(params(Ahbi=1e-5, Khbi=1e-5)))):
        for n, f in enumerate((B_.balance_zero_order, B_.balance_first_order, B_.balance_second_order,
                               B_.balance_third_order, B_.balance_fourth_order)):
            put('%s%d' % (tag, n), *f(us.copy(), vs.copy(), hs.copy()))

    # optimal balance
    for tag, kw in (('fft_cos', dict(T=1., FFT_zero_order=True, ramp_func='cos', max_iterations=3)),
                    ('fft_exp', dict(T=1., FFT_zero_order=True, ramp_func='exp', max_iterations=3)),
                    ('average', dict(T=1., FFT_zero_order=False, ramp_func='cos', max_iterations=2, T_ave=2*np.pi))):
        O = opv(params())
        O.diag(us.copy(), vs.copy(), hs.copy(), stop_criterion=1e-14, **kw)
        put('ob_' + tag, O.u[1:-1, 1:-1], O.v[1:-1, 1:-1], O.h[1:-1, 1:-1])

    # the kernel recipes: PV inversion (run_pv_table.py) and the first order
    # with truncated kernels (run_pv_first.py), R = 3
    Bu = CSQR
    kxf = np.fft.fftfreq(N, DX)*2*np.pi
    KXf, KYf = np.meshgrid(kxf, kxf, indexing='ij')
    Lhat = -(hat_ksqr(KXf, DX) + hat_ksqr(KYf, DY)) - (F0**2/Bu)*hat_epm(KXf, DX)*hat_epm(KYf, DY)
    Kd = np.real(np.fft.ifft2(1./Lhat))
    x = np.minimum(np.arange(N)*DX, LX - np.arange(N)*DX)
    X, Y = np.meshgrid(x, x, indexing='ij'); Rd = np.sqrt(X**2 + Y**2)
    taper = lambda R: 0.5*(1 + np.cos(np.pi*np.clip((Rd - R/2)/(R/2), 0, 1)))
    avg_h2c = lambda a: 0.25*(a + np.roll(a, -1, 0) + np.roll(a, -1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))
    avg_c2h = lambda a: 0.25*(a + np.roll(a, 1, 0) + np.roll(a, 1, 1) + np.roll(np.roll(a, 1, 0), 1, 1))
    zeta = (np.roll(vs, -1, 0) - vs)/DX - (np.roll(us, -1, 1) - us)/DY
    Ph = np.fft.fft2(zeta - (F0/Bu)*avg_h2c(hs))
    for tag, Kf in (('exact', 1./Lhat), ('R3', np.fft.fft2(Kd*taper(3.)))):
        psi = np.real(np.fft.ifft2(Ph*Kf))
        put('kernel0_' + tag, -(psi - np.roll(psi, 1, 1))/DY, (psi - np.roll(psi, 1, 0))/DX, F0*avg_c2h(psi))
    u0, v0, h0 = B.balance_zero_order(us.copy(), vs.copy(), hs.copy())
    DU, DV, DH = B.model_non_lin(u0, v0, h0)
    qp, qm = (B.qxp, B.qyp, B.qzp), (B.qxm, B.qym, B.qzm)
    pp_, pm = (B.pxp, B.pyp, B.pzp), (B.pxm, B.pym, B.pzm)
    dl = np.zeros((N, N)); dl[0, 0] = 1.
    Fd = my_fft(dl, DX, DY)
    W = taper(3.)
    corr = []
    for iX in range(3):
        acc = 0.
        for jD, Dj in enumerate((DU, DV, DH)):
            S = 1j*B.iom*(qp[iX]*pp_[jD] - qm[iX]*pm[jD])
            ker = np.real(my_ifft(S*Fd, DX, DY))*(2*np.pi)
            acc = acc + my_fft(ker*W, DX, DY)/(2*np.pi)/Fd*Dj
        corr.append(np.real(my_ifft(acc, DX, DY))*(2*np.pi))
    put('kernel1_R3', u0 + RO*corr[0], v0 + RO*corr[1], h0 + RO*corr[2])

    # particles (run_hires.py: velocities times Ro, integrated after each model step)
    nrec, every, nsteps = 11, 10, 100
    for sch in ('Euler', 'RK3', 'AB2', 'AB3', 'AB4'):
        P = np.zeros((2, 144, nrec))
        class sw_local(sw_particles):
            def particles_find_u(self):
                return self.pa['Ro']*sw_particles.particles_find_u(self)
            def particles_find_v(self):
                return self.pa['Ro']*sw_particles.particles_find_v(self)
            def show(self, n):
                if n % every == 0:
                    P[:, :, n//every] = self.pxy
                self.integrate_particles()
                self.particles_update_pointers()
        M = sw_local(params()); M.allocate_memory()
        M.particles_init(144, scheme=sch)
        xc = (np.arange(12) + 0.3)*LX/12
        Yc, Xc = np.meshgrid(xc + 0.11, xc)
        M.pxy[0, :], M.pxy[1, :] = Xc.ravel(), Yc.ravel()
        M.u[1:-1, 1:-1], M.v[1:-1, 1:-1], M.h[1:-1, 1:-1] = us, vs, hs
        M.boundary_conditions()
        M.particles_find_position()
        M.loop(nsteps)
        P[:, :, nrec - 1] = M.pxy
        out['particles_%s/x' % sch], out['particles_%s/y' % sch] = P[0].T.copy(), P[1].T.copy()
        for i, c in enumerate('uvh'):
            out['particles_%s/%s' % (sch, c)] = M.puvh[i, :, M.p_tau].copy()
    sys.path.pop(0)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute(sys.argv[1])
    if '--check' in sys.argv:
        ref = np.load(OUT)
        d = max(float(np.max(np.abs(res[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300)) for k in ref.files)
        print('sw references reproduced: max relative difference %.2e (%d arrays)  %s'
              % (d, len(ref.files), 'PASS' if d == 0. else 'FAIL'))
        sys.exit(0 if d == 0. else 1)
    np.savez_compressed(OUT, **res)
    print('sw references: %d arrays written to %s' % (len(res), OUT))
