"""
balance family: the balance diagnostics of pe_simple, pe_realistic and
nh_simple (Warn et al., kernel, band, optimal balance, the band window of a
raw snapshot). Small cases for the regression test 24 (regression.py,
README_test_suite.md), which records the output of every method on the
current code, so that the reorganisation into pyOM3/diagnostics can be
checked to round-off.

  ps_model(bases, **opt)     pe_simple: two opposite jets in thermal wind
                             balance with a perturbation (jets_balance1 at
                             32 x 32 x 6), periodic, buoyancy as tracer.
                             Options: free_surface, beta (variable f),
                             n2 (variable N^2), seamount (topography for the
                             band method)
  pr_model(bases)            pe_realistic, the same jets on a uniform
                             Cartesian periodic grid, buoyancy as tracer
  nh_model(**opt)            nh_simple, the jets of jets_balance_nh1
  pr_snapshot(path)          a raw snapshot (read_raw format, 2 x 2 rank
                             blocks) of the small closed box of the pe
                             family (test/pe/config.py: spherical grid,
                             topography, equation of state) after 20 steps
  run(m, n)                  n steps of the model's own loop (AB3 ramp)
  fingerprint(a)             [|a|, a.r1, a.r2, a.r3]: the 2-norm and three
                             products with fixed random vectors; a change
                             da shows in them as about |da|/sqrt(3)
  PERTURB                    relative perturbation of the initial velocity of
                             every case (0; regression.py --sensitivity sets
                             it to 1e-13 to show how round-off propagates)
"""
import os, sys, io, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))   # dir containing pyOM3/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))              # pyOM3/test/
import pyOM3 as OM
from common import line                                                    # shared by all families (regression.py imports it from here)

# the jets (jets_balance1.py at 32 x 32 x 6)
NX, NY, NZ = 32, 32, 6
LX, LY, LZ = 320e3, 320e3, 500.
DX, DY, DZ = LX/NX, LY/NY, LZ/NZ
DT = 600.
F0 = 1e-4
N0 = 100*F0                     # c_1 = 1.6 m/s, L_1 = 16 km
U0 = 0.2                        # jet width 19 km, Ro 0.1
AH = DX**4/(10*86400.)
PERTURB = 0.


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def fingerprint(a):
    a = np.asarray(a, dtype='f8').ravel()
    rng = np.random.default_rng(1234 + a.size)
    r = rng.uniform(-1., 1., (3, a.size))
    return [float(np.sqrt(np.sum(a**2)))] + [float(x) for x in r @ a]


def _jets(xu, yt, zt, H):
    """u of the two jets at the U points and the thermal-wind buoyancy
    (interior arrays, zt the height of the T levels with z = 0 at the
    surface, increasing upwards)"""
    Z, Y, X = np.meshgrid(zt, yt, xu, indexing='ij')
    jet = np.exp(-(Y - 3*LY/4)**2/(0.06*LX)**2) - np.exp(-(Y - LY/4)**2/(0.06*LX)**2)
    u = U0*(jet + 0.05*np.sin(X/LX*10*np.pi)*np.sin(Y/LY*2*np.pi))*np.cos((Z + H)/H*np.pi)
    uz = np.gradient(u, zt, axis=0)
    bt = -F0*np.cumsum(uz, axis=1)*DY
    return u, bt - bt.mean(axis=(1, 2), keepdims=True)


def _wave(xt, yt, zt, H, amp=0.02):
    """a short inertia-gravity wave packet in u and v (vertical mode 1)"""
    Z, Y, X = np.meshgrid(zt, yt, xt, indexing='ij')
    env = np.exp(-((X - LX/2)**2 + (Y - LY/2)**2)/(0.15*LX)**2)
    ph = 2*np.pi*(3*X/LX + 2*Y/LY)
    return amp*env*np.cos(ph)*np.cos(np.pi*Z/H), amp*env*np.sin(ph)*np.cos(np.pi*Z/H)


def N2_profile():
    """variable N^2 at the w levels (bottom to top): a thermocline at the top"""
    zw = (np.arange(NZ) + 1.)*DZ - LZ                            # w levels, z = 0 at the top face
    return N0**2*(0.3 + 1.7*np.exp(zw/150.))


def ps_model(bases, free_surface=False, beta=0., n2=False, seamount=False, wave=True, nsteps=30):
    """pe_simple jets after nsteps, with the balance classes `bases`"""
    class cls(*bases):
        def diagnose(self, n, tau):
            self.bal_tau = tau                                    # the time level of p_s, as warn_pe_simple.diagnose
        def set_topography(self):
            super().set_topography()
            if seamount:
                h = self.halo
                kb = np.asarray(self.kbot)*1
                j, i = np.meshgrid(np.arange(self.Ny) - h, np.arange(self.Nx) - h, indexing='ij')
                d = np.hypot(i - NX/2, j - NY/2)
                kb[d < 4] = h + 2                                    # two dry levels at the summit
                kb[d < 2] = h + 3
                self.kbot = OM.np.asarray(kb)
    par = dict(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, dt=DT,
               periodic_in_x=True, periodic_in_y=True,
               eq_of_state=0, implicit_free_surface=free_surface, explicit_free_surface=False,
               epsilon=1e-14, max_iterations=5000, Ahbi=AH, Khbi=AH, vmix_scheme='explicit')
    with quiet():
        m = cls(par)
        yt = (np.arange(NY) + 0.5)*DY
        m.coriolis_t = OM.np.asarray(_f_field(m, beta))
        m.set_topography()
        m.prepare_for_run()
        xt, xu = (np.arange(NX) + 0.5)*DX, (np.arange(NX) + 1.)*DX
        zt = (np.arange(NZ) + 0.5)*DZ - LZ
        u, b = _jets(xu, yt, zt, LZ)
        v = np.zeros_like(u)
        if wave:
            uw, vw = _wave(xt, yt, zt, LZ)
            u, v = u + uw, v + vw
        u, v = u*(1. + PERTURB), v*(1. + PERTURB)
        if n2:
            N2 = N2_profile()
            B = np.zeros(NZ)
            for k in range(NZ - 2, -1, -1):
                B[k] = B[k + 1] - N2[k]*DZ
            B -= B[-1] + N2[-1]*DZ/2.
            back = B[:, None, None]
        else:
            back = N0**2*zt[:, None, None]
        m.set_initial_conditions(u, v, b + back, np.zeros((NZ, NY, NX)), np.zeros((NY, NX)))
        m.u, m.v = m.u*m.maskU, m.v*m.maskV
        m.loop(nsteps)
    m.case_N0 = N2_profile() if n2 else N0
    return m


def _f_field(m, beta):
    h = m.halo
    yt = (np.arange(m.Ny) - h + 0.5)*DY
    return F0 + beta*(yt - LY/2)[:, None] + np.zeros((m.Ny, m.Nx))


def pr_model(bases, nsteps=30):
    """pe_realistic: the jets on a uniform Cartesian periodic grid, buoyancy as tracer"""
    class cls(*bases):
        def diagnose(self, n, tau):
            self.bal_tau = tau
    par = dict(nx=NX, ny=NY, nz=NZ, dxt=np.full(NX, DX), dyt=np.full(NY, DY), dzt=np.full(NZ, DZ),
               coord_degree=False, x_origin=0., y_origin=0., dt=DT,
               periodic_in_x=True, periodic_in_y=True,
               eq_of_state=0, implicit_free_surface=False, explicit_free_surface=False,
               epsilon=1e-14, max_iterations=5000, Ahbi=AH, Khbi=AH, vmix_scheme='explicit')
    with quiet():
        m = cls(par)
        m.coriolis_t = OM.np.asarray(F0 + np.zeros((m.Ny, m.Nx)))
        m.set_topography()
        m.prepare_for_run()
        h = m.halo
        yt = (np.arange(NY) + 0.5)*DY
        xt, xu = (np.arange(NX) + 0.5)*DX, (np.arange(NX) + 1.)*DX
        zt = np.asarray(m.zt)[h:-h] - LZ                            # height above the bottom -> z = 0 at the top
        u, b = _jets(xu, yt, zt, LZ)
        uw, vw = _wave(xt, yt, zt, LZ)
        m.set_initial_conditions((u + uw)*(1. + PERTURB), vw*(1. + PERTURB), b + N0**2*zt[:, None, None],
                                 np.zeros((NZ, NY, NX)), np.zeros((NY, NX)))
        m.loop(nsteps)
    return m


def nh_model(periodic_z=False, slave_modes=None, nsteps=20, dt=60.):
    """nh_simple: the jets of jets_balance_nh1 at 32 x 32 x 6, arrays (x, y, z).
    Non-hydrostatic, the fastest waves have the frequency N0: N0 dt = 0.6"""
    from pyOM3.diagnostics.balance.warn import warn_nh_simple
    class cls(warn_nh_simple):
        def diagnose(self, n, tau):
            pass
    par = dict(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, dt=dt, Lx=LX, Ly=LY, Lz=LZ,
               f0=F0, N0=N0, Ro=1., dsqr=1., periodic_in_z=periodic_z, Ahbi=AH, Khbi=AH)
    with quiet():
        m = cls(par)
        m.setup()
        m.init_balance(slave_modes=slave_modes)
        h = m.halo
        x = (np.arange(m.Nx) - h + 0.5)*DX
        y = (np.arange(m.Ny) - h + 0.5)*DY
        z = (np.arange(m.Nz) - h + 0.5)*DZ - LZ
        X, Y, Z = np.meshgrid(x + DX/2, y, z, indexing='ij')
        jet = np.exp(-(Y - 3*LY/4)**2/(0.06*LX)**2) - np.exp(-(Y - LY/4)**2/(0.06*LX)**2)
        kz = (2 if periodic_z else 1)*np.pi/LZ                          # periodic in z: a periodic profile
        u = U0*(jet + 0.05*np.sin(X/LX*10*np.pi)*np.sin(Y/LY*2*np.pi))*np.cos((Z + LZ)*kz)
        b = -F0*np.cumsum(np.gradient(u, DZ, axis=2), axis=1)*DY
        b = b - b[h:-h, h:-h, :].mean(axis=(0, 1))
        env = np.exp(-((X - LX/2)**2 + (Y - LY/2)**2)/(0.15*LX)**2)
        v = 0.02*env*np.sin(2*np.pi*(3*X/LX + 2*Y/LY))*np.cos(kz*Z)
        m.u = m.apply_bc(OM.np.asarray(u*(1. + PERTURB)))
        m.v = m.apply_bc(OM.np.asarray(v*(1. + PERTURB)))
        m.b = m.apply_bc(OM.np.asarray(b))
        m.loop(nsteps)
    return m


def pr_snapshot(path, nsteps=20):
    """
    the raw snapshot (read_raw format) of the small closed box of the pe
    family after nsteps, cut into 2 x 2 rank blocks as snap_raw writes them,
    and the parameters a window of it shares with the run. Returns par.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location('pe_config', os.path.join(HERE, '..', 'pe', 'config.py'))
    pecfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pecfg)
    with quiet():
        m = pecfg.step(pecfg.small_model('realistic'), nsteps)
    h = m.halo
    I = lambda a: np.asarray(a)[h:-h, h:-h, h:-h]
    tau = m._tau3[0] - 1                                           # the level written by the last step
    fields = dict(u=I(m.u)*(1. + PERTURB), v=I(m.v)*(1. + PERTURB), temp=I(m.temp), salt=I(m.salt),
                  p_h=I(m.p_h)*I(m.maskT), p_s=np.asarray(m.p_s[tau % 3])[h:-h, h:-h])
    os.makedirs(path, exist_ok=True)
    co = dict(xt=np.asarray(m.xt)[h:-h], yt=np.asarray(m.yt)[h:-h], zt=np.asarray(m.zt)[h:-h],
              xu=np.asarray(m.xu)[h:-h], yu=np.asarray(m.yu)[h:-h], zu=np.asarray(m.zw)[h:-h])
    for k, a in co.items():
        np.save(os.path.join(path, k + '.npy'), a)
    npx, npy = 2, 2
    np.save(os.path.join(path, 'decomposition.npy'), np.array([npx, npy]))
    nx, ny = len(co['xt']), len(co['yt'])
    xb, yb = (nx - 1)//npx + 1, (ny - 1)//npy + 1
    for pe in range(npx*npy):
        bx, by = pe % npx, pe//npx
        xs, ys = bx*xb, by*yb
        d = os.path.join(path, 'pe=%d' % pe, 'n=%d' % nsteps)
        os.makedirs(d, exist_ok=True)
        for k, a in fields.items():
            np.save(os.path.join(d, k + '.npy'), a[..., ys:ys + yb, xs:xs + xb])
        np.save(os.path.join(d, 't.npy'), np.array(nsteps*m.dt))
    par = dict(eq_of_state=m.eq_of_state, rho_0=m.rho_0, dt=m.dt, Ahbi=m.Ahbi, Khbi=0.,
               implicit_free_surface=True, explicit_free_surface=False)
    return par
