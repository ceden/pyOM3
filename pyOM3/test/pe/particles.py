"""Test 23: Lagrangian particles of pe_realistic (pe_realistic/particles.py).

  23a analytic    prescribed uniform flows on the sphere against the exact
                  trajectories: u alone (dx/dt = u/(a cos y), exact for the
                  scheme, 1e-12), u and v together (the latitude exact, the
                  longitude the Mercator integral) and w linear in z
                  (exponential growth of the height). Where the rate changes
                  along the path the start of the scheme (Euler, AB2, AB3)
                  dominates the error, second order in dt: 1e-6 and 1e-4
                  relative to the displacement, and halving dt must reduce
                  it at least 3.5 times
  23b land        an unmasked flow into the seamount and the east wall: no
                  particle ever sits in a dry cell, the rule acted (refused
                  steps > 0), fixed points (moving = 0) never move; with
                  the masked flow none reaches land                 exact
  23c tracers     the masked trilinear interpolation of temp and salt on a
                  field linear in x, y, z: exact away from land, and at a
                  cell next to land the value of the wet points only
                                                                   1e-12
  23d restart     2N steps against N steps, particles_state, a new set
                  restored from it, N steps; in the forced box     exact
  23e simple      pe_simple's particles (pe_simple/particles.py) against
                  these on the same uniform Cartesian box, the forced model
                  of test 1 (the two models agree to 1e-12)        1e-9 rel
  23f layout      prescribed analytic fields (the same in every layout,
                  halos included), particles everywhere, 12 steps: MPI 1x2,
                  2x1, 2x2 and jax 1x1, 1x2, 2x1, 2x2 (fake CPU devices),
                  2x2 once more with halo_points = 2 on both, against one
                  numpy PE. Each particle is interpolated by
                  exactly one PE, so this is exact on MPI; on jax XLA
                  rounds differently from numpy (1e-13 of the displacement
                  in the channel, the same in every jax layout)  exact / 1e-12
  23g model       the same with the forced model running: MPI and jax
                  against one PE                                   1e-10 rel
  23h periodic    the box made a zonal channel (periodic_in_x): a uniform u
                  carries the particles across the seam, exact after the
                  wrap (1e-12); then prescribed fields periodic in x, the
                  layouts of 23f across the seam                exact / 1e-12

    python3 particles.py            23a-h with the MPI layouts (run_fast.py; needs mpirun)
    python3 particles.py --jax      the jax layouts of 23f-h (run_slow.py, about 2 min)
"""
import os, sys, io, contextlib, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from config import small_model, step, line
import pyOM3 as OM
from pyOM3.pe_realistic.particles import particles

AB2 = (1.6, -0.6, 0.0)


def _model(kind='realistic', **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return small_model(kind, bases=(particles,), **kw)


def _set(m, name, val):
    """a field of the local layout from a numpy array of the layout's shape"""
    setattr(m, name, m.apply_bc(OM.np.asarray(val)))


def _layout_coords(m):
    """the coordinates of the local layout, broadcast to (Nz, Ny, Nx)"""
    b = lambda a, ax: np.asarray(a).reshape([-1 if i == ax else 1 for i in range(3)])
    return dict(xt=b(m.xt, 2), xu=b(m.xu, 2), yt=b(m.yt, 1), yu=b(m.yu, 1), zt=b(m.zt, 0), zw=b(m.zw, 0))


def _seeds(m, stride=2, depths=(5.0, 100.0, 600.0, 1250.0)):
    xt, yt = np.asarray(m.xt_gl), np.asarray(m.yt_gl)
    X, Y, Z = np.meshgrid(xt[::stride] + 0.13, yt[::stride] - 0.07, m.particle_depth_to_z(np.array(depths)),
                          indexing='ij')
    return X.ravel(), Y.ravel(), Z.ravel()


# ---------------------------------------------------------------------------
def analytic():
    ok = True
    m = _model()
    sh = (m.Nz, m.Ny, m.Nx)
    lon = np.array([14.0, 15.3, 17.2, 19.9]); lat = np.array([28.0, 29.5, 31.1, 30.2])
    z0 = m.particle_depth_to_z(np.array([30., 300., 700., 900.]))
    a, r2d = m.radius, 180.0/np.pi
    N = 40; t = N*m.dt
    # u alone
    m.init_particles(lon, lat, z0, quiet=True)
    U0 = 0.3
    _set(m, 'u', np.full(sh, U0)); _set(m, 'v', np.zeros(sh)); _set(m, 'w', np.zeros(sh))
    for _ in range(N):
        m.step_particles()
    P = m._p_host(m.p_pos)
    ex = lon + U0*t/(a*np.cos(lat/r2d))*r2d
    ok &= line('23a analytic: uniform u on the sphere, lon', float(np.max(np.abs(P[0] - ex))/np.max(np.abs(ex - lon))), 1e-12)
    # u and v: y(t) = y0 + V0 t/a, x(t) = x0 + U0/V0 (gd^-1(y) - gd^-1(y0)).
    # The rate of x changes along the path, so the error is that of the
    # scheme: its start (Euler, AB2, AB3) dominates, second order in dt. The
    # same time with half the step must reduce it by at least 3.5.
    V0 = 0.2
    gdi = lambda y: np.log(np.abs(1.0/np.cos(y/r2d) + np.tan(y/r2d)))
    err = []
    for k in (1, 2):
        m.dt = 900.0/k; m._p_fns = {}
        m.init_particles(lon, lat, z0, quiet=True)
        _set(m, 'u', np.full(sh, U0)); _set(m, 'v', np.full(sh, V0))
        for _ in range(N*k):
            m.step_particles()
        P = m._p_host(m.p_pos)
        ey = lat + V0*t/a*r2d
        ex = lon + U0/V0*(gdi(ey) - gdi(lat))*r2d
        if k == 1:
            ok &= line('23a analytic: uniform u and v, lat', float(np.max(np.abs(P[1] - ey))/np.max(np.abs(ey - lat))), 1e-12)
        err.append(float(np.max(np.abs(P[0] - ex))/np.max(np.abs(ex - lon))))
    m.dt = 900.0; m._p_fns = {}
    ok &= line('23a analytic: uniform u and v, lon (Mercator), dt 900 s', err[0], 1e-6)
    ok &= line('23a analytic: the same, error ratio dt/(dt/2) >= 3.5', 3.5/err[0]*err[1], 1.0)
    # w linear in z (w = 0 at the bottom face): z(t) = z0 exp(W t)
    z0 = np.array([300.0, 500.0, 800.0, 1000.0])        # heights above the bottom, below the top clamp at the end
    m.init_particles(lon, lat, z0, quiet=True)
    _set(m, 'u', np.zeros(sh)); _set(m, 'v', np.zeros(sh))
    W = 1e-6
    c = _layout_coords(m)
    _set(m, 'w', W*c['zw']*np.ones(sh))
    for _ in range(N):
        m.step_particles()
    P = m._p_host(m.p_pos)
    ez = z0*np.exp(W*t)
    ok &= line('23a analytic: w linear in z, height (scheme start, as above)', float(np.max(np.abs(P[2] - ez))/np.max(np.abs(ez - z0))), 1e-4)
    return ok


def land():
    ok = True
    m = _model()
    sh = (m.Nz, m.Ny, m.Nx)
    # eastward 1 m/s: in 40 steps of 900 s 36 km, i.e. across most of a cell;
    # particles start one cell west of the seamount (i 9..10, j 4..5, wet
    # below level 5) and of the east wall, at several depths
    xt, yt = np.asarray(m.xt_gl), np.asarray(m.yt_gl)
    X = np.concatenate([np.full(8, xt[8] + 0.2), np.full(8, 0.5*(xt[14] + xt[15]) + 0.3)])
    Y = np.concatenate([np.repeat(yt[4:6], 4), np.repeat(yt[4:6], 4)])
    Z = np.tile(m.particle_depth_to_z(np.array([20.0, 300.0, 900.0, 1250.0])), 4)
    mv = np.ones(len(X)); mv[::5] = 0.0
    m.init_particles(X, Y, Z, moving=mv, quiet=True)
    P0 = m._p_host(m.p_pos)
    # an UNMASKED eastward flow drives them into the seamount and the wall:
    # the rule has to refuse those steps
    _set(m, 'u', np.full(sh, 1.0)); _set(m, 'v', np.zeros(sh)); _set(m, 'w', np.zeros(sh))
    dry = 0
    for _ in range(40):
        m.step_particles()
        wet = np.asarray(m._p_host(m._p_wet_fn()(m.maskT, m.p_pos)))
        dry += int(np.sum(wet < 0.5))
    P = m._p_host(m.p_pos)
    fixed = m.p_moving_np < 0.5
    ok &= line('23b land: particles in dry cells over 40 steps', float(dry), 0.0)
    ok &= line('23b land: refused steps > 0 (the rule acted)', 0.0 if m.p_frozen > 0 else 1.0, 0.0)
    ok &= line('23b land: fixed points moved', float(np.max(np.abs(P[:, fixed] - P0[:, fixed]))), 0.0)
    # with the masked flow (the model's) the interpolated velocity vanishes at
    # the walls: the particles come to rest in front of them
    m.init_particles(X, Y, Z, moving=mv, quiet=True)
    _set(m, 'u', np.full(sh, 1.0)*np.asarray(m.maskU))
    for _ in range(40):
        m.step_particles()
    wet = np.asarray(m._p_host(m._p_wet_fn()(m.maskT, m.p_pos)))
    ok &= line('23b land: masked flow, particles in dry cells', float(np.sum(wet < 0.5)), 0.0)
    return ok


def tracers():
    ok = True
    m = _model()
    c = _layout_coords(m)
    lin = lambda x, y, z: 3.0 + 0.5*x - 0.25*y + 1e-3*z
    sh = (m.Nz, m.Ny, m.Nx)
    m.temp = m.apply_bc(OM.np.asarray(lin(c['xt'], c['yt'], c['zt'])*np.ones(sh)))
    m.salt = m.apply_bc(OM.np.asarray(-lin(c['xt'], c['yt'], c['zt'])*np.ones(sh)))
    # interior particles (all eight points wet), inside the T point hull
    xt, yt, zt = np.asarray(m.xt_gl), np.asarray(m.yt_gl), np.asarray(m.zt_gl)
    X = np.array([0.3*xt[5] + 0.7*xt[6], 0.5*(xt[12] + xt[13])])
    Y = np.array([0.6*yt[2] + 0.4*yt[3], 0.2*yt[6] + 0.8*yt[7]])
    Z = np.array([0.5*(zt[4] + zt[5]), 0.9*zt[7] + 0.1*zt[8]])
    m.init_particles(X, Y, Z, quiet=True)
    r = m.step_particles(record=True)
    ok &= line('23c tracers: linear field, interior', float(np.max(np.abs(r['temp'] - lin(X, Y, Z)))), 1e-12)
    # next to the shelf (i <= 3, j >= 8 wet in the top 2 levels only): a
    # particle at mid depth between the shelf column and the deep column
    # sees only the deep points
    X = np.array([0.5*(xt[3] + xt[4])]); Y = np.array([0.5*(yt[8] + yt[9])]); Z = np.array([0.5*(zt[5] + zt[6])])
    m.init_particles(X, Y, Z, quiet=True)
    r = m.step_particles(record=True)
    # the wet points of the stencil: i = 4 (deep column), j 8, 9, k 5, 6 -> the value at x = xt[4]
    ok &= line('23c tracers: next to the shelf, wet points only', float(abs(r['temp'][0] - lin(xt[4], Y[0], Z[0]))), 1e-12)
    return ok


def restart(n=6):
    """two identical models; the second saves the particles after N steps,
    re-initialises them elsewhere and restores the saved state"""
    import copy
    res = []
    for brk in (False, True):
        m = _model()
        X, Y, Z = _seeds(m)
        m.init_particles(X, Y, Z, quiet=True)
        for i in range(2*n):
            if brk and i == n:
                st = copy.deepcopy(m.particles_state())
                m.init_particles(X[::-1] + 0.01, Y, Z, quiet=True)     # something else, then the restore
                m.particles_restore(st)
            step(m, 1, AB2); m.step_particles()
        res.append((m._p_host(m.p_pos).copy(), m._p_host(m.p_hist).copy()))
    d = max(float(np.max(np.abs(res[0][0] - res[1][0]))), float(np.max(np.abs(res[0][1] - res[1][1]))))
    return line('23d restart: 2N against N + restore + N, positions and history', d, 0.0)


def simple(n=16):
    """pe_simple's particles against these, the forced box on the uniform
    Cartesian grid of test 1"""
    from pyOM3.pe_simple.particles import particles as psimple
    with contextlib.redirect_stdout(io.StringIO()):
        ms = small_model('simple', bases=(psimple,))
        mr = small_model('cartesian', bases=(particles,))
    dx, dz, H = 50e3, 100.0, 1000.0
    # interior positions (pe_simple: x of the U point i at (i + 1) dx, z in
    # (-H, 0); pe_realistic 'cartesian': x_origin = 0, i.e. xu(0) = 0, so
    # x_r = x_s - dx, and z = height above the bottom = z_s + H)
    rng = np.random.default_rng(3)
    xs = rng.uniform(3.0, 14.0, 60)*dx; ys = rng.uniform(2.0, 10.0, 60)*dx; zs = -rng.uniform(5.0, 990.0, 60)
    ms.init_particles(np.stack([xs, ys, zs]))
    mr.init_particles(xs - dx, ys - dx, zs + H, quiet=True)
    keep = np.asarray(mr.p_ids)
    P0 = mr._p_host(mr.p_pos).copy()
    for _ in range(n):
        step(ms, 1, AB2); ms.step_particles()
        step(mr, 1, AB2); mr.step_particles()
    Ps = ms.pxyz[:, keep]
    Pr = mr._p_host(mr.p_pos)
    Pr_s = np.stack([Pr[0] + dx, Pr[1] + dx, Pr[2] - H])
    disp = np.max(np.abs(Pr - P0))
    ok = line('23e simple: pe_simple against pe_realistic particles (rel to the displacement)',
              float(np.max(np.abs(Ps - Pr_s)))/disp, 1e-9)
    return ok


# ---------------------------------------------------------------------------
# 23f / 23g: the layouts, each run in its own process
# ---------------------------------------------------------------------------
def _prescribe(m):
    """analytic u, v, w, temp, salt of the layout's own coordinates; in the
    zonal channel periodic in x over the domain length (the halo coordinates
    of a periodic direction are the images, not shifted), with a mean
    eastward flow that carries the particles across the seam"""
    c = _layout_coords(m); sh = (m.Nz, m.Ny, m.Nx)
    if m.periodic_in_x:
        L = m._p_grid()['Lx']
        s = lambda a: np.sin(2*np.pi*a/L)
        c = dict(c); xt, xu = c['xt'], c['xu']
        u = 1.0 + 0.4*np.cos(np.deg2rad(3*c['yt']))*s(xu)*(0.5 + c['zt']/1300.0)
        v = 0.3*s(xt + 3.0)*np.cos(np.deg2rad(2*c['yu']))
        w = 2e-4*s(2*xt)*np.sin(np.pi*c['zw']/1300.0)
        T = 5.0 + 10.0*c['zt']/1300.0 + s(xt)*np.cos(np.deg2rad(5*c['yt']))
        m.u = m.apply_bc(OM.np.asarray(u*np.ones(sh)))*m.maskU
        m.v = m.apply_bc(OM.np.asarray(v*np.ones(sh)))*m.maskV
        m.w = m.apply_bc(OM.np.asarray(w*np.ones(sh)))*m.maskW
        m.temp = m.apply_bc(OM.np.asarray(T*np.ones(sh)))
        m.salt = m.apply_bc(OM.np.asarray((35.0 + 0.1*T)*np.ones(sh)))
        return
    s = lambda a: np.sin(np.deg2rad(a))
    u = 0.4*np.cos(np.deg2rad(3*c['yt']))*(1 + 0.3*s(5*c['xu']))*(0.5 + c['zt']/1300.0)
    v = 0.3*s(4*c['xt'] + 30)*np.cos(np.deg2rad(2*c['yu']))*(1.0 - 0.4*c['zt']/1300.0)
    w = 2e-4*s(7*c['xt'])*s(9*c['yt'] + 20)*np.sin(np.pi*c['zw']/1300.0)
    T = 5.0 + 10.0*c['zt']/1300.0 + s(6*c['xt'])*np.cos(np.deg2rad(5*c['yt']))
    m.u = m.apply_bc(OM.np.asarray(u*np.ones(sh)))*m.maskU
    m.v = m.apply_bc(OM.np.asarray(v*np.ones(sh)))*m.maskV
    m.w = m.apply_bc(OM.np.asarray(w*np.ones(sh)))*m.maskW
    m.temp = m.apply_bc(OM.np.asarray(T*np.ones(sh)))
    m.salt = m.apply_bc(OM.np.asarray((35.0 + 0.1*T)*np.ones(sh)))


def periodic():
    m = _model(periodic_x=True)
    sh = (m.Nz, m.Ny, m.Nx)
    g = m._p_grid()
    lon = np.array([g['x_west'] + g['Lx'] - 0.3, 20.0, 12.0]); lat = np.array([28.0, 29.5, 31.1])
    z0 = m.particle_depth_to_z(np.array([30., 300., 700.]))
    m.init_particles(lon, lat, z0, quiet=True)
    U0 = 1.5; N = 60
    _set(m, 'u', np.full(sh, U0)); _set(m, 'v', np.zeros(sh)); _set(m, 'w', np.zeros(sh))
    for _ in range(N):
        m.step_particles()
    P = m._p_host(m.p_pos)
    r2d = 180.0/np.pi
    ex = lon + U0*N*m.dt/(m.radius*np.cos(lat/r2d))*r2d
    ex = g['x_west'] + np.mod(ex - g['x_west'], g['Lx'])
    crossed = ex < lon
    d = float(np.max(np.abs(P[0] - ex)))/float(np.max(np.abs(U0*N*m.dt/(m.radius*np.cos(lat/r2d))*r2d)))
    ok = line('23h periodic: uniform u across the seam, lon after the wrap', d, 1e-12)
    ok &= line('23h periodic: a particle crossed the seam (control)', 0.0 if crossed.any() else 1.0, 0.0)
    return ok


def _layout_run(mode, npx, npy, comm=None, halo=1):
    kw = dict(n_pes_x=npx, n_pes_y=npy, halo=halo, periodic_x=(mode == 'periodic'))
    if comm is not None:
        kw['comm'] = comm
    m = _model(**kw)
    X, Y, Z = _seeds(m, stride=1)
    mv = np.ones(len(X)); mv[::7] = 0.0
    m.init_particles(X, Y, Z, moving=mv, quiet=True)
    rec = []
    for n in range(12):
        if mode in ('prescribed', 'periodic'):
            _prescribe(m)
        else:
            step(m, 1, AB2)
        r = m.step_particles(record=(n % 3 == 2))
        if r is not None:
            rec.append(np.stack([r[k] for k in ('x', 'y', 'z', 'u', 'v', 'w', 'temp', 'salt')]))
    out = dict(P=m._p_host(m.p_pos), H=m._p_host(m.p_hist), R=np.stack(rec), ids=np.asarray(m.p_ids),
               frozen=np.array(m.p_frozen), P0=np.stack([X, Y, Z])[:, np.asarray(m.p_ids)])
    return out


REF = lambda mode: os.path.join(HERE, 'particles_ref_%s.npz' % mode)


def worker(argv):
    """python3 particles.py worker <mode> <npx> <npy> [ref | h2]: one layout (h2: halo_points = 2)"""
    mode, npx, npy = argv[0], int(argv[1]), int(argv[2])
    halo = 2 if 'h2' in argv[3:] else 1
    comm = None
    if not OM.jax and OM.mpi is not None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    out = _layout_run(mode, npx, npy, comm, halo)
    rank = comm.Get_rank() if comm is not None else 0
    if len(argv) > 3 and argv[3] == 'ref':
        if rank == 0:
            np.savez(REF(mode), **out)
        return True
    if rank != 0:
        return True
    ref = np.load(REF(mode))
    if not np.array_equal(ref['ids'], out['ids']):
        return line('23%s %s %s %dx%d: the kept particles differ' % ({'prescribed': 'f', 'model': 'g', 'periodic': 'h'}[mode], mode,
                    'jax' if OM.jax else 'MPI', npx, npy), 1.0, 0.0)
    disp = float(np.max(np.abs(ref['P'] - ref['P0'])))
    dP = float(np.max(np.abs(out['P'] - ref['P'])))/disp
    # the recorded positions relative to the displacement, the values to their size
    dR = max(float(np.max(np.abs(out['R'][:, k] - ref['R'][:, k])))/(disp if k < 3 else max(float(np.max(np.abs(ref['R'][:, k]))), 1e-300))
             for k in range(8))
    if mode in ('prescribed', 'periodic'):
        tol = 1e-12 if OM.jax else 0.0
    else:
        tol = 1e-10
    tag = '23%s %s %s %dx%d%s' % ({'prescribed': 'f', 'model': 'g', 'periodic': 'h'}[mode],
                                   {'prescribed': 'layout', 'model': 'model', 'periodic': 'periodic'}[mode],
                                   'jax' if OM.jax else 'MPI', npx, npy, ' halo 2' if halo > 1 else '')
    ok = line(tag + ', positions', dP, tol)
    ok &= line(tag + ', records', dR, tol)
    ok &= line(tag + ', refused steps', float(abs(int(out['frozen']) - int(ref['frozen']))), 0.0)
    return ok


def layouts(jax=False):
    """the MPI layouts (jax=False, the fast suite) or the jax layouts (jax=True,
    the slow suite), each against a fresh one-PE numpy reference"""
    ok = True
    mpirun = os.environ.get('MPIRUN', '').split() or (['mpirun', '-np'] if shutil.which('mpirun') else None)
    have_jax = jax and subprocess.run([sys.executable, '-c', 'import jax'], capture_output=True).returncode == 0
    if jax:
        mpirun = None
    runs = []
    for mode in ('prescribed', 'model', 'periodic'):
        runs.append((mode, 'numpy', 1, 1, 1, 'ref'))
        if mpirun:
            runs += [(mode, 'numpy', 2, 1, 2, ''), (mode, 'numpy', 2, 2, 1, ''), (mode, 'numpy', 4, 2, 2, ''),
                     (mode, 'numpy', 4, 2, 2, 'h2')]
        if have_jax:
            runs += [(mode, 'jax', 1, 1, 1, ''), (mode, 'jax', 1, 1, 2, ''),
                     (mode, 'jax', 1, 2, 1, ''), (mode, 'jax', 1, 2, 2, ''), (mode, 'jax', 1, 2, 2, 'h2')]
    if not jax and not mpirun:
        print('23f-h MPI layouts: skipped (no mpirun on the path; set MPIRUN)')
    if jax and not have_jax:
        print('23f-h jax layouts: skipped (jax not importable)')
    for mode, backend, n, npx, npy, opt in runs:
        env = dict(os.environ, PYOM3_BACKEND=backend, XLA_FLAGS='--xla_force_host_platform_device_count=4')
        cmd = [sys.executable, os.path.abspath(__file__), 'worker', mode, str(npx), str(npy)] + ([opt] if opt else [])
        if backend == 'numpy' and mpirun and opt != 'ref':
            cmd = mpirun + [str(n)] + cmd
        r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith('23')]
        if lines:
            print('\n'.join(lines))
        if r.returncode != 0:
            ok = False
            print('   %s %s %dx%d failed (exit %d):' % (backend, mode, npx, npy, r.returncode))
            print('\n'.join('      ' + l for l in (r.stderr or r.stdout).splitlines()[-12:]))
        ok &= not any(l.endswith('FAIL') for l in lines)
    for mode in ('prescribed', 'model', 'periodic'):
        if os.path.exists(REF(mode)):
            os.remove(REF(mode))                       # the references are run products
    return ok


def run(jax=False):
    """the fast part: 23a-e, 23h and the MPI layouts; jax=True: the jax layouts only"""
    if jax:
        return layouts(jax=True)
    ok = True
    for f in (analytic, land, tracers, restart, simple, periodic, layouts):
        ok &= f()
    return ok


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        sys.exit(0 if worker(sys.argv[2:]) else 1)
    sys.exit(0 if run(jax='--jax' in sys.argv) else 1)
