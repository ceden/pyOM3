"""
pe family (pe_simple, pe_realistic). Small configuration for the routine
tests (README_test_suite.md): a closed
wind- and buoyancy-forced box, 16 x 12 x 10 cells, with the TKE closure.

  small_model(kind, **opt)   'realistic' (spherical, variable grid: the
                             default of the suite), 'stretched' (pe_realistic
                             on a stretched CARTESIAN grid, same topography)
                             and 'nh' (nh_realistic on that stretched
                             Cartesian grid, used by test/nhr), 'cartesian' (pe_realistic
                             with coord_degree=False on a uniform grid) or
                             'simple' (pe_simple on the same uniform grid,
                             for the reduction test). Options: explicit
                             (free surface), mxl_choice, tracer_flux
                             (free_surface_tracer_flux), vadv
                             (vadv_sub_cycling), n_pes_x, n_pes_y,
                             comm, nsteps-independent.
  step(m, n, ab)             n steps of the model's own time_step_*; ab = weights
                             (default pyOM2's AB2 eps 0.1) or 'ab3' (ramped)
  state(m)                   dict of the interior prognostic fields (numpy)
  checksum(state)            one float per field, order-independent sums

Every test in the suite builds its models here, so the configuration is
the same everywhere and a change to it shows in all tests at once.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))   # dir containing pyOM3/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))              # pyOM3/test/
import pyOM3 as OM
from pyOM3.density import get_rho
from common import checksum, reldiff, line                                 # shared by all families

NX, NY, NZ = 16, 12, 10
DT = 900.0
OMEGA = np.pi/43082.0
GRAV = 9.81
FIELDS = ('u', 'v', 'w', 'temp', 'salt', 'p_s', 'tke', 'K_diss_v')


def _grids(kind):
    if kind in ('stretched', 'nh'):
        # the spacings of 'realistic' in metres (nh_realistic refuses degrees)
        dxt = np.linspace(0.9, 1.1, NX)*50e3; dyt = np.linspace(0.8, 1.2, NY)*50e3
        dzt = 10.0 + 240.0*0.5*(1.0 - np.cos(np.linspace(0.0, np.pi, 10)))[::-1]
        return dict(dxt=dxt, dyt=dyt, dzt=dzt, coord_degree=False, x_origin=0.0, y_origin=0.0)
    if kind == 'realistic':
        dxt = np.linspace(0.9, 1.1, NX); dyt = np.linspace(0.8, 1.2, NY)      # degrees
        # thin surface layers (top cell 10 m) so that vertical-order errors are
        # visible in the tracer tests, not only in the exact identities: with
        # 150-600 m cells and A_v 1e-4 the friction changed u by 5e-6 per step
        # and a stale w went unnoticed (Sep 15 2026)
        dzt = 10.0 + 240.0*0.5*(1.0 - np.cos(np.linspace(0.0, np.pi, 10)))[::-1]   # 250 m bottom -> 10 m top, cosine taper (pyOM2 dzw > 0), H 1300 m
        return dict(dxt=dxt, dyt=dyt, dzt=dzt, coord_degree=True, x_origin=10.0, y_origin=25.0)
    dx = dy = 50e3; dz = 100.0
    if kind == 'cartesian':
        return dict(dxt=np.full(NX, dx), dyt=np.full(NY, dy), dzt=np.full(NZ, dz),
                    coord_degree=False, x_origin=0.0, y_origin=0.0)
    return dict(dx=dx, dy=dy, dz=dz)                                           # simple


def small_model(kind='realistic', explicit=False, mxl_choice=1, tracer_flux=False,
                n_pes_x=1, n_pes_y=1, comm=None, Ahbi=1e11, quiet=True, vadv=0):
    if kind == 'simple':
        from pyOM3.pe_simple.model import model as _base
        from pyOM3.pe_simple.tke import tke as _tke
    elif kind == 'nh':
        from pyOM3.nh_realistic.model import model as _base
        from pyOM3.pe_realistic.tke import tke as _tke
    else:
        from pyOM3.pe_realistic.model import model as _base
        from pyOM3.pe_realistic.tke import tke as _tke

    class model(_tke, _base):                       # the tke mixin, as the configurations do
        pass
    par = dict(nx=NX, ny=NY, nz=NZ, dt=DT, periodic_in_x=False, periodic_in_y=False,
               eq_of_state=1, explicit_free_surface=explicit, implicit_free_surface=not explicit,
               superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
               enable_tke=True, c_k=0.1, c_eps=0.7, alpha_tke=30.0, mxl_min=1e-8,
               kappa_T=1e-5, A_v=1e-4, Ahbi=Ahbi, biha_friction_cosPower=1.0,
               epsilon=1e-12, max_iterations=5000)
    par.update(_grids(kind))
    kw = dict(n_pes_x=n_pes_x, n_pes_y=n_pes_y)
    if comm is not None:
        kw['comm_in'] = comm
    if quiet:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            m = model(par, **kw)
    else:
        m = model(par, **kw)
    h = m.halo
    m.mxl_choice = mxl_choice
    m.free_surface_tracer_flux = tracer_flux
    if vadv:                                         # pe_realistic only (vadv_split.py)
        m.vadv_sub_cycling = vadv
    if explicit:
        m.sub_cycling = 12
    # Coriolis: f(y) on the sphere, f-plane + beta on the plane (same numbers)
    # GLOBAL interior index of every point of the local layout (halo points get
    # the index of the point they mirror or an out-of-range one; apply_bc
    # overwrites them anyway). MPI: local slice; JAX: inline per-shard halos.
    jg, ig = global_index(m)
    if kind == 'realistic':
        f = 2*OMEGA*np.sin(np.deg2rad(np.asarray(m.yt)))[:, None]*np.ones((m.Ny, m.Nx))
    else:
        yy = jg*50e3
        f = 7e-5 + 2e-11*yy
    m.coriolis_t = OM.np.asarray(f)
    # topography: closed box with a shelf and a seamount (all decompositions
    # see the same GLOBAL indices)
    kb = np.ones((m.Ny, m.Nx))
    kb[(jg >= 8) & (ig <= 3)] = NZ - 2                     # shelf, 2 wet levels
    kb[(jg >= 4) & (jg <= 5) & (ig >= 9) & (ig <= 10)] = 5  # seamount
    kb[(jg < 0) | (jg >= NY) | (ig < 0) | (ig >= NX)] = 0    # walls at the global edges
    m.kbot = m.apply_bc(OM.np.asarray(kb))
    m.prepare_for_run()
    m.init_tke()
    # initial state: stratified, a warm anomaly, uniform salt
    if hasattr(m, 'zt'):                             # pe_realistic: height above the bottom
        zt = np.asarray(m.zt); H = float(np.asarray(m.dzt)[h:-h].sum())
    else:                                            # pe_simple: uniform dz
        zt = (np.arange(m.Nz) - h + 0.5)*m.dz; H = NZ*m.dz
    T = 4.0 + 16.0*(zt/H)[:, None, None]*np.ones((m.Nz, m.Ny, m.Nx))
    T += 1.0*np.exp(-((ig - NX/2)/3.)**2 - ((jg - NY/2)/2.)**2)[None]*(zt/H)[:, None, None]
    m.temp = m.apply_bc(OM.np.asarray(T))*m.maskT
    m.salt = 35.0*m.maskT
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
    m.tke = OM.np.full((m.Nz, m.Ny, m.Nx), 1e-6, OM.prec)*m.maskW
    # forcing: zonal wind, surface cooling in the north, per GLOBAL index
    m._taux = OM.np.asarray(-2e-4*np.cos(np.pi*jg/(NY - 1)))          # 0.2 N/m2: drives the TKE
    m._qsurf = OM.np.asarray(-2e-5*np.clip((jg - NY/2)/(NY/2), 0, 1))       # K m/s
    m._dz_surf = float(np.asarray(m.dzt)[-h-1]) if hasattr(m, 'dzt') else float(m.dz)
    m.set_forcing = _set_forcing.__get__(m)
    return m


def _set_forcing(self, tau):
    h = self.halo; ks = -h - 1
    taux = self._taux*self.maskU[ks]
    self.du = OM.add_to_array(self.du, (tau, ks, slice(None), slice(None)), taux/self._dz_surf)
    self.dtemp = OM.add_to_array(self.dtemp, (tau, ks, slice(None), slice(None)),
                                 self._qsurf*self.maskT[ks]/self._dz_surf)
    self.taux, self.tauy = taux, OM.np.zeros_like(taux)
    self.forc_temp_surface = self._qsurf*self.maskT[ks]


AB3 = (23./12., -16./12., 5./12.)


def step(m, n, ab=(1.6, -0.6, 0.0)):
    """n steps. ab = the three Adams-Bashforth weights, applied from the first
    step on (pyOM2's AB2 with eps 0.1, as the coupled runs do), or 'ab3': the
    true third-order weights after the cold-start ramp of pe's own loop()
    (forward Euler, AB2, then AB3). The ramp counts the steps of THIS model
    (m._nstep), which therefore belongs to a restart like _tau3 does."""
    if not hasattr(m, '_tau3'):
        m._tau3 = (2, 1, 0)
    if not hasattr(m, '_nstep'):
        m._nstep = 0
    tau, taum1, taum2 = m._tau3
    for _ in range(n):
        if isinstance(ab, str):
            A, B, C = ((1., 0., 0.), (1.5, -0.5, 0.))[m._nstep] if m._nstep < 2 else AB3
        else:
            A, B, C = ab
        m._ab = (A, B, C)
        if getattr(m, 'explicit_free_surface', False):
            m.time_step_explicit_pressure(A, B, C, tau, taum1, taum2)
        else:
            m.time_step_implicit_pressure(A, B, C, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        m._nstep += 1
    m._tau3 = (tau, taum1, taum2)
    return m


def global_index(m):
    """(jg, ig): global 0-based interior index of each point of the model's
    local (Ny, Nx) layout, as float arrays. MPI: the local slice starts at
    global index xs_pe-1 (ys_pe-1). JAX: the global array holds n_pes
    blocks of nx/n_pes interior points, each with its own halo inline."""
    h = m.halo
    def axis(N, n, n_shards, start):
        J = np.arange(N)
        if OM.jax and n_shards > 1:
            b = n // n_shards                                 # interior points per shard
            return (J // (b + 2*h))*b + J % (b + 2*h) - h
        return J - h + start
    jg = axis(m.Ny, NY, m.n_pes_y, m.ys_pe - 1)[:, None]*np.ones((1, m.Nx))
    ig = axis(m.Nx, NX, m.n_pes_x, m.xs_pe - 1)[None, :]*np.ones((m.Ny, 1))
    return jg, ig


def _interior(m, a):
    """strip the halo: MPI/one PE -> the local interior; JAX with several
    shards -> the GLOBAL interior (unpad removes every inline halo)"""
    h = m.halo; sl = slice(h, -h)
    if OM.jax:
        return np.asarray(m.unpad(OM.np.asarray(a)))
    return a[..., sl, sl] if a.ndim == 2 else a[sl, sl, sl]


def _fields(m):
    """the prognostic fields, plus p_nh for the nh_realistic model"""
    return FIELDS + (('p_nh',) if hasattr(m, 'p_nh') else ())


def state(m):
    out = {}
    for f in _fields(m):
        a = np.asarray(getattr(m, f))
        if a.ndim == 3 and f == 'p_s':
            a = a[m._tau3[1]]
        out[f] = _interior(m, a)
    return out


def gathered_state(m):
    """state assembled on rank 0 across the model's PEs (None elsewhere)"""
    if m.n_pes == 1 or OM.jax:                        # JAX: one process, unpad gives the global interior
        return state(m)
    h = m.halo; out = {}
    for f in _fields(m):
        a = np.asarray(getattr(m, f))
        if a.ndim == 3 and f == 'p_s':
            a = a[m._tau3[1]]
        if a.ndim == 2:                                   # gather strips the k halo too: wrap in 1+2h levels
            b = np.zeros((1 + 2*h,) + a.shape); b[h] = a; a3 = OM.np.asarray(b)
        else:
            a3 = a
        g = m.gather_array_mpi(a3)
        if m.my_pe == 0:
            g = np.asarray(g)
            out[f] = g[0] if a.ndim == 2 else g
    return out if m.my_pe == 0 else None
