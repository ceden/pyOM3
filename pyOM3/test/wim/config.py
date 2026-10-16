"""
wim family (wim_simple, wim_realistic). Small configuration for the routine
tests (README_test_suite.md): a wind-forced basin, 14 x 12 cells, 16
frequencies x 12 directions, closed in y and periodic in x, with an island,
a sheared current through the Doppler machinery, the DIA and the wind input
and dissipation.

  small_model(kind, **opt)   'realistic' (spherical, variable dxt, dyt and a
                             variable dzt of the ocean column: the default of
                             the suite), 'cartesian' (wim_realistic with
                             coord_degree=False on a uniform grid) or 'simple'
                             (wim_simple on the same uniform grid, for the
                             reduction test). Options: open_coasts,
                             energy_consistent, reflection (enable_reflection),
                             current, janssen (input_scheme, wim_realistic
                             only), tail_flux, backend (propagation_backend),
                             snl (snonlin_backend), sources (False: adiabatic
                             propagation only, for the conservation
                             identities), n_pes_x, n_pes_y, comm.
  step(m, n)                 n steps of the model's own time_step with the
                             weights of the model's own loop(): forward Euler,
                             then AB2 with eps 0.05
  state(m)                   dict of the interior prognostic fields (numpy),
                             the outflow book-keeping and the EXCHANGE fields
                             the ocean receives (surface stress, Stokes drift)
  checksum(state)            one float per field, order-independent sums

Every test of the family builds its models here, so the configuration is the
same everywhere and a change to it shows in all tests at once. All fields are
set per GLOBAL index, so every decomposition sees the same discrete problem.
The numpy backends are the default of the suite (no compiled module needed);
the fortran kernels are compared against them in operators.py when built.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))   # dir containing pyOM3/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))              # pyOM3/test/
import pyOM3 as OM
from common import checksum, reldiff, line                                 # shared by all families

NX, NY, NF, ND = 14, 12, 16, 12
NZ = 12                                  # levels of the ocean column the currents live on
DT = 300.0
FIELDS = ('A', 'loss_A', 'loss_E', 'loss_Eabs', 'taux', 'tauy', 'us', 'vs')


def global_index(m):
    """(jg, ig): global 0-based interior index of each point of the model's
    local (Ny, Nx) layout, as float arrays (halo points get the index of the
    point they mirror or an out-of-range one; apply_bc overwrites them).
    MPI: the local block starts at global index xs_pe-1 (ys_pe-1). JAX: the
    global array holds n_pes blocks of nx/n_pes interior points, each with
    its own halo inline (wim_simple/decomposition_jax.py)."""
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
    """strip the halo of the trailing (y,x) axes: MPI/one PE the local
    interior, JAX the GLOBAL interior (unpad removes every inline halo)"""
    h = m.halo; sl = slice(h, -h)
    if OM.jax:
        return np.asarray(m.unpad(OM.np.asarray(a)))
    return np.asarray(a)[..., sl, sl]


def _new_instance():
    """JAX: the jitted methods take the model as a STATIC argument, so the
    jit cache keeps a reference to every instance it ever compiled for
    (with its arrays and executables, ~230 MB each here) and a test that
    builds models in a loop runs out of memory. Dropping the caches before
    a new instance frees the old ones; the persistent compilation cache on
    disk (jax_sharding.py sets it) keeps the recompilation cheap."""
    if OM.jax:
        OM.jax.clear_caches()


def _grids(kind):
    if kind == 'realistic':
        return dict(dxt=np.linspace(0.9, 1.1, NX), dyt=np.linspace(0.8, 1.2, NY),   # degrees
                    coord_degree=True, x_origin=10.0, y_origin=40.0)
    if kind == 'cartesian':
        return dict(dxt=50e3, dyt=50e3, coord_degree=False)
    return dict(dx=50e3, dy=50e3)                                                   # simple


def ocean_column(kind):
    """dz of the ocean column for the interface routines, bottom to top: a
    variable dzt (2 m at the surface) for 'realistic', which only
    wim_realistic accepts, a uniform dz else"""
    if kind == 'realistic':
        return (2.0 + 28.0*0.5*(1.0 - np.cos(np.linspace(0.0, np.pi, NZ))))[::-1].copy()
    return 10.0


def small_model(kind='realistic', open_coasts=True, energy_consistent=True, reflection=False,
                current=True, janssen=False, tail_flux='wamdi', backend='numpy', snl='numpy',
                sources=True, n_pes_x=1, n_pes_y=1, comm=None, quiet=True):
    _new_instance()
    if kind == 'simple':
        from pyOM3.wim_simple.model import model as _base
    else:
        from pyOM3.wim_realistic.model import model as _base

    class model(_base):
        def set_topography(self):                    # island, per GLOBAL index
            jg, ig = global_index(self)
            kb = np.ones((self.Ny, self.Nx))
            kb[(jg >= 5) & (jg <= 6) & (ig >= 6) & (ig <= 8)] = 0         # across both seams of a 2 x 2 layout
            self.kbot = OM.np.asarray(kb)
    par = dict(nx=NX, ny=NY, nf=NF, nd=ND, f0=0.06, df0=0.13, dt=DT,
               periodic_in_x=True, periodic_in_y=False,
               snonlin_backend=snl, propagation_backend=backend,
               open_coasts=open_coasts, energy_consistent=energy_consistent,
               enable_reflection=reflection, tail_flux=tail_flux,
               limit_growth=sources, mask_energy=sources,
               implicit_dissipation_only=True, closed_bottom=True)
    if janssen:
        par['input_scheme'] = 'janssen'
    par.update(_grids(kind))
    kw = dict(n_pes_x=n_pes_x, n_pes_y=n_pes_y, halo_points=1)
    if comm is not None:
        kw['comm_in'] = comm
    if quiet:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            m = model(par, **kw); m.setup()
    else:
        m = model(par, **kw); m.setup()
    m._kind, m._sources = kind, sources
    m._floor_warns = 20 if quiet else 0              # the floor's warnings (clip_mass keeps the number)
    h = m.halo; sl = slice(h, -h)
    jg, ig = global_index(m)
    yy, xx = jg/(NY - 1), ig/NX
    # the same on the GLOBAL interior grid (the ocean fields come unpadded)
    yy_g = (np.arange(NY)/(NY - 1))[:, None]*np.ones((1, NX))
    xx_g = (np.arange(NX)/NX)[None, :]*np.ones((NY, 1))
    # wind: a storm in the west, the direction turning with y
    m.ustar = m.apply_bc(OM.np.asarray(0.2 + 0.7*np.exp(-((xx - 0.3)/0.25)**2 - ((yy - 0.5)/0.3)**2)))
    m.udir = m.apply_bc(OM.np.asarray(0.5*np.pi*(1.0 + 0.3*yy)))
    # sheared surface-intensified jet through the Doppler projections
    m._dz, m._nz = ocean_column(kind), NZ
    if current:
        dz = np.broadcast_to(m._dz, (NZ,)); z = np.cumsum(dz) - 0.5*dz; H = dz.sum()
        prof = np.exp((z - H)/40.0)
        UJ = 0.8*np.exp(-((yy_g - 0.5)/0.25)**2)*(1.0 + 0.1*np.sin(2*np.pi*xx_g))
        VJ = 0.25*np.exp(-((yy_g - 0.4)/0.3)**2)*np.cos(2*np.pi*xx_g)
        if not OM.jax:                                   # MPI: this PE's block
            blk = (slice(m.ys_pe - 1, m.ye_pe), slice(m.xs_pe - 1, m.xe_pe))
            UJ, VJ = UJ[blk], VJ[blk]
        m._UL, m._VL = prof[:, None, None]*UJ[None], prof[:, None, None]*VJ[None]
        m.ocean_U, m.ocean_V = m.calc_doppler_vel(m._dz, NZ, m._UL, m._VL)
        m.ocean_Udf, m.ocean_Vdf = m.calc_doppler_vel_df(m._dz, NZ, m._UL, m._VL)
    # seed: a swell packet running north-east plus a young wind sea
    # (Pierson-Moskowitz shape, peak 0.13 Hz, cos^2 about the wind), so that
    # dissipation, stress and the diagnostic f^-5 tail are active from step one
    ft, th = np.asarray(m.ft)[sl], np.asarray(m.theta)[sl]
    fseed = np.exp(-((ft - 0.09)/0.03)**2)
    dseed = np.maximum(np.cos(th - np.pi/4), 0.0)**2
    pack = 0.2 + np.exp(-((xx - 0.6)/0.2)**2 - ((yy - 0.4)/0.25)**2)
    pm = 0.01*9.81**2/(2*np.pi)**4*ft**-5*np.exp(-1.25*(0.13/ft)**4)/(2*np.pi*ft)
    spread = (2/np.pi)*np.maximum(np.cos(th[:, None, None] - np.asarray(m.udir)[None]), 0.0)**2
    A = np.zeros((m.Nd, m.Nf, m.Ny, m.Nx))
    A[sl, sl] = 2.0*dseed[:, None, None, None]*fseed[None, :, None, None]*pack[None, None] \
                + spread[:, None]*pm[None, :, None, None]
    m.A = m.apply_bc(OM.np.asarray(A*np.asarray(m.maskTp)[None, None]))
    m.E = m.A*m.omega_t[None, :, None, None]
    return m


def homogeneous_model(ustar=0.5, udir=0.0, amp=1.0, fp=0.13, periodic_y=True, nx=NX, ny=NY,
                      **opt):
    """wim_simple on a doubly periodic (or y-walled) uniform grid WITHOUT the
    island, a uniform wind (ustar, udir) and a spatially uniform spectrum
    (Pierson-Moskowitz shape with peak fp, cos^2 about the wind, amplitude
    amp): the homogeneous case, where the propagation tendency vanishes
    exactly and the source terms act alone. Used by symmetry.py, sources.py
    and budget.py. Options as small_model (current off by default)."""
    _new_instance()
    from pyOM3.wim_simple.model import model as _base

    class model(_base):
        def set_topography(self):
            self.kbot = OM.np.ones((self.Ny, self.Nx), OM.prec)
    kw = dict(current=False, open_coasts=False, energy_consistent=True)
    kw.update(opt)
    par = dict(nx=nx, ny=ny, nf=NF, nd=ND, f0=0.06, df0=0.13, dt=DT,
               periodic_in_x=True, periodic_in_y=periodic_y,
               snonlin_backend=kw.pop('snl', 'numpy'), propagation_backend=kw.pop('backend', 'numpy'),
               open_coasts=kw.pop('open_coasts'), energy_consistent=kw.pop('energy_consistent'),
               enable_reflection=kw.pop('reflection', False), tail_flux=kw.pop('tail_flux', 'wamdi'),
               limit_growth=kw.pop('limit_growth', True), mask_energy=kw.pop('mask_energy', True),
               implicit_dissipation_only=kw.pop('implicit_dissipation_only', True), closed_bottom=True,
               dx=50e3, dy=50e3)
    current = kw.pop('current')
    par.update(kw)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        m = model(par, halo_points=1); m.setup()
    m._kind, m._sources = 'simple', True
    m._floor_warns = 20
    h = m.halo; sl = slice(h, -h)
    m.ustar = m.apply_bc(OM.np.full((m.Ny, m.Nx), float(ustar), OM.prec))
    m.udir = m.apply_bc(OM.np.full((m.Ny, m.Nx), float(udir), OM.prec))
    m._dz, m._nz = 10.0, NZ
    if current:
        m._UL = np.full((NZ, ny, nx), 0.3); m._VL = np.zeros((NZ, ny, nx))
        m.ocean_U, m.ocean_V = m.calc_doppler_vel(m._dz, NZ, m._UL, m._VL)
        m.ocean_Udf, m.ocean_Vdf = m.calc_doppler_vel_df(m._dz, NZ, m._UL, m._VL)
    ft, th = np.asarray(m.ft)[sl], np.asarray(m.theta)[sl]
    pm = 0.01*9.81**2/(2*np.pi)**4*ft**-5*np.exp(-1.25*(fp/ft)**4)/(2*np.pi*ft)        # action
    spread = (2/np.pi)*np.maximum(np.cos(th - float(udir)), 0.0)**2
    A = np.zeros((m.Nd, m.Nf, m.Ny, m.Nx))
    A[sl, sl] = amp*spread[:, None, None, None]*pm[None, :, None, None]
    m.A = m.apply_bc(OM.np.asarray(A))
    m.E = m.A*m.omega_t[None, :, None, None]
    return m


def _adiabatic_step(m, tA, tB, tC, tau, taum1, taum2):
    """the propagation part of model.time_step alone (sources=False): the
    same calls, the same AB weights, the same outflow book-keeping, NO
    action floor (the floor is not conservative, time_step reports it as
    clip_mass) and no source terms"""
    m.xdot, m.ydot, m.fdot, m.ddot = m.mk_advection(m.xdot, m.ydot, m.fdot, m.ddot)
    m.dA = m.add_propagation(m.A, m.dA, m.xdot, m.ydot, m.fdot, m.ddot, tau)
    m.A = m.A + m.dt*(tA*m.dA[tau] + tB*m.dA[taum1] + tC*m.dA[taum2])
    if getattr(m, 'open_coasts', False):
        for acc, lv in (('loss_A', m._lossA_lv), ('loss_E', m._lossE_lv), ('loss_Eabs', m._lossO_lv)):
            setattr(m, acc, getattr(m, acc) + m.dt*(tA*lv[tau] + tB*lv[taum1] + tC*lv[taum2]))
    m.A = m.apply_bc(m.A*m.maskTp[None, None])
    m.E = m.A*m.omega_t[None, :, None, None]


def step(m, n):
    """n steps with the weights of wim's own loop(): forward Euler first, then
    AB2 with eps 0.05. The counter m._nstep and the level pointers m._tau3
    belong to a restart."""
    if not hasattr(m, '_tau3'):
        m._tau3, m._nstep = (2, 1, 0), 0
    tau, taum1, taum2 = m._tau3
    for _ in range(n):
        tA, tB, tC = (1., 0., 0.) if m._nstep == 0 else (1.55, -0.55, 0.)
        if m._sources:
            m.time_step(tA, tB, tC, tau, taum1, taum2)
        else:
            _adiabatic_step(m, tA, tB, tC, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        m._nstep += 1
    m._tau3 = (tau, taum1, taum2)
    return m


def _local_state(m):
    """fields in the local layout: A with its frequency ghosts (imphftail's
    tail lives there and the tail stress reads it), the exchange fields padded"""
    h = m.halo; sl = slice(h, -h)
    out = {'A': np.asarray(m.A)[sl]}                       # (nd, Nf, Ny, Nx)
    if m._sources and hasattr(m, 'gamma_diss'):
        tau = np.asarray(m.calc_stress()); us = np.asarray(m.calc_stokes(m._dz, m._nz))
        if OM.jax:          # already the global interior (unpad inside the interface)
            out.update(taux=tau[0], tauy=tau[1], us=us[0], vs=us[1])
        else:
            out.update(taux=np.pad(tau[0], h), tauy=np.pad(tau[1], h),
                       us=np.pad(us[0], ((0, 0), (h, h), (h, h))), vs=np.pad(us[1], ((0, 0), (h, h), (h, h))))
    return out


def state(m):
    h = m.halo; sl = slice(h, -h)
    out = {k: (_interior(m, v) if k == 'A' or not OM.jax else v) for k, v in _local_state(m).items()}
    la, le, lo = m.coastal_loss()
    out.update(loss_A=np.array(la), loss_E=np.array(le), loss_Eabs=np.array(lo))
    return out


def gathered_state(m):
    """state assembled on rank 0 across the model's PEs (None elsewhere)"""
    if m.n_pes == 1 or OM.jax:                 # JAX: one process, unpad gives the global interior
        return state(m)
    h = m.halo; out = {}
    la, le, lo = m.coastal_loss()                          # collective
    for k, v in _local_state(m).items():
        parts = m.comm.gather((m.ys_pe - 1, m.xs_pe - 1, v[..., h:-h, h:-h]), root=0)
        if m.my_pe == 0:
            g = np.zeros(v.shape[:-2] + (NY, NX))
            for j0, i0, a in parts:
                g[..., j0:j0 + a.shape[-2], i0:i0 + a.shape[-1]] = a
            out[k] = g
    out.update(loss_A=np.array(la), loss_E=np.array(le), loss_Eabs=np.array(lo))
    return out if m.my_pe == 0 else None


# weights and totals shared by identities.py and operators.py
def weights(m):
    """(nd, nf, ny, nx) volume element of the interior: dd dft area"""
    h = m.halo; sl = slice(h, -h)
    area = _interior(m, m.area_t) if hasattr(m, 'area_t') else m.dx*m.dy*np.ones((m.ny_loc, m.nx_loc))
    return np.asarray(m.dft)[sl][None, :, None, None]*m.dd*area[None, None]


def action_total(m):
    h = m.halo; sl = slice(h, -h)
    return float(np.sum(_interior(m, np.asarray(m.A)[sl, sl])*weights(m)))


def abs_energy_total(m):
    """sum Omega A, Omega = omega + k.U with the Doppler velocity U(f)"""
    h = m.halo; sl = slice(h, -h)
    om = np.asarray(m.omega_t)[sl][None, :, None, None]*np.ones((m.nd, 1, 1, 1))
    if hasattr(m, 'ocean_U'):
        s = np.asarray(m.sinth)[sl][:, None, None, None]; c = np.asarray(m.costh)[sl][:, None, None, None]
        om = om + om**2/m.grav*(_interior(m, np.asarray(m.ocean_U)[None, sl])*s
                                + _interior(m, np.asarray(m.ocean_V)[None, sl])*c)
    return float(np.sum(om*_interior(m, np.asarray(m.A)[sl, sl])*weights(m)))
