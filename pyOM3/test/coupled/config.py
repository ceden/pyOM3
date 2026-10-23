"""
coupled family: the shared COUPLING and FORCING code of setups/
(setups/coupled_common/exp_ocean.py, exp_wave.py, exp_common.py and
setups/forcing_common/forcing_base.py) exercised on the small box of the pe
family, without run.py, without input files and without a wave model: the
wave fields the coupler would deliver are synthetic and analytic in the
GLOBAL coordinates, so every decomposition, halo width and backend sees the
same discrete problem.

  small_ocean(**opt)     the ocean of a configuration, assembled exactly as
                         run.py does it (ocean_common + tke + snap_raw on
                         pe_realistic, 16 x 12 x 10, degrees, the topography
                         of the pe family), forced by the 'synthetic' source
                         below. Options: coupled (True: the wave fields are
                         injected every step), relwind (STRESS_MODE), ice
                         (ICE_TAU_FRAC), direct (direct_stress_branch),
                         shear (STOKES_SHEAR), stress_work (use_stress_work),
                         sw_bottom, mxl_choice, hs_factor, piston, ke_budget
                         (1 full, 2 light, 0 off), average ('6h' etc., None
                         off), n_pes_x, n_pes_y, comm, halo
  step(m, n)             n coupled (or forced) steps: _gitt and _ab as the
                         coupler loop sets them, the model's own
                         time_step_implicit_pressure, diagnose, then the
                         wave fields of the next coupling (coupled only),
                         applied as coupler_loop.run_ocean applies them
  state(m)               interior fields: the prognostic ones, the applied
                         forcing (taux, tauy, forc_*_surface, tke flux)
                         and the coupling fields
  wave_fields(m, n)      the synthetic wave fields of coupling step n on this
                         PE's interior block, as the coupler would hand them
                         over: u_S (2, nz, ny, nx), stress (2, ny, nx) or
                         (2, nz, ny, nx), phi_oc, tau_in (2, ny, nx), hs
  SyntheticForcing       forcing_base.Forcing on this rank's block, analytic
                         in xt, yt and t: all eight fields plus u10, v10.
                         Registered as source 'synthetic' in
                         forcing_base.SOURCES by importing this module.
  small_wave(**opt)      the WAVE model of a configuration (wave_common on
                         wim_realistic) on the same box, the synthetic source
                         as its wind; coarsen = 1 or 2 (wave_coarsen)

The experiment module of a configuration is replaced by Setup, an object
with the names the shared classes read (exp_common.Cfg takes any object);
every small_ocean gets its own copy, so the options of one model never leak
into another.
"""
import os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))           # dir containing pyOM3/ and setups/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))        # pyOM3/test/
sys.path.insert(0, os.path.join(ROOT, 'setups', 'coupled_common'))
sys.path.insert(0, os.path.join(ROOT, 'setups', 'forcing_common'))
import pyOM3 as OM
from pyOM3.density import get_rho
from common import checksum, reldiff, line                            # shared by all families
import forcing_base
from forcing_base import Forcing
from exp_common import Cfg, parse_average, coarse_wave_grid
from exp_ocean import ocean_common
from exp_wave import wave_common

NX, NY, NZ = 16, 12, 10
DT = 900.0
OMEGA = np.pi/43082.0
GRAV = 9.81
RHO_0, CP_0, RHO_AIR = 1024.0, 3991.87, 1.225
X0, Y0 = 10.0, 25.0
DXT = np.linspace(0.9, 1.1, NX); DYT = np.linspace(0.8, 1.2, NY)     # degrees (pe family 'realistic')
DZT = 10.0 + 240.0*0.5*(1.0 - np.cos(np.linspace(0.0, np.pi, NZ)))[::-1]   # bottom -> top, 10 m top cell
FIELDS = ('u', 'v', 'w', 'temp', 'salt', 'p_s', 'tke', 'K_diss_v',
          'taux', 'tauy', 'forc_temp_surface', 'forc_salt_surface', 'forc_tke_surface')
COUPLED_FIELDS = ('u_S', 'v_S', 'phi_oc', 'tau_in_x', 'tau_in_y', 'hs')
# the tolerance of a check that is exact on numpy: on the jax backend XLA's
# fused kernels round differently from the op-by-op reference formed in the
# test (GPU: 1e-21 .. 1e-19 absolute in the forcing checks)
EXACT = 1e-15 if OM.jax else 0.0
DAY = 86400.0

# the work directory of the suite: snapshots, energy files, restarts
WORK = os.path.join(HERE, '_work')


def _global_coords():
    """T coordinates of the global interior grid [degrees], as pe_realistic
    and wim_realistic form them (u_centered_grid: x_origin is the U point of
    the first interior cell, xt(0) = x_origin - dxt(0)/2)."""
    from pyOM3.wim_realistic.grid import u_centered_grid       # the same recursion as pe_realistic's
    xu, xt, _ = u_centered_grid(DXT); yu, yt, _ = u_centered_grid(DYT)
    xt = xt + X0 - xu[0]; yt = yt + Y0 - yu[0]
    return xt, yt


def kbot_global():
    """Bottom level per global cell in the halo-1 padded convention (1 = the
    whole column, 0 = land): the closed box with the shelf and the seamount
    of the pe family, plus a land cell in the open sea (the ice/land masks
    of the forcing must not care)."""
    jg = np.arange(NY)[:, None]*np.ones((1, NX)); ig = np.arange(NX)[None, :]*np.ones((NY, 1))
    kb = np.ones((NY, NX))
    kb[(jg >= 8) & (ig <= 3)] = NZ - 2
    kb[(jg >= 4) & (jg <= 5) & (ig >= 9) & (ig <= 10)] = 5
    kb[(jg == 2) & (ig == 12)] = 0
    return kb


def zw_pyom2():
    """depth of the cell bottoms, top -> bottom, negative (pyOM2 zw)"""
    return -np.cumsum(DZT[::-1])


# ---------------------------------------------------------------------------
# THE SYNTHETIC FORCING SOURCE: analytic fields of the block's coordinates
# and the model time. Every field the interface knows is served; `fields`
# restricts the dict as a real source would. The time dependence has a
# diurnal and a longer cycle so that two steps never see the same forcing.
# ---------------------------------------------------------------------------
class SyntheticForcing(Forcing):
    """settings: 'start', 'cycle_days' (forcing_base), 'amp' (overall factor,
    default 1), 'omit' (names NOT to serve, e.g. ('qsol',) for a source
    without a solar part)."""

    def __init__(self, settings, geom, fields, who):
        st = dict(settings); st.setdefault('start', '2000-01-01'); st.setdefault('cycle_days', 0.0)
        super().__init__(st, geom, fields, who)
        self.amp = float(st.get('amp', 1.0))
        self.omit = tuple(st.get('omit', ()))
        xt, yt = np.asarray(geom['xt'], float), np.asarray(geom['yt'], float)
        xg, yg = _global_coords()
        # normalised global position of the block's cells, [0, 1]
        self.xx = ((xt - xg[0])/(xg[-1] - xg[0]))[None, :]*np.ones((len(yt), 1))
        self.yy = ((yt - yg[0])/(yg[-1] - yg[0]))[:, None]*np.ones((1, len(xt)))
        self.n_calls = 0

    def get(self, t):
        self.n_calls += 1
        xx, yy, a = self.xx, self.yy, self.amp
        d = 1.0 + 0.3*np.sin(2*np.pi*t/DAY)                         # diurnal
        s = np.cos(2*np.pi*t/(30*DAY))                              # monthly
        F = {'taux': a*(-0.2*np.cos(np.pi*yy) + 0.05*np.sin(2*np.pi*xx)*s),
             'tauy': a*(0.06*np.sin(2*np.pi*xx)*np.cos(np.pi*yy)*d),
             'qnet': a*(50.0 - 250.0*np.clip(2*yy - 1.0, 0.0, 1.0) + 150.0*d*(1.0 - yy)),
             'qsol': a*(180.0*d*(1.0 - 0.6*yy)),
             'sst_star': 4.0 + 16.0*(1.0 - yy) + 0.5*s,
             'dqdt': a*(-30.0 - 15.0*xx),
             'emp': a*(2.0e-8*np.sin(np.pi*yy)*np.cos(2*np.pi*xx) - 1.0e-8*s),
             'sss_star': 35.0 + 0.8*(yy - 0.5) + 0.2*np.sin(2*np.pi*xx),
             'u10': a*(12.0*np.cos(np.pi*yy)*(1.0 + 0.2*s) - 3.0*np.sin(2*np.pi*xx)),
             'v10': a*(4.0*np.sin(2*np.pi*xx)*np.cos(np.pi*yy)*d + 1.0*yy)}
        # ocean cells below the freezing point in the far north: the ice rule
        # must act somewhere (the restoring target does it)
        F['sst_star'] = np.where(yy > 0.9, -2.0, F['sst_star'])
        return {k: v for k, v in F.items() if k in self.fields and k not in self.omit}

    def describe(self):
        return ['forcing: synthetic analytic fields (test suite), amp %g' % self.amp]


forcing_base.SOURCES['synthetic'] = ('config', 'SyntheticForcing')
sys.modules.setdefault('config', sys.modules[__name__])


# ---------------------------------------------------------------------------
# THE EXPERIMENT MODULE, as an object: the names exp_ocean/exp_wave read
# ---------------------------------------------------------------------------
class Setup:
    __name__ = 'coupled_test'
    exp_name = 'coupled_test'
    nx, ny, nz = NX, NY, NZ
    rho_0, cp_0, rho_air = RHO_0, CP_0, RHO_AIR
    t_rest = 30.0*DAY
    omega_e = OMEGA
    FORCING_START = '2000-01-01'
    FORCING_CYCLE_DAYS = 0.0
    HS_FACTOR = 1.6
    MXL_CHOICE = 1
    MXL_BETA = 1.0
    MXL_SURF_FACTOR = 1.0
    RAW_SNAPSHOTS = True
    KE_BUDGET = 1
    AVERAGE_MLD = True
    dxt_deg, dyt_deg, x_origin, y_origin = DXT, DYT, X0, Y0
    forcing_amp = 1.0
    forcing_omit = ()

    def __init__(self, **opt):
        for k, v in opt.items():
            setattr(self, k, v)
        if getattr(self, 'wave_coarsen', 1) > 1:
            # the coarse wave grid, as a configuration builds it (exp_common)
            wg = coarse_wave_grid(DXT, DYT, X0, Y0, self.wave_coarsen)
            self.nx_wave, self.ny_wave = wg['nx'], wg['ny']
            self._ie_w, self._je_w = wg['ie'], wg['je']
            self.dxt_wave, self.dyt_wave = wg['dxt'], wg['dyt']
            self.x_origin_wave, self.y_origin_wave = wg['x_origin'], wg['y_origin']

    def kbot_global(self):
        return kbot_global()

    def dzt_global(self):
        return DZT.copy()

    def zw_pyom2(self):
        return zw_pyom2()

    def forcing_settings(self):
        return {'source': 'synthetic', 'start': self.FORCING_START,
                'cycle_days': self.FORCING_CYCLE_DAYS, 'amp': self.forcing_amp,
                'omit': self.forcing_omit}


def _work_dir():
    os.makedirs(WORK, exist_ok=True)
    return WORK


# ---------------------------------------------------------------------------
def small_ocean(coupled=True, relwind=False, ice=1.0, direct=False, shear=False,
                stress_work=False, sw_bottom=False, mxl_choice=1, hs_factor=1.6, piston=0.0,
                ke_budget=1, average=None, n_pes_x=1, n_pes_y=1, comm=None, halo=1,
                quiet=True, amp=1.0, omit=(), tag='o'):
    """The ocean of a configuration, assembled as run.py does it."""
    from pyOM3.pe_realistic.snapshot import snap_raw
    from pyOM3.pe_realistic.tke import tke as _tke
    setup = Setup(STRESS_MODE='relwind' if relwind else 'era5', ICE_TAU_FRAC=float(ice),
                  direct_stress_branch=bool(direct), STOKES_SHEAR=bool(shear),
                  SW_BOTTOM=bool(sw_bottom), MXL_CHOICE=int(mxl_choice), HS_FACTOR=float(hs_factor),
                  SSS_PISTON=float(piston), KE_BUDGET=int(ke_budget), forcing_amp=amp, forcing_omit=omit)

    class ocean(ocean_common, _tke, snap_raw):
        cfg = Cfg(setup)

        def make_initial_fields(self):
            """stratified, a warm anomaly, uniform salt, resting (global interior)"""
            jg = np.arange(NY)[:, None]*np.ones((1, NX)); ig = np.arange(NX)[None, :]*np.ones((NY, 1))
            zt = np.cumsum(DZT) - 0.5*DZT; H = DZT.sum()
            T = 4.0 + 16.0*(zt/H)[:, None, None]*np.ones((NZ, NY, NX))
            T += 1.0*np.exp(-((ig - NX/2)/3.)**2 - ((jg - NY/2)/2.)**2)[None]*(zt/H)[:, None, None]
            S = 35.0*np.ones((NZ, NY, NX)) + 0.3*(zt/H - 0.5)[:, None, None]
            return (np.zeros((NZ, NY, NX)), np.zeros((NZ, NY, NX)), T, S, np.zeros((NY, NX)))

        def harmonic_momentum(self, *a, **k):
            """book the KE change of the horizontal friction, as a
            configuration's class does (coupler_loop expects _ke_hfric)"""
            if not self.cfg.KE_BUDGET:
                return super().harmonic_momentum(*a, **k)
            u0, v0 = self.u*1.0, self.v*1.0
            k0 = self._ke_of(u0, v0)
            out = super().harmonic_momentum(*a, **k)
            self._ke_hfric = self._ke_of(self.u, self.v) - k0
            self._ke_hnum = self._ke_of(self.u - u0, self.v - v0)
            self._ke_hstencil = self._ke_hfric - self._ke_hnum
            return out

    par = dict(nx=NX, ny=NY, nz=NZ, dt=DT, dxt=DXT, dyt=DYT, dzt=DZT, coord_degree=True,
               x_origin=X0, y_origin=Y0, periodic_in_x=False, periodic_in_y=False,
               eq_of_state=5, explicit_free_surface=False, implicit_free_surface=True,
               superbee_advection=True, vmix_scheme='implicit', enable_conv_adjust=False,
               enable_tke=True, c_k=0.1, c_eps=0.7, alpha_tke=30.0, mxl_min=1e-8,
               kappa_T=1e-5, A_v=1e-4, Ahbi=1e11, biha_friction_cosPower=1.0,
               epsilon=1e-12, max_iterations=5000)
    kw = dict(n_pes_x=n_pes_x, n_pes_y=n_pes_y, halo_points=halo)
    if comm is not None:
        kw['comm_in'] = comm
    cwd = os.getcwd(); os.chdir(_work_dir())
    import io, contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
            m = ocean(par, **kw)
            m.snapint = 1e9*DT
            m.use_stress_work = stress_work
            m.coupled = coupled
            m.hs_factor = hs_factor
            m.alpha_CB = 100.0                 # Craig-Banner flux of the ocean alone (a configuration's choice)
            m.energy_file = '%s_energy_%s.txt' % (setup.exp_name, tag)
            m.set_coriolis()
            m.set_topography()
            m.make_topo_masks()
            m.set_initial_conditions(*m.make_initial_fields())
            m.tke_surf_from_waves = coupled
            if not coupled:
                m.mix_eulerian = False
            m.init_tke()
            m.localise_forcing()
            m.prepare_for_run(snap_file_name='%s_snap_%s' % (setup.exp_name, tag), show_each_time_step=False)
            m.average_setup('%s_mean_%s.cdf' % (setup.exp_name, tag), parse_average(average) if average else None)
            m.u_S = OM.np.zeros((m.Nz, m.Ny, m.Nx), np.float64)
            m.v_S = OM.np.zeros((m.Nz, m.Ny, m.Nx), np.float64)
            if stress_work:
                m.tau_work_x = OM.np.zeros((m.Nz, m.Ny, m.Nx), np.float64)
                m.tau_work_y = OM.np.zeros((m.Nz, m.Ny, m.Nx), np.float64)
            else:
                m.taux = OM.np.zeros((m.Ny, m.Nx), np.float64)
                m.tauy = OM.np.zeros((m.Ny, m.Nx), np.float64)
            m.phi_oc = OM.np.zeros((m.Ny, m.Nx), np.float64)
            m.tau_in_x = OM.np.zeros((m.Ny, m.Nx), np.float64)
            m.tau_in_y = OM.np.zeros((m.Ny, m.Nx), np.float64)
            if coupled:
                m.hs = OM.np.zeros((m.Ny, m.Nx), np.float64)
    finally:
        os.chdir(cwd)
    m._setup = setup
    m.tke = OM.np.full((m.Nz, m.Ny, m.Nx), 1e-6, OM.prec)*m.maskW
    m._tau3 = (2, 1, 0); m._nstep = 0
    return m


# ---------------------------------------------------------------------------
# THE WAVE FIELDS the coupler would deliver: analytic in the global
# coordinates and the coupling step, on this PE's interior block
# ---------------------------------------------------------------------------
def block_coords(m):
    """normalised global position (xx, yy) in [0, 1] of this PE's interior
    cells (numpy, (ny_loc, nx_loc)); JAX with several shards: the global interior"""
    h = m.halo; sl = slice(h, -h)
    xt, yt = np.asarray(m.xt), np.asarray(m.yt)
    if OM.jax and (m.n_pes_x > 1 or m.n_pes_y > 1):
        xt = np.asarray(m.unpad(OM.np.asarray(xt)[None, :]*OM.np.ones((m.Ny, 1))))[0]
        yt = np.asarray(m.unpad(OM.np.asarray(yt)[:, None]*OM.np.ones((1, m.Nx))))[:, 0]
    else:
        xt, yt = xt[sl], yt[sl]
    xg, yg = _global_coords()
    xx = ((xt - xg[0])/(xg[-1] - xg[0]))[None, :]*np.ones((len(yt), 1))
    yy = ((yt - yg[0])/(yg[-1] - yg[0]))[:, None]*np.ones((1, len(xt)))
    return xx, yy


def wave_fields(m, n):
    """u_S (2, nz, ny, nx), stress (2, ny, nx) or (2, nz, ny, nx) with
    use_stress_work, phi_oc (ny, nx), tau_in (2, ny, nx), hs (ny, nx) of
    coupling step n: a Stokes drift decaying over 8 m, turning with time,
    the released stress a fraction of the wind stress, tau_in exceeding tau_a
    where the sea is young (so the direct branch's clipping acts)."""
    xx, yy = block_coords(m)
    ph = 2*np.pi*n/50.0
    zt = DZT.sum() - (np.cumsum(DZT) - 0.5*DZT)     # depth below the surface, bottom -> top
    prof = np.exp(-zt/8.0)[:, None, None]
    us0 = 0.08*(1.0 + 0.5*np.cos(np.pi*yy))*(1.0 + 0.2*np.sin(2*np.pi*xx + ph))
    vs0 = 0.03*np.sin(2*np.pi*xx)*np.cos(np.pi*yy - ph)
    u_S = np.stack([prof*us0[None], prof*vs0[None]])
    tx = -0.6*0.2*np.cos(np.pi*yy)*(1.0 + 0.1*np.sin(ph)) + 0.02*np.cos(2*np.pi*xx + ph)
    ty = 0.6*0.06*np.sin(2*np.pi*xx)*np.cos(np.pi*yy) + 0.01*np.sin(ph)
    tin = np.stack([0.9*tx*(1.0 + 0.5*np.exp(-((xx - 0.3)/0.15)**2)), 0.9*ty])/RHO_0
    if m.use_stress_work:
        wprof = np.exp(-zt/4.0); wprof = wprof/np.sum(wprof*DZT)
        stress = np.stack([wprof[:, None, None]*tx[None], wprof[:, None, None]*ty[None]])/RHO_0
    else:
        stress = np.stack([tx, ty])/RHO_0
    phi = 1e-5*(1.0 + np.cos(np.pi*yy))*(1.0 + 0.3*np.cos(ph))
    hs = 2.0*(1.0 + 0.5*np.cos(np.pi*yy))*(1.0 + 0.2*np.sin(2*np.pi*xx - ph))
    return u_S, stress, phi, tin, hs


def _set_block(m, a, val):
    """the coupler's modify_array of the interior block + apply_bc"""
    h = m.halo; hb = slice(h, -h)
    if OM.jax and (m.n_pes_x > 1 or m.n_pes_y > 1):
        return m.apply_bc(m.pad(OM.np.asarray(val)))
    if val.ndim == 2:
        return m.apply_bc(OM.modify_array(a, (hb, hb), OM.np.asarray(val)))
    return m.apply_bc(OM.modify_array(a, (hb, hb, hb), OM.np.asarray(val)))


def inject_wave_fields(m, n):
    """as coupler_loop.run_ocean after the receive"""
    u_S, stress, phi, tin, hs = wave_fields(m, n)
    m.u_S_old, m.v_S_old = m.u_S.copy(), m.v_S.copy()
    m.u_S = _set_block(m, m.u_S, u_S[0]); m.v_S = _set_block(m, m.v_S, u_S[1])
    if m.use_stress_work:
        m.tau_work_x = _set_block(m, m.tau_work_x, stress[0])
        m.tau_work_y = _set_block(m, m.tau_work_y, stress[1])
    else:
        m.taux = _set_block(m, m.taux, stress[0]); m.tauy = _set_block(m, m.tauy, stress[1])
    m.phi_oc = _set_block(m, m.phi_oc, phi)
    m.tau_in_x = _set_block(m, m.tau_in_x, tin[0]); m.tau_in_y = _set_block(m, m.tau_in_y, tin[1])
    m.hs = _set_block(m, m.hs, hs)
    m.set_wave_roughness()
    return


def _ab_weights(n):
    return (1.6, -0.6, 0.0)


def step(m, n, diagnose=True, quiet=True):
    """n steps of the coupled (or forced) ocean, the sequence of
    coupler_loop.run_ocean / run_ocean_only"""
    import io, contextlib
    cwd = os.getcwd(); os.chdir(_work_dir())
    try:
      with contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
        for _ in range(n):
            tau, taum1, taum2 = m._tau3
            m._gitt = m._nstep
            m._ab = _ab_weights(m._nstep)
            m.time_step_implicit_pressure(*m._ab, tau, taum1, taum2)
            if diagnose:
                m.diagnose(m._nstep, tau)
            m._tau3 = ((tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3)
            m._nstep += 1
            if m.coupled:
                inject_wave_fields(m, m._nstep)
    finally:
        os.chdir(cwd)
    return m


def _interior(m, a):
    h = m.halo; sl = slice(h, -h)
    if OM.jax:
        return np.asarray(m.unpad(OM.np.asarray(a)))
    return a[..., sl, sl] if a.ndim == 2 else a[sl, sl, sl]


def state(m, fields=None):
    out = {}
    names = fields or (FIELDS + (COUPLED_FIELDS if m.coupled else ()))
    for f in names:
        if not hasattr(m, f):
            continue
        a = np.asarray(getattr(m, f))
        if f == 'p_s':
            a = a[m._tau3[1]]
        out[f] = _interior(m, a)
    return out


def diagnostics(m):
    """the scalar and reduced diagnostics of the shared code after a step
    (numpy): the exact bookings, the KE budget accumulators, the online
    means' partial sums, the mixed layer depth. Compared across backends and
    layouts by jax_backend.py, since none of them is in the state."""
    out = {'stress_work': np.array([m.stress_work_diag()]),
           'exchange_C': np.array([m.exchange_work_C()]),
           'vortex': np.array([m.vortex_work()])}
    if m.coupled:
        out['stress_work_eul'] = np.array([m.stress_work_eul_diag()])
        out['stress_work_field'] = _interior(m, np.asarray(m.stress_work_field()))
    if hasattr(m, '_tke_lhs'):
        out['tke_lhs_rhs'] = np.array([m._tke_lhs, m._tke_rhs])
    if hasattr(m, 'eb_acc'):
        out['eb_acc'] = m.eb_acc.copy()         # (eb_chk, eb_trc are round-off residuals: test 19)
    if getattr(m, '_avg', None) is not None:
        for nme, a in m.average_state()[0]:
            out['avg_' + nme] = np.asarray(a)
    out['mld'] = _interior(m, np.asarray(m.mixed_layer_depth()))
    return out


def gathered_state(m, fields=None):
    """state assembled on rank 0 across the model's PEs (None elsewhere)"""
    if m.n_pes == 1 or OM.jax:
        return state(m, fields)
    h = m.halo; out = {}
    names = fields or (FIELDS + (COUPLED_FIELDS if m.coupled else ()))
    for f in names:
        if not hasattr(m, f):
            continue
        a = np.asarray(getattr(m, f))
        if f == 'p_s':
            a = a[m._tau3[1]]
        if a.ndim == 2:
            b = np.zeros((1 + 2*h,) + a.shape); b[h] = a; a3 = OM.np.asarray(b)
        else:
            a3 = a
        g = m.gather_array_mpi(a3)
        if m.my_pe == 0:
            g = np.asarray(g)
            out[f] = g[0] if a.ndim == 2 else g
    return out if m.my_pe == 0 else None


# ---------------------------------------------------------------------------
# THE WAVE MODEL of a configuration (exp_wave.wave_common on wim_realistic)
# ---------------------------------------------------------------------------
def small_wave(coarsen=1, n_pes_x=1, n_pes_y=1, comm=None, halo=1, quiet=True, relwind=False):
    from pyOM3.wim_realistic.snapshot import snap_raw as _base
    setup = Setup(wave_coarsen=int(coarsen), STRESS_MODE='relwind' if relwind else 'era5')

    class wave(wave_common, _base):
        cfg = Cfg(setup)

    if coarsen > 1:
        g = dict(nx=setup.nx_wave, ny=setup.ny_wave, dxt=setup.dxt_wave, dyt=setup.dyt_wave,
                 x_origin=setup.x_origin_wave, y_origin=setup.y_origin_wave)
    else:
        g = dict(nx=NX, ny=NY, dxt=DXT, dyt=DYT, x_origin=X0, y_origin=Y0)
    par = dict(nf=16, nd=12, f0=0.06, df0=0.13, dt=300.0, coord_degree=True,
               periodic_in_x=False, periodic_in_y=False,
               snonlin_backend='numpy', propagation_backend='numpy',
               open_coasts=True, energy_consistent=True, enable_reflection=False,
               tail_flux='input', implicit_dissipation_only=True, closed_bottom=True)
    par.update(g)
    kw = dict(n_pes_x=n_pes_x, n_pes_y=n_pes_y, halo_points=halo)
    if comm is not None:
        kw['comm_in'] = comm
    cwd = os.getcwd(); os.chdir(_work_dir())
    import io, contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
            m = wave(par, **kw)
            m.snapint = 1e9*300.0
            m.setup(snap_file_name='coupled_test_wsnap', show_each_time_step=False)
            m.set_initial()
    finally:
        os.chdir(cwd)
    m._setup = setup
    return m


def wave_state(m, nstep=1):
    """the wave side of the shared code after `nstep` steps (numpy, global
    interior): the mask, u*, udir, the wind input field and the two work
    diagnostics; compared across backends and layouts by jax_backend.py"""
    from block_io import interior
    tau, taum1, taum2 = 2, 1, 0
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(nstep):
            tA, tB, tC = (1., 0., 0.) if k == 0 else (1.55, -0.55, 0.)
            m.time_step(tA, tB, tC, tau, taum1, taum2)
            tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
    m.update_forcing(nstep*m.dt)
    return {'kbot': interior(m, m.kbot), 'ustar': interior(m, m.ustar), 'udir': interior(m, m.udir),
            'p_in': np.asarray(m.wind_input_field()),            # (ny, nx), interior already
            'wind_work': np.array([m.wind_work()]), 'E': interior(m, m.E, lead=2)}
