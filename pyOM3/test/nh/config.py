"""
nh family (nh_simple). Small configuration for the routine tests
(README_test_suite.md): a doubly periodic box, 16 x 12 x 10 cells, a
stratified fluid with a buoyancy anomaly and a sheared jet, rotation, a
surface stress and a surface buoyancy loss, harmonic friction and mixing.

  small_model(kind, **opt)   'lid' (rigid lids top and bottom: the default of
                             the suite, the 2D-FFT + tridiagonal pressure
                             solver) or 'periodic' (periodic in z, the 3D-FFT
                             solver, no surface forcing). Options:
                             b_adv ('2nd', 'superbee', 'upwind3'),
                             les (None, 'smag', 'tke', 'ducros'),
                             biharmonic (friction operator), forcing, b_mix
                             (explicit mixing of b; off for the reduction),
                             n_pes_x, n_pes_y (p3dfft: n_pes_x splits y,
                             n_pes_y splits z), comm, quiet.
  step(m, n)                 n steps of the model's own time_step with the
                             cold-start ramp of its loop() (forward Euler,
                             AB2, then AB3); m._tau3 and m._nstep carry the
                             level pointers and the ramp and belong to a
                             restart
  state(m)                   dict of the interior prognostic fields (numpy)
  gathered_state(m)          the same assembled on rank 0
  checksum(state)            one float per field, order-independent sums

Arrays of nh_simple are fortran ordered, (Nx, Ny, Nz) with one halo point on
each side; the interior is [h:-h, h:-h, h:-h]. The model is dimensional
here (Ro = dsqr = 1), which the LES closures require.

The LES closure is hooked in through set_forcing, where the model calls it
after the harmonic friction and before the Adams-Bashforth step, as les.py
asks; the surface forcing is applied in the same hook.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))   # dir containing pyOM3/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))              # pyOM3/test/
import pyOM3 as OM
from common import checksum, reldiff, line                                 # shared by all families

NX, NY, NZ = 16, 12, 10
DX, DY, DZ = 4.0, 4.0, 2.0          # m
DT = 2.0                            # s
N0 = 5e-3                           # 1/s, background stratification
F0 = 1e-4                           # 1/s
FIELDS = ('u', 'v', 'w', 'b', 'p')
LES_FIELDS = ('tke', 'diff', 'shear_prod', 'buoy_prod', 'stokes_prod')   # TKE closure state carried across steps


def small_model(kind='lid', b_adv='2nd', les=None, biharmonic=False, forcing=True, b_mix=True,
                n_pes_x=1, n_pes_y=1, comm=None, quiet=True, **extra):
    from pyOM3.nh_simple.model import model as _base
    from pyOM3.nh_simple.les import les as _les

    class model(_les, _base):                      # the les mixin, as the configurations do
        pass
    par = dict(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, dt=DT, Ro=1.0, dsqr=1.0,
               N0=N0, f0=F0, periodic_in_z=(kind == 'periodic'))
    if biharmonic:
        par.update(Ahbi=2e-2, Avbi=5e-3)                             # m^4/s
        if b_mix:
            par.update(Khbi=2e-2, Kvbi=5e-3)
    else:
        par.update(Ah=2e-3, Av=1e-3)                                 # m^2/s
        if b_mix:
            par.update(Kh=2e-3, Kv=1e-3)
    par['buoyancy_superbee_advection'] = (b_adv == 'superbee')
    par['buoyancy_upwind3_advection'] = (b_adv == 'upwind3')
    if les == 'smag':
        par.update(enable_smagorinsky=True)
    elif les == 'tke':
        par.update(enable_tke_closure=True, les_mix_b=(b_adv != 'superbee'))
    elif les == 'ducros':
        par.update(enable_ducros_96=True)
    elif les is not None:
        raise ValueError('les must be None, smag, tke or ducros')
    par.update(extra)
    kw = dict(n_pes_x=n_pes_x, n_pes_y=n_pes_y)
    if comm is not None:
        kw['comm_in'] = comm
    if quiet:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            m = model(par, **kw)
    else:
        m = model(par, **kw)
    m.setup()
    if les is not None:
        m.init_les()
    h = m.halo
    # GLOBAL coordinates of the local block (all decompositions see the same fields)
    x = (np.arange(m.Nx) - h + m.xs_pe - 1 + 0.5)*DX
    y = (np.arange(m.Ny) - h + m.ys_pe - 1 + 0.5)*DY
    z = (np.arange(m.Nz) - h + m.zs_pe - 1 + 0.5)*DZ
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    Lx, Ly, Lz = NX*DX, NY*DY, NZ*DZ
    # initial state: a buoyancy anomaly (zero mean), a sheared jet with a
    # vertical structure, a weak perturbation so that every term is active
    gauss = lambda X, Y, Z: 2e-4*np.exp(-((X - 0.5*Lx)/(0.2*Lx))**2 - ((Y - 0.5*Ly)/(0.2*Ly))**2 - ((Z - 0.6*Lz)/(0.25*Lz))**2)
    XG, YG, ZG = np.meshgrid((np.arange(NX) + 0.5)*DX, (np.arange(NY) + 0.5)*DY, (np.arange(m.nz) + 0.5)*DZ, indexing='ij')
    b = gauss(X, Y, Z) - gauss(XG, YG, ZG).mean()                  # zero GLOBAL mean, the same on every rank
    u = 0.05*np.sin(2*np.pi*Y/Ly)*(0.5 + Z/Lz) + 0.01*np.cos(2*np.pi*X/Lx)*np.sin(np.pi*Z/Lz)
    v = 0.01*np.sin(2*np.pi*X/Lx)*np.cos(2*np.pi*Y/Ly)
    m.b = m.apply_bc(OM.np.asarray(b))
    m.u = m.apply_bc(OM.np.asarray(u))
    m.v = m.apply_bc(OM.np.asarray(v))
    m.w = m.apply_vertical_flux_bc(m.apply_bc(OM.np.zeros_like(m.u)))
    m.pressure_solver()                              # project the initial velocity
    if les == 'tke':
        m.tke = m.apply_bc(OM.np.full_like(m.u, 1e-5))
        m.diff = OM.np.sqrt(m.tke)*(DX*DY*DZ)**(1./3.)
    # forcing, per GLOBAL index: a zonal stress and a surface buoyancy loss
    # (cooling), both in the top cell. Only with rigid lids.
    m._forcing = forcing and kind == 'lid'
    m._taux = OM.np.asarray(1e-4*(1.0 + 0.5*np.cos(2*np.pi*Y[:, :, 0]/Ly)))     # m^2/s^2
    m._bflux = OM.np.asarray(-2e-7*np.ones_like(m._taux))                      # m^2/s^3
    m._les = les
    m.set_forcing = _set_forcing.__get__(m)
    return m


def _set_forcing(self, tau):
    """surface stress and buoyancy flux into the top cell (du, db at level
    tau, Adams-Bashforth weighted by the model), then the LES closure"""
    if self._forcing and self.my_blk_z == self.n_pes_z:
        ks = -self.halo - 1
        self.du = OM.add_to_array(self.du, (tau, slice(None), slice(None), ks), self._taux/self.dz)
        self.db = OM.add_to_array(self.db, (tau, slice(None), slice(None), ks), self._bflux/self.dz)
    if self._les is not None:
        self.les_closure()


AB3 = (23./12., -16./12., 5./12.)


def ab_weights(nstep):
    """the weights loop() uses at step nstep of a cold start"""
    return ((1., 0., 0.), (1.5, -0.5, 0.))[nstep] if nstep < 2 else AB3


def step(m, n):
    """n steps with the cold-start ramp of the model's loop(): forward
    Euler, AB2, then AB3. The pointers m._tau3 and the counter m._nstep are
    restart state."""
    if not hasattr(m, '_tau3'):
        m._tau3 = (2, 1, 0)
    if not hasattr(m, '_nstep'):
        m._nstep = 0
    tau, taum1, taum2 = m._tau3
    for _ in range(n):
        A, B, C = ab_weights(m._nstep)
        m._ab = (A, B, C)
        m.time_step(A, B, C, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        m._nstep += 1
    m._tau3 = (tau, taum1, taum2)
    return m


def interior(m, a):
    h = m.halo
    return np.array(np.asarray(a)[h:-h, h:-h, h:-h])               # a copy: the model steps in place


def state(m):
    out = {f: interior(m, getattr(m, f)) for f in FIELDS}
    if getattr(m, '_les', None) == 'tke':
        out['tke'] = interior(m, m.tke)
    return out


def gathered_state(m):
    """state assembled on rank 0 across the model's PEs (None elsewhere)"""
    if m.n_pes == 1:
        return state(m)
    fields = FIELDS + (('tke',) if getattr(m, '_les', None) == 'tke' else ())
    out = {}
    for f in fields:
        g = m.gather_array_mpi(getattr(m, f))
        if m.my_pe == 0:
            out[f] = np.asarray(g)
    return out if m.my_pe == 0 else None
