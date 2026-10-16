"""Test 5: operator unit tests on a model object with a prescribed state.

  pressure solver   on a prescribed divergent velocity, for the lid and the
                    periodic solver: the projected velocity is divergence
                    free (1e-13) and p satisfies the discrete Poisson
                    equation with a hand-written stencil, Neumann at the
                    lids (1e-11, FFT accuracy)
  lid convention    apply_vertical_flux_bc zeroes a T-grid flux exactly at
                    the lid faces (index 0 and nz) and nowhere else
  uniform tracer    the three buoyancy advection tendencies vanish for a
                    uniform b in a divergence-free flow                1e-13
  monotonicity      superbee keeps a top hat within [0, 1] under 1D
                    translation (forward Euler, CFL 0.5)              1e-15
                    control: the 2nd-order scheme must overshoot
  Smagorinsky       pure shear u = S z: visc_00 = (C delta)^2 |S| in the
                    interior levels                                   1e-12
  lid mirror        every operator of the rigid-lid model must map the
                    vertically mirrored state (u, v even, w, b odd) to the
                    mirrored tendency: momentum and buoyancy advection,
                    harmonic and biharmonic friction of a T-grid field and
                    of w, pressure solver, Smagorinsky viscosities   1e-13
                    les_mix_momentum, les_mix_buoyancy (from the
                    increments, 1e-13 of cancellation)               1e-11
                    w-grid fields are compared on the faces 1..nz-1.
                    The TKE closure is excluded: its w-grid has a TKE
                    point at the top lid and none at the bottom (ps3D
                    convention), so it is asymmetric by design.
"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, line, interior, NX, NY, NZ, DX, DY, DZ
import pyOM3 as OM

rel = lambda a, b: float(np.max(np.abs(a)))/max(float(np.max(np.abs(b))), 1e-300)


def mirror_T(a):
    return np.asarray(a)[:, :, ::-1].copy()


def mirror_W(a):
    """face k (above cell k) <-> face nz - k; the lid faces 0 and nz swap"""
    a = np.asarray(a); b = np.zeros_like(a); Nz = a.shape[2]
    for k in range(Nz):
        kk = Nz - 2 - k
        if 0 <= kk < Nz:
            b[:, :, kk] = a[:, :, k]
    return b


def pressure_solver_test(kind):
    m = small_model(kind, forcing=False); h = m.halo; I = (slice(h, -h),)*3
    rx, ry, rz = m.rollx, m.rolly, m.rollz
    x = (np.arange(m.Nx) - h + 0.5)*DX; y = (np.arange(m.Ny) - h + 0.5)*DY; z = (np.arange(m.Nz) - h + 0.5)*DZ
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = 0.1*np.sin(2*np.pi*X/(NX*DX))*np.cos(2*np.pi*Y/(NY*DY)) + 0.02*np.cos(np.pi*Z/(NZ*DZ))
    w = 0.03*np.sin(np.pi*Z/(NZ*DZ))*np.cos(2*np.pi*X/(NX*DX))
    m.u = m.apply_bc(OM.np.asarray(u)); m.v = m.apply_bc(OM.np.asarray(0.5*u))
    m.w = m.apply_vertical_flux_bc(m.apply_bc(OM.np.asarray(w)))
    us, vs, ws = (np.asarray(f).copy() for f in (m.u, m.v, m.w))
    rhs = ((us - rx(us, 1))/m.dx + (vs - ry(vs, 1))/m.dy + (ws - rz(ws, 1))/m.dz)/m.dt
    m.pressure_solver()
    u, v, w, p = (np.asarray(f) for f in (m.u, m.v, m.w, m.p))
    div = ((u - rx(u, 1))/m.dx + (v - ry(v, 1))/m.dy + (w - rz(w, 1))/m.dz)[I]
    d_div = np.abs(div).max()/np.abs(rhs[I]*m.dt).max()
    # hand stencil: periodic in x, y; z periodic or Neumann (zero gradient across the lids)
    gz = (rz(p, -1) - p)/m.dz
    if kind == 'lid':
        gz = np.asarray(m.apply_vertical_flux_bc(gz))
    lap = ((rx(p, -1) - 2*p + rx(p, 1))/m.dx**2 + (ry(p, -1) - 2*p + ry(p, 1))/m.dy**2 + (gz - rz(gz, 1))/m.dz/m.dsqr)
    d_poisson = np.abs(lap[I] - rhs[I]).max()/np.abs(rhs[I]).max()
    return d_div, d_poisson


def uniform_tracer(kind):
    """(scheme, max |tendency| / (max |w| / dz)) for a uniform b"""
    m = small_model(kind, forcing=False); step(m, 2); h = m.halo; I = (slice(h, -h),)*3
    m.b = m.apply_bc(OM.np.full_like(m.b, 1.0))
    scale = np.abs(np.asarray(m.w)[I]).max()/m.dz
    return [(name, np.abs(np.asarray(f())[I]).max()/scale) for name, f in
            (('2nd', m.buoyancy_advection_2nd), ('superbee', m.buoyancy_advection_superbee), ('upwind3', m.buoyancy_advection_upwind3))]


def monotonicity(scheme, nsteps=20):
    m = small_model('lid', forcing=False); h = m.halo; I = (slice(h, -h),)*3
    m.u = m.apply_bc(OM.np.full_like(m.u, 0.5*m.dx/m.dt)); m.v = 0.0*m.v; m.w = 0.0*m.w       # CFL 0.5
    i = np.arange(m.Nx) - h
    b = np.where((i >= 4) & (i < 9), 1.0, 0.0)[:, None, None]*np.ones_like(np.asarray(m.b))
    m.b = m.apply_bc(OM.np.asarray(b)); m.Ro = 1.0
    f = {'superbee': m.buoyancy_advection_superbee, '2nd': m.buoyancy_advection_2nd}[scheme]
    for n in range(nsteps):
        m.b = m.apply_bc(m.b + m.dt*f())
    bi = np.asarray(m.b)[I]
    return max(0.0, -bi.min(), bi.max() - 1.0)


def smagorinsky():
    m = small_model('lid', les='smag', forcing=False); h = m.halo
    S = 0.01; z = (np.arange(m.Nz) - h + 0.5)*DZ
    m.u = m.apply_bc(OM.np.asarray(S*z[None, None, :]*np.ones_like(np.asarray(m.u)))); m.v = 0.0*m.v; m.w = 0.0*m.w
    m.les_smagorinsky(m.les_strain(m.u, m.v, m.w))
    ref = (m.smag_coeff*(DX*DY*DZ)**(1./3.))**2*S
    v = np.asarray(m.visc_00)[h:-h, h:-h, h+1:-h-1]                 # interior levels: the lid levels see the zeroed shear
    return np.abs(v - ref).max()/ref


def mirror_tests():
    """(name, asymmetry) for each operator of the rigid-lid model"""
    out = []
    m = small_model('lid', les='smag', forcing=False); step(m, 3)
    m2 = small_model('lid', les='smag', forcing=False)
    m2.u, m2.v = mirror_T(m.u), mirror_T(m.v)
    m2.b, m2.w = -mirror_T(m.b), -mirror_W(m.w)
    h = m.halo; I = (slice(h, -h),)*3
    IW = (slice(h, -h), slice(h, -h), slice(h, -h-1))             # w-grid: faces 1..nz-1, the lid face nz is not interior
    r = lambda a, b: float(np.max(np.abs(a - b)[I]))/max(float(np.max(np.abs(a)[I])), 1e-300)
    rW = lambda a, b: float(np.max(np.abs(a - b)[IW]))/max(float(np.max(np.abs(a)[IW])), 1e-300)

    def op(name, f, grid, sign):
        a = np.asarray(f(m)); b = np.asarray(f(m2))
        out.append((name, r(a, sign*mirror_T(b)) if grid == 'T' else rW(a, sign*mirror_W(b))))
    op('momentum_advection_u', lambda m: m.momentum_advection_u(), 'T', +1)
    op('momentum_advection_v', lambda m: m.momentum_advection_v(), 'T', +1)
    op('momentum_advection_w', lambda m: m.momentum_advection_w(), 'W', -1)
    op('buoyancy_advection_2nd', lambda m: m.buoyancy_advection_2nd(), 'T', -1)
    op('buoyancy_advection_superbee', lambda m: m.buoyancy_advection_superbee(), 'T', -1)
    op('buoyancy_advection_upwind3', lambda m: m.buoyancy_advection_upwind3(), 'T', -1)
    op('harmonic(u)', lambda m: m.harmonic(m.u, 1e-3, 1e-3), 'T', +1)
    op('harmonic(b)', lambda m: m.harmonic(m.b, 1e-3, 1e-3), 'T', -1)
    op('biharmonic(u)', lambda m: m.biharmonic(m.u, 1e-3, 1e-3), 'T', +1)
    # the model applies the friction operators to w as well: on_wgrid when the tree has it
    kw = dict(on_wgrid=True) if 'on_wgrid' in m.harmonic.__code__.co_varnames else {}
    op('harmonic(w)' + (' on_wgrid' if kw else ''), lambda m: m.harmonic(m.w, 1e-3, 1e-3, **kw), 'W', -1)
    op('biharmonic(w)' + (' on_wgrid' if kw else ''), lambda m: m.biharmonic(m.w, 1e-3, 1e-3, **kw), 'W', -1)
    # pressure solver: a mirrored divergent perturbation, then the projection
    pert = 0.01*np.cos(2*np.pi*(np.arange(m.Nx) - h + 0.5)/m.nx)[:, None, None]*((np.arange(m.Nz) - h + 0.5)/m.nz)[None, None, :]*np.ones_like(np.asarray(m.u))
    m.u = m.apply_bc(m.u + OM.np.asarray(pert)); m2.u = m2.apply_bc(m2.u + OM.np.asarray(mirror_T(pert)))
    m.pressure_solver(); m2.pressure_solver()
    pm = np.asarray(m.p) - np.asarray(m.p)[I].mean(); pm2 = np.asarray(m2.p) - np.asarray(m2.p)[I].mean()
    out.append(('pressure solver (u, w, p)', max(r(np.asarray(m.u), mirror_T(m2.u)), rW(np.asarray(m.w), -mirror_W(m2.w)), r(pm, mirror_T(pm2)))))
    # Smagorinsky: viscosities, then the mixing increments
    for mm in (m, m2):
        mm.les_smagorinsky(mm.les_strain(mm.u, mm.v, mm.w))
        for f in ('visc_00', 'visc_12', 'visc_13', 'visc_23', 'kappah', 'kappav'):
            setattr(mm, f, mm.apply_bc(getattr(mm, f)))
    out.append(('smagorinsky viscosities', max(r(np.asarray(m.visc_00), mirror_T(m2.visc_00)), r(np.asarray(m.visc_12), mirror_T(m2.visc_12)),
                                               rW(np.asarray(m.visc_13), mirror_W(m2.visc_13)), rW(np.asarray(m.visc_23), mirror_W(m2.visc_23)))))
    u0, w0, u20, w20, b0, b20 = (np.asarray(x).copy() for x in (m.u, m.w, m2.u, m2.w, m.b, m2.b))
    m.les_mix_momentum(m.u, m.v, m.w); m2.les_mix_momentum(m2.u, m2.v, m2.w)
    # the increments are 1e-3 of the fields, so u + du - u carries 1e-13 relative round-off: marked
    out.append(('les_mix_momentum (du, dw), increment', max(r(np.asarray(m.u) - u0, mirror_T(np.asarray(m2.u) - u20)), rW(np.asarray(m.w) - w0, -mirror_W(np.asarray(m2.w) - w20)))))
    m.les_mix_buoyancy(); m2.les_mix_buoyancy()
    out.append(('les_mix_buoyancy (db), increment', r(np.asarray(m.b) - b0, -mirror_T(np.asarray(m2.b) - b20))))
    return out


def run():
    ok = True
    for kind in ('lid', 'periodic'):
        d_div, d_poisson = pressure_solver_test(kind)
        ok &= line('5 pressure solver: projected velocity divergence free, %s' % kind, d_div, 1e-13)
        ok &= line('5 pressure solver: Poisson residual (hand stencil), %s' % kind, d_poisson, 1e-11)
    m = small_model('lid', forcing=False); h = m.halo
    f = np.asarray(m.apply_vertical_flux_bc(OM.np.ones_like(m.u)))
    conv = (f[:, :, 0].max() == 0) and (f[:, :, -2].max() == 0) and (f[:, :, 1:-2].min() == 1) and (f[:, :, -1].min() == 1)
    ok &= line('5 lid convention: T-grid flux zeroed at faces 0 and nz only', 0.0 if conv else 1.0, 0.0)
    for kind in ('lid', 'periodic'):
        for name, d in uniform_tracer(kind):
            ok &= line('5 uniform tracer: advection tendency vanishes, %s, %s' % (name, kind), d, 1e-13)
    ok &= line('5 superbee: top hat stays in [0, 1] (1D, CFL 0.5)', monotonicity('superbee'), 1e-15)
    ok &= line('5 control: 2nd order must overshoot (> 1e-3)', 1.0 if monotonicity('2nd') <= 1e-3 else 0.0, 0.0)
    ok &= line('5 smagorinsky: pure shear, visc_00 = (C delta)^2 |S|', smagorinsky(), 1e-12)
    for name, d in mirror_tests():
        ok &= line('5 lid mirror symmetry: %s' % name, d, 1e-11 if 'increment' in name else 1e-13)
    return ok


if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
