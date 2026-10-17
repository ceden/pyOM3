"""Test 1b: nh_realistic against nh_simple in the box where both apply.

nh_realistic (masked, variable grid, iterative solver) on a uniform
Cartesian grid without topography, doubly periodic under rigid lids, must
reproduce nh_simple (FFT solver, no masks) to the tolerance of the
iterative solve. nh_simple's buoyancy b is nh_realistic's temp with
eq_of_state = 0; the background stratification N0 is put into b itself
(N0 = 0 in nh_simple), so that the buoyancy equations are the same flux
form. Horizontal harmonic friction only (Ah on u, v, w), no explicit
mixing of b, 2nd-order advection, f-plane, Ro = dsqr = 1.

nh_simple solves one Poisson equation for the full pressure with the
buoyancy in the w equation; nh_realistic splits p = p_h + p_s + p_nh and
carries no buoyancy in the w equation. The two predictors differ by
-dt grad p_h, a discrete gradient, which the projection removes, so the
velocities agree exactly up to the solver tolerance (p up to its gauge
and the split).

    tolerance   1e-11 relative, u, v, w, b after 10 steps
"""
import numpy as np
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the box, its spacings and the AB ramp of the nh family (test/nh/config.py)
_spec = importlib.util.spec_from_file_location('nh_config', os.path.join(HERE, '..', 'nh', 'config.py'))
nh_config = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(nh_config)
line, NX, NY, NZ, DX, DY, DZ, DT, F0, ab_weights = (nh_config.line, nh_config.NX, nh_config.NY, nh_config.NZ,
                                                   nh_config.DX, nh_config.DY, nh_config.DZ, nh_config.DT,
                                                   nh_config.F0, nh_config.ab_weights)
import pyOM3 as OM

STEPS = 10


def initial_fields():
    """(x,y,z) interior fields of the nh config (b anomaly on a background
    N0^2 z, jet, perturbation), as plain numpy, fortran order"""
    N0 = 5e-3
    Lx, Ly, Lz = NX*DX, NY*DY, NZ*DZ
    x = (np.arange(NX) + 0.5)*DX; y = (np.arange(NY) + 0.5)*DY; z = (np.arange(NZ) + 0.5)*DZ
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    gauss = 2e-4*np.exp(-((X - 0.5*Lx)/(0.2*Lx))**2 - ((Y - 0.5*Ly)/(0.2*Ly))**2 - ((Z - 0.6*Lz)/(0.25*Lz))**2)
    b = gauss - gauss.mean() + N0**2*Z
    u = 0.05*np.sin(2*np.pi*Y/Ly)*(0.5 + Z/Lz) + 0.01*np.cos(2*np.pi*X/Lx)*np.sin(np.pi*Z/Lz)
    v = 0.01*np.sin(2*np.pi*X/Lx)*np.cos(2*np.pi*Y/Ly)
    return u, v, b


def model_simple(quiet=True):
    from pyOM3.nh_simple.model import model
    par = dict(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, dt=DT, Ro=1.0, dsqr=1.0,
               N0=0.0, f0=F0, periodic_in_z=False, Ah=2e-3)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        m = model(par)
    m.setup()
    u, v, b = initial_fields()
    h = m.halo
    pad = lambda a: np.pad(a, h)
    m.b = m.apply_bc(OM.np.asarray(pad(b)))
    m.u = m.apply_bc(OM.np.asarray(pad(u)))
    m.v = m.apply_bc(OM.np.asarray(pad(v)))
    m.w = m.apply_vertical_flux_bc(m.apply_bc(OM.np.zeros_like(m.u)))
    m.pressure_solver()
    return m


def model_realistic(quiet=True, **extra):
    from pyOM3.nh_realistic.model import model
    par = dict(nx=NX, ny=NY, nz=NZ, dt=DT, dxt=DX, dyt=DY, dzt=DZ, coord_degree=False,
               periodic_in_x=True, periodic_in_y=True, eq_of_state=0,
               explicit_free_surface=False, implicit_free_surface=False,
               vmix_scheme='explicit', enable_conv_adjust=False, enable_tke=False,
               enable_neutral_diffusion=False, Ah=2e-3,
               epsilon=1e-16, max_iterations=5000, epsilon_nh=1e-16, max_iterations_nh=5000)
    par.update(extra)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        m = model(par)
    h = m.halo
    m.coriolis_t = OM.np.full((m.Ny, m.Nx), F0, OM.prec)
    m.kbot = OM.np.full((m.Ny, m.Nx), float(h), OM.prec)          # flat bottom, all levels wet
    m.prepare_for_run()
    u, v, b = initial_fields()
    tr = lambda a: np.pad(a.transpose(2, 1, 0), h)                # (x,y,z) -> (z,y,x) with halo
    m.temp = m.apply_bc(OM.np.asarray(tr(b)))*m.maskT
    m.salt = OM.np.zeros_like(m.temp)
    m.u = m.apply_bc(OM.np.asarray(tr(u)))*m.maskU
    m.v = m.apply_bc(OM.np.asarray(tr(v)))*m.maskV
    m.w = OM.np.zeros_like(m.u)
    from pyOM3.density import get_rho
    m.rho = get_rho(m.salt, m.temp, m.p0, m.eq_of_state)
    m.nh_projection()                                            # project the initial velocity
    return m


def interior_simple(m):
    h = m.halo; s = slice(h, -h)
    return {k: np.asarray(getattr(m, k))[s, s, s] for k in ('u', 'v', 'w', 'b')}


def interior_realistic(m):
    h = m.halo; s = slice(h, -h)
    f = {k: np.asarray(getattr(m, k))[s, s, s].transpose(2, 1, 0) for k in ('u', 'v', 'w')}
    f['b'] = np.asarray(m.temp)[s, s, s].transpose(2, 1, 0)
    return f


def run():
    ok = True
    ms, mr = model_simple(), model_realistic()
    d0 = {k: np.abs(interior_simple(ms)[k] - interior_realistic(mr)[k]).max()/max(1e-30, np.abs(interior_simple(ms)[k]).max())
          for k in ('u', 'v', 'w', 'b')}
    for k in ('u', 'v', 'w', 'b'):
        ok &= line('nh_realistic = nh_simple, projected initial %s' % k, d0[k], 1e-11)
    tau, taum1, taum2 = 2, 1, 0
    for n in range(STEPS):
        A, B, C = ab_weights(n)
        ms.time_step(A, B, C, tau, taum1, taum2)
        mr.time_step_implicit_pressure(A, B, C, tau, taum1, taum2)
        tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
    fs, fr = interior_simple(ms), interior_realistic(mr)
    for k in ('u', 'v', 'w', 'b'):
        d = np.abs(fs[k] - fr[k]).max()/max(1e-30, np.abs(fs[k]).max())
        ok &= line('nh_realistic = nh_simple after %d steps, %s' % (STEPS, k), d, 1e-11)
    print('      itts_nh of the last step: %d (error %.1e)' % (mr.congr_itts_nh, mr.estimated_error_nh))
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
