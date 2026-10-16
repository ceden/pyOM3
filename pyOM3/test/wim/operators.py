"""Test 5: operator unit tests, no model run or a single tendency.

  backends     fortran propagation kernels = numpy form, a full forced run,
               bit for bit (skipped when wim_simple/Propagation is not built);
               fortran DIA = numpy DIA (skipped likewise). Measured 1e-8, not
               round-off: Snonlin.f90 has PI = 3.1415927 without d0, a
               single-precision literal (3.14159274101), the numpy and jax
               ports carry the double 3.1415927; with float32(PI) in the port
               the two agree to 1.5e-15
  tail         after a step every diagnostic bin (freq_mask False) and the
               frequency ghost hold E(f_c) (f_c/f)^5 exactly (imphftail)
  Doppler      a depth-independent current is projected exactly, U(f) = u_0,
               with closed_bottom, on the variable dzt
  sphere       a state uniform in (lambda, phi, theta) is a steady state of
               the continuous equations: the metric divergence of the cos phi
               flux cancels the great-circle theta flux. Discretely the
               residual is second order: measured at two resolutions, the
               ratio must be near 4 (sphere_test.py of the slow part follows
               a packet along a great circle; this is the cheap guard of the
               same terms)
  land         the tendency of a land cell is zero and no flux crosses a
               closed coast (open_coasts off): action of every wet cell
               column changes only by fluxes between wet cells
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, state, line, NZ, ocean_column
from reduction import fortran_built
import pyOM3 as OM
R_EARTH = 6370.0e3


def _uniform_residual(dy, nd):
    import io, contextlib
    from pyOM3.wim_realistic.model import model
    par = dict(nx=8, ny=int(80/dy) + 1, nf=2, nd=nd, f0=0.05, df0=0.1, dt=600.0, dxt=5.0, dyt=dy,
               coord_degree=True, y_origin=-40.0 - dy/2, periodic_in_x=True, periodic_in_y=False,
               snonlin_backend='numpy', propagation_backend='numpy', floor_action=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m = model(par, halo_points=1); m.setup()
    h = m.halo
    m.A = OM.modify_array(m.A, (slice(h, -h),)*4, 1.0); m.A = m.apply_bc(m.A)
    m.xdot, m.ydot, m.fdot, m.ddot = m.mk_advection(m.xdot, m.ydot, m.fdot, m.ddot)
    m.dA = m.add_propagation(m.A, m.dA, m.xdot, m.ydot, m.fdot, m.ddot, 0)
    cg = float(m.grav/2/m.omega_t[h])
    return float(np.max(np.abs(m.dA[0][h:-h, h, 5:-5, h:-h])))/(cg/R_EARTH)


def run():
    ok = True
    # backends
    if fortran_built():
        a = state(step(small_model(backend='numpy'), 8)); b = state(step(small_model(backend='fortran'), 8))
        ok &= line('5 fortran propagation = numpy, sphere, 8 steps', max(float(np.max(np.abs(a[k] - b[k]))) for k in a), 0.0)
    else:
        print('%-56s skipped (wim_simple/Propagation not built)' % '5 fortran propagation = numpy')
    try:
        from pyOM3.wim_simple.Snonlin import Snonlin      # noqa
        ma, mb = small_model(snl='numpy'), small_model(snl='fortran')
        sa, da = ma.Snonlin(ma.E); sb, db = mb.Snonlin(mb.E)
        d = max(float(np.max(np.abs(sa - sb))/np.max(np.abs(sa))), float(np.max(np.abs(da - db))/np.max(np.abs(da))))
        ok &= line('5 fortran DIA = numpy DIA (S_nl and dS/dE)', d, 1e-7)   # 1e-8: the single-precision PI literal of Snonlin.f90, see README
    except ImportError:
        print('%-56s skipped (wim_simple/Snonlin not built)' % '5 fortran DIA = numpy DIA')
    # imphftail
    m = step(small_model(), 3); h = m.halo
    E = np.asarray(m.E)[h:-h]; mask = np.asarray(m.freq_mask); ft = np.asarray(m.ft)
    m0 = (m.Nf - 1) - np.argmax(mask[::-1], axis=0)
    Ec = np.take_along_axis(E, m0[None, None], axis=1)
    ref = np.where(mask[None], E, Ec*(ft[m0][None, None]/ft[None, :, None, None])**5)
    wet = np.asarray(m.maskTp)[None, None]*np.ones_like(E)
    ndiag = int(np.sum(~mask[h:-h, h:-h, h:-h]))
    ok &= line('5 f^-5 tail in the %d diagnostic bins and the ghost' % ndiag,
               float(np.max(np.abs(E - ref)*wet)/np.max(E)), 1e-15)
    ok &= ndiag > 0
    # Doppler projection of a barotropic current
    dz = ocean_column('realistic'); one = np.ones((NZ, m.ny_loc, m.nx_loc))
    U, V = m.calc_doppler_vel(dz, NZ, 0.3*one, -0.2*one)
    d = max(float(np.max(np.abs(np.asarray(U)[h:-h, h:-h, h:-h] - 0.3))), float(np.max(np.abs(np.asarray(V)[h:-h, h:-h, h:-h] + 0.2))))
    ok &= line('5 Doppler velocity of a barotropic current = u_0', d, 1e-15)
    # uniform state on the sphere
    r1, r2 = _uniform_residual(2.0, 24), _uniform_residual(1.0, 48)
    print('%-56s %10.2e  -> %8.2e' % ('5 uniform state on the sphere, residual/(c_g/R)', r1, r2))
    ok &= line('5 ... second order: |ratio - 4| / 4', abs(r1/r2 - 4.0)/4.0, 0.25)
    # land
    m = small_model(open_coasts=False, sources=False); step(m, 1)
    land = 1.0 - np.asarray(m.maskTp)
    ok &= line('5 tendency on land = 0', float(np.max(np.abs(np.asarray(m.dA)[m._tau3[1]]*land[None, None]))), 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
