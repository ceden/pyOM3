"""
Twin test of pe_realistic.damping.biharmonic_momentum against pyOM2's
biharmonic_friction (for_src/main/biharmonic_friction.f90) via the f2py
build of pyOM2 (py_src/pyOM_code.so; $PYOM2 = repo root).

Spherical grid with variable dxt(i), dyt(j), cyclic in x, reaching 85 N,
a coastline (kbot = 0 blob + island + one closed row), random u, v,
A_hbi and biha_friction_cosPower as the POP setup. One call each:
  pyOM2:  du_mix = 0; call biharmonic_friction  ->  du_mix, dv_mix
  pe:     (u_new - u)/dt from biharmonic_momentum
compared on the interior U/V points. Expected: agreement at the pyOM2
pi floor (2.8e-8 relative, pi single precision in main_module.f90)
when the metric arrays are involved, round-off otherwise.

  PYOM2=/path/to/pyOM2 PYOM3=/path/to/dir_containing_pyOM3 python3 biharm_twin.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get('PYOM3', '.'))
sys.path.insert(0, os.path.join(os.environ.get('PYOM2', '../pyOM2'), 'py_src'))
import pyOM3 as OM
from pyOM3.pe_realistic.model import model
from pyOM import pyOM

nx, ny, nz = 40, 50, 3
x0, y0 = 10.0, 30.0
rng = np.random.default_rng(7)
dxt_gl = np.full(nx, 1.0)            # deg; uniform: pe and pyOM2 build DIFFERENT u-grids for variable cyclic dxt (see report)
dyt_gl = np.linspace(0.8, 1.4, ny)                                       # deg, variable -> ~85 N
dzt_gl = np.array([300., 200., 100.])
dt = 600.0
A_hbi, cosp = 2.7e10, 1.5
print('grid: y from %.1f to %.1f N' % (y0, y0 + dyt_gl.sum()))

def make_kbot(nx, ny):
    kb = np.full((ny, nx), nz)
    kb[:, 12:15] = 0                     # closed strip (wraps cyclic domain into a channel with a wall)
    kb[20:26, 25:31] = 0                 # island
    kb[35:40, 5:9] = 1                   # shallow shelf (one wet level)
    kb[0, :] = 0                         # closed southern row
    return kb
kbot_gl = make_kbot(nx, ny)


# ---------------------------------------------------------------- pyOM2
class twin(pyOM):
    def set_parameter(self):
        M = self.fortran.main_module
        M.nx, M.ny, M.nz = nx, ny, nz
        M.dt_mom = dt; M.dt_tracer = dt
        M.coord_degree = 1; M.enable_cyclic_x = 1
        M.enable_biharmonic_friction = 1
        M.a_hbi = A_hbi; M.biha_friction_cospower = cosp
        M.enable_conserve_energy = 0
        M.enable_streamfunction = 0; M.enable_free_surface = 1
        M.enable_hydrostatic = 1
        M.eq_of_state_type = 1
    def set_grid(self):
        M = self.fortran.main_module
        M.x_origin, M.y_origin = x0, y0
        M.dxt[M.onx:-M.onx] = dxt_gl
        M.dyt[M.onx:-M.onx] = dyt_gl
        M.dzt[:] = dzt_gl
    def set_coriolis(self):
        M = self.fortran.main_module
        M.coriolis_t[:, :] = 2*M.omega*np.sin(M.yt[None, :]/180.*np.pi)
    def set_topography(self):
        M = self.fortran.main_module
        M.kbot[M.onx:-M.onx, M.onx:-M.onx] = kbot_gl.T
    def set_initial_conditions(self):
        pass
    def set_forcing(self):
        pass

P = twin()
M2 = P.fortran.main_module
onx = M2.onx
u_rand = rng.standard_normal((nz, ny, nx))
v_rand = rng.standard_normal((nz, ny, nx))
maskU2 = np.transpose(M2.masku, (2, 1, 0))[:, onx:-onx, onx:-onx]     # (k,j,i)
maskV2 = np.transpose(M2.maskv, (2, 1, 0))[:, onx:-onx, onx:-onx]
u_rand *= maskU2; v_rand *= maskV2
tau = M2.tau - 1
M2.u[onx:-onx, onx:-onx, :, tau] = np.transpose(u_rand, (2, 1, 0))
M2.v[onx:-onx, onx:-onx, :, tau] = np.transpose(v_rand, (2, 1, 0))
bnd = (M2.is_pe - onx, M2.ie_pe + onx, M2.js_pe - onx, M2.je_pe + onx)
for a in (M2.u[:, :, :, tau], M2.v[:, :, :, tau]):
    P.fortran.border_exchg_xyz(*bnd, a, nz); P.fortran.setcyclic_xyz(*bnd, a, nz)
M2.du_mix[...] = 0.0; M2.dv_mix[...] = 0.0
P.fortran.biharmonic_friction()
du2 = np.transpose(M2.du_mix, (2, 1, 0))[:, onx:-onx, onx:-onx]
dv2 = np.transpose(M2.dv_mix, (2, 1, 0))[:, onx:-onx, onx:-onx]

# ---------------------------------------------------------------- pe_realistic
par = dict(nx=nx, ny=ny, nz=nz, dxt=dxt_gl, dyt=dyt_gl, dzt=dzt_gl,
           coord_degree=True, x_origin=x0, y_origin=y0, dt=dt,
           periodic_in_x=True, periodic_in_y=False, eq_of_state=0,
           explicit_free_surface=False, implicit_free_surface=True,
           superbee_advection=True, vmix_scheme='implicit',
           Ahbi=A_hbi, biha_friction_cosPower=cosp)
m = model(par)
h = m.halo
m.kbot = np.zeros((m.Ny, m.Nx)); m.kbot[h:-h, h:-h] = kbot_gl
m.kbot = m.apply_bc(m.kbot)
m.prepare_for_run()
sl = slice(h, -h)
maskU3 = np.asarray(m.maskU)[sl, sl, sl]; maskV3 = np.asarray(m.maskV)[sl, sl, sl]
print('masks identical: U %s V %s' % (np.array_equal(maskU3, maskU2), np.array_equal(maskV3, maskV2)))
for nm, a2, a3 in (('cost', M2.cost[onx:-onx], np.asarray(m.cost)[sl]), ('cosu', M2.cosu[onx:-onx], np.asarray(m.cosu)[sl]),
                   ('dxu', M2.dxu[onx:-onx], np.asarray(m.dxu)[sl]), ('dyu', M2.dyu[onx:-onx], np.asarray(m.dyu)[sl])):
    print('  %-5s max rel diff pyOM2 vs pe: %.2e' % (nm, np.max(np.abs(a2 - a3)/np.abs(a3))))

u3 = np.zeros((m.Nz, m.Ny, m.Nx)); u3[sl, sl, sl] = u_rand; u3 = m.apply_bc(u3)
v3 = np.zeros((m.Nz, m.Ny, m.Nx)); v3[sl, sl, sl] = v_rand; v3 = m.apply_bc(v3)
un, vn = m.biharmonic_momentum(u3, v3, A_hbi, m.maskU, m.maskV)
du3 = (np.asarray(un) - u3)[sl, sl, sl]/dt
dv3 = (np.asarray(vn) - v3)[sl, sl, sl]/dt

def report(name, a3, a2):
    scale = np.max(np.abs(a2))
    d = np.abs(a3 - a2)
    k, j, i = np.unravel_index(np.argmax(d), d.shape)
    print('%s: max|pyOM2| %.3e  max|pe - pyOM2| %.3e  rel %.2e  at k=%d j=%d i=%d (lat %.1f)'
          % (name, scale, d.max(), d.max()/scale, k, j, i, np.asarray(m.yt)[h + j]))
    return d.max()/scale
r1 = report('du', du3, du2)
r2 = report('dv', dv3, dv2)
# same with cosPower = 0 and a Cartesian-like check of the pi floor: rerun pe with pyOM2's single-precision pi?
# ---- diagnostics of the pattern
i_bad = np.where(np.abs(M2.dxu[onx:-onx] - np.asarray(m.dxu)[sl])/np.asarray(m.dxu)[sl] > 1e-6)[0]
print('dxu differs at i =', i_bad, ' pyOM2', M2.dxu[onx:-onx][i_bad], ' pe', np.asarray(m.dxu)[sl][i_bad])
for j in range(ny):
    d = np.abs(du3[:, j] - du2[:, j]).max(); s = np.abs(du2[:, j]).max()
    if s > 0 and d/s > 1e-6: print('  du row j=%2d lat %.1f: rel %.2e' % (j, np.asarray(m.yt)[h+j], d/s))

ok = max(r1, r2) < 1e-7
print('biharmonic operator twin:', 'PASS (pyOM2 pi floor)' if ok else 'FAIL')
sys.exit(0 if ok else 1)

