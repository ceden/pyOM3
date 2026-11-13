"""
sw_simple against the original 2-D shallow-water code (test 24b of
README_test_suite.md): the references of sw_reference.npz (written by
sw_reference.py from the original code) recomputed with sw_simple and the
balance classes warn_sw_simple, kernel_sw_simple, optimal_sw_simple.

The reference case is nondimensional (31 x 31 on 2 pi, f0 = 1, c^2 = 1,
Ro = 0.1, dt = 0.01). sw_simple runs it dimensional, with f0 = 1e-4 1/s,
L = 50 km (the domain 2 pi L), g = 0.02 m/s^2: U = Ro f0 L = 0.5 m/s, the
scale of h N = f0 U L/g = 125 m, H0 = Bu (f0 L)^2/g = 1250 m, dt = 100 s,
Ahbi = 1e-5 f0 L^4. The results are scaled back and compared, the
deviation is the maximum difference relative to the maximum of the
reference array (positions: relative to the domain). The Warn orders use
eps = Ro (warn_sw_simple.init_balance), as the original code.

    python3 sw_check.py            all cases
    python3 sw_check.py warn ob    single groups (model, warn, kernel, ob, particles)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
import io, contextlib, time
import numpy as np

REF = os.path.join(HERE, 'sw_reference.npz')
N, RO, BU, DT_ND = 31, 0.1, 1., 0.01
F0, L, G = 1e-4, 5e4, 0.02
U = RO*F0*L
NH = F0*U*L/G
H0 = BU*(F0*L)**2/G
DXN = 2*np.pi/N                       # the grid spacing of the reference
TOL = 1e-12
TOL_HIGH = 1e-10                      # Warn orders 3 and 4: the time derivatives
                                      # of the orders 2 and 3 (one step each) take
                                      # the round-off up by 1/(f0 dt) per order


def params(**kw):
    p = dict(nx=N, ny=N, dx=2*np.pi*L/N, dy=2*np.pi*L/N, dt=DT_ND/F0, f0=F0, g=G, H0=H0)
    p.update(kw)
    return p


T = lambda a: np.asarray(a).T         # (ny, nx) of sw_simple <-> (Nx, Ny) of the reference


def dev(R, tag, X):
    """the deviation of the state X = (u, v, h) of sw_simple from the reference tag"""
    return max(float(np.max(np.abs(T(x)/s - R['%s/%s' % (tag, c)]))/np.max(np.abs(R['%s/%s' % (tag, c)])))
               for x, s, c in zip(X, (U, U, NH), 'uvh'))


def state(M, R, tag='state'):
    M.set_initial_conditions(T(R[tag + '/u'])*U, T(R[tag + '/v'])*U, T(R[tag + '/h'])*NH)


def case_model(R):
    from pyOM3.sw_simple.model import model
    out = []
    for sch, key in (('AB2', 'AB'), ('AB3', 'AB3'), ('AB4', 'AB4'), ('RK3', 'RK3')):
        M = model(params(time_scheme=sch)); state(M, R, 'init')
        M.loop(50)
        out.append(('model step50 %s' % sch, dev(R, 'step50_' + key, (M.u, M.v, M.h)), TOL))
    M = model(params()); state(M, R, 'init'); M.loop(200)
    out.append(('model state (200 AB3 steps)', dev(R, 'state', (M.u, M.v, M.h)), TOL))
    return out


def case_warn(R):
    from pyOM3.diagnostics.balance.warn import warn_sw_simple
    out = []
    for tag, A in (('warn', 0.), ('warn_damped', 1e-5*F0*L**4)):
        M = warn_sw_simple(params(Ahbi=A, Khbi=A)); state(M, R)
        M.init_balance(eps=RO, verbose=False)
        X = M.decompose(4)
        for n in range(5):
            S = [sum(x[i] for x in X[:n + 1]) for i in range(3)]
            out.append(('%s order %d' % (tag, n), dev(R, '%s%d' % (tag, n), S), TOL if n < 3 else TOL_HIGH))
    return out


def case_kernel(R):
    from pyOM3.diagnostics.balance.kernel import kernel_sw_simple
    M = kernel_sw_simple(params()); state(M, R)
    S = (M.u, M.v, M.h)
    M.init_kernel(None, verbose=False)
    out = [('kernel zero order exact', dev(R, 'kernel0_exact', M.kernel_zero(S)), TOL)]
    M.init_kernel(3*L, verbose=False)
    out.append(('kernel zero order R = 3 L', dev(R, 'kernel0_R3', M.kernel_zero(S)), TOL))
    # the first order alone, against the reference's first order minus its zero order
    X0 = M.zero_order(S)
    X1 = M.kernel_first(X0)
    d = max(float(np.max(np.abs(T(x)/s - (R['kernel1_R3/' + c] - R['warn0/' + c])))
                  /np.max(np.abs(R['kernel1_R3/' + c] - R['warn0/' + c]))) for x, s, c in zip(X1, (U, U, NH), 'uvh'))
    out.append(('kernel first order R = 3 L', d, TOL))
    return out


def case_ob(R):
    from pyOM3.diagnostics.balance.optimal import optimal_sw_simple
    out = []
    for tag, kw in (('fft_cos', dict(OB_projection='fft', OB_ramp='cos', OB_max_itts=3)),
                    ('fft_exp', dict(OB_projection='fft', OB_ramp='exp', OB_max_itts=3)),
                    ('average', dict(OB_projection='average', OB_ramp='cos', OB_max_itts=2, OB_average=2*np.pi/F0))):
        M = optimal_sw_simple(params()); state(M, R)
        M.balance_it(1./F0, OB_tol=1e-14, verbose=False, **kw)
        out.append(('optimal balance %s' % tag, dev(R, 'ob_' + tag, (M.u, M.v, M.h)), TOL))
    return out


def case_particles(R):
    from pyOM3.sw_simple.particles import particles
    out = []
    xc = (np.arange(12) + 0.3)*2*np.pi/12
    Yc, Xc = np.meshgrid(xc + 0.11, xc)
    wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
    for sch in ('Euler', 'RK3', 'AB2', 'AB3', 'AB4'):
        P = np.zeros((2, 144, 11))
        class rec(particles):
            def diagnose(self, n, tau):
                if n % 10 == 0: P[:, :, n//10] = self.pxy
                super().diagnose(n, tau)
        M = rec(params()); state(M, R)
        # the reference's x is ours + one grid spacing (its domain starts after the halo)
        M.init_particles(np.array([(Xc.ravel() - DXN)*L, (Yc.ravel() - DXN)*L]), scheme=sch)
        M.loop(100)
        P[:, :, 10] = M.pxy
        d = max(float(np.max(np.abs(wrap(P[k].T/L + DXN - R['particles_%s/%s' % (sch, c)]))))/(2*np.pi)
                for k, c in ((0, 'x'), (1, 'y')))
        uvh = M.puvh[:, :, M.p_lev[0]]
        # the reference's velocities are Ro times the nondimensional ones
        d = max([d] + [float(np.max(np.abs(uvh[k]/s - R['particles_%s/%s' % (sch, c)]))/np.max(np.abs(R['particles_%s/%s' % (sch, c)])))
                       for k, s, c in ((0, U/RO, 'u'), (1, U/RO, 'v'), (2, NH, 'h'))])
        out.append(('particles %s (positions, u v h)' % sch, d, TOL))
    return out


CASES = dict(model=case_model, warn=case_warn, kernel=case_kernel, ob=case_ob, particles=case_particles)


def run(groups=None):
    R = np.load(REF)
    ok = True
    for g in (groups or list(CASES)):
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            res = CASES[g](R)
        for name, d, tol in res:
            ok &= d <= tol
            print('24b sw_simple %-36s %9.2e  (tol %7.1e)  %s' % (name, d, tol, 'PASS' if d <= tol else 'FAIL'))
        print('   [%s %.0f s]' % (g, time.time() - t))
    return ok


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv[1:] or None) else 1)
