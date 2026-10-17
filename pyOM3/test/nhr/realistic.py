"""Tests of nh_realistic on the pe configuration (stretched spherical grid,
shelf and seamount, free surface, TKE closure), see config.py.

 1. hydrostatic limit: enable_nonhydrostatic = False against pe_realistic,
    implicit and explicit free surface, 8 steps        exact (0)
    (the time steps of nh_realistic/model.py are pe_realistic's with
    the nh lines inserted; this keeps the two in step)
 2. the operator -V div grad of solver.py is symmetric on the masked,
    stretched grid: <A p, q> = <p, A q> for random fields, and so is the
    columnwise preconditioner M                        1e-14 relative
 3. the projection removes a discrete gradient: a predictor changed by
    dt grad q (q arbitrary) gives the same (u,v,w)      1e-10 relative
    (both solves run to epsilon_nh = 1e-22)
    (the property behind the twin with nh_simple: there the buoyancy sits
    in the w equation, here in -grad p_h)
 4. divergence: after 8 steps (implicit and explicit free surface) the
    divergence of every wet cell, including the surface cells with the
    free-surface w, is below the solver tolerance epsilon_nh [1/s]
 5. gauge: the wet-cell mean of p_nh is zero            1e-12 relative
"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, small_model_pe, step, state, line, OM
AB2 = (1.6, -0.6, 0.0)


def _rel(a, b):
    return float(np.max(np.abs(a - b))/max(np.max(np.abs(b)), 1e-300))


def hydrostatic_limit():
    ok = True
    for explicit, name in ((False, 'implicit'), (True, 'explicit')):
        m = small_model(explicit=explicit); m.enable_nonhydrostatic = False
        r = small_model_pe(explicit=explicit)
        sn, sr = state(step(m, 8, AB2)), state(step(r, 8, AB2))
        d = max(_rel(sn[k], sr[k]) for k in sr)
        ok &= line('1 hydrostatic limit = pe_realistic, %s, 8 steps' % name, d, 0.0)
        ok &= line('1 ... p_nh stays zero', float(np.max(np.abs(sn['p_nh']))), 0.0)
    return ok


def operator_symmetry():
    ok = True
    m = small_model()
    rng = np.random.default_rng(1)
    shape = (m.Nz, m.Ny, m.Nx)
    p = m.apply_bc(OM.np.asarray(rng.standard_normal(shape)))*m.maskT
    q = m.apply_bc(OM.np.asarray(rng.standard_normal(shape)))*m.maskT
    Ap, Aq = m._op_nh(p, m.cf_nh), m._op_nh(q, m.cf_nh)
    s1, s2 = float(m._dot(Ap, q)), float(m._dot(p, Aq))
    ok &= line('2 operator symmetry <Ap,q> = <p,Aq>', abs(s1 - s2)/abs(s1), 1e-14)
    Mp, Mq = m._pre_nh(p, m.cf_nh, m.maskT), m._pre_nh(q, m.cf_nh, m.maskT)
    s1, s2 = float(m._dot(Mp, q)), float(m._dot(p, Mq))
    ok &= line('2 preconditioner symmetry <Mp,q> = <p,Mq>', abs(s1 - s2)/abs(s1), 1e-14)
    # A annihilates the constant on the wet cells (Neumann everywhere)
    one = m.maskT*1.0
    ok &= line('2 A 1 = 0 on the wet cells', float(np.max(np.abs(m._op_nh(one, m.cf_nh)))/np.max(np.abs(m.cf_nh[3]))), 1e-14)
    return ok


def projection_of_gradient():
    ok = True
    m = step(small_model(), 3, AB2)
    m.epsilon_nh = 1e-22                      # converge the two solves fully
    rng = np.random.default_rng(2)
    q = m.apply_bc(OM.np.asarray(1e-3*rng.standard_normal((m.Nz, m.Ny, m.Nx))))*m.maskT
    gu = (m.rollx(q, -1) - q)/(m.cost_y*m.dxu_x)*m.maskU
    gv = (m.rolly(q, -1) - q)/m.dyu_y*m.maskV
    gw = (m.rollz(q, -1) - q)/m.dzw_z*m.maskWc
    zero = OM.np.zeros_like(q)
    args = (m.maskU, m.maskV, m.maskWc, m.maskT, m.cf_nh, m.n_wet)
    u1, v1, w1, p1, e1, n1 = m.solve_for_nh_pressure(m.u, m.v, m.w, zero, *args)
    u2, v2, w2, p2, e2, n2 = m.solve_for_nh_pressure(m.apply_bc(m.u + m.dt*gu), m.apply_bc(m.v + m.dt*gv),
                                                     m.apply_bc(m.w + m.dt*gw), zero, *args)
    for a, b, nm in ((u1, u2, 'u'), (v1, v2, 'v'), (w1, w2, 'w')):
        ok &= line('3 projection removes dt grad q, %s' % nm, _rel(np.asarray(a), np.asarray(b)), 1e-10)
    return ok


def divergence_and_gauge():
    ok = True
    for explicit, name in ((False, 'implicit'), (True, 'explicit')):
        m = step(small_model(explicit=explicit), 8, AB2)
        div = m.volume_divergence(m.u, m.v, m.w, m.maskU, m.maskV, m.maskT)/(m.area_t*m.dzt_z)
        ok &= line('4 max |div(u,v,w)| of the wet cells [1/s], %s, 8 steps' % name,
                   float(np.max(np.abs(div))), m.epsilon_nh)
        mean = float(m._dot(m.p_nh, m.maskT))/float(m.n_wet)
        ok &= line('5 wet-cell mean of p_nh = 0, %s' % name, abs(mean)/float(np.max(np.abs(m.p_nh))), 1e-12)
        print('      itts_nh of the last step: %d (error %.1e)' % (m.congr_itts_nh, m.estimated_error_nh))
    return ok


def run():
    ok = hydrostatic_limit()
    ok &= operator_symmetry()
    ok &= projection_of_gradient()
    ok &= divergence_and_gauge()
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
