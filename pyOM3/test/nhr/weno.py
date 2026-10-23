"""Test 17: WENO5 tracer advection in nh_realistic. nh_realistic takes the
tracer advection of pe_realistic unchanged (tracer_flux_weno of
pe_realistic/thermodynamics.py), so this checks that the scheme works with
the non-hydrostatic velocity: the configuration of the family (stretched
Cartesian grid, shelf and seamount, TKE closure), halo_points = 3.

  17a  a uniform tracer has no advective tendency after N steps (the
       velocity of the 3-D projection is divergence free on the T cells to
       the solver tolerance, set to epsilon_nh = 1e-16 here: what is left
       is 3e-13 of a tracer of 5 per step, the round-off divergence of the
       3-D solve), and that tendency is the centred flux's to round-off,
       i.e. WENO5 adds nothing to a uniform tracer
  17b  the advective tendency times the cell volume area_t dzt sums to zero
       over the wet cells
  17c  the result does not depend on the halo width: halo 4 against halo 3
       after N steps, bit for bit, every field including p_nh
  17d  20 AB3 steps stay finite
  17g  boundedness at a salinity front across the shelf and the seamount
       (as test 17g of the pe family): after 60 steps inside [35, 36] to
       2e-4 (measured Oct 18 2026: 6e-5; superbee 2e-7; the centred
       fallback next to land, replaced that day, 1.9e-3)
  17e  the hydrostatic limit: with the non-hydrostatic pressure switched off
       (enable_nonhydrostatic = False) nh_realistic with WENO5 is
       pe_realistic with WENO5 on the same stretched grid, to round-off

Under MPI (mpirun -np 4 python3 weno.py 2 2) the run after N steps is
compared with one PE, at the tolerances of the family's decomposition test
4 (the 3-D CG reduces in rank order; 1e-9 for the fields, 1e-5 for p_nh)."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, small_model_pe, step, state, gathered_state, line, global_index, OM
from common import reldiff


def weno(m):
    m.weno_advection = True
    m.weno_ref = 10.0                      # the temperature range of the box
    return m


def run(N=6):
    ok = True; tau = 0
    # epsilon_nh 1e-16 as in the layout tests: with the default 1e-12 the
    # T-cell divergence left by the 3-D solve, times T dt, is 5e-9 per step
    m = weno(small_model(halo=3, tracer_flux=True, eps_nh=1e-16))
    h = m.halo; sl = slice(h, -h)
    step(m, N)
    m.dtemp = m.tracer_advection(5.0*m.maskT, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)
    dW = np.asarray(m.dtemp[tau])[sl, sl, sl]*m.dt
    # the same with the centred flux: T times the divergence. On a SEPARATE
    # model with the option off, evaluated on m's fields: tracer_advection is
    # jitted with the model as a static argument, so toggling the attribute
    # on m after the first call would not reach the compiled function (jax)
    m2 = small_model(halo=3, tracer_flux=True, eps_nh=1e-16)
    d2 = np.asarray(m2.tracer_advection(5.0*m.maskT, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)[tau])[sl, sl, sl]*m.dt
    ok &= line('17a nh_realistic: uniform tracer, |dT| per step, %d steps' % N, float(np.max(np.abs(dW))), 1e-12)
    ok &= line('17a ... equal to the centred flux (T x divergence)', float(np.max(np.abs(dW - d2))), 1e-14)
    # closed surface face for the budget: again a separate model (same reason)
    m3 = weno(small_model(halo=3, tracer_flux=False, eps_nh=1e-16))
    dT = np.asarray(m3.tracer_advection(m.temp, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)[tau])[sl, sl, sl] \
         *np.asarray(m.area_t*m.dzt_z)[sl, sl, sl]
    ok &= line('17b nh_realistic: sum of volume x tendency / sum |...|',
               abs(float(np.sum(dT)))/float(np.sum(np.abs(dT))), 1e-13)
    # exact on numpy. On the jax backend XLA compiles each halo width on its
    # own and the round-off differs; the 3-D pressure solve then stops at a
    # different iteration, and with the default eps_nh that is 1e-7 in u and
    # 6e-4 in the weakly determined p_nh. So on jax the solve is run to 1e-16
    # as in jax_sharding.py and its tolerances apply (1e-9, p_nh 1e-5)
    eps = dict(eps_nh=1e-16) if OM.jax else {}
    ref = state(step(weno(small_model(halo=3, **eps)), N))
    d = reldiff(state(step(weno(small_model(halo=4, **eps)), N)), ref)
    if OM.jax:
        ok &= line('17c nh_realistic: halo 4 = halo 3, WENO5, %d steps (jax, eps_nh 1e-16)' % N, max(v for k, v in d.items() if k != 'p_nh'), 1e-9)
        ok &= line('17c ... p_nh (weakly determined, see decomposition.py)', d.get('p_nh', 0.0), 1e-5)
    else:
        ok &= line('17c nh_realistic: halo 4 = halo 3, WENO5, %d steps' % N, max(d.values()), 0.0)
    m = weno(small_model(halo=3, tracer_flux=True)); jg, ig = global_index(m)
    m.salt = m.apply_bc(OM.np.asarray(35.0 + 1.0*(ig >= 7)[None]*np.ones((m.Nz, 1, 1))))*m.maskT
    step(m, 60)
    sal = np.asarray(m.salt)[sl, sl, sl][np.asarray(m.maskT)[sl, sl, sl] > 0]
    ok &= line('17g nh_realistic: salinity front, max excursion out of [35,36]',
               float(max(35.0 - sal.min(), sal.max() - 36.0, 0.0)), 2e-4)
    st = state(step(weno(small_model(halo=3)), 20, 'ab3'))
    d = 0.0 if all(np.all(np.isfinite(v)) for v in st.values()) else np.inf
    ok &= line('17d nh_realistic: 20 AB3 steps finite (0 = yes)', d, 0.0)
    m = weno(small_model(halo=3)); m.enable_nonhydrostatic = False
    a = state(step(m, N)); a.pop('p_nh', None)
    b = state(step(weno(small_model_pe(halo=3)), N))
    ok &= line('17e nh_realistic hydrostatic = pe_realistic, WENO5, %d steps' % N,
               max(reldiff(a, b).values()), 1e-12)
    return ok


def run_mpi(npx, npy, N=6):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    st = gathered_state(step(weno(small_model(halo=3, comm=comm, n_pes_x=npx, n_pes_y=npy, eps_nh=1e-16)), N))
    ref = gathered_state(step(weno(small_model(halo=3, comm=MPI.COMM_SELF, eps_nh=1e-16)), N)) \
          if comm.Get_rank() == 0 else None
    ok = True
    if comm.Get_rank() == 0:
        pnh = reldiff(st, ref).pop('p_nh')
        st.pop('p_nh'); ref.pop('p_nh')
        d = max(reldiff(st, ref).values())
        # the tolerances of decomposition.py (test 4): the 3-D CG reduces in
        # rank order, p_nh is weakly determined (see the comment there)
        ok = line('17 decomposition %dx%d vs 1 PE, nh_realistic WENO5, %d steps' % (npx, npy, N), d, 1e-9)
        ok &= line('17 ... p_nh', pnh, 1e-5)
    return bool(comm.bcast(ok, root=0))


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        sys.exit(0 if run_mpi(int(sys.argv[1]), int(sys.argv[2])) else 1)
    sys.exit(0 if run() else 1)
