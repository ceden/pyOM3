"""Test 20: restart of the coupled ocean through coupler_io
(ocean_restart_write / ocean_restart_read, the gathered netCDF form and the
raw per-rank form): 2N steps against N steps, a restart written and read
into a NEW model, N more steps. Every field of the state must agree bit for
bit, and so must the partial sums of the online means and the KE budget's
accumulators after the restart. The coupling fields (u_S and u_S_old, the
stress, phi_oc, tau_in, hs), tke and K_diss_v are restart state, as the
restart reader documents. Cases: coupled default with 6-hourly means,
coupled with the distributed stress, forced with the relative-wind stress.

    tolerance   exact, no epsilon
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_ocean, step, state, line, reldiff, WORK, NX, NY
import coupler_io
from coupler_io import ocean_restart_write, ocean_restart_read

CASES = (('coupled, means 6h', dict(coupled=True, average='6h')),
         ('coupled, mxl 2, direct, shear', dict(coupled=True, mxl_choice=2, direct=True, shear=True)),
         ('coupled, distributed stress', dict(coupled=True, stress_work=True, ke_budget=0)),
         ('forced, relwind', dict(coupled=False, relwind=True, average='6h')))


def _restart_pair(kw, N, raw):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    fname = os.path.join(WORK, 'restart_test%s.cdf' % ('_raw' if raw else ''))
    coupler_io.RESTART_RAW = raw
    cwd = os.getcwd(); os.chdir(WORK)
    try:
        ref = step(small_ocean(comm=comm, tag='rs_a', **kw), 2*N)
        a = step(small_ocean(comm=comm, tag='rs_b', **kw), N)
        tau, taum1, taum2 = a._tau3
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ocean_restart_write(a, fname, NY, NX, a.use_stress_work, a._nstep, tau, taum1, taum2, *a._ab)
            b = small_ocean(comm=comm, tag='rs_c', **kw)
            att = ocean_restart_read(b, fname, NY, NX, b.use_stress_work, comm)
        b._nstep = int(att['n_cpl']); b._tau3 = (int(att['tau']), int(att['taum1']), int(att['taum2']))
        b.set_wave_roughness()
        step(b, N)
    finally:
        os.chdir(cwd)
    return ref, a, b


def run(N=4):
    ok = True
    for raw in (False, True):
        for name, kw in CASES:
            ref, a, b = _restart_pair(kw, N, raw)
            d = max(reldiff(state(b), state(ref)).values())
            ok &= line('20 restart %s%s, %d + %d vs %d steps' % (name, ' (raw)' if raw else '', N, N, 2*N), d, 0.0)
            if getattr(ref, '_avg', None) is not None:
                sa, sb = ref.average_state(), b.average_state()
                d = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for (_, x), (_, y) in zip(sa[0], sb[0]))
                d += abs(sa[1] - sb[1]) + abs(sa[2] - sb[2])
                ok &= line('20 restart %s%s: partial sums of the means' % (name, ' (raw)' if raw else ''), d, 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
