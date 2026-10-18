"""Test 17: WENO5 tracer advection (pe_simple/thermodynamics.py
tracer_flux_weno, pyOM3/advection_weno.py) on the small configuration with
its shelf and seamount, pe_simple, halo_points = 3 (the scheme refuses
less). Serial part:

  17a  a uniform tracer has no advective tendency (with the surface flux
       option, so that the top cell is included): the face value of a
       constant is the constant whatever the order chosen at a face, and
       the velocity is divergence free
  17b  the scheme is conservative: the advective tendency sums to zero over
       the wet cells (ft = 0 at the surface, masked faces at the walls and
       the topography)
  17c  the result does not depend on the halo width: halo 4 against halo 3
       after N steps, bit for bit
  17d  the scheme runs with AB3 at the suite's time step and stays finite,
       with the LES-free closure of the suite (TKE)
  17e  accuracy: a Gaussian advected once around a periodic channel
       (pe_simple, flat bottom, prescribed u, no dynamics, the AB2 weights
       of the runs) has a smaller error with WENO5 than with superbee,
       and no over- or undershoot beyond 1e-3 of the amplitude

Under MPI (mpirun -np 4 python3 weno.py 2 2) the WENO run after N steps is
compared with one PE, bit for bit for a y split (no global sums in the
tracer advection; the CG of the implicit free surface reduces in rank
order, 1e-12 for an x split), the halo-slot fluxes being taken from the
owning block."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, gathered_state, line, reldiff, NX, NY, NZ, OM


def weno_model(**kw):
    m = small_model(kind='simple', halo=kw.pop('halo', 3), **kw)
    m.weno_advection = True
    m.weno_ref = 10.0                  # the temperature range of the box
    return m


def run_serial(N=6):
    ok = True
    # 17a uniform tracer
    m = weno_model(tracer_flux=True)
    h = m.halo; sl = slice(h, -h)
    tau = 0
    m.dtemp = m.tracer_advection(5.0*m.maskT, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)
    d = float(np.max(np.abs(np.asarray(m.dtemp[tau])[sl, sl, sl])))*m.dt
    ok &= line('17a uniform tracer, |dT| per step after one step', d, 1e-13)
    step(m, 3)                                  # a nontrivial velocity
    m.dtemp = m.tracer_advection(5.0*m.maskT, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)
    d = float(np.max(np.abs(np.asarray(m.dtemp[tau])[sl, sl, sl])))*m.dt
    ok &= line('17a uniform tracer, |dT| per step after 3 steps', d, 1e-13)
    # 17b conservation (ft = 0 at the surface)
    m = weno_model()
    step(m, 3)
    m.dtemp = m.tracer_advection(m.temp, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)
    dT = np.asarray(m.dtemp[tau])[sl, sl, sl]
    d = abs(float(np.sum(dT)))/float(np.sum(np.abs(dT)))
    ok &= line('17b sum of the advective tendency / sum |tendency|', d, 1e-13)
    # 17c halo independence
    ref = state(step(weno_model(), N))
    d = max(reldiff(state(step(weno_model(halo=4), N)), ref).values())
    ok &= line('17c halo 4 = halo 3, WENO5, %d steps' % N, d, 0.0)
    # 17d AB3, finite
    st = state(step(weno_model(), 20, 'ab3'))
    d = 0.0 if all(np.all(np.isfinite(v)) for v in st.values()) else np.inf
    ok &= line('17d 20 AB3 steps finite (0 = yes)', d, 0.0)
    # 17e accuracy in a periodic channel
    ok &= _channel()
    return ok


def _channel(nx=64, cfl=0.2):
    """Gaussian (sigma 6 cells) advected once around a periodic channel by
    a prescribed uniform u; superbee against WENO5, both with the AB2
    weights of the runs (1.6, -0.6): superbee's limiter is built for a
    forward-in-time step and blows up under AB3 at this Courant number,
    WENO5 under AB3 is test 17d."""
    from pyOM3.pe_simple.model import model
    import io, contextlib
    out = {}
    for scheme in ('superbee', 'weno'):
        par = dict(nx=nx, ny=4, nz=3, dx=1e3, dy=1e3, dz=10., dt=cfl*1e3/0.1,
                   periodic_in_x=True, periodic_in_y=True, eq_of_state=1,
                   implicit_free_surface=True, explicit_free_surface=False, superbee_advection=True,
                   epsilon=1e-12, max_iterations=100)
        with contextlib.redirect_stdout(io.StringIO()):
            m = model(par, halo_points=3)
        m.weno_advection = (scheme == 'weno'); m.weno_ref = 1.0
        m.kbot = OM.np.ones((m.Ny, m.Nx)); m.make_topo_masks()
        h = m.halo; x = (np.arange(m.Nx) - h + 0.5)*1e3
        T0 = np.exp(-((x - nx*1e3/2)/(6e3))**2)[None, None, :]*np.ones((m.Nz, m.Ny, 1))
        m.temp = m.apply_bc(OM.np.asarray(T0))*m.maskT
        m.salt = 35.0*m.maskT
        m.u = 0.1*m.maskU; m.v = OM.np.zeros_like(m.u); m.w = OM.np.zeros_like(m.u)
        m.dtemp = OM.np.zeros_like(m.dtemp)
        nsteps = int(round(nx*1e3/(0.1*m.dt)))
        tau, taum1, taum2 = 2, 1, 0
        for n in range(nsteps):
            A, B, C = (1.6, -0.6, 0.0)
            m.dtemp = m.tracer_advection(m.temp, m.dtemp, m.u, m.v, m.w, m.maskU, m.maskV, m.maskW, m.maskT, tau)
            m.temp = m.apply_bc(m.temp + m.dt*(A*m.dtemp[tau] + B*m.dtemp[taum1] + C*m.dtemp[taum2])*m.maskT)
            tau, taum1, taum2 = (tau+1) % 3, (taum1+1) % 3, (taum2+1) % 3
        T = np.asarray(m.temp)[h, h, h:-h]; T0i = T0[0, 0, h:-h]
        out[scheme] = (float(np.mean(np.abs(T - T0i))), float(T.max() - 1.0), float(-T.min()))
    ok = line('17e channel L1 error WENO5 / superbee', out['weno'][0]/out['superbee'][0], 0.5)
    ok &= line('17e channel WENO5 overshoot (max - 1)', out['weno'][1], 1e-3)
    ok &= line('17e channel WENO5 undershoot (-min)', out['weno'][2], 1e-3)
    return ok


def run_mpi(npx, npy, N=6):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    st = gathered_state(step(weno_model(comm=comm, n_pes_x=npx, n_pes_y=npy), N))
    ref = gathered_state(step(weno_model(comm=MPI.COMM_SELF), N)) if comm.Get_rank() == 0 else None
    ok = True
    if comm.Get_rank() == 0:
        d = max(reldiff(st, ref).values())
        ok = line('17 decomposition %dx%d vs 1 PE, WENO5, %d steps' % (npx, npy, N), d, 0.0 if npx == 1 else 1e-12)
    return bool(comm.bcast(ok, root=0))


def run():
    return run_serial()


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        sys.exit(0 if run_mpi(int(sys.argv[1]), int(sys.argv[2])) else 1)
    sys.exit(0 if run() else 1)
