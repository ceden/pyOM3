"""
Profiling for the coupled run: where each component spends a coupling step,
and how much of it is waiting for the other one.

Two levels, both living HERE and never in the model package:

  level 1  SECTION TIMERS. The two coupled loops bracket their phases with
           `with prof.section('...')`, so the report splits a coupling step
           into stepping / diagnostics / gather / exchange / scatter /
           restart / barrier. The barrier and the blocking intercomm receive
           are the two places a component waits for the other, so those rows
           ARE the load imbalance.

  level 2  WAVE-MODEL INTERNALS, obtained by wrapping bound methods on the
           model INSTANCE (`prof.wrap(M, [...])`). Nothing is added to
           wim_realistic / wim_simple / pe_realistic: the wrapper replaces
           `M.method` with a timing closure on that one object, so the
           package files stay clean and an unprofiled run is byte-for-byte
           the same code path.

Nesting is handled with a stack, so each row carries BOTH an exclusive
(self) time and an inclusive time. Exclusive times sum to the total; the
inclusive column tells you what a method costs including everything it
calls (e.g. add_propagation includes its flux kernels and apply_bc).

Cost of the instrumentation itself: one perf_counter pair per section and
per wrapped call. With the default wrap list that is O(10) pairs per wave
sub-step, i.e. microseconds against a step that costs seconds. The section
timers alone (level 1) are free.
"""
import time
from collections import OrderedDict
from contextlib import contextmanager


def jax_sync():
    """Wait until every kernel queued on every device has run. JAX
    dispatches asynchronously: a timed section returns as soon as its
    kernels are QUEUED, and the time is paid in whichever later section
    first needs a result on the host (the CG's iteration count, say). The
    exclusive times of a jax run are therefore only meaningful if each
    section ends with a device sync, which this provides: one trivial
    jitted op per device, blocked on -- XLA runs one compute stream per
    device in order, so it completes after everything queued before it.
    About 0.1 ms per device and section; profile level 2 on jax pays
    ~5 ms per step for exact rows."""
    import pyOM3 as OM
    if not OM.jax:
        return None
    jax = OM.jax
    ops = []
    for d in jax.devices():
        if d.process_index != jax.process_index():
            continue
        x = jax.device_put(OM.np.zeros(()), d)
        ops.append((jax.jit(lambda a: a + 1.0), x))
    def sync():
        for f, x in ops:
            y = f(x)
            if not hasattr(y, 'block_until_ready'):
                return                      # inside a jit trace: nothing to wait for
            y.block_until_ready()
    return sync


class Profiler:
    """Named accumulators with correct nesting. One instance per component."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.excl = OrderedDict()      # self time, sums to the total
        self.incl = OrderedDict()      # including callees
        self.cnt  = OrderedDict()
        self._child = []               # per-level accumulated child time
        self.steps = 0
        self.sync = None               # device sync before each timing (jax_sync)

    # ------------------------------------------------------------------
    def _enter(self):
        if self.sync is not None:
            self.sync()
        self._child.append(0.0)

    def _exit(self, name, t0):
        if self.sync is not None:
            self.sync()
        dt = time.perf_counter() - t0
        child = self._child.pop()
        self.excl[name] = self.excl.get(name, 0.0) + (dt - child)
        self.incl[name] = self.incl.get(name, 0.0) + dt
        self.cnt[name]  = self.cnt.get(name, 0) + 1
        if self._child:
            self._child[-1] += dt

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        self._enter()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._exit(name, t0)

    # ------------------------------------------------------------------
    def wrap(self, obj, names, prefix=''):
        """Replace bound methods on ONE instance with timing closures.
        Silently skips names the object does not have, so the same list can
        be used for models with different feature sets."""
        if not self.enabled:
            return
        for nm in names:
            fn = getattr(obj, nm, None)
            if fn is None or not callable(fn):
                continue
            if getattr(fn, '_profiled', False):
                continue
            label = prefix + nm

            def make(fn=fn, label=label):
                def timed(*a, **k):
                    self._enter()
                    t0 = time.perf_counter()
                    try:
                        return fn(*a, **k)
                    finally:
                        self._exit(label, t0)
                timed._profiled = True
                return timed
            setattr(obj, nm, make())

    # ------------------------------------------------------------------
    def reset(self):
        self.excl.clear(); self.incl.clear(); self.cnt.clear()
        self._child = []
        self.steps = 0

    # ------------------------------------------------------------------
    def report(self, comm, label, wall, top=None):
        """Collective over `comm` -- every rank of the component must call.
        Rank 0 returns the formatted table, the others None.

        `wall` is the wall time of the reporting window, so the rows are
        per coupling step and the percentages are of real elapsed time,
        not of the sum of the rows (which would hide the unprofiled
        remainder)."""
        if not self.enabled:
            return None
        nst = max(self.steps, 1)
        keys = sorted(set().union(*comm.allgather(set(self.excl))))
        rows = []
        for k in keys:
            e = comm.allgather(self.excl.get(k, 0.0))
            i = comm.allgather(self.incl.get(k, 0.0))
            c = comm.allgather(self.cnt.get(k, 0))
            rows.append((k, min(e), sum(e)/len(e), max(e),
                         sum(i)/len(i), sum(c)/len(c)))
        if comm.Get_rank() != 0:
            return None
        rows.sort(key=lambda r: -r[2])
        if top:
            rows = rows[:top]
        acc = sum(r[2] for r in rows)
        out = ['', '=== %s: %d coupling steps, %.3f s/step wall ==='
               % (label, nst, wall/nst),
               '%-26s %9s %9s %9s %9s %7s %6s'
               % ('section', 'ms/step', 'min', 'max', 'incl', '% wall', 'n/step'),
               '-'*82]
        for k, mn, av, mx, ic, c in rows:
            out.append('%-26s %9.2f %9.2f %9.2f %9.2f %7.1f %6.1f'
                       % (k, 1e3*av/nst, 1e3*mn/nst, 1e3*mx/nst,
                          1e3*ic/nst, 100*av/max(wall, 1e-30), c/nst))
        out.append('%-26s %9.2f %53.1f' % ('  [sum of rows]', 1e3*acc/nst,
                                           100*acc/max(wall, 1e-30)))
        out.append('min/max are across the PEs of this component: a wide '
                   'spread is internal imbalance,')
        out.append('a large barrier/exchange row is imbalance against the '
                   'OTHER component.')
        return '\n'.join(out)


# Methods worth wrapping on a wim instance. Anything absent is skipped, so
# the same list serves wim_simple, wim_realistic and future variants.
WAVE_METHODS = (
    'mk_advection', 'add_propagation', 'energy_fix_flux',
    'tracer_flux_superbee', 'tracer_flux_upwind',
    'open_coast_flux', 'open_coast_loss', 'reflect_flux',
    'Sinput', 'Sdiss', 'Snonlin',
    'apply_bc', 'check_monotonicity', 'update_forcing', 'diagnose',
    'calc_stokes', 'calc_stress', 'calc_stress_work', 'calc_phioc',
    'calc_doppler_vel', 'calc_doppler_vel_df', 'coastal_loss',
)

OCEAN_METHODS = (
    'tracer_advection', 'vadv_tracer', 'momentum_tendency_impl', 'solve_for_pressure',
    # the split-explicit path (explicit_free_surface): the tendencies and the
    # barotropic sub cycle, the counterparts of the two rows above
    'momentum_tendency_expl', 'solve_expl_surface', 'bottom_friction',
    'vertical_friction', 'vertical_mixing', 'integrate_tke',
    'set_tke_diffusivities', 'set_forcing', 'set_forcing_direct',
    'vertical_velocity', 'update_thermodynamics_AB3', 'update_momentum_AB3',
    'harmonic_momentum', 'biharmonic_momentum', 'apply_bc', 'diagnose',
    'gather_array_mpi',
    # isoneutral mixing (pe_realistic/isoneutral.py): the driver and its
    # two jitted kernels (tensor/triad construction, explicit flux
    # divergence) plus the implicit K_33 solve it shares with vertical_mixing
    'isoneutral_diffusion', 'isoneutral_diffusion_pre',
    'isoneutral_diffusion_tendency', 'implicit_vdiff',
)
