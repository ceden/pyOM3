# JAX backend on GPUs: notes

Things learned on Levante (A100, jax 0.11, one process per GPU), Oct 2026.
They apply to every pyOM3 package with the JAX backend (pe_*, wim_*).

## Running

- `PYOM3_BACKEND=jax`. Several GPUs: one process per GPU with
  `srun -l python3 -u script.py npx npy` (jax.distributed is initialised in
  pyOM3/__init__); the mesh then spans the processes. The test suite is a
  single process (`python3 -u run_fast.py pe wim`), never under srun with
  several tasks.
- `--xla_gpu_autotune_level=0`: pyOM3's `__init__.py` adds this to
  `XLA_FLAGS` itself before importing jax, unless the user's `XLA_FLAGS`
  already contain an autotune level; nothing is needed in job scripts. XLA's
  autotuner benchmarks every fused kernel with real-size buffers at compile
  time; for the wim fields at fac=16 that takes minutes per step (the job
  looks stuck after the `fac = 16` line). Without it the timings are the
  same to within noise (no GEMMs outside the DIA).
- Persistent compilation cache: `JAX_COMPILATION_CACHE_DIR=<dir>` and
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5`; compiled steps are reused
  between jobs with the same configuration.
- Every model instance traces its jitted step anew (self is a static
  argument) and the jit cache keeps every instance alive. Scripts that build
  several models call `jax.clear_caches()` between them (timing scripts,
  test config).

## Writing jitted model code

- A jitted method has self static: attributes read inside it are trace-time
  constants. Time-varying fields (ocean_U, the Janssen state, ...) are
  arguments; results are returned, never written to attributes inside.
- With one process per GPU a jitted function may not close over a sharded
  array ("Closing over jax.Array that spans non-addressable devices"). The
  sharded run constants (masks, bc tables, loss weights) are therefore
  passed as the `consts` argument (`shard_consts()`) and bound with
  `bind_consts(consts)` inside the jitted wrapper. Any new sharded constant
  used inside a jitted step goes into `_CONST_NAMES`.
- Global numpy fields go to the devices with `from_global(a)`
  (make_array_from_callback: every process supplies its shards); `to_device`
  does the same for a padded single-PE array.

## XLA fusion of a large step (wim fac=16, Oct 12 2026)

Symptom: one resolution step (wim_simple basin, nx=ny=384, 24x25 spectral
bins, 2x2 GPUs, 0.2 GB per field per GPU) took minutes per call at 100 %
GPU utilisation while the next smaller resolution took 10 ms; before that
the XLA fusion autotuner hung in the compile of the same step
("autotune-fusion-emitters" in an `--xla_dump_to` dump). Every piece of the
step on its own (flux scheme, energy fixer, loss sums, divergence) ran in
milliseconds; it was XLA's fusion of all of them into one program that
produced the bad kernel. Memory was not involved.

Fix: `jax.lax.optimization_barrier` between the pieces inside the jitted
body (`_barrier()` in wim_simple/propagation.py, identity on numpy). The
step stays one jitted program, XLA just may not fuse across the barriers;
no cost at smaller resolutions.

The pe packages do not have the barriers and do not need them at the sizes
run so far (pe fields are much smaller per column). If a pe step ever shows
the same symptom, a step that grinds at one resolution while the next
smaller one is fine, the same barriers between the pieces of that step are
the answer. Diagnosis in that case: a script that runs the pieces of the
step one by one, each jitted on its own, with a print before each
(testing/probe_wim16.py is the template). `--xla_gpu_autotune_level=0` is
set by pyOM3's import (see Running); without it the compile does not get past the
autotuner at these sizes.
