# Working on pyOM3

This note collects the conventions and design decisions of pyOM3 that are not
visible in the code itself. Read it before changing a package. The state of
individual experiments is not recorded here.

## Layout

Repository root:

- `pyOM3/` the package
- `pyOM3/test/` the routine test suite
- `doc/` the LaTeX model documentation
- `setups/` the configurations: `coupled_common/`, `forcing_common/`, `simple/`
  and one directory per configuration

Packages:

| package | content | relation |
|---|---|---|
| `pe_simple` | hydrostatic primitive equations, uniform Cartesian grid | holds the grid-metric-free infrastructure (boundary, solver, MPI and JAX decomposition) |
| `pe_realistic` | as pe_simple with variable dx(x), dy(y), dz(z), optional pseudo-spherical grid, topography, TEOS-10, isoneutral diffusion and GM | imports pe_simple's infrastructure directly |
| `nh_simple` | non-hydrostatic, FFT pressure solver (p3dfft), LES closures | |
| `nh_realistic` | non-hydrostatic with topography mask and variable Cartesian grid | subclass of pe_realistic, no spherical grid (not consistent) |
| `wim_simple` | spectral wave model, WAMDI sources, DIA nonlinear transfer | reference, kept stable |
| `wim_realistic` | wim_simple on the pseudo-spherical grid, Janssen/WAM4 sources as option, variable-dzt ocean interface | inherits from wim_simple | |

Conventions shared by all packages:

- vertical index 0 is the bottom, in the models and in the wave-ocean interface
- wave direction theta is measured clockwise from +y, k_hat = (sin theta, cos theta), as in WAM
- pe_realistic still misses many parameterisations like IDEMIX modules

## Rules for model code

1. **Realistic imports simple.** Code that does not depend on the grid metric
   exists once, in the simple package, and the realistic package imports it
   directly. No stub modules that only re-export. New wave physics goes into
   wim_realistic. A model bug found in one package is fixed in its sibling too.
2. **Ported code does what pyOM2 does.** A routine ported from pyOM2 follows the
   Fortran, with the pyOM2 names. Every deviation is a stated decision. The
   current deviations are the surface-buoyancy TKE source for eq_of_state 0
   (drho/db = -rho_0/g, pyOM2 has none) and the momentum splitting order
   (implicit vertical friction after the AB update, less dt-sensitive than
   pyOM2). Convective adjustment runs only when no TKE closure is active.
3. **No new names.** Follow the existing pattern (e.g. `X` and `X_fortran`).
   Never rename a file, function or option without saying so.
4. **Model communicator.** Module code uses `self.global_sum` and
   `self.global_max`. `OM.global_*` reduce over COMM_WORLD and deadlock when the
   ocean runs on a sub-communicator next to the wave PEs. Collective calls (e.g.
   `get_full_domain_extent`) never sit inside a `my_pe == 0` branch.
5. **Halo exchanges cost time.** Use as few `apply_bc` calls as possible.
   `apply_bc` works in place, so snapshot anything that must survive it (e.g. the
   frequency ghost bins of wim). A field that is read in the halo needs a valid
   halo. Results must not depend on `halo_points` (test 16).
6. **Density once per time step**, at the place pyOM2 evaluates it (after
   vertical mixing, before the TKE integration).
7. **Honest energy bookkeeping.** Every term is booked and written out.
   Numerical dissipation (e.g. 1/2 |du|^2/dt of the implicit friction) is booked
   separately and never added to the physical TKE production. The models are
   energetically consistent wherever possible.
8. **Restart state is everything the next step reads.** Examples: `K_diss_v`
   with TKE, `U_bt`, `V_bt` with the explicit free surface, `taux`, `tauy` with
   `mxl_choice = 2`, the wave frequency ghost bins, `tau_w` and `z0` with the
   Janssen input. A continued run must be bit-identical to an uninterrupted one.
   A new restart field goes into the family's `RESTART_FIELDS`.
9. **Backends.** numpy with MPI is the reference. Fortran kernels must be
   bit-identical to it (compile with `-ffp-contract=off`, keep `np.sum` where its
   pairwise order is part of the result). JAX work must not make the numpy/MPI
   path slower. Production runs use one MPI rank per core, no threading.
10. **Shared methods.** A zero-argument `super()` binds to the class it was
    defined in. A method assigned from one package's class to another's must not
    contain it.

## Tests

`pyOM3/test/` has one directory per package family (`pe/`, `nh/`, `nhr/`,
`wim/`) and `common.py`. Every family `config.py` provides `small_model`,
`step`, `state`, `RESTART_FIELDS` and `TESTS`.

- `python3 run_fast.py` runs in minutes. Run it before and after every change.
- `python3 run_slow.py` runs the twins against pyOM2 (needs `PYOM2` pointing to
  a pyOM2 clone with its f2py build), the free-surface harnesses, the wim
  benchmarks and the JAX sharding tests.
- Test 4 runs under `mpirun`.

| number | test |
|---|---|
| 1 | reduction: simple = realistic on a uniform Cartesian grid |
| 2 | conservation identities |
| 3 | restart, exact |
| 4 | decomposition, exact in every layout |
| 5 | operators |
| 6 | golden numbers (`golden.py --update` only for an intended physics change, with the reason stated) |
| 7 | twins against pyOM2 |
| 8 | free surface harnesses |
| 12-15 | wim symmetry, sources, budget, benchmarks |
| 16 | halo width |

A refactoring must be bit-identical. A new feature gets its own test in the
family. Not covered yet are a coupled smoke test, wim_idemix, and the Fortran
kernels with more than one halo point.

## Known floors when comparing with pyOM2

- The pi literal in pyOM2's `main_module.f90` is single precision. Metric arrays
  differ by 2.8e-8 from pyOM3.
- `wim_simple/Snonlin/Snonlin.f90` has a single-precision PI. Fortran and numpy
  DIA differ by about 1e-8.
- Dynamics differ at first order in dt (operator splitting). The difference
  vanishes with dt or at steady state.
- dzt must vary gently. The u-centred recurrence for dzw oscillates and can go
  non-positive for strongly stretched grids (pe raises an error, pyOM2 does not).

## Writing a configuration

Coupled and realistic runs: `setups/<exp>/` holds only `<exp>_setup.py`, the
batch scripts and inputs, and runs `python3 -u ../coupled_common/run.py`.
Generic coupler code lives in `coupled_common`, forcing code in
`forcing_common` (selected in `forcing_settings()`), never in the model or
experiment classes.

Checklist for a driver:

- add the forcing to `du`, `dv`, `dtemp`, `dsalt` at `tau` in `set_forcing`
- set `coriolis_t` (it is allocated as zeros)
- with TKE set `forc_temp_surface` and `forc_salt_surface` (K m/s and psu m/s,
  positive into the ocean). There is no background `kappaM_min` or
  `kappaH_min` unless the configuration sets one
- apply bottom friction in one place only
- the implicit free surface is the robust choice. The rigid lid does not
  converge on global domains with land. The explicit free surface needs
  `sub_cycling` set by the configuration, `fb_theta` about 0.5 at a large
  external-wave CFL, and `free_surface_tracer_flux = True`
- shallow coupled configurations set `closed_bottom` in the wave interface

Educational configurations in `setups/simple/` are as simple as possible, with
few comments. Short experiments are notebooks, long runs are scripts with
`plot_snap.py` for their snapshots.

## Documentation (`doc/*.tex`)

- Set the scene first (grid, coordinate range, what a cell is), then the physics.
- No undefined shorthand for a step. Write the relation out. When an equation is
  integrated, write the integral and say which equation it is.
- When two quantities look contradictory, say which plays which role.
- One thought per paragraph, short sentences, no colons or semicolons.
- No new names. Use the macros of the existing files (`\beq`/`\eeq`, `\v`,
  `\vn`, `\p`, `\lk`/`\rk`, `\eq{}`).

## Asking an AI assistant for changes

What helps, whoever asks:

- Upload the current tree, or at least the files involved. The upload is the
  reference, not an earlier version.
- Say which package the change belongs to ("pe_realistic only").
- Ask for individual changed files, each with its target directory and an
  account of what changed and why. No archives unpacked over the tree.
- Require `run_fast.py` to pass, bit-identical where the physics is not meant to
  change.
- For a model problem, ask to check pyOM2 and the literature first.
- For a review, ask for all findings: what was checked and found consistent,
  what was not, and the caveats.
