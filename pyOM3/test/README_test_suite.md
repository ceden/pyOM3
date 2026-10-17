# Routine test suite for pyOM3

What should be run when the code changes, how long it takes, and what each
test guarantees. The suite is split into a FAST part that runs on every
commit and a SLOW part that runs nightly or before a release.

All fast tests use a small configuration, of the order of 16 x 12 x 10 points
and a handful of time steps, not a production grid. The whole fast part
should finish in a few minutes on one node. Every test prints ONE line with
the measured number and the threshold it is compared against, so a failure
says what moved and by how much.

A note on options. The package now carries several switches that change the
code path: `mxl_choice` (pyOM2 or NEMO mixing length),
`free_surface_tracer_flux`, the implicit and the explicit free surface,
`enable_bottom_friction`, `use_stress_work`, and the coupled and uncoupled
ocean. Tests 2 to 4 must run for each combination that is in use, not only
for the defaults. The halo dependence of the NEMO mixing length existed in
one option only and was invisible in the default path.


## Fast part, every commit

### 1. pe_simple against pe_realistic

On a uniform Cartesian grid the two packages must agree to round-off. This
is the test that catches metric errors in pe_realistic, because pe_simple
has no metrics to get wrong.

    tolerance   1e-12 relative, all prognostic fields
    existing    pe_reduction_expl.py (explicit free surface),
                the mixing-length comparison written for mxl_choice

Run it for the implicit and the explicit free surface, and for both mixing
lengths.

### 2. Discrete identities in a short run

Each of these is an identity of the discrete equations, so it holds to
round-off and not merely approximately.

    KE_ph = conversion                          1e-16 W/m2
    D_h   = stencil + forward-Euler part        1e-16 W/m2
    TKE budget closes                           1e-17 W/m2
    heat and salt conserved                     1e-15 flux units

These have found real errors: TKE diffusing into the dry cell below the
deepest wet level, a convective branch acting on top of the closure, and a
vertical friction term booked twice. The run needs about ten steps.

### 3. Restart

One run of 2N steps against the same run stopped and restarted after N
steps. Every field of the restart file must agree bit for bit, and so must
the online time means when they are switched on.

    tolerance   exact, no epsilon

This catches state that is used by the closure but not written to the
restart file, which is easy to introduce and silent in a single run.

### 4. Decomposition

One PE against two and four, split in y and in x.

    y-split     exact
    x-split     1e-12, the pressure solver sums in a different order

The mixing length, the surface forcing and every field whose halo a horizontal
average reaches into are covered by this. Run it with the coupled ocean too,
where the fields delivered by the coupler must be broadcast to all ocean
ranks.

### 5. Operator unit tests

No model run, seconds each.

    mixing length vs an independent implementation       exact
    bottom friction: only the lowest wet cell, -dt r u   1e-15
    mixed layer depth vs a brute-force vertical search   the search spacing
    surface TKE flux vs its definition                   1e-8

They need only a model object and a prescribed state, so they are the
cheapest place to check a new operator.

### 6. Golden regression

A checksum of a short run with the default settings, stored in the
repository. Every new option must leave it unchanged. This is the cheapest
guard against an option that quietly alters the default path, and it is the
test that tells a reviewer that a large diff changes nothing for existing
setups.


### 10. Tracer advection outside the AB step (`pe/vadv.py`)

Option `vadv_sub_cycling = M` of pe_realistic (`vadv_split.py`; 0 = off, and
then tests 1 to 6 are the test that nothing changed). Numbered 10 because 7
to 9 are taken by the slow part; it runs in the fast part, 5 s.

* 10a a uniform tracer stays uniform in the full model, both free surface
  paths, M = 1, 2, 4 (round-off). Control: the same sweeps without the pseudo
  density must FAIL, otherwise 10a proves nothing.
* 10b per step, content change = AB-weighted forcing + the surface-face flux
  the sub-steps report (`vadv_fs_flux`), to round-off of the content.
* 10c kinematic overturning cell with a narrow downwelling (vertical CFL ten
  times the horizontal one), a halocline advected through it. The unsplit
  scheme must DIE at CFL 0.64 (the icon01 crash), M = 1 must hold 0.64 and 0.9
  with no new extremum, at 1.8 M = 1 must die and M = 2 hold, M = 4 holds 3.6.
* 10d split against unsplit in the forced box: the mean difference must shrink
  like the first-order splitting error when dt is halved.
* 10e KE_ph = CONV and the exact continuity with the option on.
* 10f restart bit for bit with the option on (no new restart state).
* 10g M = 1 against M = 4, diagnostic.

Test 4 has a third case, 'split' (implicit path, M = 2).

### 11. Third-order Adams-Bashforth (`pe/ab3.py`)

`step(m, n, 'ab3')` of `pe/config.py`: the weights (23/12, -16/12, 5/12) after
the cold-start ramp of pe's own `loop()` (forward Euler, AB2, AB3). All other
tests use (1.6, -0.6, 0), pyOM2's AB2 with eps 0.1, as the coupled runs do
(`EXP_OCEAN_AB=3` switches those).

* 11a restart bit for bit with AB3, both paths, with and without
  `vadv_sub_cycling` (the ramp counter `m._nstep` belongs to the restart).
* 11b heat content change = forcing with the three weights actually used
  (+ the surface-face flux), round-off.
* 11c uniform tracer, 11d KE_ph = CONV and the exact continuity, with AB3.
* 11e LEAK, the energy the time stepping loses, in a frictionless unforced
  adjustment box, relative to the energy that went through KE_ph, measured
  after the barotropic waves of the adjustment are gone (the implicit free
  surface removes them at any dt and would hide the AB part). AB2 eps: 1.1
  percent at dt 900, halving with dt (first order). AB3: 1.4e-4. The test
  asks for the factor 2 and for AB3 below 5 percent of AB2 eps.

Test 4 has a fourth case, 'ab3' (implicit path, M = 2, AB3).

## Slow part, nightly or before a release

### 7. pyOM2 twins

The column TKE cases, the biharmonic friction, the Munk gyre, and the
variable-grid twin. They need the compiled fortran, which is why they cannot
sit in the fast part.

    tolerance   round-off where the schemes are identical,
                O(dt) where the time stepping differs

### 8. Free surface, implicit against explicit

The Munk gyre and the isolated barotropic bump harness, over a range of
sub-cycle counts and for deep, shallow and polar boxes.

    tolerance   3 % in the velocities (measured 1-2 %), no growth of the
                difference with the sub-cycle count, bump variance decaying;
                the surface elevation carries the gravity waves the implicit
                path damps (5-13 %) and is reported, not tested

### 9. Coupled smoke test

Two short segments of the coupled model over a restart, on a few ranks per
component. Check that the exchange fields arrive on every rank, that the
restart continues, and that the identities of test 2 still hold.

This is the test that would have caught the missing broadcast of the
significant wave height, in one minute instead of a whole ensemble.


## Deliberately not in the suite

The forcing preprocessing checks need the ERA5 pool. The mixed-layer
comparison needs the observed climatology and ten model years. The ensemble
notebooks need judgement about a flat minimum. All three are calibration
work, not regression testing.

The debug scripts written during an investigation belong to that
investigation and should not be kept.


## Families

pyOM3 is five packages in three families, and each family needs its own
small configuration, restart list and golden checksum:

    pe    pe_simple, pe_realistic        hydrostatic ocean (built, below)
    nh    nh_simple                      non-hydrostatic (built, below)
    nhr   nh_realistic                   non-hydrostatic with topography and
                                         variable Cartesian grid (built, below)
    wim   wim_simple, wim_realistic      surface waves (built, below);
          wim_idemix                     wim_idemix not built: its column and
                                         calibration scripts
    coupled                              test 9, the wave-ocean coupler

Every family directory carries a `config.py` with the same contract
(`small_model`, `step`, `state`, `RESTART_FIELDS`, `TESTS`) so that
`run_fast.py` treats them alike. Test names inside a family directory are the
generic ones (reduction, identities, restart, ...) because the family
prefix carries the meaning: `pe/restart.py`, later `wim/restart.py`.

## State of the implementation (Sep 15 2026), pe family

nh|nh|wim]` runs tests 1, 2, 3, 5, 6 of every built
family and, when mpirun is on the path (or MPIRUN is set), test 4 through
pe/run_decomposition.py; for pe on the small configuration (`pe/config.py`: 16 x 12 x 10,
closed box with a shelf and a seamount, wind + surface cooling, TKE, 900 s
steps, spherical variable grid; 'cartesian'/'simple' variants for the
reduction) in about 2 s, one line per check with the measured number and
its threshold. `python3 run_slow.py` runs 8 (pe/free_surface/) and, with PYOM2 set, 7
(pe/twins/); 9 is not written. Test 4 runs under mpirun, from run_fast.py when a launcher is
available, else standalone (see decomposition.py). State Sep 15 2026: tests 1-6 and 8 run and pass on the pe
family; 7 needs the pyOM2 build; 9 (coupled smoke test) is the one not
written -- planned as a wrapper around coupled_global4/run.py: two short
segments over a restart against one of double length, restart files compared
bit for bit, the log checked for the exchange fields arriving on every ocean
rank.

The first run of the fast part found two bugs: the explicit free-surface
path dropped the surface heat/salt flux (set_forcing ran before the
overwriting tracer_advection; fixed in both packages by carrying the tracer
forcing increments across the advection), and `mxl_choice = 2` needs `taux`,
`tauy` in the restart (the closure reads last step's stress before
set_forcing). Both are covered by the suite now; the golden checksum of the
default path was unchanged by either fix.

    common.py            line / reldiff / checksum, shared by all families
    run_fast.py          loops over the families' fast tests
    run_slow.py          8 via pe/free_surface/, 7 via pe/twins/ when PYOM2 is set
    pe/config.py         small_model / step / state / RESTART_FIELDS
    pe/reduction.py      test 1: implicit + explicit, mxl 1 + 2, 1e-12 (measured 1e-14)
    pe/identities.py     test 2: KE_ph = CONV_int + CONV_top (pressure work =
                         conversion, 1e-15), KE_adv = KE_div (flux form telescopes,
                         1e-15), biharmonic step dKE = stencil + fwd-Euler (1e-18),
                         salt content exact, heat content change = applied flux
                         (AB2 end effects, bound 0.3), uniform tracer under
                         free_surface_tracer_flux (1e-12); both free-surface paths,
                         from the model state with the package operators. Writing
                         it found that the explicit step applied vertical friction
                         AFTER diagnosing w (stale w, 1.6e-8); fixed in both
                         model.py (friction before w and the tracers). Still in
                         global4_setup.py only: the TKE budget identity (needs the
                         closure's lhs/rhs booked inside integrate_tke)
    pe/restart.py        test 3: exact, both paths, both mixing lengths
    pe/run_decomposition.py  test 4 in one go: serial references + 1x2, 2x1, 2x2,
                         implicit and explicit, each its own mpirun (MPIRUN env
                         var for srun); prints the eight result lines
    pe/decomposition.py  test 4: needs mpirun (serial reference + NPX NPY runs),
                         both free-surface paths since Sep 15 2026;
                         run Sep 15 2026: 1x2 exact, 2x1 and 2x2 7e-14 -- the CG's
                         global dot products are reduced in rank order, so an
                         x-split changes the summation order where a y-split
                         with this layout does not; a y-split that stops being
                         exact means something other than the solver depends on
                         the layout (what the mixing-length halo bug looked like)
    pe/operators.py      test 5: surface TKE flux; bottom friction (lowest wet
                         cell only, u/(1+dt r), exact, live since Sep 15 2026);
                         mixing length: the script of that work
    pe/golden.py         test 6: pe/golden.json, --update after an intended change
    pe/vadv.py           test 10: vadv_sub_cycling (pe_realistic/vadv_split.py)
    pe/ab3.py            test 11: AB3 weights, restart, identities, LEAK against AB2 eps
    pe/free_surface/     the explicit-free-surface harnesses (test 8): the five
                         box harnesses, shelf_coast_test.py, and the restart-window
                         pair restart_window.py + biafra_box.py that reproduces a
                         POP event on the laptop (pe README 2b item 3). Since
                         Sep 15 2026 the two run_slow harnesses carry their
                         expectations and exit nonzero on a miss:
                         munk_free_surface_test.py -- explicit = implicit to 3 % in
                         u at every stable M >= 1.2 M_cfl (measured 1-2 %), the
                         difference does not grow with M by more than 1.5x
                         (measured 0.9-1.1; 100x before the carried barotropic
                         state), explicit stable above 1.2 M_cfl, shelf case within
                         10 %, bump p_s variance ratio < 1 for both paths; eta is
                         reported only (gravity-wave content). pe_reduction_expl.py
                         -- reduction at 1e-12, bump harness decays at every
                         M > 1.2 M_cfl and pe_simple == pe_realistic Cartesian
    pe/twins/            biharm_twin.py, munk_biharm_twin.py (test 7, need PYOM2)
    coupled/             contract only (nh: see its own section below)

## State of the implementation (Sep 20 2026), wim family

`python3 run_fast.py wim` runs tests 1, 2, 3, 5, 6 and, with mpirun, 4, for
wim_simple and wim_realistic, in about 20 s. The tests carry the numbers of
the pe family and mean the same thing. Small configuration (`wim/config.py`):
14 x 12 cells, 16 frequencies (0.06 Hz, ratio 1.13) x 12 directions, 300 s
steps, periodic in x, walls in y, an island across both seams of a 2 x 2
layout; a storm (u* 0.2 to 0.9 m/s), a swell packet plus a young wind sea as
initial state, so that the dissipation stress, the outflow through the open
coasts and the diagnostic f^-5 tail INSIDE the grid (38 bins) are active from
the first step; a sheared current on a 12-level ocean column through the
Doppler projections. 'realistic' = the sphere with variable dxt, dyt and a
variable dzt of the ocean column (2 m at the surface); 'cartesian' and
'simple' = the uniform grid and uniform dz of the reduction. Defaults: open
coasts, energy fixer, current, limit_growth, mask_energy, closed_bottom,
numpy backends (no compiled module needed). All fields per GLOBAL index.

    wim/config.py        small_model / step / state / gathered_state, totals.
                         state = A WITH its frequency ghosts, the outflow
                         totals, and the exchange fields the ocean receives
                         (surface stress, Stokes drift), so tests 1, 3, 4, 6
                         cover the interface too
    wim/reduction.py     test 1: wim_simple = wim_realistic on the uniform
                         Cartesian grid, 1e-12 (measured 2e-15): fixer + open
                         coasts + current, wall reflection, the plain scheme,
                         tail_flux input, and the fortran kernels when built
    wim/identities.py    test 2, propagation alone (no floor, no sources),
                         sphere and wim_simple: action + outflow constant;
                         energy + outflow constant without a current; ABSOLUTE
                         energy + outflow constant with current and fixer, open
                         and closed coasts (all 2e-16); control: without the
                         fixer it must NOT close (3e-5); action under wall
                         reflection. Interface, after five forced steps, for
                         the WAMDI tail, the input tail and janssen: column sum
                         of calc_stress_work = calc_stress on the variable dzt;
                         the EXCHANGE identity sum_z dz u_L . tau(z) =
                         sum_f dft U(f) . tau(f), the Doppler projection as the
                         adjoint of the stress profile (resolved bins; the
                         analytic tail and the cascade credit have no U(f));
                         Stokes transport = spectral integral + analytic tail
    wim/restart.py       test 3: exact; default, janssen, input tail,
                         reflection, wim_simple. RESTART_FIELDS is the
                         reference list: A (WITH the frequency ghosts), dA, the
                         three outflow totals and their three per-level rates
                         (_lossA_lv, _lossE_lv, _lossO_lv: AB2 combines two
                         levels), tau_w under janssen (z0 follows from it), the
                         level pointers and the Euler-start counter. Control:
                         A without its frequency ghosts must FAIL (1e-5)
    wim/decomposition.py, run_decomposition.py
                         test 4: 1x2, 2x1, 2x2 against 1 PE; default,
                         reflection, janssen. No global solver in the wave
                         model, so EVERY layout is exact, exchange fields
                         included; the outflow totals are one Allreduce, 1e-14
    wim/operators.py     test 5: fortran propagation = numpy bit for bit;
                         fortran DIA = numpy DIA (1e-8, see below); the f^-5
                         tail in the diagnostic bins and the ghost, exact; a
                         barotropic current is projected as U(f) = u_0; the
                         uniform state on the sphere (metric divergence against
                         great-circle turning) second order; tendency on land 0
    wim/golden.py        test 6: wim/golden.json, --update after an intended
                         change
    wim/jax_sharding.py  the JAX backend (Oct 12 2026): numpy reference, then
                         PYOM3_BACKEND=jax with 1x1, 1x2, 2x1, 2x2 shards on
                         four fake CPU devices, each in its own process;
                         simple, realistic, cartesian x default, reflection,
                         janssen, tail_flux input, adiabatic; 8 steps, the
                         fields of config.FIELDS on the global interior,
                         1e-12 (measured 3e-15). Not in run_fast.py: with
                         every model instance tracing its jitted step anew
                         it takes about 12 min on two cores (a persistent
                         compilation cache is set inside, see the docstring).
                         The whole fast family also runs under
                         PYOM3_BACKEND=jax (config.py and four tests were made
                         layout-aware: global_index, _interior, no in-place
                         writes), slowly, for the same reason.
    run_slow.py          8w sphere_test.py (great-circle ray, conservation over
                         3 d, closure; 23 s) and 7w janssen_ref_test.py when the
                         ecwam reference is built; both stay in
                         wim_realistic/development/ and are run from there

Writing the suite found one thing: `wim_simple/Snonlin/Snonlin.f90` line 16
has `PI = 3.1415927` without `d0`. That is a single-precision literal, so the
fortran DIA works with 3.14159274101, while the numpy and jax ports carry the
double 3.1415927 (chosen on purpose to mimic WAM). The two backends differ by
1e-8 in S_nl through the quadruplet angles DELPHI1/2; with float32(PI) in the
port they agree to 1.5e-15. Nothing was changed; the operator test has the
tolerance 1e-7 until one of the two is aligned. `g = 9.81` on line 13 is
single too but does not enter.

Not covered: wim_idemix, snapshot files, calc_dt_us
(carries ust_last, which a coupler restart must hold), the source-term energy
budget (wind work, dissipation work: booked in the coupler, test 9).

## State of the implementation (Oct 11 2026), wim_simple source terms

The wim family of Sep 20 covers propagation, the fixer, outflow,
reflection, restart, decomposition, golden and the wave-ocean interface with
the strategy of the pe family (tests 1-6). Everything in `source.py` and the
source time stepping is physics, not a discrete identity, so three fast
tests (12-14) and a slow one (15) use other strategies. `python3 run_fast.py
wim` now takes about 30 s; `run_slow.py` runs 15 in 70 s.

    wim/config.py        + homogeneous_model: wim_simple on a doubly periodic
                         (or y-walled) uniform grid without the island, a
                         uniform wind and a spatially uniform spectrum, so the
                         propagation tendency vanishes exactly and the sources
                         act alone (the homogeneous case of the growth
                         literature)
    wim/symmetry.py      test 12, the analogue of the nh lid mirror test:
                         theta mirror symmetry of the spectrum under a wind
                         along theta = 0 (bin d <-> nd+1-d); the wind turned
                         by one directional bin gives the spectrum rolled by
                         one bin (nothing in the sources knows the absolute
                         direction); x <-> y symmetry of propagation (a packet
                         on the transposed field with the directions reflected,
                         theta -> pi/2 - theta); positivity under strong wind
                         on a near-empty sea (floor not needed, E >= 0). All
                         round-off (1e-15): an index error in the DIA
                         quadruplets, the input/dissipation direction weights
                         or the tail shows here and nowhere else
    wim/sources.py       test 13: Sinput_wamdi and Sdiss_wamdi against the
                         WAMDI formulae with the spectral means by direct
                         quadrature, frcutindex, calc_phioc (exact, 1e-15).
                         DIA: on a spectrum confined to the interior of the
                         frequency grid it conserves action and energy to
                         round-off (3e-16) and momentum to 1.2e-2 (the
                         quadruplet is interpolated onto the directional grid,
                         the known property of the WAM DIA; tolerance 3e-2);
                         on the JONSWAP sea the interactions reaching the
                         cut-off are dropped (net/gross 5e-2 action, 1.4e-1
                         energy, 3e-1 momentum, printed) and the cascade
                         credit the interface books equals that net loss over
                         the prognostic bins (exact). Source integration:
                         wind input alone, the explicit (implicit_dissipation_
                         only) and the implicit step are first order (error
                         ratio 2 for dt -> dt/2, measured 1.94/2.07) and
                         bracket exp(gamma t) from below/above. limit_growth:
                         the fraction of prognostic bins cut is printed (12 %
                         at u* 0.3 and 23 % at 1.5 on a PM sea, from 0.12-0.14
                         Hz up; 0 on a near-empty sea): the cap is active in
                         the upper prognostic range of any developing sea, as
                         in WAM, so it is a diagnostic, not a PASS/FAIL
    wim/budget.py        test 14: the energy budget of a full time step from
                         the model's own booking. Needs the booking of the
                         APPLIED source increments in wim_simple/model.py
                         time_step (dE_src and its split dE_in, dE_ds, dE_nl in
                         proportion to the explicit rates; dE_tail for the
                         f^-5 overwrite; A_clip for the action floor; 23 lines,
                         inherited by wim_realistic; no result changes, golden
                         and restart unchanged). Without it the test says so
                         and passes. With it: split exact; the budget sum
                         omega dA = AB-weighted propagation (incl. outflow) -
                         floor + sources + tail closes to 1e-16 for the
                         homogeneous case, wim_simple with island, current and
                         open coasts, wim_realistic on the sphere and with
                         janssen. SOURCE-STEP CLOSURE (Oct 12 2026): the
                         semi-implicit step dE = S dt/(1 - dt dS/dE) is the
                         linearised implicit step, so the sources act at
                         E^{n+1}, and the fluxes the coupler builds AFTER the
                         step (calc_stress, calc_phioc, wind_work: gamma_in E,
                         gamma_ds E, S_nl at E^{n+1}) are the consistent
                         booking -- gamma_in is in dS/dE like gamma_ds, the
                         input is as implicit as the dissipation, and tau_in
                         from gamma_in E is right. What remains between the
                         booked rates at E^{n+1} and the applied increment is
                         decomposed exactly: the DIA's cubic nonlinearity,
                         gamma_ds(E), the limiter (cuts S, not dS/dE), the
                         dropped positive part of dS/dE, and the share in the
                         diagnostic bins. Spun-up homogeneous sea, dt 300 s, in
                         units of the wind work: 1e-4 at u* 0.3; +0.16 at u*
                         0.6, of which the limiter +0.156, sitting in the
                         diagnostic (overwritten) bins (+0.159): the known tail
                         inconsistency, quantified; with the limiter off +0.11
                         from the DIA nonlinearity over one step, half of it
                         prognostic. Both are dt effects. In the 6-day global4
                         run (monthly climatology, u* mostly < 0.4) the
                         coupler's momentum budget closes to 1 %.
                         RETRACTED (Oct 11-12 2026): a first version of this
                         test compared the explicit rates at E^n with the
                         applied increment and reported "the explicit booking
                         exceeds the applied dissipation by 15-35 %"; that was
                         the factor between E^n- and E^{n+1}-based rates, not
                         an inconsistency. The flux_booking switch delivered
                         on Oct 11 implemented that wrong attribution and was
                         withdrawn on Oct 12; the interfaces are back to their
                         Oct 10 state.
    wim/benchmarks.py    test 15, slow part, the "twin" is the literature:
                         15a duration-limited growth, homogeneous u* 0.5, 72 h:
                         stationary (0.9 % change over the last 6 h), f_p u*/g
                         within 21 % of the PM value 0.13/28, E within 1.12 of
                         the PM level alpha g^2/(5 omega_p^4) at the measured
                         f_p; 15b fetch-limited 1D (closed coast upwind, open
                         coast downwind, 48 x 5 km, 24 h): JONSWAP exponents
                         fitted over X* = 1.2e3-6.5e3 (beyond 1e4 the growth
                         rolls off towards full development): E ~ X^0.90
                         (JONSWAP 1.0), f_p ~ X^-0.25 (-0.33), eps = 0.77 of
                         1.6e-7 X* at mid fetch; 15c a one-bin swell packet
                         travels at c_g to 4e-4 in x and y for theta 45 deg
                         (the superbee spreading doubles its variance in 10 h,
                         printed); 15d a swell packet crossing a zonal jet U(y)
                         = 1.5 m/s: at the jet core the frequency shift is
                         within 10 % and the direction shift within 21 % of
                         Snell's law (k_x and the absolute frequency
                         conserved); 15e the exact S_nl battery
                         wim_simple/snl_exact/test_physics.py run as is (four
                         checks; it sets no exit code and its final export has
                         a hard-coded path /home/claude/snl_exact/, which fails
                         harmlessly after the checks -- the wrapper reads the
                         PASS/FAIL lines)

Not covered after this round: the jax backends, snapshot files, calc_dt_us,
wim_idemix; the Janssen input of wim_realistic beyond the budget and the
stress identities (its formula test would need the ecwam twin, 7w).

## State of the implementation (Oct 10 2026), nh family

`python3 run_fast.py nh` runs tests 1, 2, 3, 5, 6 and, with mpirun, 4, for
nh_simple, in about 10 s. The tests carry the numbers of the pe family and
mean the same thing; where nh_simple has no counterpart the test says what
takes its place. Small configuration (`nh/config.py`): 16 x 12 x 10 cells of
4 x 4 x 2 m, 2 s steps, doubly periodic, N0 5e-3 1/s, f0 1e-4 1/s, Ro =
dsqr = 1 (the LES closures require it); a zero-mean buoyancy anomaly, a
sheared jet, a weak perturbation; a zonal surface stress and a surface
buoyancy loss in the top cell; harmonic friction and mixing (or biharmonic).
'lid' = rigid lids (the default, 2D-FFT + tridiagonal solver), 'periodic' =
periodic in z (3D-FFT solver, no forcing). Options: b_adv ('2nd', 'superbee',
'upwind3'), les (None, 'smag', 'tke', 'ducros'), biharmonic, forcing, b_mix.
The LES closure is hooked in through set_forcing (after the friction, before
the Adams-Bashforth step, as les.py asks). step() uses the cold-start ramp of
the model's own loop() (forward Euler, AB2, AB3); the level pointers and the
ramp counter are restart state. Arrays are fortran ordered (Nx, Ny, Nz).

p3dfft holds ONE plan per process: two models of different size cannot
coexist (heap corruption); the suite finishes one model before it creates
the next, and the reduction test runs its two models one after the other.

    nh/config.py         small_model / step / state / gathered_state
    nh/reduction.py      test 1: no second package, so the LID takes the place
                         of the metrics: the rigid-lid model against the
                         periodic model on the mirror-extended domain (2 nz
                         levels, u, v even, w, b odd), 1e-12 (measured 1e-14
                         once the lid fixes below are in). Compares the two
                         pressure solvers at the same time. Harmonic and
                         biharmonic friction, without and with Smagorinsky.
                         b runs without explicit mixing: the odd mirror image
                         implies b = 0 at the lid, the lid model imposes zero
                         FLUX, so b's diffusion is outside the reduction, as are
                         the limited schemes (they need a slope across the lid)
                         and the TKE closure (its w-grid has a TKE point at the
                         top lid and none at the bottom, by design)
    nh/identities.py     test 2, lid and periodic, three advection schemes:
                         continuity of the carried (u, v, w) (1e-16); pressure
                         work of the projected velocity = 0; KE advection = 0,
                         u+v part and w part; buoyancy work on w = exchange
                         with the background PE (the KE_ph = CONV of pe);
                         Coriolis work = 0; b content: sum of the advection
                         tendency = 0 (all schemes), sum of the background term
                         = 0, content conserved unforced, content change =
                         AB-weighted surface flux forced; variance conserved
                         (2nd; the sign for superbee/upwind3 is printed only:
                         the CFL-dependent antidiffusive part of the limited
                         flux is no instantaneous identity); harmonic: sum of
                         tendency = 0, dKE = stencil + fwd-Euler (harmonic,
                         biharmonic), biharmonic = harmonic(harmonic) (exact).
                         TKE closure: budget closes with tke_min 0 (1e-14,
                         the diffusion telescopes), sum shear_prod = resolved
                         KE removed (the documented identity, 1e-13 from the
                         increment), booked les_ke_loss = KE change, dz sum
                         buoy_prod = resolved PE gain of les_mix_buoyancy
                         (TKE buoyancy sink = PE gain, 1e-14)
    nh/restart.py        test 3: exact; lid default, superbee + tke,
                         smagorinsky, biharmonic, periodic. RESTART_FIELDS: u,
                         v, w, b, the three levels of du, dv, dw, db, _tau3,
                         _nstep, and with the TKE closure tke, diff (last
                         step's diffusivity, used before it is recomputed),
                         shear_prod, buoy_prod, stokes_prod (last step's
                         production). Control: without shear_prod and diff it
                         must FAIL (does, 1e-5)
    nh/decomposition.py, run_decomposition.py
                         test 4: p3dfft splits y (n_pes_x) and z (n_pes_y), so
                         1x2 (z split: lids and TKE surface point on different
                         ranks), 2x1, 2x2 against 1 PE; lid, periodic, les
                         (superbee + tke), smag + harmonic. EXACT in every
                         layout: each 1D transform runs over a complete line
                         in every layout and the tridiagonal is local in
                         spectral space, no global reduction; a difference is a
                         halo read before its exchange (the les paths, below)
    nh/operators.py      test 5: pressure solver on a prescribed divergent
                         velocity, lid and periodic: projection divergence
                         free (7e-15), p satisfies the hand-written Poisson
                         stencil, Neumann at the lids (7e-15); the lid
                         convention of apply_vertical_flux_bc (a T-grid flux is
                         zeroed at faces 0 and nz only); uniform tracer: each
                         advection tendency vanishes, per scheme; superbee
                         keeps a top hat in [0, 1] under 1D translation (CFL
                         0.5), control: 2nd order must overshoot; Smagorinsky
                         in pure shear u = S z: visc_00 = (C delta)^2 |S|
                         (4e-16); LID MIRROR SYMMETRY: every operator of the
                         rigid-lid model maps the vertically mirrored state to
                         the mirrored tendency (momentum and buoyancy
                         advection, harmonic and biharmonic of a T-grid field
                         and of w, pressure solver, Smagorinsky viscosities,
                         les_mix_momentum, les_mix_buoyancy; the TKE closure is
                         asymmetric by design, see above)
    nh/golden.py         test 6: nh/golden.json, three paths (lid default,
                         les = superbee + tke, periodic); --update after an
                         intended change. The lid fixes below change all three.

Writing the suite found four things in nh_simple (Oct 10 2026), all in the
rigid-lid treatment of w-grid quantities and in the halo state; none was
changed in the package, the tests report them as FAIL until they are:

1. `advection.py` momentum_advection_w zeroes the vertical flux of w at the
   two T levels next to the lids (`apply_vertical_flux_bc(top,
   flux_on_wgrid=True)`). The averaged transport 1/2 w_1 at those levels is
   not zero, so the flux form no longer telescopes: sum w adv_w = 7e-5 of
   sum |w adv_w| (test 2, lid), a spurious KE source in the lid cells;
   periodic in z it is round-off. With w = 0 at the lids the out-of-domain
   fluxes vanish by themselves: the bc call is not needed. Without it the
   identity closes at 4e-18.
2. `diabatic.py` harmonic / harmonic_v_only / biharmonic applied to w
   (model.py): apply_vertical_flux_bc is the T-grid lid convention (faces 0
   and nz); for the flux of a w-grid field (at T levels) it zeroes the flux at
   T level 1 but keeps the one at T level nz -- top/bottom asymmetric (test 5,
   harmonic(w) mirror 1.5). Fix tried: an `on_wgrid` argument that skips the
   flux bc (and, in biharmonic, zeroes the inner Laplacian at the lid faces
   instead, where w is odd); model.py passes on_wgrid=True for w. The same
   zeroing sits in les.py les_mix_momentum (w flux, flux_on_wgrid) and in
   les_shear_production (d_w[2]): symmetric there, but the mirror reduction
   (test 1, smag) needs the flux kept; the shear-production identity still
   holds without the bc.
3. `advection.py` superbee and upwind3 take the limiter slope rj at the top
   lid from the zero halo (b_{nz+1} - b_nz = -b_nz) while the bottom one is
   zeroed by apply_bc: asymmetric (test 5, 3e-3), and upwind3 does not keep a
   uniform tracer uniform with lids (test 5, 0.12: the top cell is pushed
   towards the b = 0 above the lid). Fix tried: zero the slope across both
   lids, `rj = apply_vertical_flux_bc(apply_bc(rj))` in both schemes.
4. `model.py` time_step: the explicit friction and mixing increments
   `u += dt*harmonic(u)` also change the halo points (with wrapped data), and
   set_forcing -- where the configurations call the LES closure -- runs before
   the next apply_bc, so the closure reads stale halos in x, y and z: the LES
   increment differs by 1e-3 (smag) / 1e-4 (tke) of itself from the one on
   fresh halos, and the result depends on the layout (test 4, les and smag
   1e-5 in every split; exact for the other paths). Fix tried: apply_bc on u,
   v, w, b after the friction block, before set_forcing. Only runs with both
   an explicit friction and the LES closure are affected.

With the four fixes in a scratch copy the nh fast part passes completely:
reduction 1e-14, every identity round-off, every layout exact. The golden
checksums then have to be re-recorded (--update).

Not covered: the Ducros closure beyond the restart list (no state), the
Peclet bound, les_mix_eulerian / les_stokes_prod (need u_S, coupled runs),
tke_source, snapshot files, the jax path.


## State of the implementation (Oct 16 2026), nhr family

`python3 run_fast.py nhr` runs the tests below and, with mpirun, the
decomposition, for nh_realistic, in about 2 min. The configuration is the
pe family's (`nhr/config.py` loads `pe/config.py` and builds its
`small_model('nh')`: the closed box on the stretched CARTESIAN grid, the
spacings of the spherical one in metres, since nh_realistic refuses
coord_degree (no spherical metric terms in the w equation), the shelf and
the seamount, wind and cooling, the TKE closure, implicit or split-explicit
free surface), so the hydrostatic limit of the package is pe_realistic on
the same grid (kind 'stretched') and every pe test pattern applies. `state` carries
p_nh in addition to the pe fields. The 3-D solve stops at a remaining
divergence epsilon_nh = 1e-12 1/s (its own criterion, see
nh_realistic/solver.py), 50-150 iterations per step with the columnwise
preconditioner (500-1400 without it in this box).

    nhr/config.py          small_model / small_model_pe, the rest from pe/config.py
    nhr/realistic.py       1 hydrostatic limit: enable_nonhydrostatic = False
                           against pe_realistic, implicit and explicit, exact
                           (the time steps are pe_realistic's with the nh
                           lines inserted; this keeps the two in step);
                           2 the operator -V div grad and the columnwise
                           preconditioner are symmetric on the masked
                           stretched grid, A 1 = 0 (1e-14); 3 the projection
                           removes a discrete gradient dt grad q, 1e-10 (the
                           property behind the twin with nh_simple; while the
                           grid was still spherical it found a missing 1/cost
                           in the east coefficient, which the divergence test
                           cannot see: the vertical coefficients are 1e6 times
                           the horizontal ones in this box); 4 the divergence of every wet cell after
                           8 steps, including the surface cells with the
                           free-surface w, is below epsilon_nh; 5 the wet-cell
                           mean of p_nh is zero
    nhr/twin_nh_simple.py  1b nh_realistic against nh_simple in the nh family's
                           box (uniform grid, no topography, doubly periodic,
                           rigid lids): temp with eq_of_state = 0 is b, N0 put
                           into b, harmonic friction Ah only, 2nd order
                           advection, f-plane; 10 steps, 1e-11 (measured
                           1e-13). nh_simple carries the buoyancy in the w
                           equation and solves one Poisson equation with the
                           FFT; nh_realistic carries -grad p_h and solves 2-D
                           then 3-D; the predictors differ by a discrete
                           gradient, which the projection removes
    nhr/restart.py         3 exact, implicit and explicit; the pe list plus dw
                           and p_nh (the first guess of the 3-D solve: the
                           control without p_nh must differ)
    nhr/decomposition.py, run_decomposition.py
                           4 the pe paths (implicit, explicit, split, ab3),
                           1x2, 2x1, 2x2, the 3-D solve run to eps_nh = 1e-16
                           (300 iterations): 1e-9 for all fields but p_nh
                           (measured 1e-10, in tke and K_diss_v) and 1e-5 for
                           p_nh (measured 1e-6). No layout is exact: the 3-D
                           CG's global sums reduce in rank order on every
                           path. The stop criterion is the remaining
                           divergence, which the vertical part of p_nh fixes;
                           the horizontal structure of p_nh is weakly
                           determined in this hydrostatic-aspect box
                           (vertical coefficients 1e6 times the horizontal
                           ones) and still moves when the criterion is met,
                           hence the tight eps_nh here. With the default
                           1e-12 the fields differ by 1e-8 between layouts,
                           physically irrelevant but not reproducible; an FFT
                           preconditioner would be the remedy if it matters
    nhr/jax_sharding.py    jax 1x1, 1x2, 2x1, 2x2 against numpy on 4 fake CPU
                           devices, by hand as for pe, same eps_nh and
                           tolerances (measured 2e-10 and 1e-6)

## Layout

    pyOM3/test/
        common.py            helpers shared by the families
        run_fast.py          tests 1-6 of every built family, prints the table
        run_slow.py          tests 7-9
        pe/                  reduction, identities, restart, decomposition,
                             operators, golden (+ golden.json), free_surface/, twins/
        nh/                  reduction, identities, restart, decomposition,
                             operators, golden (+ golden.json)
        nhr/                 realistic, twin_nh_simple, restart, decomposition,
                             jax_sharding (on the pe configuration)
        wim/                 reduction, identities, restart, decomposition,
                             operators, golden (+ golden.json), symmetry, sources,
                             budget; benchmarks (slow); the sphere and ecwam slow
                             tests stay in wim_realistic/development/
        coupled/             test 9

The harnesses of tests 1, 5 and 8 exist already in
`pyOM3/pe_realistic/development/` and in the mixing-length work; tests 2, 3,
4 and 6 are currently done by hand with short global4 runs and would have to
be written against the small configuration.
