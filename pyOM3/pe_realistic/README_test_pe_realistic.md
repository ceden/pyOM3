# Tests of pe_realistic

Status as of Sep 15 2026. Every test below has been run; the numbers quoted
are the ones obtained. "Round-off" means 1e-14 relative or better.
pe_simple shares eleven of the source files with pe_realistic (uniform-grid
special case); where a test covers both it is said so.

Reference model throughout: pyOM2 (github.com/ceden/pyOM2), run either as the
fortran executable or step by step through the f2py module
(`py_src/pyOM_code.so`, driver `py_src/pyOM.py`; every `time_step()` must be
followed by `time_goes_by()`).

Known floor for any metric comparison against pyOM2: the `pi` literal in
`main_module.f90` is single precision, so pyOM2's `degtom`, `cost` and cell
areas are off by 2.8e-8 relative; likewise `grav = 9.81` is a single-precision
literal (9.8100004196167). Tests that want more than 1e-8 against pyOM2 use
pyOM2's values.

---------------------------------------------------------------------------

## 1. Construction tests (Aug 2026, delivered with the package)

| test | what | result |
|---|---|---|
| uniform-grid reduction | pe_realistic with dx=dy=dz const, Cartesian, vs pe_simple; both solver families | ~1e-15 |
| stencil check | pressure, Coriolis+metric, tracer-flux stencils vs pyOM2 index formulas, incl. variable dxt/dyt | exact |
| momentum/tracer advection conservation | flux form, closed domain | ~1e-18 |
| implicit free surface identity | w_surface = d(p_s/g)/dt | ~1e-14 |
| stretched-dz vertical diffusion | tracer content conserved with dzt(z) | round-off |
| stability | 60 steps sphere + channel, eq_of_state 100 (atmosphere) | runs |
| MPI layout | Cartesian Munk gyre, pe_simple/pe_realistic x 1 PE/2x2, 2000 steps (28 d), identical discrete problem via global-index forcing; harness `gyre.py`, `compare_gyre.py`, `gyre_diagnostics.ipynb` (gyre_experiment.tar) | ~1e-15 all four pairs, identical CG iteration counts |

Grid bug found by pop14 and fixed: the u-centred z recurrence ran across the
z halo (dzw=0 at the surface for non-uniform dzt); now built on the nz
interior as pyOM2, raises on dzw <= 0. Caveat inherited from pyOM2: the
recurrence dzw(k) = 2 dzt(k) - dzw(k-1) oscillates for constant-ratio
stretched dzt; dzt must vary gently.

## 2. Twins against pyOM2 fortran

| twin | configuration | result |
|---|---|---|
| spherical Munk gyre | 55 d, pe_realistic vs pyOM2 | psi 5.8e-4, temp 3.3e-4, u 8.5e-3; residual is the O(dt) vertical-mixing splitting order, vanishes at steady state |
| variable grid | `vartwin_pyom2_test.py` (needs the f2py build): dx(x), dy(y), dz(z) all non-uniform on the Munk twin | coordinates exact, dzt/dzw/zt bitwise, dynamics O(dt) splitting (5.5e-4 at dt=1200, halves with dt) |
| GSW / TEOS-10 EOS | eq_of_state=5, `density.py` vs fortran | digit for digit |
| isoneutral (Redi + GM skew) | `isoneutral.py` vs pyOM2 f2py | tensors/triads 4e-15, tracer increment 3e-11, spherical 3.6e-8 (pi floor), MPI 2x2 / 2x1 vs 1 PE 2e-14 |
| pop14 conversion | serial vs 2-rank MPI with restart resume | 4e-19 |
| biharmonic momentum friction, operator | `biharm_twin.py`: `damping.biharmonic_momentum` vs pyOM2 `biharmonic_friction`, one call each on the same random u, v; sphere 30-85 N, variable dyt, cyclic x, closed strip + island + one-level shelf, A_hbi 2.7e10, cosPower 1.5 (POP values) | masks identical; du, dv 2.4e-7 relative, maximum at 83.5 N = pyOM2 pi floor (cos error 3.6e-7 there) amplified by the cos^3 / cos^-2 factors |
| biharmonic momentum friction, Munk gyre | `munk_biharm_twin.py [ndays]`: closed spherical basin 10-40 E, 20-52 N, dxt 0.8->1.2, dyt 0.8->1.4 deg, dzt 325/275/225/175 m, linear EOS, zonal wind, A_hbi 1e12 cos^1.5, kappaH_0 = kappaM_0 = 1e-4 implicit, AB2 eps 0.1 and implicit free surface in both, 2 d at dt 1200 and 600 | u, v 1.3e-5 -> 6.5e-6, p_s 1.5e-7 -> 8e-8, temp 1e-8 -> 5e-9: halves with dt, i.e. the momentum-splitting order only (see 3) |

Four operator fixes came out of the Munk twin, applied to both packages, all
one cause (maskW opened at the surface for the free-surface solver was used as
a closed-face mask): implicit vertical mixing surface leak (delta(nz)=0
unconditional), momentum advection mask-pair factors removed and apply_bc
dropped from the fluxes (pyOM2/MITgcm wall self-flux), tracer advective
surface flux zeroed, superbee surface-gradient zeroed.

## 2b. Split-explicit free surface (pe_realistic/development/, Sep 10-14 2026)

Started from the POP 1/10 deg run, which blew up within ~11 steps on the
split-explicit path (Demange et al 2019) while the implicit path ran.
Harnesses (all need only PYOM3=<dir containing pyOM3>, no MPI, no pyOM2
except where said):

| script | what | result |
|---|---|---|
| `pe_reduction_expl.py [nsteps M]` | pe_simple vs pe_realistic, explicit path, uniform Cartesian Munk box, with/without biharmonic friction | u, v, p_s 1e-13 (metric-aware form reduces exactly) |
| `pe_reduction_expl.py bump` | barotropic harness: f = 0, uniform density, no wind, no friction, a 0.5 m p_s bump released from rest in a POP-like box (dx 3.5 km, dy 11 km, H 5000 m, dt 216 s, c dt/dx = 13.6, M_cfl = 14); pe_simple, pe_realistic Cartesian and spherical | p_s variance ratio after 50 steps 0.2 at M = 31 and 62 (decay, as the 2-D toy); pe_simple = pe_realistic Cartesian digit for digit; M below M_cfl blows up |
| `toy_split.py`, `toy_split2d.py` | 1-D / 2-D linear shallow water with the pyOM3 sub cycle only, restart from the averaged vs the end-of-cycle transport | average-restart grows with M at high CFL (140 -> 4742 for M 31 -> 124 at c dt/dx 13.6), end-restart decays (0.40 -> 0.66) |
| `bump_variants.py` | implicit control in the same box, friction, theta, single layer | implicit 0.11; friction does not stop the growth; theta 0/0.14/0.3/0.5 changes it by orders of magnitude but never removes it; a single layer reproduces it |
| `shelf_coast_test.py` | ragged 90 m shelf at 73 S with land blobs, explicit vs implicit, bottom friction on/off | does NOT reproduce the POP failure (agree to 3 digits); kept as the coast harness |
| `restart_window.py`, `biafra_box.py` | POP restart -> lon/lat window -> pe_realistic on the real grid_topo window; the procedure that found fb_theta | see 2b item 3 |
| `munk_free_surface_test.py [ndays]` | implicit vs explicit on the spherical variable-grid Munk gyre: M sweep in a deep (H 1000) and a shallow (H 100) basin, flat vs shelf topography, bump adjustment, and the POP-like polar box (0.1 deg at 70 N, H 5000, dt 216) | explicit = implicit to 0.8-1.4 % in u at every M >= 1.2 M_cfl, eta 4-13 % (the gravity waves the implicit path damps), temp 1e-5..1e-4; polar box: u 1.2e-2 at M 18, 23, 31, 46 with no growth; the one-level shelf strip gives u 0.29 (not understood, the shallow cells adjust on scales the implicit path damps) |

Two causes, both fixed (both packages):

1. the sub cycle was restarted every step from U = sum(u dz), i.e. from the
   sub-cycle AVERAGE <<U>> the 3-D velocities are trimmed to, paired with the
   end-of-cycle p_s. FESOM2 (Banerjee et al 2024, arXiv:2311.08241) carries
   the END-of-cycle transport as state and uses <<U>> only to trim the 3-D
   field. solve_expl_surface now takes and returns U_bt, V_bt (model state,
   RESTART state; init_barotropic_state on a cold start). Growth in the bump
   harness 5e4 -> 5.5 at M = 31.
2. the remaining growth was a p_h gradient in a run that started with uniform
   density: flux-form tracer advection with the advective surface flux zeroed
   (pyOM2's adv_ft(:,:,nz) = 0, matched in the Munk twin) changes a uniform
   tracer in the top cell by -T dt div_h(u) every step under the linear free
   surface (correlation 1.000 measured); the surface density anomaly gives
   dp_h/dx in phase with d(eta)/dt, which anti-damps the barotropic mode.
   Option `free_surface_tracer_flux = True` (thermodynamics.py, both flux
   schemes, both packages, default False = pyOM2 form) sets the surface-face
   flux to -w_surf*var_surf, so div_h(u var) + (ft_surf - ft_below)/dz =
   var (div_h u + dw/dz) = 0 and a uniform tracer stays uniform (temp ptp
   exactly 0 in the harness). Growth 5.5 -> 0.18. The option affects the
   implicit path as well (it is a conservation error there too, only hidden
   by the solver's damping) and breaks the bit-match to pyOM2 by exactly
   this term; it is off unless a configuration sets it. pop.py sets it.

Also fixed on the way: the metric terms of the barotropic mode (+tantr U V/h,
-tantr U^2/h) were missing from the sub cycle (formulation error, 1e-6 effect
in the polar box, guarded by coord_degree so pe_simple is untouched), and the
explicit path's biharmonic friction now comes from biharmonic_momentum.

Sep 15 2026, the explicit path at 1/10 deg (POP). Two more things were needed
before it ran, both found by bringing the failing state to the laptop instead
of guessing on 1536 ranks:

3. `fb_theta` (free_surface.py, both packages, default 0.14 = Demange et al):
   the dissipation of the forward-backward sub cycle. At 1/10 deg with dt 216 s
   (external CFL c dt/dx ~ 5, 10 m surface cells) 0.14 left grid-scale
   barotropic waves whose surface divergence gave w ~ 1e-2 m/s (implicit path:
   1e-3), a vertical advective CFL of 0.4 for the tracers over the Biafra
   slope, temperature overshoot 29 -> 47 C in 20 steps, then the dynamics.
   Reproduced exactly from a POP restart 28 steps before the event
   (`test/pe/free_surface/restart_window.py` cuts a lon/lat window out of
   restart_pe/*.npz, `biafra_box.py` runs pe_realistic on the real grid_topo
   window from it; explicit blew up at the same step as POP, implicit stayed
   at 0.2 m/s): theta 0.14 dies at step 380, 0.3 at 404, 0.5 runs on; the polar
   box keeps u agreement with the implicit path at 0.8 % and halves the eta
   difference. POP runs with fb_theta = 0.5 (7 d and counting at the time of
   writing). Switching tracer flux, TKE or bottom friction off did not help;
   doubling M made it EARLIER (finer sub cycle preserves the grid-scale waves).
4. Bottom friction (damping.py, both packages, off by default):
   `enable_bottom_friction`/`r_bot` (pyOM2 linear) and
   `enable_quadratic_bottom_friction`/`r_quad_bot` (C_d), implicit split step
   on the lowest wet cell, applied BEFORE solve_expl_surface on the explicit
   path so the trimmed column matches the damped transport, and the same
   stress on the barotropic transport inside the sub cycle. Test 5 of the
   suite checks the operator; reduction with it on 1e-13. It was not the
   cure for the POP failure (see 3) but is the standard damping of shallow
   barotropic modes and POP runs with C_d 2e-3.

Two POP initial-condition faults found in the same week, both in the file,
not the model: the WGHC-derived file's horizontal fill put Pacific 4000 m
water into the South China Sea (statically unstable by 1.5 kg/m3, 5 m/s
bottom jets); the first WAGHC regeneration's downward extension put
thermocline water on seamount flanks (20 C at 4000 m, 8 m surface pressure in
one step). `make_woce_ic.py` now fills by connectivity at each depth (nearest
same-depth water in the connected region, sill water in pockets) and the
stability check is part of its output. A cold start still produces a
2.5-5 m/s deep adjustment in the Celebes basin and 4-6 m of surface pressure
in the first two days on both paths.

Notes from the biharmonic twins (Sep 10 2026):
- pyOM2 applies cos(y)^cosPower in EACH of the two Laplacian passes (total
  cos^(2 cosPower)); `biharmonic_momentum` does the same. The forward-Euler
  split step of pe is stable up to A dt (4/dx^2 + 4/dy^2)^2 <= 2, i.e.
  A dt/dx^4 <= 1/32 on a square grid (measured: stable at 0.031, unstable at
  0.032); with A cos^3, 0.1 deg cells and dt 216 s the POP grid is stable to
  89.8 N.
- variable dxt on a CYCLIC grid: pe and pyOM2 build DIFFERENT xt and dxu
  (pyOM2 starts the u-centred recurrence in the x halo, pe on the interior;
  both carry a doubled dxu entry at the seam). Coordinates agree for
  non-cyclic variable dxt (variable-grid twin) and for uniform cyclic dxt.
  Not resolved; irrelevant for configurations with uniform dxt.
- pe_realistic allocates `coriolis_t` as zeros: the configuration must set it
  (global4_setup.py `set_coriolis`; pop.py set none until Sep 10 2026 and ran
  with f = 0). A twin's driver must set it in both models.

pyOM2 bug found by the isoneutral twin (Carsten agreed, patch on GitHub):
`isoneutral_skew_diffusion` added the K_11/K_22 diagonal a second time when
skew diffusion was on. pe_realistic adds it once. Test cases `isopyc2.f90`,
`isopyc2_redi.f90`, `isopyc2_redi_eos5.f90` with notebook.

## 3. TKE closure, single column vs pyOM2 (`tke_test/`, Sep 8-9 2026)

Scripts: `tke_cases.py`, `run_pyom2.py` ($PYOM2 = repo root), `run_pe.py`
($PYOM3 = directory containing pyOM3/), `run_all.py [ndays] [dt]` runs
everything and writes `tke_column_summary.txt` + one figure per case.
Configuration = pyOM2 `for_config/test_tke1.f90`: one column, nz 50, dz 1 m,
dt 360 s, 30N, GSW EOS, N0 0.02, alpha_tke 30, mxl choice 2, tke0 1e-12,
kappaM_min 0, AB_eps 0.1. pe uses pyOM2's forward-Euler surface-flux placement
(vertical_mixing override in the harness) and pyOM2's grav literal.

| case | forcing | pyOM2 vs pe_realistic (1 day) |
|---|---|---|
| A | wind 1e-4 | u 36 %, tke 30 % at dt 360; 13 % at 90; 4.5 % at 30 (splitting, see below) |
| B | cooling 200 W/m2 | round-off: temp 5e-15, tke 9e-12, kappaM 2e-12, P_diss_v 6e-11 |
| C | wind + heating | as A |
| D | wind + cooling | as A |
| E | wind + buoyancy loss, eos 0 | as A, plus the eos-0 difference below |
| A2/B2/D2 | as A/B/D on a stretched grid (40 levels, 2 m surface, +6 %/level, 310 m, N0 0.005) | B2 round-off (temp 3e-15, tke 9e-12); A2/D2 as A |

pe_realistic vs pe_simple: 4e-10 or better in all uniform cases, 0 in B.

Deviations from tke.f90 found and fixed by this test (both packages):
kappaH=1 override where N2<0 with TKE on (removed); missing surface buoyancy
flux term P_diss_v(nz) = -(g/rho_0) forc_rho_surface and stale rho in N2
(configurations now hand `forc_temp_surface`/`forc_salt_surface` to the
model; rho evaluated once per step after vertical mixing, as pyOM2);
K_diss_E/L U/V->T averaging without metric weights (pe_realistic; now
ugrid_to_tgrid/vgrid_to_tgrid); N2 at the surface face taken from the halo
(now copied from the face below); mxl sweep order; TKE floor (pyOM2 has none;
tke_min default 0, tke_init 1e-12); kappaM_min/kappaH_min default 0; missing
face mask maskU(k)maskU(k+1) on the momentum diffusivity (spurious bottom drag
over topographic steps and half the dissipation lost into dry cells); missing
mask on the TKE diffusion coupling (TKE leaked into the ground);
pe_realistic p0 for the EOS built from zt+dzt/2 instead of the surface face
(off by up to dzt/2 on stretched grids); pe_simple eq_of_state 5 unusable
(referenced a non-existent zt). K_diss_v is restart state (see 5).

Deliberate differences to pyOM2, decided by Carsten:
- momentum splitting: pe applies the implicit friction to the AB-updated
  velocity, pyOM2 to u(tau) with du_mix added to the AB tendencies. Both first
  order; pe's solution is nearly dt-independent (u_top 0.0517/0.0501/0.0498 at
  dt 360/90/30), pyOM2's is not (0.0811/0.0574/0.0522); same limit.
- eos 0 (buoyancy as tracer): pe uses drho/db = -rho_0/g in the surface
  buoyancy TKE term; pyOM2's get_drhodT returns 0 there.
- no background diffusivity (pyOM2 gets it from IDEMIX, which pe lacks); a
  configuration sets kappaM_min/kappaH_min if wanted.

Energy of the implicit friction (both models, dt 360, 1 m cells): KE removed
by the friction / K_diss_v booked = pe 1.17-1.25 (backward-Euler term
|u+ - u|^2/(2 dt) is not booked), pyOM2 0.83-0.87 (K_diss_v formed with
u(taup1) while the friction acted on u(tau)). Neither exact at finite dt.

## 3b. Mixing-length options `mxl_choice` (Sep 12 2026)

`tke.py` (both packages) offers two sets of limits for l = sqrt(2 tke)/N:

| | `mxl_choice = 1` (default) | `mxl_choice = 2` |
|---|---|---|
| source | pyOM2 tke.f90 `tke_mxl_choice = 2` | NEMO 4.2 `nn_mxl = 3`, `ln_mxl0 = .true.` (manual 9.1.3) |
| order | downward sweep with the UNLIMITED surface value, then the surface W point limited to mxl_min + z0_surf + dz(top), then the upward sweep | the surface value is set FIRST, so it bounds the whole near-surface profile |
| lengths | one l for the diffusivity and the dissipation | l_k = sqrt(l_up l_dwn) for kappaM, l_eps = min(l_up, l_dwn) for the dissipation |
| surface | mxl_min + z0_surf + dz(top) | Charnock l0 = kappa beta \|tau\|/(g rho_0), kappa 0.4, beta 2e5, floor mxl_min0 = 0.04 m; with z0_surf > 0 instead l0 = kappa z0_surf (NEMO `ln_mxhsw`) |
| bottom | the sweep starts from the dry cells (0) | the dry cells are seeded with mxl_min0 (NEMO's bottom condition) |

pyOM2's `tke_mxl_choice = 1` (bounded by the distance to the boundaries level
by level) is NOT ported: every pyOM2 configuration sets choice 2, so it was
never exercised.

| test | what | result |
|---|---|---|
| operator | `_mxl_nemo` vs an independent implementation in NEMO's own index order, random tke/N^2, topography, one unstable level | 0.0 (exact), l_k >= l_eps everywhere |
| reduction | `_mxl_nemo` and `surface_mxl` of pe_simple vs pe_realistic, uniform dz, same state, shallow column | 0.0 for l0, l_k, l_eps |
| regression | global4 ocean 1 d, `mxl_choice = 1` with the new tke.py vs the old package and setup | bit-identical in every restart field |
| identities (see 4) | global4 ocean 5 d, `mxl_choice = 2` | TKE budget 3.0e-18 W/m2, conversion 1.1e-17, harmonic friction 1.5e-17, heat/salt unchanged |
| restart + decomposition (see 5, 6) | 1 d: 1 PE in one go vs 2 PEs (y-split) with a restart in the middle, `mxl_choice = 2` | bit-identical, restart fields and monthly means |
| cost | same 1 d run, 1 PE, 24 steps | choice 1 9.1 s, choice 2 10.8 s wall |

DECOMPOSITION BUG FOUND BY THIS TEST. With choice 2 the run first depended on
the PE layout: after three steps tke differed by 4e-6 relative at exactly the
PE boundary row (choice 1 was exact). The mixing length of the HALO row enters
the U/V -> T averages of `implicit_vfric`, and a configuration sets taux/tauy
without a halo exchange, so l0 was wrong there. `surface_mxl` therefore applies
`apply_bc` to the finished l0 -- ONE 2-D exchange per step, and only for
choice 2. A configuration whose stress carries a valid halo can set
`tau_halo_ok = True` to skip it, or hand over the T-point modulus as
`tau_modulus`; note that the closure runs BEFORE the configuration's
set_forcing in a step, so such a field must be restart state, otherwise the
first step after a restart uses a stale value (measured: 0.14 m/s after 1 d).

Single column, 5 d, |tau| = 0.1 Pa, alpha_CB = 100, N = 0.02 1/s, alpha_tke =
30, dz = 1 m (the geometry of the section 3 cases, injection only): choice 2
gives a SHORTER length at the surface (0.80 m against 1.00 m) and everywhere
above 15 m, so kappaM is 20-30 % smaller and the TKE 4 % lower at the surface
but 60 % lower at 13 m; the column integral is 12 % smaller. Below the
injection layer choice 2 keeps l_eps at mxl_min0 = 0.04 m instead of letting
it fall to mxl_min, which leaves a floor of kappaM ~ 1e-9 m2/s there.

In the global 4 degree run (5 d, alpha_CB = 100, alpha_tke = 30) the same
shows up as a REDISTRIBUTION: the surface TKE is 24 times larger with
choice 2 (median 2.3e-2 against 9.4e-4 m2/s2, the small l0 suppresses the
dissipation in the 10 m top cell), while one to three levels below it is 0.35,
0.30 and 0.03 times the choice-1 value. Choice 2 thus concentrates the energy
near the surface and mixes less deeply -- the same direction as reducing
alpha_tke, and consistent with NEMO needing its Langmuir and below-ML
penetration terms to reach deep enough.

## 4. Discrete identities in the coupled global4 run (global4_setup.py, Sep 9 2026)

The ocean class books, every step, and prints daily ("KE-budget ocn" and
"identities ocn" lines, `global4_energy.txt`):

| identity | residual in the run |
|---|---|
| momentum advection: u.adv = -1/2 sum u^2 (div T) with the flux form's own transports | 0 to 4e-22 W/m2 |
| Coriolis work | 1e-19 |
| vortex-force work | 1e-24 |
| surface pressure: KE_ps + PEs_dot = 0 | exact |
| conversion: u.(-grad p_h) = sum_faces w (p_h(k+1)-p_h(k)) A_t - w_surf p_h(surf) A_t | 1e-16 |
| harmonic friction: KE change = stencil adjoint -A_h sum(grad u)^2 (metric weights) + dt/2 |F|^2 | 3e-17 |
| vertical friction: KE change = -int K_diss_v dV - int |u+ - u|^2/(2 dt) dV | 1e-14..1e-16 |
| TKE: sum V (tke+ - tke)/dt = sum V (K_diss_v - P_diss_v) + A (forc_surf + tke_surf_corr) - sum V c_eps sqrt(tke)/mxl tke+ | 2e-17 |
| heat content change = AB-weighted applied surface flux + shortwave column integral | 6e-16 (flux units) |
| salt content change = AB-weighted applied surface flux | 4e-15 |

Non-closing terms, all measured and named, none a bug:
- LEAK of the tendency-based KE budget (dE/dt minus the re-evaluated terms):
  the AB/splitting difference between applied and re-evaluated tendencies,
  ~3 % of the largest terms after the initial adjustment.
- KE_adv ~1e-6 W/m2: the linear free surface's top-layer divergence (the
  identity above shows it is exactly -1/2 sum u^2 div T).
- vertical friction: 20-45 % of the KE it removes is the backward-Euler term
  (dt 3600 s, 10 m surface cell, K dt/dz^2 > 1); pyOM2 has the same term.
- harmonic friction: forward-Euler under-dissipation 4 % of D_h at dt 3600.
- shortwave leak 2.2e-8 K m/s (0.3 % of the surface heat flux): light
  absorbed below the bottom of shallow columns is lost (pyOM2 setup1 same).

## 5. Restart (test_restart.sh + compare_restart.py, Sep 8 2026)

Three runs from one cold start: A (2 d + 1 step), A2 (identical repeat), B
(2 d, restart, 1 step). Result: A = A2 bit-identical (the run is
reproducible); A = B bit-identical after adding `K_diss_v` to the restart
(the next step's Prandtl number is built from it: 6e-2 K in temp after one
step without it) and after the implicit_vfric halo fix (with mix_eulerian the
u_S part added back after the solve left the outer halo column unrefreshed;
the restart's apply_bc had masked the bug). Same K_diss_v rule applied to
global4_pe.py, pop14_pe.py, the acc1 and HS couplers.

## 6. Decomposition invariance (test_decomp.sh, Sep 9 2026)

Same 1 d + 1 step from one cold start, layouts wave x ocean: A 4x4/1x2,
B 2x8/2x1, C 4x4/1x1, D 2x8/1x2, E 4x4/2x1; plus A1/E1 = A/E for one step.

- A = C bit-identical (ocean y-split vs single PE); A = D bit-identical (wave
  layout, coupler): no decomposition-dependent operation on those paths.
- A vs E = A vs B to the last digit: the whole layout dependence is the ocean
  x-split; after ONE step p_s 1e-15, u/v 7e-16, tke 4e-20 relative with
  temp/salt/wave fields exactly 0, identical CG iteration counts (87, 62,
  1655 total) -> summation order of the CG dot products, not a halo/mask
  error. Amplifies to 1e-11 over 25 steps.

## 7. Rules that the tests established

- pe_realistic module code reduces with the model's `self.global_sum` /
  `self.global_max` (model communicator), never `OM.global_*` (COMM_WORLD),
  otherwise the coupled runs deadlock.
- Restart state of the TKE closure: tke AND K_diss_v.
- Density is evaluated once per step, after vertical mixing, before the TKE.
- Diagnostics, instrumentation and energy bookkeeping live in the
  configuration or the coupler (instance-level overrides), not in the package.
- Rigid lid (both free-surface flags False) does not converge on a cyclic
  domain with land; use the implicit free surface.

## 8. Production-run findings that are tests in themselves (POP 1/10 deg, Sep 10 2026)

- I/O at 1536 ranks: per-rank netCDF hyperslab reads of contiguous float32
  files are thousands of 360-byte strided reads on one shared file (15 min
  setup); rank-0 read of one level + `MPI_Scatterv` (pop.py `BlockScatter`)
  brings the setup to 41 s. `snap_nc4.diagnose` is a serial round-robin over
  the ranks (barrier, open r+, strided write, close, per rank), minutes per
  snapshot; `snap_raw` (per-rank .npy, `raw2nc.py` reassembles, float32) is
  the output for large rank counts.
- `epsilon` of the surface-pressure CG is max|delta p_s| per iteration (not
  a squared residual): 1e-18 costs ~650 iterations per step, 1e-9 (pyOM2's
  congr_epsilon) is the intended value.
- The 19 h blow-up of the first POP run (CG iterations climbing over three
  steps, then NaN) is consistent with the missing Coriolis force (one-level
  shelf cells accelerate freely to CFL ~0.8 in the 190 m cells at 89 N);
  friction, time stepping (pyOM2 has AB2 with AB_eps only; pe's (1.6,-0.6,0)
  is AB2 with eps 0.1) and the restart (all three du/dv/dtemp/dsalt levels
  and p_s levels stored) were checked and cleared. Rerun with Coriolis
  pending.

## 9. Not tested / not covered

The split-explicit path in a long realistic run (POP rerun pending), and
the shelf-strip difference in 2b.
JAX layout of pe_realistic; TEM, EKE, IDEMIX (not in pe); the PE side of the
energy cycle (dynamic enthalpy, cabbeling, isoneutral/skew dissipation);
mirror-symmetry test for pe; a decomposition test with n_pes_z > 1 (pe has
none).
