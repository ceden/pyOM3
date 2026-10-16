# wim_realistic — wim_simple on the pseudo-spherical grid

Sibling of wim_simple exactly as pe_realistic is of pe_simple: variable
spacings `dxt(x)`, `dyt(y)` and, with `coord_degree=True`, the
pseudo-spherical metric of pyOM2 (`cost`, `cosu`, `tantr`,
`degtom = R/180·π`, R = 6370 km). The spectral axes (frequency,
direction) are untouched. With `coord_degree=False` and uniform spacings
the model reproduces wim_simple **bit-identically** (energy-fixer variants
to 9e-16: its arithmetic is reordered by the area factors).

## What changed relative to wim_simple

1. **grid.py** (new): horizontal-only port of pe_realistic/grid.py
   (pyOM2 `u_centered_grid` / `calc_grid`), local layout from the wave
   model's own decomposition (`xs_pe`/`ys_pe`). Trailing-axis broadcast
   views (`(Ny,1)`/`(1,Nx)`) work against both the `(Nd,Nf,Ny,Nx)` state
   and the `(Nf,Ny,Nx)` Doppler fields. `dx`/`dy` are tripwired to None.
2. **Metric flux divergence** (pyOM2/pe_realistic tracer convention):
   the y flux carries `cosu` at the face; both horizontal divergences
   divide by `cost` times the spacing. Applies to superbee, upwind, the
   open-coast replacement flux and the divergence in `add_propagation`.
   Face-integrated transports are then `fx·dyt·dd·dft` (x face, meridian
   arc) and `fy·dxt·dd·dft` (y face, `cosu` already inside) — used by
   `open_tables`/`open_coast_loss`.
3. **Spatial gradients** of the refraction terms via the metric
   operators: `d/dx → (R cosφ)^-1 d/dλ`, `d/dy → R^-1 d/dφ`, with
   centre-to-centre distances from `dxu`/`dyu` (2nd order on the
   stretched grid).
4. **Great-circle terms.** Derived exactly (sympy, residual 0) from the
   canonical ray equations in (λ,φ) with conjugate momenta
   (k_e R cosφ, k_n R) and Ω = σ(k) + k·U, deep water:

       θ̇ = θ̇_Cartesian-form(spherical gradients) + (c_g + s·U) sinθ tanφ/R
       σ̇ = σ̇_Cartesian-form(spherical gradients) − (σ/2)(U·s⊥) sinθ tanφ/R

   with s = (sinθ, cosθ), s⊥ = ds/dθ. ECMWF (IFS doc Part VII, eq.
   2.25c) keeps only the c_g part of the turning and no σ̇ term — a
   truncation at O(U tanφ/R) ≈ 2e-7 s⁻¹ (U = 1 m/s, 60°); the exact
   forms cost nothing and are implemented. The turning is evaluated at
   the **theta faces** (`sinthu`), same convention and same
   mirror-symmetry reason as the refraction rate; the σ̇ term at theta
   centres like fdot. xdot/ydot are unchanged. No depth refraction
   (deep water), matching the Cartesian model.
5. **energy_fix_flux in absolute (area-integrated) units**: with varying
   cell area no common factor can be dropped; the per-face defect terms
   carry `dyt` (x faces), `dxt` (y faces, `cosu` inside the flux) and
   `area_t` (spectral faces), the denominator carries `area_t`. Same
   correction shape, same conditioning guard, same open-coast masking.
6. Snapshot coordinates from the true grid (degrees when
   `coord_degree`).

## File inventory: shared vs local

Shared from wim_simple by inheritance (single source of truth, no
copies): the grid-metric-free infrastructure -- `wim_simple.main`
(workspace, spectral grid, backend selection; `main.setup` here calls
`calc_grid` first and then `wim_simple.main.setup`), `wim_simple.boundary`
(halo exchange; `boundary` here subclasses it), `wim_simple.model` (time
stepping and the jitted source update), the WAMDI source pair and the
vertical/spectral coupling integrals of `wim_simple.source` and
`wim_simple.interface` (subclassed, see below), and the S_nl backends
including the fortran build. `__init__.py` holds only the package
docstring, there are no re-exports.

Local to wim_realistic: `grid.py` (new), `boundary.py`, `main.py`,
`propagation.py`, `snapshot.py` (each metric-touched), `source.py`
(subclass of `wim_simple.source`: the Janssen input + WAM4 dissipation
option and its tails), `interface.py` (subclass of
`wim_simple.interface`: the coupling integrals on an array dzt and the
sphere), and `model.py`, which stitches the class chain
`model(propagation, interface, source, wim_simple.model)`. In this MRO
the metric-aware methods of wim_realistic resolve first; the nominal
base chain `wim_simple.main`/`boundary` of the parent classes sits
behind it and is never used for geometry.

Not ported (deliberately, matching pe_realistic's scope): reduced
(coarsened-longitude) grids and any pole treatment — a global domain
must stop short of the poles; the zonal CFL shrinks with cosφ.

## Tests (development/)

- `reduction_test.py` — Cartesian reduction gate: five forced variants
  (currents+DIA+wind, energy fixer, wall reflection, open coasts with
  loss booking, no current) against wim_simple; bitwise except the fixer
  at ≤9e-16.
- `sphere_test.py` — on the sphere: (1) area-weighted action
  conservation to 1e-16 over 3 d; (2) packet follows the great-circle
  ray reference (RK4) to 0.15° position / 0.6° direction after 36° of
  arc at 5°×2°×15° resolution, Clairaut invariant drift −0.6%;
  (3) uniform state in (λ,φ,θ) — an exact continuous steady state — has
  a discrete residual converging at 2nd order (ratio 3.56 for 2×
  refinement); (4) absolute-energy closure Σ Ω A dV with a sheared
  current and `energy_consistent=True`: 2e-16, closed and open+booked.

The two U tanφ/R terms themselves are validated by the symbolic
derivation (machine-checked residual 0) and by vanishing in the
Cartesian gate; they are ~3 orders below the gradient refraction for
mesoscale currents and no test here isolates them dynamically.

## Source option: Janssen input + WAM4 dissipation (source.py, Sep 2026)

`input_scheme = 'wamdi'` (default, wim_simple's pair, bit-identical to
before) or `'janssen'`: the pre-2019 ecWAM pair (ecwam IPHYS=0, Janssen
1991 input with the wave-induced roughness `z0 = alpha u*^3/(g sqrt(u*^2 -
tau_w))`, Bidlot et al. 2007 dissipation in wavenumber form), constants of
ecwam `setwavphys.F90`. `source.py` is a mixin before `wim_simple.model` in
the MRO; it overrides `Sinput`, `Sdiss`, `frcutindex` and
`_tail_input_factors`, so `calc_stress`, `calc_phioc` and `calc_stress_work`
use the ecwam tail quadrature when the scheme is on. u* stays prescribed
(coupler); the z0-tau_w feedback is iterated `janssen_ncall=2` times per
step as ecwam's two `sinflx` calls. State between steps: `tau_w`, `z0`
(not in restarts; a restart starts from Charnock and recovers in one step).

Reference: `development/Makefile` (with `wrap.F90`, `yow_stubs.F90`,
`dummies.F90` and the `*.intfb.h` files in `development/`) compiles the
unchanged ecwam routines with f2py into `development/ecwref*.so`
(`make ECWAM=<clone of github.com/ecmwf-ifs/ecwam>`);
`development/janssen_ref_test.py` checks the port against them column by
column (growth rate, dissipation, spectral means, windsea mean frequency,
tail stress and energy flux, z0, cut-off bin, depth weights: all at
round-off) and runs a 24 h duration-limited growth with both schemes.
Known deviations from ecwam, all documented in `source.py`: wim's end-bin
frequency weights in the spectral means (0.05-0.5% on E, fm1, f1, kbar),
the tail anchored at the last real bin for tau_w and at the ghost bin for
the ocean stress (wim convention), the tail stress booked per direction
(ecwam drops the cross-wind part).
