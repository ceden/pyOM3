"""
sw_simple: the 2-D shallow-water model of the Lagrangian-filtering draft
(sw_model.py, sw_particles.py of the grav2D code), dimensional, ported to
pyOM3 (Nov 2026).

  model.py      the model: state, tendencies, time steps, loop
  particles.py  Lagrangian particles (Euler, RK3, AB2, AB3, AB4)
  snapshot.py   netCDF snapshots (snap_nc4)

The balance methods of the model are in pyOM3/diagnostics/balance
(warn_sw_simple, kernel_sw_simple, optimal_sw_simple). The test against
the original code is pyOM3/test/balance/sw_check.py.
"""
