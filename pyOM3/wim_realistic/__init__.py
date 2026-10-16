"""
 Realistic WIM model: wim_simple generalised to variable / pseudo-spherical
 grids (grid.py) with the metric-aware propagation and coupling interface.

 The grid-metric-free infrastructure (setup, halo exchange, source terms,
 time stepping) lives in wim_simple and is imported from there directly
 by boundary.py, main.py and model.py.
"""
