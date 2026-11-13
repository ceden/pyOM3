"""
Balance diagnostics of the pyOM3 models, ordered by method:
  warn.py      Warn et al. (1995), asymptotic to fourth order: warn_pe_simple, warn_nh_simple,
               warn_sw_simple
  kernel.py    truncated kernels of the Warn et al. orders: kernel_pe_simple, kernel_sw_simple
  band.py      frequency bands of the model's linear operator: band_pe_simple, band_pe_realistic
  optimal.py   optimal balance: optimal_pe_simple, optimal_pe_realistic, optimal_sw_simple
  window.py    a window of a pe_realistic raw snapshot as a band_pe_realistic model,
               and run_params.json, the parameters of the run beside its raw snapshots
  run_window.py  the driver of the band method on such a window:
               python3 -m pyOM3.diagnostics.balance.run_window <raw dir> <n> <lon0> <lon1> <lat0> <lat1> [order] ...
Each class is added to the model class bases, e.g.
  from pyOM3.diagnostics.balance.warn import warn_pe_simple
  class my_model(warn_pe_simple): ...
"""
