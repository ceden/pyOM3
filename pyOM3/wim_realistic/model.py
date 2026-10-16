"""
 Brings together all components for the model.

 wim_realistic contributes the metric-aware propagation and interface and
 the source option input_scheme (source.py: Janssen/WAM4 pair); the
 time-stepping methods are wim_simple's, taken by inheriting
 wim_simple.model. MRO: propagation -> main -> boundary -> interface ->
 source -> wim_simple.model -> wim_simple.propagation -> wim_simple.source
 -> wim_simple.interface -> wim_simple.main -> wim_simple.boundary -> setup
 -> grid, so every method of this package wins.
"""
from   pyOM3.wim_realistic.propagation import propagation
from   pyOM3.wim_realistic.interface   import interface
from   pyOM3.wim_realistic.source      import source
from   pyOM3.wim_simple.model          import model as _model_simple


class model(propagation, interface, source, _model_simple):
      pass
