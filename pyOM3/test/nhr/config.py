"""
nhr family (nh_realistic). The configuration is the pe family's
(test/pe/config.py: closed wind- and buoyancy-forced box, 16 x 12 x 10
cells, shelf and seamount, TKE closure) on the stretched CARTESIAN grid
(the spacings of the spherical one in metres; nh_realistic refuses the
pseudo-spherical grid) with the nh_realistic model in place of
pe_realistic, so that every pe test pattern applies unchanged and the
hydrostatic limit is the pe model on the same grid (kind 'stretched').

  small_model(eps_nh=1e-12, **opt)
                          pe_config.small_model('nh', **opt) with epsilon_nh = eps_nh
                          (the layout tests use 1e-16, see decomposition.py)
  small_model_pe(**opt)   pe_config.small_model('stretched', **opt), the reference
  step, state, gathered_state, line, global_index   from the pe config
                          (state carries p_nh in addition to FIELDS)
"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('pe_config', os.path.join(HERE, '..', 'pe', 'config.py'))
pe_config = importlib.util.module_from_spec(_spec)
sys.modules['pe_config'] = pe_config
_spec.loader.exec_module(pe_config)
step, state, gathered_state, line, global_index = (pe_config.step, pe_config.state,
                                                   pe_config.gathered_state, pe_config.line,
                                                   pe_config.global_index)
OM, FIELDS, NX, NY, NZ, DT = pe_config.OM, pe_config.FIELDS, pe_config.NX, pe_config.NY, pe_config.NZ, pe_config.DT


def small_model(eps_nh=1e-12, **kw):
    m = pe_config.small_model('nh', **kw)
    m.epsilon_nh = eps_nh
    return m


def small_model_pe(**kw):
    return pe_config.small_model('stretched', **kw)
