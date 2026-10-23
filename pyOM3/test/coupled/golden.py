"""Test 9: golden regression of the shared coupling and forcing code
(setups/coupled_common, setups/forcing_common): a checksum of a short run
for every code path a configuration can switch on, against the stored
value. The run is config.small_ocean + config.step: the synthetic forcing
source through surface_forcing_fields / set_forcing, and coupled, the
synthetic wave fields through set_forcing_direct, the vortex force, the
TKE hooks and the diagnostics. 8 steps each, tolerance 1e-13.

Every option that changes the default path has its own case, so a change
to one branch shows in one line:
  forced            ocean alone, 'era5' stress mode, all eight forcing fields
  forced_relwind    relative-wind stress, ice stress factor 0.3, SW_BOTTOM,
                    SSS piston, the light KE budget and 6-hourly online means
  forced_nosolar    a source without qsol, dqdt, sst_star, emp (the absent-key
                    rule: every one of those terms off)
  coupled           the wave fields, the vortex force, phi_oc in the closure
  coupled_full      + the direct stress branch, the Stokes shear force and
                    NEMO's mixing length with z0 = HS_FACTOR Hs
  coupled_stress    the depth-distributed stress (use_stress_work); KE budget
                    off, see README (ke_budget reads taux there)
`python3 golden.py --update` rewrites golden.json after an intended change.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_ocean, step, state, checksum, line, HERE, OM
GOLD = os.path.join(HERE, 'golden.json')
CASES = (('forced', dict(coupled=False)),
         ('forced_relwind', dict(coupled=False, relwind=True, ice=0.3, sw_bottom=True, piston=0.5/86400.0,
                                 ke_budget=2, average='6h')),
         ('forced_nosolar', dict(coupled=False, omit=('qsol', 'dqdt', 'sst_star', 'emp'))),
         ('coupled', dict(coupled=True)),
         ('coupled_full', dict(coupled=True, direct=True, shear=True, mxl_choice=2)),
         ('coupled_stress', dict(coupled=True, stress_work=True, ke_budget=0)))


def run(nsteps=8, update=False):
    cs = {}
    for name, kw in CASES:
        cs[name] = checksum(state(step(small_ocean(tag=name, **kw), nsteps)))
    if update or not os.path.exists(GOLD):
        json.dump(cs, open(GOLD, 'w'), indent=1)
        print('9 golden checksums %s in %s' % ('rewritten' if update else 'created', GOLD))
        return True
    ref = json.load(open(GOLD)); ok = True
    for name, _ in CASES:
        if name not in ref:
            ok &= line('9 golden %s: no stored checksum (run --update)' % name, 1.0, 0.0); continue
        d = max(abs(cs[name][k] - ref[name][k])/max(abs(ref[name][k]), 1e-300) for k in ref[name])
        # golden.json is a NUMPY reference; on the jax backend the closure's
        # round-off gives 1e-12 (jax_backend.py is the jax test, 1e-10)
        ok &= line('9 golden regression, %s, %d steps' % (name, nsteps), d, 1e-10 if OM.jax else 1e-13)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run(update='--update' in sys.argv) else 1)
