"""Test 6: checksums of short runs against the stored values, three paths:
rigid lid with the default settings, the LES path (superbee + TKE closure,
the Langmuir configurations), and periodic in z. Every new option must
leave them unchanged. `python3 golden.py --update` rewrites golden.json
after an intended change."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, checksum, line, HERE
GOLD = os.path.join(HERE, 'golden.json')
CASES = (('lid', dict()), ('les', dict(b_adv='superbee', les='tke')), ('periodic', dict(kind='periodic')))


def run(nsteps=10, update=False):
    cs = {name: checksum(state(step(small_model(**kw), nsteps))) for name, kw in CASES}
    if update or not os.path.exists(GOLD):
        json.dump(cs, open(GOLD, 'w'), indent=1)
        print('6 golden checksums %s in %s' % ('rewritten' if update else 'created', GOLD))
        return True
    ref = json.load(open(GOLD)); ok = True
    for name, _ in CASES:
        d = max(abs(cs[name][k] - ref[name][k])/max(abs(ref[name][k]), 1e-300) for k in ref[name])
        ok &= line('6 golden regression, %s, %d steps' % (name, nsteps), d, 1e-13)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run(update='--update' in sys.argv) else 1)
