"""Test 6: checksum of a short default run (wim_realistic on the sphere, numpy
backends) against the stored value.
`python3 golden.py --update` rewrites golden.json after an intended change."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, checksum, line, HERE
GOLD = os.path.join(HERE, 'golden.json')


def run(nsteps=12, update=False):
    cs = checksum(state(step(small_model(), nsteps)))
    if update or not os.path.exists(GOLD):
        json.dump(cs, open(GOLD, 'w'), indent=1)
        print('6 golden checksums %s in %s' % ('rewritten' if update else 'created', GOLD))
        return True
    ref = json.load(open(GOLD))
    d = max(abs(cs[k] - ref[k])/max(abs(ref[k]), 1e-300) for k in ref)
    return line('6 golden regression, default settings, %d steps' % nsteps, d, 1e-12)


if __name__ == '__main__':
    sys.exit(0 if run(update='--update' in sys.argv) else 1)
