"""Test 1: the rigid-lid model against the periodic-in-z model on the
mirror-extended domain.

nh_simple has no second package to reduce to. What takes the place of
"pe_simple against pe_realistic" is the lid: a rigid-lid problem on nz
levels is the periodic problem on 2 nz levels with the state extended by
its mirror image (u, v even in z, w and b odd, the lids at the symmetry
planes). The periodic model has no lids to get wrong, so this is the test
that catches an error in the lid treatment of any operator -- the
counterpart of pe_simple having no metrics to get wrong. The periodic
model uses the 3D-FFT pressure solver, the lid model the 2D-FFT +
tridiagonal one, so the two solvers are compared at the same time.

    tolerance   1e-12 relative, all prognostic fields (p up to its gauge)

Unforced, without LES and with Smagorinsky (mirror symmetric by
construction). The mirror image of b is odd (gravity flips with z), which
implies b = 0 at the lid, whereas the lid model imposes zero buoyancy FLUX;
the two conditions differ, so b is run without explicit mixing (b_mix off,
les_mix_b off) and with the 2nd-order advection, whose lid flux vanishes
with w. The limited schemes and the TKE closure are outside the reduction:
superbee and upwind3 need a slope across the lid (the mirror has a jump
there), and the TKE closure's w-grid has a TKE point at the top lid and none
at the bottom (the ps3D convention, the surface point receives the
wave-breaking flux). Their lid treatment is covered by the mirror-symmetry
operator test (5) instead."""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, step, state, line, NZ, FIELDS
import pyOM3 as OM


def extend_T(a, sign):
    """T-grid field (Nx, Ny, nz+2) -> (Nx, Ny, 2nz+2), interior mirrored at the top lid"""
    nz = a.shape[2] - 2
    out = np.zeros(a.shape[:2] + (2*nz + 2,), a.dtype)
    out[:, :, 1:nz+1] = a[:, :, 1:nz+1]
    out[:, :, nz+1:2*nz+1] = sign*a[:, :, nz:0:-1]
    return out


def extend_W(a, sign):
    """w-grid field: face k (above cell k) -> faces k and 2nz-k; the lid faces stay zero"""
    nz = a.shape[2] - 2
    out = np.zeros(a.shape[:2] + (2*nz + 2,), a.dtype)
    out[:, :, 1:nz] = a[:, :, 1:nz]
    out[:, :, nz+1:2*nz] = sign*a[:, :, nz-1:0:-1]
    return out


def run(nsteps=6):
    ok = True
    for les in (None, 'smag'):
        for biharmonic in (False, True):
            kw = dict(les=les, forcing=False, b_mix=False, biharmonic=biharmonic)
            if les:
                kw['les_mix_b'] = False
            # p3dfft holds ONE plan per process: the lid model must be finished
            # before the model of the other size is created
            m = small_model('lid', **kw)
            init = {f: np.array(getattr(m, f)) for f in ('u', 'v', 'w', 'b')}
            a = state(step(m, nsteps))
            mp = small_model('periodic', nz=2*NZ, **kw)
            mp.u = mp.apply_bc(OM.np.asarray(extend_T(init['u'], +1)))
            mp.v = mp.apply_bc(OM.np.asarray(extend_T(init['v'], +1)))
            mp.b = mp.apply_bc(OM.np.asarray(extend_T(init['b'], -1)))
            mp.w = mp.apply_bc(OM.np.asarray(extend_W(init['w'], -1)))
            for f in ('du', 'dv', 'dw', 'db'):
                setattr(mp, f, OM.np.zeros((3,) + mp.u.shape, OM.prec))
            b = state(step(mp, nsteps))
            d = 0.0
            for f in FIELDS:
                x, y = a[f], b[f][:, :, :NZ]         # lower half; w: faces 1..nz-1 free, face nz = lid in both
                if f == 'p':
                    x = x - x.mean(); y = y - y.mean()
                d = max(d, float(np.max(np.abs(x - y)))/max(float(np.max(np.abs(x))), 1e-300))
            ok &= line('1 reduction lid = periodic on the mirror-extended domain, %s%s'
                       % ('biharmonic' if biharmonic else 'harmonic', ', smag' if les else ''), d, 1e-12)
    return ok


if __name__ == '__main__':
    import sys; sys.exit(0 if run() else 1)
