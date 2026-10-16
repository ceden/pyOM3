"""
Cartesian reduction gate for wim_realistic.

With coord_degree=False and uniform spacings (dxt=dx, dyt=dy) every metric
factor is 1 (cost=cosu=1, tantr=0) and every grid array is the constant
spacing, so wim_realistic must reproduce wim_simple to round-off, term by
term. Exercised together in one forced run: wind input + dissipation + DIA
(numpy backend), a vertically sheared current through the full Doppler
machinery (advection, radial correction, refraction ddot, frequency
refraction fdot), the superbee fluxes, the AB3 loop with the action floor,
and -- in separate variants -- the energy fixer (energy_consistent), the
open radiating coasts with the outflow book-keeping, and specular wall
reflection with land.

Run from the pyOM3 parent directory:
    python3 -m pyOM3.wim_realistic.development.reduction_test
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', '..')))
import numpy as np
import pyOM3 as OM


def build(pkg, par, extra):
    if pkg == 'simple':
        from pyOM3.wim_simple.model import model
    else:
        from pyOM3.wim_realistic.model import model
    p = dict(par); p.update(extra)
    M = model(p, halo_points=1)
    M.setup()
    return M


def run(M, nsteps, with_current=True):
    ny, nx = M.ny_loc, M.nx_loc
    h = M.halo
    # wind: uniform speed, direction turning slowly in y
    yy = np.arange(ny) / max(ny - 1, 1)
    M.ustar = np.zeros((M.Ny, M.Nx)); M.ustar[h:-h, h:-h] = 0.5
    M.udir  = np.zeros((M.Ny, M.Nx))
    M.udir[h:-h, h:-h] = 0.5 * np.pi * (1.0 + 0.2 * yy)[:, None]
    M.ustar = M.apply_bc(M.ustar); M.udir = M.apply_bc(M.udir)

    if with_current:
        # sheared jet current through the Doppler machinery
        nz, dz = 30, 5.0
        z = (np.arange(nz) + 0.5) * dz            # height above bottom
        prof = np.exp((z - nz * dz) / 40.0)       # surface intensified
        xj = np.arange(nx) / max(nx - 1, 1)
        UJ = 0.8 * np.exp(-((yy[:, None] - 0.5) / 0.25) ** 2) \
                 * (1.0 + 0.1 * np.sin(2 * np.pi * xj)[None, :])
        UL = prof[:, None, None] * UJ[None, :, :]
        VL = 0.3 * np.roll(UL, 3, axis=-1)
        M.ocean_U,   M.ocean_V   = M.calc_doppler_vel(dz, nz, UL, VL)
        M.ocean_Udf, M.ocean_Vdf = M.calc_doppler_vel_df(dz, nz, UL, VL)

    # seed
    sl = slice(h, -h)
    fseed = np.exp(-((M.ft[sl] - 0.1) / 0.03) ** 2)
    dseed = np.maximum(np.cos(M.theta[sl] - np.pi / 2), 0.0) ** 2
    M.A[sl, sl, sl, sl] = 1e-2 * dseed[:, None, None, None] * \
                          fseed[None, :, None, None]
    M.A = M.apply_bc(M.A)

    tA, tB, tC = 1., 0., 0.
    tau, taum1, taum2 = 2, 1, 0
    for n in range(nsteps):
        M.time_step(tA, tB, tC, tau, taum1, taum2)
        tau, taum1, taum2 = (tau + 1) % 3, (taum1 + 1) % 3, (taum2 + 1) % 3
        if tA == 1. and tB == 0. and tC == 0:
            tA, tB, tC = 1.5 + 0.05, -0.5 - 0.05, 0.
    return M


def compare(name, extra, nsteps=20, with_current=True):
    par = dict(nx=16, ny=18, nf=28, nd=24, f0=0.04, df0=0.1,
               dt=60.0, dx=5e3, dy=5e3,
               periodic_in_x=True, periodic_in_y=True,
               snonlin_backend='numpy', implicit_dissipation_only=True)
    A = run(build('simple',    par, extra), nsteps, with_current)
    B = run(build('realistic', par, extra), nsteps, with_current)
    da = np.max(np.abs(A.A - B.A))
    ref = np.max(np.abs(A.A))
    rel = da / ref
    ok = rel < 1e-14
    print('%-34s max|dA| = %.3e  rel = %.3e  %s'
          % (name, da, rel, 'PASS' if ok else 'FAIL'))
    if getattr(A, 'open_coasts', False):
        la, le, lo = A.coastal_loss(); lb = B.coastal_loss()
        dl = max(abs(la - lb[0]), abs(le - lb[1]), abs(lo - lb[2]))
        print('%-34s coastal-loss diff = %.3e  %s'
              % ('', dl, 'PASS' if dl <= 1e-12 * max(abs(la), 1e-30) else 'FAIL'))
        ok = ok and dl <= 1e-12 * max(abs(la), 1e-30)
    return ok


if __name__ == '__main__':
    allok = True
    allok &= compare('periodic, current',            {})
    allok &= compare('periodic, current, energyfix', {'energy_consistent': True})
    allok &= compare('closed y, reflection',
                     {'periodic_in_y': False, 'enable_reflection': True})
    allok &= compare('closed y, open coasts',
                     {'periodic_in_y': False, 'open_coasts': True,
                      'energy_consistent': True})
    allok &= compare('no current, no fix',           {}, with_current=False)
    print('reduction test:', 'ALL PASS' if allok else 'FAILURES')
    sys.exit(0 if allok else 1)
