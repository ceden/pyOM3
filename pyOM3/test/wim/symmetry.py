"""Test 12: symmetries of the spectral discretisation (wim family). The
counterpart of the lid mirror test of the nh family: an index error in the
direction bookkeeping of the DIA quadruplets, the wind input, the
dissipation or the tail is invisible to every conservation identity and
shows here. Homogeneous case (config.homogeneous_model: doubly periodic, no
island, uniform wind and spectrum), where the propagation tendency vanishes
exactly and the source terms act alone; propagation is tested separately
with a packet.

  mirror       wind along theta = 0 (north): the spectrum must stay
               symmetric under theta -> -theta (bin d <-> nd+1-d)   round-off
  rotation     the wind turned by one directional bin and the initial
               spectrum rolled by one bin give the rolled spectrum (theta is
               periodic, nothing in the sources knows the absolute
               direction)                                          round-off
  x <-> y      a swell packet on the square doubly periodic grid against
               the packet on the transposed field with the directions
               reflected, theta -> pi/2 - theta, propagation only  round-off
  positivity   strong wind on a near-empty sea: the action floor must not
               be needed (clip_mass = 0) and E >= 0 after the source step
"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import small_model, homogeneous_model, step, line, NX, NY, ND
import pyOM3 as OM


def mirror(A):
    """theta -> -theta on the padded (Nd, ...) array: interior bin d <-> nd+1-d"""
    return np.asarray(A)[::-1]


def interior(A):
    return np.asarray(A)[1:-1, 1:-1, 1:-1, 1:-1]


def run(nsteps=12):
    ok = True
    # mirror symmetry
    m = homogeneous_model(ustar=0.6, udir=0.0); step(m, nsteps)
    A = np.asarray(m.A)
    ok &= line('12 theta mirror symmetry, wind north, %d steps' % nsteps,
               float(np.abs(interior(mirror(A)) - interior(A)).max()/interior(A).max()), 1e-13)
    # rotation by one bin
    m1 = homogeneous_model(ustar=0.6, udir=0.0)
    m2 = homogeneous_model(ustar=0.6, udir=0.0)
    m2.udir = m2.udir + m2.dd
    m2.A = m2.apply_bc(OM.np.roll(m1.A, 1, axis=0)); m2.E = m2.A*m2.omega_t[None, :, None, None]
    step(m1, nsteps); step(m2, nsteps)
    rolled = np.asarray(m1.apply_bc(OM.np.roll(np.asarray(m1.A), 1, axis=0)))
    d = float(np.abs(interior(rolled) - interior(m2.A)).max()/interior(m1.A).max())
    ok &= line('12 rotation by one directional bin, %d steps' % nsteps, d, 1e-13)
    # x <-> y: propagation only, square doubly periodic grid. Transposing
    # x and y reflects the directions, theta -> pi/2 - theta (north <-> east):
    # interior bin d -> nd/4 + 1 - d (mod nd)
    n = max(NX, NY)
    refl = lambda d: (ND//4 + 1 - d) % ND
    pk = {}
    for name in ('north', 'east'):
        mm = homogeneous_model(ustar=0.0, udir=0.0, amp=0.0, nx=n, ny=n)
        mm._sources = False
        h = mm.halo; sl = slice(h, -h)
        ig = np.arange(n)[None, :]*np.ones((n, 1)); jg = np.arange(n)[:, None]*np.ones((1, n))
        ft, th = np.asarray(mm.ft)[sl], np.asarray(mm.theta)[sl]
        fseed = np.exp(-((ft - 0.09)/0.03)**2)
        lobe = lambda t: np.exp(-((np.angle(np.exp(1j*(t - 0.35))))/0.3)**2)     # a lobe at 20 deg east of north
        dseed = lobe(th) if name == 'north' else lobe(0.5*np.pi - th)
        pack = np.exp(-((ig - 4.0)/2.0)**2 - ((jg - 3.0)/2.0)**2)
        if name == 'east':
            pack = pack.T
        A = np.zeros((mm.Nd, mm.Nf, mm.Ny, mm.Nx))
        A[sl, sl, sl, sl] = dseed[:, None, None, None]*fseed[None, :, None, None]*pack[None, None]
        mm.A = mm.apply_bc(OM.np.asarray(A)); mm.E = mm.A*mm.omega_t[None, :, None, None]
        step(mm, nsteps)
        pk[name] = interior(mm.A)
    north = pk['north'][[refl(d + 1) - 1 for d in range(ND)]].transpose(0, 1, 3, 2)
    ok &= line('12 x <-> y symmetry of propagation (packet, %d steps)' % nsteps,
               float(np.abs(north - pk['east']).max()/pk['east'].max()), 1e-13)
    # positivity
    m = homogeneous_model(ustar=1.5, udir=0.0, amp=1e-6); clip = 0.0; emin = 0.0
    for _ in range(nsteps):
        step(m, 1); clip = max(clip, m.clip_mass)
        emin = min(emin, float(interior(m.E).min()))
    ok &= line('12 positivity: action floor not needed under strong wind (clip_mass)', clip, 0.0)
    ok &= line('12 positivity: min E after the source step', -emin, 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
