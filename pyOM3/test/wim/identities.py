"""Test 2: discrete identities in a short run on the small basin (wim family).
Each one is an identity of the discrete equations, so it holds to round-off.

Propagation alone (sources=False in config.step: the model's own
mk_advection / add_propagation with the AB weights and the outflow
book-keeping, no action floor, no source terms), wim_realistic on the sphere
with variable dxt, dyt, and wim_simple:

  action       sum A dd dft area + loss_A is constant: the flux form
               telescopes, x periodic, land and walls closed, the frequency
               axis closed, the open coasts booked by open_coast_loss
  energy       no current: sum omega A + loss_E is constant (fdot = 0)
  abs. energy  with a current and energy_consistent: sum Omega A + loss_Eabs,
               Omega = omega + k.U, is constant (energy_fix_flux)
  control      the same current WITHOUT the energy fixer must NOT close,
               otherwise the line above proves nothing
  reflection   wall reflection returns what reaches the wall: action constant

Interface, full model after a few forced steps:

  stress       sum_k tau(z_k) dz_k of calc_stress_work = calc_stress (closed
               bottom), for the WAMDI and the input-based tail flux and for
               input_scheme janssen, on the variable dzt of 'realistic'
  exchange     sum_k dz_k u_L . tau(z_k) = sum_f dft U(f) . tau(f): the Doppler
               projection is the adjoint of the stress profile, so the energy
               the ocean receives from the wave stress is what the waves lose
               to the current (resolved bins; the analytic tail and the cascade
               credit have no U(f))
  Stokes       sum_k u_S dz_k = sum omega E s dd dft + (2 pi/3) f_c^2 sum E_c s dd
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import small_model, step, line, action_total, abs_energy_total, NZ
import pyOM3 as OM


def _drift(kw, total, loss, nsteps=20):
    m = small_model(sources=False, **kw)
    t0 = total(m); step(m, nsteps)
    la, le, lo = m.coastal_loss()
    return abs(total(m) + dict(A=la, E=le, O=lo)[loss] - t0)/abs(t0), (la if loss == 'A' else lo)/abs(t0)


def _interface(m):
    """column sum of the stress profile against the surface stress, the
    exchange adjointness, the Stokes transport"""
    h = m.halo; sl = slice(h, -h)
    dz = np.broadcast_to(np.asarray(m._dz, float), (NZ,))[:, None, None]
    tau_s = np.asarray(m.calc_stress()); tau_z = np.asarray(m.calc_stress_work(m._dz, NZ))
    col = float(np.max(np.abs((tau_z*dz[None]).sum(1) - tau_s))/np.max(np.abs(tau_s)))
    # Stokes transport
    E = np.asarray(m.E); om = np.asarray(m.omega_t); dft = np.asarray(m.dft)
    us = np.asarray(m.calc_stokes(m._dz, NZ))
    # the f^-5 tail E_c (f/f_c)^-5 is anchored at the last real bin f_c =
    # ft[-h-1] and starts at that bin's upper face f_u = fu[-h-1] (the
    # resolved sum covers the bin up to f_u); its transport is
    # 2 pi int_{f_u}^inf f E df = (2 pi/3) f_c^5/f_u^3 E_c
    fc = float(m.ft[-h-1]); fu = float(m.fu[-h-1])
    ref = []
    for w in (np.asarray(m.sinth), np.asarray(m.costh)):
        W = w[sl, None, None, None]
        res = np.sum(E[sl, sl, sl, sl]*W*om[None, sl, None, None]*dft[None, sl, None, None], axis=(0, 1))*m.dd
        ref.append(res + 2*np.pi/3*fc**5/fu**3*np.sum(E[sl, -h-1, sl, sl]*W[:, 0], axis=0)*m.dd)
    ref = np.array(ref)
    sto = float(np.max(np.abs((us*dz[None]).sum(1) - ref))/np.max(np.abs(ref)))
    # exchange: resolved bins only -> switch the analytic tail off by emptying
    # its anchor, the frequency ghost, and the cascade credit of the input
    # tail (deposited in the top cell, it has no U(f)) by hiding S_nl; both
    # restored afterwards
    keep = m.E.copy(); m.E = OM.modify_array(m.E, (slice(None), slice(-h, None)), 0.0)
    snl = m.__dict__.pop('S_nl', None)
    tau_z = np.asarray(m.calc_stress_work(m._dz, NZ))
    ocean = np.sum(dz*(m._UL*tau_z[0] + m._VL*tau_z[1]), axis=0)
    Sd = np.asarray(m.gamma_diss)[None]*np.asarray(m.E)
    if getattr(m, 'tail_flux', 'wamdi') == 'input':
        Sd = np.where(np.asarray(m.freq_mask)[None], Sd, -np.asarray(m.gamma_in)*np.asarray(m.E))
    tx = -np.sum(Sd[sl]*np.asarray(m.sinth)[sl, None, None, None], axis=0)*m.dd*om[:, None, None]
    ty = -np.sum(Sd[sl]*np.asarray(m.costh)[sl, None, None, None], axis=0)*m.dd*om[:, None, None]
    wave = np.sum((np.asarray(m.ocean_U)*tx + np.asarray(m.ocean_V)*ty)[sl, sl, sl]*dft[sl, None, None], axis=0)
    m.E = keep
    if snl is not None:
        m.S_nl = snl
    exc = float(np.max(np.abs(ocean - wave))/np.max(np.abs(wave)))
    return col, sto, exc


def run():
    ok = True
    for kind in ('realistic', 'simple'):
        d, l = _drift(dict(kind=kind, energy_consistent=False), action_total, 'A')
        ok &= line('2 action + outflow constant, %s (outflow %.0e)' % (kind, l), d, 1e-13)
        d, l = _drift(dict(kind=kind, current=False, energy_consistent=False), abs_energy_total, 'E')
        ok &= line('2 energy + outflow constant, no current, %s' % kind, d, 1e-13)
        d, l = _drift(dict(kind=kind), abs_energy_total, 'O')
        ok &= line('2 absolute energy + outflow, current, fixer, %s' % kind, d, 1e-13)
        d, l = _drift(dict(kind=kind, open_coasts=False), abs_energy_total, 'O')
        ok &= line('2 absolute energy, closed coasts, fixer, %s' % kind, d, 1e-13)
        d, l = _drift(dict(kind=kind, energy_consistent=False), abs_energy_total, 'O')
        good = d > 1e-8
        print('%-56s %10.2e  (must not close)     %s' % ('2 control: the same without the fixer, %s' % kind, d, 'PASS' if good else 'FAIL'))
        ok &= good
        d, l = _drift(dict(kind=kind, open_coasts=False, reflection=True, energy_consistent=False), action_total, 'A')
        ok &= line('2 action constant under wall reflection, %s' % kind, d, 1e-13)
    for name, kw in (('wamdi tail', {}), ('input tail', dict(tail_flux='input')), ('janssen', dict(janssen=True))):
        col, sto, exc = _interface(step(small_model(**kw), 5))
        ok &= line('2 stress: column sum = surface, %s' % name, col, 1e-12)
        ok &= line('2 exchange: ocean side = wave side, %s' % name, exc, 1e-12)
    ok &= line('2 Stokes transport = spectral integral + tail', sto, 1e-12)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
