"""
Janssen/WAM4 port (wim_realistic/source.py, input_scheme='janssen') against
the ecwam IPHYS=0 routines compiled unchanged (development/Makefile, f2py).

Every ecwam routine is called with wim's own spectral grid and the same
inputs as the python port, column by column (a 3x3 field of columns with
different u_*, wind direction and spectral level, so the direction and
frequency loops are exercised):

  1  sinput_jan   growth rate gamma(theta,f) given z0          -> round-off
  2  sdissip_jan  gamma_d(f) given (E, f1, kbar)                -> round-off
  3  fkmean       E, fm1, f1, kbar incl. tail; the spectrum is zero in the
                  first and last real bins so the end-bin quadrature weights
                  (wim dft vs ecwam DFIM) do not enter        -> round-off
     fkmean       full spectrum: the end-bin weight difference, reported
  4  femeanws     windsea mean frequency                        -> round-off
  5  tau_phi_hf   tail stress along the wind and tail energy flux, anchored
                  at the bin MIJ = last real bin                -> round-off
  6  z0wave       roughness from u_* and tau_w                  -> round-off
  7  stresso      total tau_w: resolved part (ecwam weights the anchor bin
                  by 1/2, wim uses its full dft) + tail; reported
  8  depth weights of the tail (calc_stress_work): column sum = surface
                  tail factor, closed bottom                    -> round-off
  9  frcutindex   cut-off bin (nearest in log f) for the 3x3 columns -> equal

Then a duration-limited growth run of wim_realistic with both schemes
(uniform periodic 2x2 grid, u10 ~ 10 m/s) prints energy, mean period,
tau_w/u_*^2, the Charnock parameter and tau_o/tau_a after 6, 12, 24 h as
a physical sanity check (Janssen: tau_w/u_*^2 ~ 0.6-0.9, g z0/u_*^2
~ 0.01-0.03 decreasing with wave age).

Run from the pyOM3 parent directory:
    python3 -m pyOM3.wim_realistic.development.janssen_ref_test
"""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, '..', '..', '..')))
sys.path.insert(0, here)                   # ecwref*.so built by development/Makefile
import numpy as np
import pyOM3 as OM
from pyOM3.wim_realistic.model import model

try:
    import ecwref
except ImportError:
    print('ecwref not built: cd wim_realistic/development; make ECWAM=<ecwam clone>')
    sys.exit(2)

PAR = dict(nx=3, ny=3, nf=28, nd=24, f0=0.04, df0=0.1, dt=60.0, dx=5e3, dy=5e3,
           periodic_in_x=True, periodic_in_y=True, snonlin_backend='numpy',
           implicit_dissipation_only=True, mask_energy=True, tail_flux='input',
           input_scheme='janssen')


def build(extra={}):
    p = dict(PAR); p.update(extra)
    M = model(p, halo_points=1)
    M.setup()
    return M


def jonswap(M, fp, alpha, udir, spread=2.0):
    """E(theta,f) on the padded grid, real bins set, ghosts zero"""
    h = M.halo
    f = M.ft[h:-h]
    g = M.grav
    sig = np.where(f <= fp, 0.07, 0.09)
    S = alpha*g**2/(2*np.pi)**4/f**5*np.exp(-1.25*(fp/f)**4) \
        * 3.3**np.exp(-0.5*((f - fp)/(sig*fp))**2)
    cs = np.maximum(np.cos(M.theta[h:-h, None, None] - udir[None]), 0.0)**spread
    cs = cs/np.sum(cs*M.dd, axis=0)[None]
    E = np.zeros((M.Nd, M.Nf, M.Ny, M.Nx))
    E[h:-h, h:-h] = S[None, :, None, None]*cs[:, None]
    return E


def to_ecw(M, A):
    """(Nd,Nf,Ny,Nx) padded -> (KIJL,NANG,NFRE) fortran order, real bins"""
    h = M.halo
    a = A[h:-h, h:-h, h:-h, h:-h]
    return np.asfortranarray(np.transpose(a, (2, 3, 0, 1)).reshape(-1, M.nd, M.nf))


def cols(M, a):
    h = M.halo
    return a[h:-h, h:-h].reshape(-1)


def rel(a, b):
    return np.max(np.abs(a - b))/max(np.max(np.abs(b)), 1e-300)


def verdict(name, r, tol=1e-12):
    print('%-44s %.3e  %s' % (name, r, 'PASS' if r < tol else 'FAIL'))
    return r < tol


def main():
    M = build()
    h, sl = M.halo, slice(M.halo, -M.halo)
    nd, nf, kijl = M.nd, M.nf, M.ny_loc*M.nx_loc
    jp = M._jp
    ecwref.w_setup(M.ft[sl], M.theta[sl], M.grav, jp('rho_a'), jp('rho_w'),
                   jp('betamax'), jp('zalp'), jp('alpha_charnock'), jp('cdis'),
                   jp('delta_sdis'), jp('cdisvis'), jp('tailfactor'), 0.0)
    x0, w = ecwref.w_get_tauhf_table()
    ok = verdict('x0tauhf', abs(M._x0tauhf() - x0)/x0)

    # 3x3 columns: u_* 0.2..0.8, wind direction, spectral level
    ny, nx = M.ny_loc, M.nx_loc
    us = np.zeros((M.Ny, M.Nx)); ud = np.zeros((M.Ny, M.Nx))
    us[sl, sl] = np.linspace(0.2, 0.8, ny*nx).reshape(ny, nx)
    ud[sl, sl] = np.linspace(0.3, 5.5, ny*nx).reshape(ny, nx)
    fp = 0.14
    E = np.zeros((M.Nd, M.Nf, M.Ny, M.Nx))
    for j in range(ny):
        for i in range(nx):
            Ej = jonswap(M, fp*(1 + 0.15*(i - 1)), 0.01*(1 + 0.5*j), ud[h + j, h + i])
            E[:, :, h + j, h + i] = Ej[:, :, h, h]
    M.ustar, M.udir, M.E = us, ud, E
    M.A = E/M.omega_t[None, :, None, None]
    tau_w0 = 0.35*us**2
    z0 = M.z0wave(us, tau_w0)

    om = M.omega_t[sl]
    wavnum = np.asfortranarray(np.tile(om**2/M.grav, (kijl, 1)))
    cinv   = np.asfortranarray(np.tile(om/M.grav, (kijl, 1)))
    cg     = M.grav/(2*om)
    xk2cg  = np.asfortranarray(np.tile((om**2/M.grav)**2*cg, (kijl, 1)))
    thr = M.theta[sl]
    coswdif  = np.asfortranarray(np.cos(thr[None, :] - cols(M, ud)[:, None]))
    sinwdif2 = np.asfortranarray(np.sin(thr[None, :] - cols(M, ud)[:, None])**2)
    raorw = np.full(kijl, jp('rho_a')/jp('rho_w'))
    fl1 = to_ecw(M, E)

    # 1 sinput_jan
    fld, slw, spos, xllws = ecwref.w_sinput_jan(fl1, wavnum, cinv, xk2cg, cols(M, us),
                                                cols(M, z0), coswdif, sinwdif2, raorw)
    gam, act = M._janssen_gamma(us, ud, z0)
    ok &= verdict('1 sinput_jan: gamma', rel(to_ecw(M, gam), fld))
    ok &= verdict('1 sinput_jan: active-input mask', rel(to_ecw(M, act*1.0), xllws))

    # 3 fkmean, interior spectrum (first/last real bin zero) -> exact weights
    Ei = E.copy(); Ei[:, h] = 0.0; Ei[:, -h - 1] = 0.0
    em, fm1, f1, ak, xk = ecwref.w_fkmean(to_ecw(M, Ei), wavnum)
    pem, pfm1, pf1, pxk = M.calc_fkmean(Ei)
    ok &= verdict('3 fkmean interior: E', rel(cols(M, pem), em))
    ok &= verdict('3 fkmean interior: fm1', rel(cols(M, pfm1), fm1))
    ok &= verdict('3 fkmean interior: f1', rel(cols(M, pf1), f1))
    ok &= verdict('3 fkmean interior: kbar', rel(cols(M, pxk), xk))
    em, fm1, f1, ak, xk = ecwref.w_fkmean(fl1, wavnum)
    pem, pfm1, pf1, pxk = M.calc_fkmean(E)
    print('   fkmean full spectrum (end-bin weights differ): E %.2e fm1 %.2e f1 %.2e kbar %.2e'
          % (rel(cols(M, pem), em), rel(cols(M, pfm1), fm1),
             rel(cols(M, pf1), f1), rel(cols(M, pxk), xk)))

    # 2 sdissip_jan given the means
    fldd, sld = ecwref.w_sdissip_jan(fl1, wavnum, em, f1, xk)
    k = om**2/M.grav
    X = k[:, None]/xk[None]
    d = jp('delta_sdis')
    gd_ref = fldd[:, 0, :].T                                  # (nf,kijl), theta independent
    pem, pfm1, pf1, pxk = M.calc_fkmean(E)
    # port formula with ecwam's means
    M2 = M
    sds = jp('cdis')*2*np.pi*f1*em**2*xk**4
    gd_port_formula = sds[None]*X*((1 - d) + d*X)
    ok &= verdict('2 sdissip_jan: gamma_d (ecwam means)', rel(gd_port_formula, gd_ref))
    ok &= verdict('2 sdissip_jan: theta independence', rel(fldd, np.repeat(fldd[:, :1, :], nd, axis=1)))
    _, _, gd_port = M.Sdiss_janssen(E)
    print('   Sdiss_janssen vs sdissip_jan with own means (end-bin weights): %.2e'
          % rel(cols_f(M, gd_port), gd_ref))

    # 4 femeanws
    fmws, emws = ecwref.w_femeanws(to_ecw(M, Ei), to_ecw(M, act*1.0))
    ok &= verdict('4 femeanws (interior spectrum)', rel(cols(M, M.calc_femeanws(Ei, act)), fmws))

    # 5 tau_phi_hf anchored at MIJ = last real bin
    mij = np.full(kijl, nf, dtype=np.int32)
    aird = np.full(kijl, jp('rho_a'))
    tauhf, phihf = ecwref.w_tau_phi_hf(mij, cols(M, us), cols(M, z0), fl1, aird, coswdif, sinwdif2)
    fc = M.ft[-h - 1]
    fac_tau, fac_phi = M._janssen_tail(fc, us, ud, z0)
    Ec = E[:, -h - 1]
    tx = np.sum(Ec[sl]*fac_tau[sl]*M.sinth[sl, None, None], axis=0)*M.dd
    ty = np.sum(Ec[sl]*fac_tau[sl]*M.costh[sl, None, None], axis=0)*M.dd
    along = (tx*np.sin(ud) + ty*np.cos(ud))/(jp('rho_a')/jp('rho_w'))
    ok &= verdict('5 tau_phi_hf: tail stress along wind', rel(cols(M, along), tauhf))
    phi = np.sum(Ec[sl]*fac_phi[sl], axis=0)*M.dd
    ok &= verdict('5 tau_phi_hf: tail energy flux', rel(cols(M, phi), phihf/jp('rho_w')))
    cross = (tx*np.cos(ud) - ty*np.sin(ud))/(jp('rho_a')/jp('rho_w'))
    print('   cross-wind tail stress (ecwam drops it): max |cross|/|along| = %.3f'
          % np.max(np.abs(cols(M, cross))/np.abs(tauhf)))

    # 6 z0wave
    z0r, z0b, chrnck = ecwref.w_z0wave(cols(M, us), cols(M, tau_w0))
    ok &= verdict('6 z0wave', rel(cols(M, z0), z0r))

    # 7 stresso: total tau_w
    mij = np.full(kijl, nf, dtype=np.int32)
    _, rhowgdfth = ecwref.w_frcutindex(nf, np.full(kijl, 1e3), fmws, cols(M, us))
    assert np.all(_ == nf)
    tauw, tauwdir, phiwa = ecwref.w_stresso(mij, rhowgdfth, fl1, slw, spos, cinv, cols(M, ud),
                                            cols(M, us), cols(M, z0), aird, coswdif, sinwdif2)
    S = (gam*E)[sl, sl]*om[None, :, None, None]
    ptx = np.sum(np.sum(S*M.sinth[sl, None, None, None], axis=0)*M.dft[sl, None, None], axis=0)*M.dd + tx
    pty = np.sum(np.sum(S*M.costh[sl, None, None, None], axis=0)*M.dft[sl, None, None], axis=0)*M.dd + ty
    ptau = np.sqrt(ptx**2 + pty**2)/(jp('rho_a')/jp('rho_w'))
    ptau = np.minimum(ptau, us**2/(1 + jp('eps1')))
    print('   stresso total tau_w (ecwam halves the anchor bin): rel diff %.2e, cap active in %d of %d columns'
          % (rel(cols(M, ptau), tauw), np.sum(tauw >= cols(M, us)**2/(1 + jp('eps1'))*(1 - 1e-12)), kijl))

    # 8 depth weights
    M.z0 = z0
    M.closed_bottom = True
    nz = 12
    dzt = np.array([50., 30, 20, 10, 8, 6, 4, 3, 2, 1, 0.5, 0.5])   # bottom to top
    _, zt, zb, _, _ = M._wave_kernel(dzt, nz)
    fac_tau_g, _, _, _ = M._tail_input_factors()
    w = M._janssen_tail_depth_weights(zt, zb)
    ok &= verdict('8 tail depth weights: column sum = surface',
                  rel(np.sum(w*M._dzc[:, None, None, None], axis=0), fac_tau_g))
    del M.closed_bottom

    # 9 frcutindex
    M._fmws = np.zeros((M.Ny, M.Nx)); M._fmws[sl, sl] = fmws.reshape(ny, nx)
    fm_field = np.zeros((M.Ny, M.Nx)); fm_field[sl, sl] = fm1.reshape(ny, nx)
    mijr, _ = ecwref.w_frcutindex(nf, fm1, fmws, cols(M, us))
    mask = M.frcutindex(fm_field, us)
    mij_port = np.sum(mask[sl], axis=0)                       # number of prognostic bins
    ok &= verdict('9 frcutindex: cut-off bin', np.max(np.abs(cols(M, mij_port) - mijr)))
    print('   cut-off bins (ecwam MIJ):', mijr)

    print('reference comparison:', 'ALL PASS' if ok else 'FAILURES')
    return ok


def cols_f(M, a):
    """(Nf,Ny,Nx) padded -> (nf,kijl)"""
    h = M.halo
    return a[h:-h, h:-h, h:-h].reshape(M.nf, -1)


def growth_run(scheme, hours=24, u10=10.0):
    p = dict(PAR); p.update(dict(nx=2, ny=2, input_scheme=scheme, snonlin_backend='fortran'))
    try:
        M = model(p, halo_points=1); M.setup()
    except Exception:
        p['snonlin_backend'] = 'numpy'
        M = model(p, halo_points=1); M.setup()
    h, sl = M.halo, slice(M.halo, -M.halo)
    cd = 1.2e-3
    ustar = np.sqrt(cd)*u10
    M.ustar = np.zeros((M.Ny, M.Nx)); M.ustar[sl, sl] = ustar
    M.udir  = np.zeros((M.Ny, M.Nx))
    M.ustar = M.apply_bc(M.ustar); M.udir = M.apply_bc(M.udir)
    E0 = jonswap(M, 0.35, 0.02, M.udir)*1e-2
    M.A = M.apply_bc(E0/M.omega_t[None, :, None, None])
    M.E = M.A*M.omega_t[None, :, None, None]
    nsteps = int(hours*3600/M.dt)
    tA, tB, tC = 1., 0., 0.
    tau, taum1, taum2 = 2, 1, 0
    out = []
    print('%s: u10 %.1f  u_* %.3f' % (scheme, u10, ustar))
    for n in range(1, nsteps + 1):
        M.time_step(tA, tB, tC, tau, taum1, taum2)
        tau, taum1, taum2 = (tau + 1) % 3, (taum1 + 1) % 3, (taum2 + 1) % 3
        if tA == 1. and tB == 0. and tC == 0:
            tA, tB, tC = 1.5 + 0.05, -0.5 - 0.05, 0.
        if n*M.dt in (6*3600, 12*3600, 24*3600, 48*3600):
            e = float(M.e_mean[h, h]); om = float(M.om_mean[h, h])
            tauo = M.calc_stress()
            tauo = float(np.hypot(tauo[0, 0, 0], tauo[1, 0, 0]))/(1.225/1000.)  # per rho_a
            line = 'ˇt=%5.1f h  Hs %.2f m  Tm %.2f s  tau_o/u_*^2 %.3f' % (
                n*M.dt/3600, 4*np.sqrt(e), 2*np.pi/om, tauo/ustar**2)
            if scheme == 'janssen':
                line += '  tau_w/u_*^2 %.3f  g z0/u_*^2 %.4f  cut-off bins %d' % (
                    M.tau_w[h, h]/ustar**2, M.grav*M.z0[h, h]/ustar**2, int(np.sum(M.freq_mask[sl, h, h])))
            print(line.replace('ˇ', '  '))
            out.append((e, om))
    return out


if __name__ == '__main__':
    ok = main()
    for s in ('wamdi', 'janssen'):
        growth_run(s)
    sys.exit(0 if ok else 1)
