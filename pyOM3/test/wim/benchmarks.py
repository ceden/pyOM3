"""Test 15 (slow part): wim_simple against the classical wave-growth and
kinematics results. The pe family compares with pyOM2; the wave model has
no twin, so the twin is the literature, and the tolerances are those of a
coarse spectral grid (16 frequencies, 12 directions), not round-off.

  15a duration-limited growth, homogeneous, u* 0.5: the sea must reach a
      stationary state (relative change of E over the last 6 h below 2 %),
      its peak frequency must be near the Pierson-Moskowitz one,
      f_p u*/g = 0.13/28 (PM, U10 = 28 u*) within 30 %, and its energy
      within a factor 2 of the PM level alpha g^2/(5 omega_p^4) with the
      measured f_p (alpha 0.0081)
  15b fetch-limited growth, 1D: wind blowing off a closed coast, open
      coast downwind, stationary state: the JONSWAP power laws
      E ~ X^1 and f_p ~ X^-1/3 (Hasselmann et al. 1973), fitted over
      X* = g X/U10^2 in 1.2e3 .. 6.5e3 (beyond 1e4 the growth rolls off
      towards full development): exponents 1.0 +- 0.3 and -0.33 +- 0.12,
      and the energy coefficient eps = E g^2/U10^4 = 1.6e-7 X* within a
      factor 2.5 at the middle fetch (U10 = 28 u*)
  15c swell kinematics: a packet in one frequency bin travels at its group
      velocity: centroid speed within 1 % of g/(4 pi f) in x and y for a
      direction off the axes; the spreading (growth of the packet's
      variance) is printed
  15d current refraction: a swell packet crossing a zonal jet U(y). With
      k_x and the absolute frequency conserved (Snell), inside the jet
      f_1 = f_0 - k_0 sin(theta_0) U_0 / 2 pi and sin(theta_1) = k_0
      sin(theta_0)/k_1: the measured shifts of the spectral centroid at the
      jet core within 35 % of the predicted ones
  15e DIA against the exact S_nl: wim_simple/snl_exact/test_physics.py
      (equilibrium nulls, cancellation, magnitude anchor), run as is

    python3 benchmarks.py            all
    python3 benchmarks.py a c        a selection
"""
import numpy as np
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import homogeneous_model, step, line, HERE, NF, ND
import pyOM3 as OM

G = 9.81


def spectrum_stats(m, j=None, i=None):
    """total energy E, peak frequency (parabolic fit of the direction
    integrated spectrum), centroid frequency and centroid direction at cell
    (j, i) (or the domain mean)"""
    h = m.halo; sl = slice(h, -h)
    ft = np.asarray(m.ft)[sl]; th = np.asarray(m.theta)[sl]; dft = np.asarray(m.dft)[sl]
    E = np.asarray(m.E)[sl, sl, sl, sl]
    if j is not None:
        E = E[:, :, j:j+1, i:i+1]
    Ef = np.mean(np.sum(E, axis=0)*m.dd, axis=(1, 2))            # (nf,)
    Et = float(np.sum(Ef*dft))
    p = int(np.argmax(Ef))
    if 0 < p < len(ft) - 1:                                       # parabolic peak in log f
        y0, y1, y2 = np.log(np.maximum(Ef[p-1:p+2], 1e-300)); x = np.log(ft[p-1:p+2])
        a, b, c = np.polyfit(x, [y0, y1, y2], 2); fp = np.exp(-b/(2*a)) if a < 0 else ft[p]
    else:
        fp = ft[p]
    fc = float(np.sum(Ef*ft*dft)/max(Et, 1e-300))
    Ed = np.mean(np.sum(E*dft[None, :, None, None], axis=1), axis=(1, 2))
    tc = float(np.arctan2(np.sum(Ed*np.sin(th)), np.sum(Ed*np.cos(th))))
    return Et, float(fp), fc, tc


def duration_limited(ustar=0.5, hours=72):
    m = homogeneous_model(ustar=ustar, udir=0.0, amp=0.05, nx=4, ny=4)
    n = int(hours*3600/m.dt); E = []
    for k in range(n):
        step(m, 1)
        if k % 12 == 0:
            E.append(spectrum_stats(m)[0])
    Et, fp, fc, tc = spectrum_stats(m)
    nlast = int(6*3600/m.dt/12)
    change = abs(E[-1] - E[-1-nlast])/E[-1]
    nu = fp*ustar/G; nu_pm = 0.13/28.0
    om_p = 2*np.pi*fp; E_pm = 0.0081*G**2/(5*om_p**4)
    return change, abs(nu - nu_pm)/nu_pm, abs(np.log(Et/E_pm))/np.log(2.0), (Et, fp, nu, E_pm)


def fetch_limited(ustar=0.5, ny=48, dy=5e3, hours=24):
    """wind north, closed coast south (y = 0), open coast north; 1D in x"""
    m = homogeneous_model(ustar=ustar, udir=0.0, amp=1e-4, nx=4, ny=ny, periodic_y=False, open_coasts=True, dx=dy, dy=dy)
    n = int(hours*3600/m.dt)
    for k in range(n):
        step(m, 1)
    h = m.halo
    X = (np.arange(ny) + 0.5)*dy
    Et = np.zeros(ny); fp = np.zeros(ny)
    for j in range(ny):
        Et[j], fp[j], _, _ = spectrum_stats(m, j, 1)
    U10 = 28.0*ustar
    Xs_all = G*X/U10**2
    sel = (Xs_all > 1.2e3) & (Xs_all < 6.5e3)                 # the JONSWAP range, away from full development
    pE = np.polyfit(np.log(X[sel]), np.log(Et[sel]), 1)[0]
    pf = np.polyfit(np.log(X[sel]), np.log(fp[sel]), 1)[0]
    jm = int(np.where(sel)[0].mean())
    Xs = G*X[jm]/U10**2; eps = Et[jm]*G**2/U10**4
    coef = abs(np.log(eps/(1.6e-7*Xs)))/np.log(2.5)
    return pE, pf, coef, (X, Et, fp, eps, 1.6e-7*Xs)


def swell_packet(nsteps=120, nx=48, ny=48, dx=10e3):
    """one frequency bin, one direction bin (theta = 45 deg + dd/2 ... the
    nearest bin centre), Gaussian in x and y on the doubly periodic grid"""
    m = homogeneous_model(ustar=0.0, udir=0.0, amp=0.0, nx=nx, ny=ny, dx=dx, dy=dx)
    m._sources = False
    h = m.halo; sl = slice(h, -h)
    ft, th = np.asarray(m.ft)[sl], np.asarray(m.theta)[sl]
    kf = 4; kd = int(np.argmin(np.abs(th - np.pi/4)))
    x = (np.arange(nx) + 0.5)*dx; y = (np.arange(ny) + 0.5)*dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    pack = np.exp(-((X - 8*dx)/(2.5*dx))**2 - ((Y - 8*dx)/(2.5*dx))**2)
    A = np.zeros((m.Nd, m.Nf, m.Ny, m.Nx)); A[h+kd, h+kf, sl, sl] = pack
    m.A = m.apply_bc(OM.np.asarray(A)); m.E = m.A*m.omega_t[None, :, None, None]
    cg = G/(4*np.pi*ft[kf]); vx, vy = cg*np.sin(th[kd]), cg*np.cos(th[kd])

    def centroid():
        a = np.asarray(m.A)[h+kd, h+kf, sl, sl]
        # unwrap on the periodic grid relative to the predicted position
        return (np.sum(a*X)/np.sum(a), np.sum(a*Y)/np.sum(a),
                np.sum(a*(X - np.sum(a*X)/np.sum(a))**2)/np.sum(a) + np.sum(a*(Y - np.sum(a*Y)/np.sum(a))**2)/np.sum(a))
    x0, y0, v0 = centroid()
    # a run short enough that the packet does not wrap around
    nsteps = min(nsteps, int(0.6*nx*dx/max(vx, vy)/m.dt))
    step(m, nsteps)
    x1, y1, v1 = centroid(); T = nsteps*m.dt
    return abs((x1 - x0)/T - vx)/cg, abs((y1 - y0)/T - vy)/cg, (v1 - v0)/v0, (vx, vy, (x1 - x0)/T, (y1 - y0)/T, T)


def current_refraction(U0=1.5, nx=4, ny=60, dy=5e3, f0=0.1, th0=np.pi/4):
    """swell packet running north-east across a zonal jet U(y), 1D in x"""
    m = homogeneous_model(ustar=0.0, udir=0.0, amp=0.0, nx=nx, ny=ny, periodic_y=False, open_coasts=True, dx=dy, dy=dy)
    m._sources = False
    h = m.halo; sl = slice(h, -h)
    ft, th = np.asarray(m.ft)[sl], np.asarray(m.theta)[sl]
    y = (np.arange(ny) + 0.5)*dy
    jc = int(0.55*ny); U = U0*np.exp(-((y - y[jc])/(6*dy))**2)
    NZ = m._nz; UL = np.broadcast_to(U[None, :, None], (NZ, ny, nx)).copy(); VL = np.zeros_like(UL)
    m.ocean_U, m.ocean_V = m.calc_doppler_vel(m._dz, NZ, UL, VL)
    m.ocean_Udf, m.ocean_Vdf = m.calc_doppler_vel_df(m._dz, NZ, UL, VL)
    # packet: smooth in f and theta around (f0, th0), Gaussian in y south of the jet
    fseed = np.exp(-((np.log(ft/f0))/0.08)**2); dseed = np.exp(-((th - th0)/0.25)**2)
    pack = np.exp(-((y - 0.2*ny*dy)/(4*dy))**2)
    A = np.zeros((m.Nd, m.Nf, m.Ny, m.Nx))
    A[sl, sl, sl, sl] = dseed[:, None, None, None]*fseed[None, :, None, None]*pack[None, None, :, None]
    m.A = m.apply_bc(OM.np.asarray(A)); m.E = m.A*m.omega_t[None, :, None, None]
    _, _, fc0, tc0 = spectrum_stats(m, int(0.2*ny), 1)
    cg = G/(4*np.pi*f0); T = (y[jc] - 0.2*ny*dy)/(cg*np.cos(th0))
    best = (0.0, None)
    for k in range(int(1.6*T/m.dt)):
        step(m, 1)
        Et, _, fc, tc = spectrum_stats(m, jc, 1)
        if Et > best[0]:
            best = (Et, (fc, tc))
    fc1, tc1 = best[1]
    k0 = (2*np.pi*f0)**2/G
    f1 = f0 - k0*np.sin(th0)*U0/(2*np.pi); k1 = (2*np.pi*f1)**2/G
    th1 = np.arcsin(min(1.0, k0*np.sin(th0)/k1))
    df_pred, dth_pred = f1 - f0, th1 - th0
    df_meas, dth_meas = fc1 - fc0, tc1 - tc0
    return abs(df_meas - df_pred)/abs(df_pred), abs(dth_meas - dth_pred)/abs(dth_pred), (fc0, fc1, f0, f1, tc0, tc1, th0, th1)


def run(which='abcde'):
    ok = True
    if 'a' in which:
        change, dnu, dE, info = duration_limited()
        ok &= line('15a duration-limited: stationary (rel. change over the last 6 h)', change, 0.03)
        ok &= line('15a duration-limited: f_p u*/g vs PM 0.13/28 (rel.)', dnu, 0.3)
        ok &= line('15a duration-limited: E vs PM level at the measured f_p (|log2 ratio|)', dE, 1.0)
        print('     E %.3e m2, f_p %.3f Hz, nu %.2e, E_PM %.3e' % info)
    if 'b' in which:
        pE, pf, coef, info = fetch_limited()
        ok &= line('15b fetch-limited: exponent of E(X), JONSWAP 1.0', abs(pE - 1.0), 0.3)
        ok &= line('15b fetch-limited: exponent of f_p(X), JONSWAP -0.33', abs(pf + 0.33), 0.12)
        ok &= line('15b fetch-limited: eps / (1.6e-7 X*) at mid fetch (|log2.5 ratio|)', coef, 1.0)
        print('     eps %.2e, JONSWAP %.2e at X %.0f km' % (info[3], info[4], info[0][len(info[0])//2]/1e3))
    if 'c' in which:
        dx, dy, spread, info = swell_packet()
        ok &= line('15c swell packet: centroid speed x vs c_g sin(theta)', dx, 0.01)
        ok &= line('15c swell packet: centroid speed y vs c_g cos(theta)', dy, 0.01)
        print('%-56s %10.3f  (relative growth of the packet variance over %.1f h; diagnostic)' % ('15c swell packet: numerical spreading', spread, info[4]/3600))
    if 'd' in which:
        df, dth, info = current_refraction()
        ok &= line('15d refraction by a jet: frequency shift vs Snell (rel.)', df, 0.35)
        ok &= line('15d refraction by a jet: direction shift vs Snell (rel.)', dth, 0.35)
        print('     f %.4f -> %.4f Hz (Snell %.4f -> %.4f), theta %.1f -> %.1f deg (Snell %.1f -> %.1f)'
              % (info[0], info[1], info[2], info[3], np.degrees(info[4]), np.degrees(info[5]), np.degrees(info[6]), np.degrees(info[7])))
    if 'e' in which:
        # the battery prints PASS/FAIL per check and sets no exit code; its
        # final export has a hard-coded path that may not exist (ignored here)
        d = os.path.join(HERE, '..', '..', 'wim_simple', 'snl_exact')
        env = dict(os.environ, PYTHONPATH=os.path.abspath(os.path.join(HERE, '..', '..', '..')))
        r = subprocess.run([sys.executable, 'test_physics.py'], cwd=d, capture_output=True, text=True, env=env)
        res = [l for l in r.stdout.splitlines() if l.startswith(('PASS', 'FAIL'))]
        for l in res:
            print('     ' + l)
        ok &= line('15e DIA vs exact S_nl: checks of snl_exact/test_physics.py failed', float(sum(l.startswith('FAIL') for l in res) + (0 if len(res) == 4 else 1)), 0.0)
    return ok


if __name__ == '__main__':
    sys.exit(0 if run(''.join(sys.argv[1:]) or 'abcde') else 1)
