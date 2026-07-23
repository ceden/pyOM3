"""
wam_source.py
-------------
Python translation of selected subroutines from the WAM wave model
(wam_source_module.f90 / wam_fre_dir_module.f90).

Public API
----------
  airsea(utop, tauw)                        – friction velocity & roughness
  wsigstar(ustar, z0, wstar)                – gustiness std-dev of u*
  sinput(F, SL, ustar, udir, z0, ...)       – wind-input source function
  sdissip(emean, fmean, fr)                 – whitecapping dissipation growth rate
  emean_fmean(F, fr, th)                    – total energy and mean frequency
  jonswap(fr, alphaj, fp, gamma=3.0, sa=0.07, sb=0.09)  – 1-D JONSWAP spectrum
  fetch_law(fetch, fpmax, u10)              – JONSWAP alpha, fp from fetch law

Array layout (spectral)
-----------------------
  (KL, ML, ny, nx)  –  direction, frequency, y, x

Wind / surface fields
---------------------
  (ny, nx)

Assumptions / simplifications
------------------------------
  - Deep water only (SHALLOW_RUN = False).
  - Functional derivative (FL), SPOS, and LLWS are not computed.

References
----------
  P.A.E.M. Janssen, J.P.O., 1989, 1991.   [AIRSEA, SINPUT]
  G. Komen, S. Hasselmann & K. Hasselmann, JGR, 1984.  [SDISSIP]
  H. Günther et al., ECMWF, 1991.
  J. Bidlot, ECMWF, 2012.
  Hersbach (2011) / Edson et al. (2013).   [WSIGSTAR Cd(U10)]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared physical constants  (wam_general_module.f90)
# ---------------------------------------------------------------------------
G        = 9.806        # gravitational acceleration          [m s^-2]
ROWATER  = 1000.0       # water density                       [kg m^-3]
XKAPPA   = 0.40         # von Kármán constant
PI       = np.pi
ZPI      = 2.0 * PI     # 2π

# ---------------------------------------------------------------------------
# SINPUT / WSIGSTAR constants  (wam_general_module.f90, ECMWF CY45R1)
# ---------------------------------------------------------------------------
BETAMAX  = 1.20         # wind-input parameter
ZALP     = 0.0080       # growth-curve shift

# ---------------------------------------------------------------------------
# AIRSEA constants  (wam_general_module.f90)
# ---------------------------------------------------------------------------
ALPHA    = 0.0060       # minimum Charnock constant
RNUAIR   = 1.5e-5       # kinematic air viscosity             [m^2 s^-1]
RNUAIRM  = 0.11*RNUAIR  # viscosity for momentum transfer     [m^2 s^-1]
XNLEV    = 10.0         # reference wind-speed level          [m]
EPS1     = 1.0e-30      # numerical floor  (WAM_TABLES_MODULE)


# ===========================================================================
# 1.  AIRSEA
# ===========================================================================

def airsea(utop, tauw):
    """
    Compute friction velocity and roughness length (Janssen 1991).

    Solves iteratively for the self-consistent (u*, Z0) pair in the
    quasi-linear wave–boundary-layer model.  The total wind stress is
    split into a wave-induced part (tauw) and a residual turbulent part;
    the Charnock roughness is modified accordingly.

    Parameters
    ----------
    utop : array_like, shape (ny, nx)
        Wind speed at reference level XNLEV = 10 m [m s^-1].
    tauw : array_like, shape (ny, nx)
        Wave-induced stress [m^2 s^-2].

    Returns
    -------
    ustar : ndarray, shape (ny, nx)
        Friction velocity [m s^-1].
    z0    : ndarray, shape (ny, nx)
        Roughness length (Charnock term only) [m].
    """
    # Local parameters (PARAMETER in Fortran AIRSEA)
    ACD      = 8.0e-4   # drag-law first-guess: Cd = ACD + BCD*U10
    BCD      = 8.0e-5
    NITER    = 15        # max Newton–Raphson iterations
    TWOXMP1  = 3.0       # exponent in dZ0/du* derivative
    EPSUS    = 1.0e-6    # floor on u* to avoid 1/0

    utop = np.asarray(utop, dtype=float)
    tauw = np.asarray(tauw, dtype=float)

    xlogxl  = np.log(XNLEV)
    alphaog = ALPHA / G
    xkutop  = XKAPPA * utop                                    # (ny, nx)

    # Initial guess from simple drag law
    tauold = np.maximum((utop * np.sqrt(ACD + BCD * utop))**2,
                        tauw + EPS1)
    us     = np.sqrt(tauold)
    ustm1  = 1.0 / np.maximum(us, EPSUS)

    z0ch   = np.zeros_like(us)
    active = np.ones(utop.shape, dtype=bool)

    for _ in range(NITER):
        if not active.any():
            break

        x       = tauw / tauold
        z0ch    = alphaog * tauold / np.sqrt(np.maximum(1.0 - x, EPS1))
        z0vis   = RNUAIRM * ustm1
        z0tot   = z0ch + z0vis

        xologz0 = 1.0 / (xlogxl - np.log(z0tot))
        f       = us - xkutop * xologz0
        zz      = (ustm1 * (z0ch * (2.0 - TWOXMP1 * x)
                            / np.maximum(1.0 - x, EPS1) - z0vis) / z0tot)
        delf    = 1.0 - xkutop * xologz0**2 * zz

        us_new  = us - f / delf
        taunew  = np.maximum(us_new**2, tauw + EPS1)
        us_new  = np.sqrt(taunew)

        converged = (taunew == tauold)
        active   &= ~converged
        us     = np.where(active, us_new, us)
        tauold = np.where(active, taunew, tauold)
        ustm1  = 1.0 / np.maximum(us, EPSUS)

    return us, z0ch


# ===========================================================================
# 2.  WSIGSTAR  (helper for SINPUT)
# ===========================================================================

def wsigstar(ustar, z0, wstar):
    """
    Relative standard deviation of friction velocity due to gustiness.

    Parameters
    ----------
    ustar : array_like, shape (ny, nx)
        Friction velocity [m s^-1].
    z0    : array_like, shape (ny, nx)
        Roughness length [m].
    wstar : array_like, shape (ny, nx)
        Free-convection velocity scale [m s^-1].

    Returns
    -------
    sig_n : ndarray, shape (ny, nx)
        Relative standard deviation of u* (dimensionless, capped at 0.1).
    """
    ustar = np.asarray(ustar, dtype=float)
    z0    = np.asarray(z0,    dtype=float)
    wstar = np.asarray(wstar, dtype=float)

    LOG10    = np.log(10.0)
    C1, C2   = 1.03e-3, 0.04e-3
    P1, P2   = 1.48, -0.21

    u10     = np.maximum(ustar / XKAPPA * (LOG10 - np.log(z0)), 1.0)
    u10m1   = 1.0 / u10
    c2u10p1 = C2 * u10**P1
    u10p2   = u10**P2

    c_d      = (C1 + c2u10p1) * u10p2
    dc_ddu   = (P2*C1 + (P1+P2)*c2u10p1) * u10p2 * u10m1
    sig_conv = 1.0 + 0.5 * u10 / c_d * dc_ddu

    # BG_GUST = 0 → ustar³ term vanishes
    return np.minimum(
        0.1,
        sig_conv * u10m1 * (0.5 * XKAPPA * wstar**3)**(1.0/3.0)
    )


# ===========================================================================
# 3.  SINPUT
# ===========================================================================

def sinput(ustar, udir, z0, roairn, wstar, th, fr):
    """
    Wind-input growth rate (Janssen 1989/1991, deep water).

    Returns the spectral growth rate GAMMA such that the wind-input source
    function is  S_in = GAMMA * F.  The caller multiplies by the spectrum F
    and accumulates into the total source function as required.

    Loops over directions (KL) and frequencies (ML) are vectorised via
    NumPy broadcasting.  The gustiness quadrature loop (isin = 2) is kept
    explicit to halve peak memory.

    Parameters
    ----------
    ustar  : array_like, shape (ny, nx)   Friction velocity [m s^-1].
    udir   : array_like, shape (ny, nx)   Wind direction [rad].
    z0     : array_like, shape (ny, nx)   Roughness length [m].
    roairn : array_like, shape (ny, nx)   Air density [kg m^-3].
    wstar  : array_like, shape (ny, nx)   Free-convection scale [m s^-1].
    th     : array_like, shape (KL,)      Direction bins [rad].
    fr     : array_like, shape (ML,)      Frequency bins [Hz].

    Returns
    -------
    GAMMA : ndarray, shape (KL, ML, ny, nx)
        Spectral growth rate [s^-1].  Wind-input source function = GAMMA * F.
    """
    ustar  = np.asarray(ustar,  dtype=float)
    udir   = np.asarray(udir,   dtype=float)
    z0     = np.asarray(z0,     dtype=float)
    roairn = np.asarray(roairn, dtype=float)
    wstar  = np.asarray(wstar,  dtype=float)
    th     = np.asarray(th,     dtype=float)
    fr     = np.asarray(fr,     dtype=float)

    KL = th.size
    ML = fr.size
    ny, nx = ustar.shape
    WSIN = np.array([0.5, 0.5])

    # 1. cos(θ_k − φ_ij)                                         (KL, ny, nx)
    TEMP = np.cos(th[:, np.newaxis, np.newaxis] - udir[np.newaxis, :, :])

    # 2. Frequency-dependent constants (deep water)
    omega = ZPI * fr                                              # (ML,)
    xk    = omega**2 / G
    cm    = omega / G
    CONST = omega * BETAMAX / XKAPPA**2 / ROWATER
    cnsn  = CONST[:, np.newaxis, np.newaxis] * roairn[np.newaxis, :, :]
    #       (ML, ny, nx)

    # 3. Two-point gustiness quadrature
    SIG_N  = wsigstar(ustar, z0, wstar)                           # (ny, nx)
    SIGDEV = np.stack([1.0 - SIG_N, 1.0 + SIG_N], axis=0)       # (2, ny, nx)
    USN    = ustar[np.newaxis, :, :] * SIGDEV                    # (2, ny, nx)
    Z0N    = np.broadcast_to(z0[np.newaxis, :, :], USN.shape).copy()

    # 4. Growth-rate accumulation
    cos_th = TEMP[:, np.newaxis, :, :]                            # (KL,  1, ny, nx)
    mask   = cos_th > 0.01
    GAMMA  = np.zeros((KL, ML, ny, nx))

    for isin in range(2):
        UCN = (USN[isin, np.newaxis, :, :] * cm[:, np.newaxis, np.newaxis]
               + ZALP)[np.newaxis, :, :, :]                       # (1, ML, ny, nx)
        ZCN = np.log(xk[:, np.newaxis, np.newaxis]
                     * Z0N[isin, np.newaxis, :, :]
                     )[np.newaxis, :, :, :]                        # (1, ML, ny, nx)

        X      = cos_th * UCN                                     # (KL, ML, ny, nx)
        ZLOG   = ZCN + XKAPPA / np.where(X != 0.0, X, 1.0)
        active = mask & (ZLOG < 0.0)
        ZLOG2X = np.where(active, ZLOG**2 * X, 0.0)
        GAMMA += np.where(
            active,
            WSIN[isin] * np.exp(np.clip(ZLOG, -700.0, 0.0)) * ZLOG2X**2,
            0.0
        )

    # 5. Scale by cnsn: GAMMA[k,m,y,x] = cnsn[m,y,x] * UFAC1[k,m,y,x]
    GAMMA *= cnsn[np.newaxis, :, :, :]

    return GAMMA


# ===========================================================================
# 4.  SDISSIP
# ===========================================================================

def sdissip(emean, fmean, fr):
    """
    Whitecapping dissipation growth rate (Komen et al. 1984), deep water.

    Returns the spectral growth rate GAMMA such that the dissipation source
    function is  S_dis = GAMMA * F.  Consistent with sinput, which returns
    the wind-input growth rate in the same form.

    Parameters
    ----------
    emean : array_like, shape (ny, nx)
        Total wave energy [m^2].
    fmean : array_like, shape (ny, nx)
        Mean frequency based on first moment [Hz].
    fr    : array_like, shape (ML,)
        Frequency bins [Hz].

    Returns
    -------
    GAMMA : ndarray, shape (ML, ny, nx)
        Dissipation growth rate [s^-1]  (≤ 0 everywhere).
        Dissipation source function = GAMMA * F;
        direction-independent, broadcast over KL as needed.
    """
    # Local parameters (PARAMETER in Fortran SDISSIP)
    CDIS  = 1.33
    DELTA = 0.5
    CONSD = -CDIS * ZPI**9 / G**4

    emean = np.asarray(emean, dtype=float)
    fmean = np.asarray(fmean, dtype=float)
    fr    = np.asarray(fr,    dtype=float)

    # SDS(ny, nx): scalar prefactor per grid point
    SDS = CONSD * emean**2 * fmean**9                         # (ny, nx)

    # GAMMA(ML, ny, nx): frequency-shaped dissipation growth rate
    fr_ratio2 = (fr[:, np.newaxis, np.newaxis] / fmean[np.newaxis, :, :])**2
    GAMMA = SDS[np.newaxis, :, :] * (
        (1.0 - DELTA) * fr_ratio2 + DELTA * fr_ratio2**2
    )                                                          # (ML, ny, nx)

    return GAMMA


# ===========================================================================
# 5.  EMEAN_FMEAN
# ===========================================================================

def emean_fmean(F, fr, th):
    """
    Total wave energy and first-moment mean frequency (TOTAL_ENERGY + FEMEAN).

    Integrates the variance density spectrum over all directions and
    frequencies using the WAM bin-area weights DFIM = DELTH * df, where
    df(m) is the trapezoidal frequency bandwidth and DELTH = 2π / KL is
    the constant directional spacing.

    emean = Σ_m Σ_k  DFIM(m) * F(k,m)
    fmean = Σ_m Σ_k  DFIM(m) * FR(m) * F(k,m)  /  emean

    Parameters
    ----------
    F  : array_like, shape (KL, ML, ny, nx)
        Variance density spectrum [m^2 Hz^-1 rad^-1].
    fr : array_like, shape (ML,)
        Frequency bins [Hz].
    th : array_like, shape (KL,)
        Direction bins [rad].

    Returns
    -------
    emean : ndarray, shape (ny, nx)   Total wave energy [m^2].
    fmean : ndarray, shape (ny, nx)   Mean frequency (1st moment) [Hz].
    """
    F  = np.asarray(F,  dtype=float)
    fr = np.asarray(fr, dtype=float)
    th = np.asarray(th, dtype=float)

    # Directional bin width (uniform spacing)
    DELTH = ZPI / th.size                                      # scalar

    # Frequency bin widths: trapezoidal, same formula WAM uses for DFIM
    df       = np.empty_like(fr)
    df[1:-1] = 0.5 * (fr[2:] - fr[:-2])
    df[0]    = 0.5 * (fr[1]  - fr[0])
    df[-1]   = 0.5 * (fr[-1] - fr[-2])

    # DFIM(ML,): combined bin-area weight
    DFIM = DELTH * df                                          # (ML,)

    # Sum F over directions → (ML, ny, nx), then weight by DFIM
    Fsum  = F.sum(axis=0)                                     # (ML, ny, nx)
    emean = (DFIM[:, np.newaxis, np.newaxis] * Fsum).sum(axis=0)   # (ny, nx)

    # First-moment mean frequency
    fmean = (
        (DFIM[:, np.newaxis, np.newaxis] * fr[:, np.newaxis, np.newaxis] * Fsum)
        .sum(axis=0)
        / np.maximum(emean, EPS1)
    )                                                          # (ny, nx)

    return emean, fmean


# ===========================================================================
# 6.  JONSWAP
# ===========================================================================

def jonswap(fr, alphaj, fp, gamma=3.0, sa=0.07, sb=0.09):
    """
    1-D JONSWAP spectrum (Hasselmann et al. 1976).

    Vectorised over an arbitrary set of grid points (ny, nx) given per-point
    JONSWAP parameters ALPHAJ and FP, evaluated at the shared frequency
    bins FR.

    ET(f) = ALPHAJ * G^2 / (2pi)^4 * f^-5 * exp(-1.25*(FP/f)^4)
            * GAMMA^exp(-(f/FP - 1)^2 / (2*sigma^2))

    where sigma = SA for f <= FP (left peak width) and SB for f > FP
    (right peak width).

    Parameters
    ----------
    fr     : array_like, shape (ML,)
        Frequency bins [Hz].
    alphaj : array_like, shape (ny, nx)
        JONSWAP alpha (overall energy level), per grid point.
    fp     : array_like, shape (ny, nx)
        JONSWAP peak frequency [Hz], per grid point.
    gamma  : float, optional
        Overshoot (peak enhancement) factor.  Default 3.0
        (matches the hardcoded value used by WAM's cold-start SPECTRA).
    sa     : float, optional
        Left peak width (f <= fp).  Default 0.07.
    sb     : float, optional
        Right peak width (f > fp).  Default 0.09.

    Returns
    -------
    ET : ndarray, shape (ML, ny, nx)
        1-D JONSWAP variance density spectrum [m^2 Hz^-1].
    """
    fr     = np.asarray(fr,     dtype=float)
    alphaj = np.asarray(alphaj, dtype=float)
    fp     = np.asarray(fp,     dtype=float)

    # Broadcast shapes: fr -> (ML, 1, 1), alphaj/fp -> (1, ny, nx)
    fr_b = fr[:, np.newaxis, np.newaxis]                       # (ML, 1, 1)
    alphaj_b = alphaj[np.newaxis, :, :]                        # (1, ny, nx)
    fp_b     = fp[np.newaxis, :, :]                            # (1, ny, nx)

    # Pierson-Moskowitz shape: G^2/(2pi)^4 * f^-5 * exp(-1.25*(fp/f)^4)
    g2zpi4frm5 = G**2 / ZPI**4 * fr_b**(-5)                    # (ML, 1, 1)
    arg_pm = 1.25 * (fp_b / fr_b)**4                           # (ML, ny, nx)

    ET = np.where(
        arg_pm < 50.0,
        alphaj_b * g2zpi4frm5 * np.exp(-np.minimum(arg_pm, 50.0)),
        0.0
    )                                                          # (ML, ny, nx)

    # Peak enhancement: gamma^exp(-(f/fp-1)^2 / (2*sigma^2))
    saa = 0.5 / sa**2
    sbb = 0.5 / sb**2
    sigma2_inv = np.where(fr_b > fp_b, sbb, saa)               # (ML, ny, nx)
    arg_peak = sigma2_inv * (fr_b / fp_b - 1.0)**2             # (ML, ny, nx)

    enhancement = np.where(
        arg_peak < 99.0,
        gamma**np.exp(-np.minimum(arg_peak, 99.0)),
        1.0
    )                                                          # (ML, ny, nx)

    ET *= enhancement

    return ET


# ===========================================================================
# 7.  FETCH_LAW
# ===========================================================================

def fetch_law(fetch, fpmax, u10):
    """
    JONSWAP parameters from the fetch law (Hasselmann et al. 1976).

    Computes, for each grid point, the JONSWAP peak frequency from a
    fetch law and the JONSWAP alpha from the alpha-nu relation.  Used by
    WAM's cold-start initialisation (wam_jonswap_module.f90: FETCH_LAW)
    to derive alphaj/fp for jonswap() from wind speed and fetch distance.

    FP = A * (G*FETCH/U10^2)^D     A = 2.84, D = -3/10
    FP = max(FP, 0.13)
    FP = min(FP, FPMAX / (G/U10))
    ALPHAJ = max(0.0081, B * FP^E)  B = 0.033, E = 2/3
    FP = FP * (G/U10)

    Where U10 <= 0 (calm), ALPHAJ = 0.0081 and FP = FPMAX.

    Parameters
    ----------
    fetch : array_like, shape (ny, nx)
        Fetch distance [m].
    fpmax : float
        Upper bound on the peak frequency [Hz].  In WAM this is FM,
        the "PEAK FREQUENCY" entry in WAM_User's cold-start parameter
        line (ALPHA FM GAMMA SIGMAA SIGMAB THETAQ) -- typically 0.2 Hz,
        NOT the lowest resolvable frequency of the grid.
    u10   : array_like, shape (ny, nx)
        Wind speed at 10 m [m/s].

    Returns
    -------
    alphaj : ndarray, shape (ny, nx)   JONSWAP alpha.
    fp     : ndarray, shape (ny, nx)   JONSWAP peak frequency [Hz].
    """
    fetch = np.asarray(fetch, dtype=float)
    u10   = np.asarray(u10,   dtype=float)

    A, D = 2.84, -3.0 / 10.0
    B, E = 0.033, 2.0 / 3.0

    calm = u10 <= 1e-8
    u10s  = np.where(calm, 1.0, u10)        # avoid 0-division; overwritten below
    fetchs = np.where(calm, 1.0, fetch)     # avoid 0**negative at calm points

    ug = G / u10s
    fp = np.maximum(0.13, A * (G * fetchs / u10s**2)**D)
    fp = np.minimum(fp, fpmax / ug)
    alphaj = np.maximum(0.0081, B * fp**E)
    fp = fp * ug

    alphaj = np.where(calm, 0.0081, alphaj)
    fp     = np.where(calm, fpmax,  fp)

    return alphaj, fp
