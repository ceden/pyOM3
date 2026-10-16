"""Shape ansatz and kernel profiles for wim_idemix.

Self-similar action shape (per dk, per direction):
    A~(k) = (Acal/kbar) phihat(k/kbar),   int phihat dx = 1,  s_1 = 1
PM-type family: phihat(x) ~ x^{-7/2} exp(-(5/4)(x_p/x)^2), x_p fixed by
the mean condition s_1 = 1 (from E(nu) ~ nu^-5 exp(-5/4 (nu_p/nu)^4)).

Provides:
  * moment constants s(p) and derived (beta_A = s(1/2) s(-1/2), ...)
  * the kernel profiles of the wave-mean flow coupling (paper eqs
    following eq:Aeq, eq:Eeq and eq:qM), universal functions of
    zeta = 2 kbar z <= 0, in the PAPER normalisation:
      q_A(zeta) = int x       e^{zeta x} phihat dx
      q_E(zeta) = int x^{3/2} e^{zeta x} phihat dx / s_{1/2}
      q_M(zeta) = int x^2     e^{zeta x} phihat dx
      p_X(zeta) = d/dzeta [ zeta q_X ]          for X = A, E, M
    so that q_A(0) = s_1 = 1, q_E(0) = s_{3/2}/s_{1/2}, q_M(0) = s_2,
    and int_{-inf}^0 q_X dzeta = 1 for all three: a barotropic current
    is sampled exactly (U_A = U_E = U_M = U), while the dispersion
    kernels integrate to zero (V_A = V_E = V_M = 0 without shear).
  * the antiderivatives r_X(zeta) = int_{-inf}^{zeta} q_X dzeta', for
    the cell-integrated coupling as in wim_simple._wave_kernel: an
    ocean cell with faces zeta_b < zeta_t receives
        ( r_X(zeta_t) - r_X(zeta_b) ) / (zeta_t - zeta_b)
    in place of the point value, so the column sum telescopes and a
    barotropic current is projected exactly, independent of dz.  The
    cell integral of p_X is zeta q_X evaluated at the faces, exact.
  * spectrum builder E(nu, theta) on a WIM grid from (Acal(theta),
    kbar(theta)) for calibration and comparison runs.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


class Shape:
    def __init__(self, tail=-3.5, cut=2.0, nzeta=400, zeta_max=40.0):
        # phihat(x) = N x^tail exp(-1.25 (x_p/x)^cut); PM: tail=-3.5, cut=2
        self.tail, self.cut = tail, cut
        f = lambda x, xp: x ** tail * np.exp(-1.25 * (xp / x) ** cut)
        def mean(xp):
            n = quad(lambda x: f(x, xp), 0, np.inf, limit=400)[0]
            m = quad(lambda x: x * f(x, xp), 0, np.inf, limit=400)[0]
            return m / n
        self.x_p = brentq(lambda v: mean(v) - 1.0, 0.02, 5.0)
        self._norm = quad(lambda x: f(x, self.x_p), 0, np.inf, limit=400)[0]
        self.phi0 = 1.0/self._norm    # flank amplitude phi^0_peak (paper Eq. 5)
        self._f = lambda x: f(x, self.x_p) / self._norm

        # tabulated kernel profiles on zeta in [-zeta_max, 0]; graded grid
        # (quadratic clustering at the surface) resolves the weak
        # zeta*ln(zeta) cusp of p_E and the barotropic nulls.  The
        # moments m_p(zeta) with p >= 5/2 diverge at zeta = 0 (the
        # x^{-7/2} tail), but only the products zeta*m_p enter, which
        # vanish there; the surface point is therefore set from the
        # analytic limit p_X(0) = q_X(0).
        u = np.linspace(1.0, 0.0, nzeta)
        self.zeta = -zeta_max * u ** 2
        mom = lambda p: np.array([self.moment(p, z) if z < 0 else
                                  (self.moment(p, 0.0) if p < 2.5 else 0.0)
                                  for z in self.zeta])
        s12 = self.s(0.5)
        m0, m12, m1  = mom(0.0), mom(0.5), mom(1.0)
        m32, m2      = mom(1.5), mom(2.0)
        m52, m3      = mom(2.5), mom(3.0)
        # paper kernels
        self._qA = m1
        self._qE = m32 / s12
        self._qM = m2
        # dispersion kernels p_X = d/dzeta [zeta q_X] = q_X + zeta m_{X+1}
        self._pA = self._qA + self.zeta * m2
        self._pE = self._qE + self.zeta * m52 / s12
        self._pM = self._qM + self.zeta * m3
        # antiderivatives r_X = int_{-inf}^{zeta} q_X dzeta'
        self._rA = m0
        self._rE = m12 / s12
        self._rM = m1

    # ---------------------------------------------------------- basics
    def phihat(self, x):
        return self._f(np.asarray(x, float))

    def s(self, p):
        """Moment constant s_p = int x^p phihat dx."""
        return quad(lambda x: x ** p * self._f(x), 0, np.inf, limit=400)[0]

    def moment(self, p, zeta=0.0):
        """int x^p e^{zeta x} phihat dx  (zeta <= 0)."""
        return quad(lambda x: x ** p * np.exp(zeta * x) * self._f(x),
                    0, np.inf, limit=400)[0]

    # ------------------------------------------------- derived constants
    @property
    def beta_A(self):
        return self.s(0.5) * self.s(-0.5)

    @property
    def s_half(self):
        return self.s(0.5)

    # ------------------------------------------------------ kernel eval
    def _ev(self, tab, zeta):
        return np.interp(zeta, self.zeta, tab, left=0.0)

    def qA(self, zeta):
        return self._ev(self._qA, zeta)

    def qE(self, zeta):
        return self._ev(self._qE, zeta)

    def qM(self, zeta):
        return self._ev(self._qM, zeta)

    def pA(self, zeta):
        return self._ev(self._pA, zeta)

    def pE(self, zeta):
        return self._ev(self._pE, zeta)

    def pM(self, zeta):
        return self._ev(self._pM, zeta)

    def rA(self, zeta):
        return self._ev(self._rA, zeta)

    def rE(self, zeta):
        return self._ev(self._rE, zeta)

    def rM(self, zeta):
        return self._ev(self._rM, zeta)

    # --------------------------------------------------- spectrum builder
    def build_E(self, f, th, Acal, kbar, g=9.81):
        """E(nu, theta) [m^2/Hz/rad] from directional (Acal, kbar).

        Acal(theta): action per rad; kbar(theta): mean wavenumber.
        E(nu,theta) dnu = sigma A~ dk  ->  E = sigma A~ dk/dnu.
        """
        f = np.asarray(f, float)[:, None]
        k = (2 * np.pi * f) ** 2 / g
        dkdf = 8 * np.pi ** 2 * f / g
        sig = 2 * np.pi * f
        At = Acal[None, :] / kbar[None, :] \
            * self.phihat(k / kbar[None, :])
        return sig * At * dkdf

    def moments_of_E(self, E, f, g=9.81):
        """(Acal(theta), Ecal(theta)) from a gridded E(nu,theta)."""
        f = np.asarray(f, float)
        df = np.gradient(f)
        sig = 2 * np.pi * f
        Ecal = np.sum(E * df[:, None], axis=0)
        Acal = np.sum(E / sig[:, None] * df[:, None], axis=0)
        return Acal, Ecal
