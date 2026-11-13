"""
Balanced flow decomposition by frequency bands: band_pe_simple for
pe_simple and band_pe_realistic for pe_realistic (add the class to the model
class bases). Moved here from pe_simple/diag_band_balance.py and
pe_realistic/diag_band_balance.py (Nov 2026). The parts that both families
share (the Lanczos method, the band weights, the slow projection, the band
inverse) are in the class _band. band_pe_simple no longer inherits
kernel_pe_simple and warn_pe_simple: its init_balance only sets the
background (_stratified of _common.py), init_kernel is not needed.

band_pe_simple
--------------

 Balanced flow decomposition by frequency bands of the model's own linear
 operator (the band method), for pe_simple with topography, variable f or
 any other feature of the linear dynamics that the vertical modes of
 warn.py and the kernels of kernel.py cannot represent.

 The linear operator L of the model (Coriolis, pressure gradient with the
 rigid lid, hydrostatic pressure, advection of the background
 stratification), with its masks, acts on the state X = (u, v, b). It is
 skew-adjoint in the energy inner product
   <X, Y> = sum u u' maskU + sum v v' maskV + sum b b' maskT / N^2
 (constant N^2; for variable N^2 the modal energy of the column on a flat
 bottom, or the diagonal weight 1/N^2 at the T levels with any topography,
 see init_band), so B = L^2 is self-adjoint with eigenvalues -omega^2, the
 squared frequencies of the linear modes. The slow subspace is spanned by the
 modes with |omega| < om_cut (geostrophic flow, topographic and Rossby waves;
 default 0.5 f0 with a transition of half width 0.3 f0, i.e. 0.2 to 0.8 f0),
 the fast subspace by those above (inertia-gravity waves). The zero order is
 the projection P on the slow subspace, the first and second order use
   K = L q(B),  q(-omega^2) = 1/omega^2 on the fast band, 0 on the slow band,
 which is 1/(i omega) on the fast modes (L v = -i omega v), minus the inverse
 of L on the fast subspace, as the nine kernels of
 kernel.py on the f-plane with a flat bottom, where the band method
 reproduces the kernel method. Both P X and K X are computed by the Lanczos
 method on B: m steps (2 m applications of L) give the Ritz values and the
 functions are applied to them exactly, so no spectral bounds are needed.
 The discrete modes fill the frequencies continuously (on the C-grid the high
 vertical modes have waves near the Nyquist wavenumber with omega down to
 2 f L_n/dx), so the transition between the bands is a smooth step of half
 width om_width, for which the Lanczos method converges in a few
 hundred steps (eddy100: 150 steps to round-off, without reorthogonalization),
 where a sharp cut does not converge at all. The sub-inertial discrete modes
 count as slow, in the kernel method they are waves slaved with 1/omega: on
 the flat f-plane the two methods differ by the energy of these modes (1e-4 of
 the balanced flow in eddy100 at the cutoff 0.6 f).

 The orders follow kernel.py: X0 = P X, X1 = K N(X0), X2 = K [N(X0 +
 X1) - N(X0) - N(X1) - dX1/dt + D X1] with N the advection tendencies (projected
 on the rigid lid), D the biharmonic damping of the model and dX1/dt from
 model steps (pTf1_numeric). Needs init_balance (the background
 stratification and the model steps, _init_stratification of _common.py),
 then init_band. One process, numpy.


band_pe_realistic
-----------------

 The band method of flow decomposition for pe_realistic (Nov 2026), the port
 of band_pe_simple (band.py) to the spherical or variable grid with
 topography and variable f.

 The method: the model's own linear operator L (Coriolis with coriolis_t,
 hydrostatic and rigid-lid surface pressure, advection of a background
 buoyancy by the vertical velocity of continuity) with the model's masks and
 metrics is skew-adjoint in the energy inner product (volume weighted,
 b with the weight 1/N^2 of the background), so B = L^2 is self-adjoint with
 eigenvalues -omega^2. The Lanczos method on B gives the projection P on the
 modes with |omega| < om_cut (geostrophic flow, topographic and Rossby waves;
 default 0.5 f0 with a transition of half width 0.3 f0, i.e. the band 0.2 to
 0.8 f0) and the inverse of L on the fast band, K = L q(B) with q = (1 -
 slow)/omega^2, without ever forming the modes. The orders are those of the
 kernel method of pe_simple: X0 = P X, X1 = K N(X0), X2 = K [N'(X0) X1 -
 dX1/dt + D X1], the time derivative analytic from the slow tendency of X0 +
 X1. A forcing (the slow part of the wind) is slaved like N. See
 band_pe_simple (band.py) for the derivations and
 balance_cmp1/RESULTS.md for the tests.

 Differences to pe_simple:
 * the tracer is buoyancy (temp = b, eq_of_state = 0) or T and S with an
   equation of state, then b = -g (rho - rho_back(z))/rho_0, N^2 from the EOS
   derivatives at the mean state, and band_apply_TS puts a balanced b back
   into T at fixed S (init_balance)
 * the background b_back(z) and N^2(z) are the horizontal mean of the state
   over the wet cells of each level (init_balance), or given; the inner
   product has the diagonal weight 1/N^2 at the T levels with any topography
   ('diag' of pe_simple, which was within 1e-3 of the exact modal norm there)
 * the rigid-lid operator also when the model runs with the implicit free
   surface (the external gravity waves are fast and leave with the divergent
   barotropic flow); the model keeps its free surface
 * the metric terms of coriolis_and_metric (u v tan phi/a) are quadratic and
   go with the nonlinear tendencies
 * single PE (the inner products are plain sums over the local arrays)

 Usage:
   M = band_pe_realistic(parameter); ... M.prepare_for_run(); M.set_initial_conditions(...)
   M.init_balance()                        # b_back, N^2, f0 from the state
   M.init_band(om_cut=0.5, om_width=0.3)   # the band and the inner product
   u, v, w, b, p, p_s = M.band_balanced(order, forcing=None, forcing_dot=None)
 b is the buoyancy perturbation against b_back; the model state is not changed.

"""
import numpy as np
import pyOM3 as OM
from functools import partial
from pyOM3.pe_realistic.model import model as _pe_realistic
from pyOM3.density import get_rho, get_drhodT, get_drhodS
from pyOM3.pe_simple.model import model as _pe_simple
from pyOM3.diagnostics.balance._common import _grid, _stratified


class _band:
    """
    the parts of the band method shared by the families: the Lanczos
    method for f(B) X, the band weights, the projection on the slow band,
    the band inverse, the fields of a state and a random state
    """

    def _band_B(self, X):
        return self._band_L(self._band_L(X))

    def _band_fields(self, X):
        """u, v, w, b, p, p_s with halo from interior (u, v, b)"""
        u = self._pad(X[0])*self.maskU; v = self._pad(X[1])*self.maskV; b = self._pad(X[2])*self.maskT
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        # the surface pressure of the state: that of its linear tendency, which
        # for a slow state is the pressure of the geostrophic balance
        _, _, _, p_s = self.lin_tend(u, v, b)
        p = p_s[None] + self.hydrostatic(b)
        return u, v, w, b, p, p_s

    def band_inverse(self, F):
        """K F = L q(B) F for the tendency triple F (interior arrays), interior arrays"""
        Q, = self._lanczos(F, 'q')
        return self._band_L_np(Q)

    def _band_random(self, rng):
        """a random state in the constrained space (interior arrays)"""
        I = self._interior
        shp = (self.nz, self.ny, self.nx)
        u, v, b = [rng.standard_normal(shp) for _ in range(3)]
        du, dv, _ = self.rigid_project(self._pad(u)*self.maskU, self._pad(v)*self.maskV)
        return [I(du), I(dv), b*I(self.maskT)]

    def _lanczos(self, X, which):
        """
        f(B) X for the functions in which ('P': the slow band projection,
        'q': q(B) with q = (1 - slow)/omega^2, 'S': slow^2), by the Lanczos
        method on B = L^2 in the energy inner product (band_reorth: full
        reorthogonalization). X and the results are interior arrays.
        """
        m = self.band_m
        f2 = self.bal_f0**2                                # B/f0^2 has eigenvalues -(omega/f0)^2
        X = [OM.np.asarray(x) for x in X]
        n0 = np.sqrt(self._band_dot(X, X))
        if n0 == 0:
            return [[np.zeros(x.shape) for x in X] for _ in which]
        V = [[x/n0 for x in X]]
        a, bet = [], []
        for j in range(m):
            W = [w/f2 for w in self._band_B(V[j])]
            if j > 0:
                W = [w - bet[-1]*vv for w, vv in zip(W, V[j-1])]
            a.append(self._band_dot(W, V[j]))
            W = [w - a[-1]*vv for w, vv in zip(W, V[j])]
            if self.band_reorth:
                for Vi in V:
                    c = self._band_dot(W, Vi)
                    W = [w - c*vv for w, vv in zip(W, Vi)]
            ww = self._band_dot(W, W)
            bj = np.sqrt(max(ww, 0.)) if np.isfinite(ww) else 0.
            if bj < self.band_tol or j == m - 1:                # invariant subspace, or breakdown
                break
            bet.append(bj)
            V.append([w/bj for w in W])
        k = len(a)
        T = np.diag(a) + np.diag(bet[:k-1], 1) + np.diag(bet[:k-1], -1)
        ev, U = np.linalg.eigh(T)
        om = np.sqrt(np.maximum(-ev, 0.))*abs(self.bal_f0)
        slow = self._band_slow(om)
        e1 = U[0, :]*n0                                    # U^T e1 n0
        out = []
        for wh in which:
            if wh == 'P':
                g = slow
            elif wh == 'S':
                # the slow part of the tendency of X0 + X1 due to F: P F + P L K F
                # = (slow + slow (-omega^2) (1 - slow)/omega^2) F = slow^2 F
                g = slow**2
            else:
                # zero on the slow band (slow = 1 exactly): a Ritz value
                # omega = 0 there gave 0/0 = NaN (1e-300**2 underflows),
                # the second order of eddy100t1_all
                fast = slow < 1.
                g = np.zeros_like(om)
                g[fast] = (1. - slow[fast])/om[fast]**2
            c = U @ (g*e1)
            out.append([np.asarray(sum(c[i]*V[i][l] for i in range(k))) for l in range(3)])
        self.band_its = k
        self.band_ritz = om
        return out

    def _band_slow(self, om):
        """
        the weight of the slow band at the frequency om: 1 below om_cut -
        om_width, 0 above om_cut + om_width, in between the C-infinity step
        exp(-1/x)/(exp(-1/x) + exp(-1/(1-x))). It is exactly 1 and 0 outside
        the transition, so that (1 - slow)/omega^2 of the band inverse stays
        bounded on the geostrophic modes (an error function tail there,
        divided by omega^2, blows up the first order).
        """
        if self.band_om_width <= 0:
            return np.where(om >= self.band_om_cut, 0., 1.)
        x = np.clip((om - self.band_om_cut + self.band_om_width)/(2*self.band_om_width), 0., 1.)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            f0 = np.where(x > 0, np.exp(-1./np.maximum(x, 1e-300)), 0.)
            f1 = np.where(x < 1, np.exp(-1./np.maximum(1. - x, 1e-300)), 0.)
        return f1/(f0 + f1)

    def band_project(self, u, v, b):
        """
        zero order balanced state (slow band) of the fields with halo,
        returns u, v, w, b, p, p_s. The blocked throughflow of a window is
        kept as slow (see band_balanced)
        """
        I = self._interior
        du, dv, _ = self.rigid_project(u*self.maskU, v*self.maskV)
        X0, = self._lanczos([I(du), I(dv), I(b)], 'P')
        return self._band_fields([X0[0] + I(u) - I(du), X0[1] + I(v) - I(dv), X0[2]])


class band_pe_simple(_band, _stratified, _pe_simple):

    # ------------------------------------------------------------------
    def init_balance(self, N0, nsteps=10, verbose=True, f0=None):
        """
        the background of the band method: N0 the background stratification
        (a number, or N^2 at the w levels, rigid lid), temp = background + b,
        nsteps the model steps of dt_mode 'model', f0 the reference Coriolis
        parameter (see init_balance of warn_pe_simple). Then init_band.
        """
        self._init_stratification(N0, nsteps, verbose, f0)
        return

    def _band_L_np(self, X):
        """L applied to interior arrays, numpy result (pe_simple: numpy only)"""
        return self._band_L(X)

    def init_band(self, om_cut=0.5, om_width=0.3, m_lanczos=300, tol=1e-9, reorth=False, n2_norm='auto',
                  dt_mode='linear', verbose=True, check=True):
        """
        om_cut     centre of the transition between the slow and the fast band,
                   in units of f0
        om_width   half width of the transition (a smooth step in omega, see
                   _band_slow), in units of f0; 0 for a sharp cut. The band
                   om_cut - om_width to om_cut + om_width must stay between
                   about 0.2 f0 and the lowest wave frequency, 0.8 f0 on the
                   C-grid: below 0.2 f0 the band inverse (1 - slow)/omega^2
                   amplifies the near-zero topographic modes (cut 0.3 with
                   width 0.4 blew up the first order on a seamount), above
                   0.8 f0 the waves are partly counted as slow (cut 0.6 with
                   width 0.4 kept 1.3e-2 of the wave state). Defaults 0.5 and
                   0.3 (Nov 8 2026; 0.6 and 0.2 before Nov 7, 0.6 and 0.4 on
                   Nov 7): the wide transition halves the reach of the filter
                   into a window (the first order then follows the zero order
                   from the wall), and the Lanczos method converges faster for
                   a smooth transition (the discrete modes fill the frequencies
                   continuously, high vertical modes near the Nyquist
                   wavenumber have omega down to 2 f L_n/dx)
        m_lanczos  maximal number of Lanczos steps per filter application
        tol        the Lanczos iterations stop when the next basis vector has a
                   norm below tol (B is scaled by 1/f0^2, the basis vectors have
                   unit energy norm): an invariant subspace is found
        reorth     full reorthogonalization of the Lanczos vectors (costs m^2 n
                   operations). Not needed for the smooth band functions: the
                   recurrence alone gives the same result to round-off (ghost
                   Ritz values carry the same function values)
        n2_norm    treatment of a variable N^2 (bal_N2w of init_balance):
                   'modal'   the modal energy of the column as inner product,
                             exact for the model's operator, flat bottom only
                   'diag'    the model's operator with the diagonal weight
                             1/N^2_T, N^2 averaged to the T levels (any
                             topography). Not exactly skew-adjoint (3e-8 on
                             B1, 1e-16 for 'modal'), which the Lanczos method
                             tolerates: on B1 with a wave the states differ
                             from 'modal' by 4e-4 (order 0) to 1e-3 (order 2)
                             of the flow. The choice for pe_realistic
                   'tlevel'  the operator with N^2 at the T levels, exactly
                             skew-adjoint with the weight 1/N^2_T, and the
                             difference to the model's operator, -(N2_k -
                             N2_k-1)(w_k - w_k-1)/4, as an expansion term in
                             the orders 1 and 2 like (f - f0) u in
                             warn.py (_band_dL). Inferior: the waves of
                             the model are not modes of this operator, and
                             their projection on its slow band (5e-3 of the
                             flow on B1 w34, at every order) is not corrected
                             by the expansion
                   'auto'    'modal' with a flat bottom, else 'diag'
                   (constant N^2: all coincide, nothing is done)
        dt_mode    the time derivative of the first order in the second order:
                   'linear'  (default since Nov 8 2026) from the slow tendency
                             of X0 + X1, dX0/dt = L X0 + P [N(X0) + L X1] + D X0
                             (D the damping of the model), and the derivative of the quadratic N, dF1/dt = K
                             [N(X0 + dX0/dt) - N(X0) - N(dX0/dt)], no model
                             steps; in a window the model steps carried the
                             walls c_1 dt into the domain, a floor of the
                             second order
                   'model'   from bal_nsteps model steps started at X0 + X1, as
                             pTf1_numeric of warn.py
        check      test the skew-adjointness of the linear operator on random fields
        """
        self.band_om_cut = om_cut*abs(self.bal_f0)
        self.band_om_width = om_width*abs(self.bal_f0)
        # the rigid-lid projection also when the model runs with the implicit
        # free surface: its cf carries the term -1/(g dt^2), which makes the
        # solve a damped step and not a projection. The band method's linear
        # operator is the rigid-lid one (the external gravity waves are fast
        # and leave with the divergent barotropic flow), the model may keep
        # its free surface in the reference and emission runs
        if getattr(self, 'implicit_free_surface', False):
            # make_coeff_surf_press is jitted with the flag baked in, so the
            # term is removed by hand (pressure.py)
            self.band_cf = OM.add_to_array(self.cf, (1, 1, slice(None), slice(None)),
                                           1./(self.grav*self.dt**2)*self.maskT[-self.halo - 1])
        else:
            self.band_cf = self.cf
        if om_width > 0 and om_cut - om_width < 0.1 - 1e-6:        # 0.3 - 0.2 is below 0.1 in floating point
            raise RuntimeError('band: the transition reaches below 0.1 f0 (om_cut - om_width = %.2f), '
                               'the band inverse is unbounded there' % (om_cut - om_width))
        self.band_m, self.band_tol, self.band_verbose = m_lanczos, tol, verbose
        self.band_reorth = reorth
        h = self.halo
        I = lambda a: np.asarray(a)[h:-h, h:-h, h:-h]
        self.band_w = (I(self.maskU), I(self.maskV), I(self.maskT)/self.bal_N0**2)
        self.band_Wu = self.band_Wb = None
        self.band_N2T = None
        if self.bal_N2w is not None:
            flat = not np.any(I(self.maskT) == 0)
            if n2_norm == 'auto':
                n2_norm = 'modal' if flat else 'diag'
            if n2_norm == 'modal':
                self._band_modal_norm()
            elif n2_norm == 'tlevel':
                self._band_tlevel_norm()
            elif n2_norm == 'diag':
                # the model's operator with the diagonal weight 1/N2T (the
                # weight of _band_tlevel_norm, the operator unchanged)
                self._band_tlevel_norm()
                self.band_N2T = None
            else:
                raise ValueError('band: n2_norm must be modal, diag, tlevel or auto')
        self.band_n2_norm = n2_norm
        if dt_mode not in ('linear', 'model'):
            raise ValueError('band: dt_mode must be linear or model')
        self.band_dt_mode = dt_mode
        self.band_its = 0
        if check:
            rng = np.random.default_rng(0)
            X = self._band_random(rng); Y = self._band_random(rng)
            LX, LY = self._band_L(X), self._band_L(Y)
            s = abs(self._band_dot(Y, LX) + self._band_dot(LY, X))/np.sqrt(self._band_dot(LX, LX)*self._band_dot(Y, Y))
            self.band_skew = s
            if verbose and self.my_pe == 0:
                print('band: cutoff %.2f f, width %.2f f, <Y,LX> + <LY,X> = %.1e of |LX||Y|' % (om_cut, om_width, s))
        return

    def _band_modal_norm(self):
        """
        the energy inner product for variable N^2. The background advection
        of the model puts N^2 at the w levels, db_k = -(N2_k w_k + N2_k-1 w_k-1)/2,
        the hydrostatic pressure averages b to the w levels, so the exchange
        with the kinetic energy, sum w (b_k + b_k+1)/2, matches the change of
        sum b^2/N^2 only for constant N^2: no diagonal weight of b makes L
        skew-adjoint. The vertical modes of the column (warn.py, from
        the probe of the model's column operator) separate it, and
        L is skew-adjoint in the modal energy
          sum_n U_n^2 + V_n^2 + P_n^2/c_n^2,  u = phi U, b = psi P,
        a column-local inner product (nz x nz matrices on u, v and b). The
        buoyancy of the vertical checkerboard (b_k = (-1)^k, no pressure
        gradient, a stationary mode) gets the weight 1/N0^2. Flat bottom only:
        with topography the modes differ from column to column.
        """
        h = self.halo
        if np.any(np.asarray(self.maskT)[h:-h, h:-h, h:-h] == 0):
            raise RuntimeError('band: variable N^2 only with a flat bottom (the modal energy)')
        nz = self.nz
        phi, psi, c2 = self.bal_phi, self.bal_psi, self.bal_c2
        Pi = np.linalg.inv(phi)
        self.band_Wu = Pi.T @ Pi
        cb = (-1.)**np.arange(nz)/np.sqrt(nz)
        Bb = np.concatenate([psi[:, 1:], cb[:, None]], axis=1)
        Bi = np.linalg.inv(Bb)
        wb = np.concatenate([1./c2[1:], [1./self.bal_N0**2]])
        self.band_Wb = Bi.T @ (wb[:, None]*Bi)
        return

    def _band_tlevel_norm(self):
        """
        N^2 at the T levels for the operator L_T, whose buoyancy tendency is
          db_k = -N2T_k (w_k + w_k-1)/2
        with the vertical velocity at the two faces of the cell. The kinetic
        energy source of the pressure term is sum_k w_k (b_k + b_k+1)/2 over
        the w levels, which is sum_k b_k (w_k + w_k-1)/2 over the T levels, so
        L_T is skew-adjoint in the diagonal energy sum b^2/N2T, with any
        positive N2T and any masks. N2T_k is the mean of N^2 at the two faces;
        at the bottom and the top cell (one face closed) it is N^2 of the open
        face, so that L_T equals the model's operator there and the
        difference (_band_dL) lives in the interior cells only.
        """
        h = self.halo
        nz = self.nz
        N2w = np.asarray(self.bal_N2w, dtype='f8')              # at the w level above T cell k
        if N2w.ndim == 1:
            N2w = N2w[:, None, None] + np.zeros((nz, self.ny, self.nx))
        N2lo = np.concatenate([N2w[:1], N2w[:-1]], axis=0)      # at the face below T cell k
        mW = np.asarray(self.maskW)[h:-h, h:-h, h:-h]
        mWlo = np.concatenate([np.zeros_like(mW[:1]), mW[:-1]], axis=0)
        open_hi = mW > 0; open_lo = mWlo > 0
        N2T = np.where(open_hi & open_lo, 0.5*(N2w + N2lo), np.where(open_hi, N2w, N2lo))
        mT = np.asarray(self.maskT)[h:-h, h:-h, h:-h]
        self.band_N2T = np.where(mT > 0, N2T, self.bal_N0**2)
        self.band_N2Tfull = OM.np.asarray(self._pad(self.band_N2T))
        self.band_w = (self.band_w[0], self.band_w[1], mT/self.band_N2T)
        return

    def _band_dL(self, X):
        """
        the difference between the model's linear operator and L_T (interior
        arrays): zero in u and v, in b -(N2_k - N2_k-1)(w_k - w_k-1)/4 in the
        interior cells. The expansion term of the T-level treatment, zero for a
        geostrophic state (w = 0) and for constant N^2
        """
        if self.band_N2T is None:
            return [0.*x for x in X]
        h = self.halo
        u = self._pad(X[0])*self.maskU; v = self._pad(X[1])*self.maskV; b = self._pad(X[2])
        _, _, db, _ = self.lin_tend(u, v, b)
        db_T = self._band_LT_b(u, v)
        return [0.*X[0], 0.*X[1], np.asarray(db - db_T)[h:-h, h:-h, h:-h]]

    def _band_LT_b(self, u, v):
        """the buoyancy tendency of L_T (fields with halo)"""
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        return -self.band_N2Tfull*(w + self.rollz(w, 1))*0.5*self.maskT

    def _band_dot(self, X, Y):
        if self.band_Wu is None:
            return float(sum(np.sum(w*x*y) for w, x, y in zip(self.band_w, X, Y)))
        col = lambda W, a: np.einsum('kl,lji->kji', W, a)
        return float(np.sum(X[0]*col(self.band_Wu, Y[0])) + np.sum(X[1]*col(self.band_Wu, Y[1]))
                     + np.sum(X[2]*col(self.band_Wb, Y[2])))

    def rigid_project(self, du, dv):
        """
        removes the gradient of the surface pressure from the tendencies
        (fields with halo), so that the depth integrated tendency is
        nondivergent, returns du, dv, p_s
        """
        h = self.halo
        fpx = OM.np.sum(du*self.maskU, axis=0)*self.dz
        fpy = OM.np.sum(dv*self.maskV, axis=0)*self.dz
        fpx, fpy = self.apply_bc(fpx), self.apply_bc(fpy)
        fc = ((fpx - self.rollx(fpx, 1))/self.dx + (fpy - self.rolly(fpy, 1))/self.dy)*self.maskT[-h-1]
        # the solver stops at an absolute change of the solution (epsilon), so
        # the problem is scaled to a solution of order one (the gravest mode)
        sc = (float(OM.np.max(OM.np.abs(fc))) or 1.)*(self.nx*self.dx)**2/(4*np.pi**2*self.nz*self.dz)
        p_s, err, its = self.cg(fc/sc, OM.np.zeros_like(fc), getattr(self, 'band_cf', self.cf))
        self.band_cg_its = its
        p_s = self.apply_bc(p_s)*sc
        du = du - (self.rollx(p_s[None], -1) - p_s[None])/self.dx*self.maskU
        dv = dv - (self.rolly(p_s[None], -1) - p_s[None])/self.dy*self.maskV
        return self.apply_bc(du), self.apply_bc(dv), p_s

    def lin_tend(self, u, v, b):
        """
        the linear tendencies (du, dv, db) of the fields (with halo): Coriolis,
        hydrostatic and surface pressure gradient, advection of the background
        buoyancy by the vertical velocity of continuity
        """
        f = self.coriolis_t
        du = self.maskU*(f[None]*(v + self.rolly(v, 1))
                         + self.rollx(f, -1)[None]*self.rollx(v + self.rolly(v, 1), -1))*0.25
        dv = -self.maskV*(f[None]*(u + self.rollx(u, 1))
                          + self.rolly(f, -1)[None]*self.rolly(u + self.rollx(u, 1), -1))*0.25
        p_h = self.hydrostatic(b)
        du = du - (self.rollx(p_h, -1) - p_h)/self.dx*self.maskU
        dv = dv - (self.rolly(p_h, -1) - p_h)/self.dy*self.maskV
        du, dv, p_s = self.rigid_project(du, dv)
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        fe, fn, ft = self.tracer_flux_2nd(self.bal_b_back, u, v, w, self.maskU, self.maskV, self.maskW)
        db = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskT
        return du, dv, self.apply_bc(db), p_s

    def _band_L(self, X):
        """L applied to interior arrays"""
        h = self.halo
        u = self._pad(X[0])*self.maskU; v = self._pad(X[1])*self.maskV; b = self._pad(X[2])
        du, dv, db, _ = self.lin_tend(u, v, b)
        if self.band_N2T is not None:
            db = self._band_LT_b(u, v)
        return [np.asarray(a)[h:-h, h:-h, h:-h] for a in (du, dv, db)]

    def damp_tend(self, u, v, b):
        """the biharmonic damping tendencies of the model (fields with halo)"""
        Ah = getattr(self, 'Ahbi', 0.); Kh = getattr(self, 'Khbi', 0.)
        z = lambda a: OM.np.zeros_like(a)
        du = self.biharmonic_horizontal_tendency(u, Ah, self.maskU) if Ah else z(u)
        dv = self.biharmonic_horizontal_tendency(v, Ah, self.maskV) if Ah else z(v)
        db = self.biharmonic_horizontal_tendency(b, Kh, self.maskT) if Kh else z(b)
        return du, dv, db

    def _band_nonlin(self, u, v, b):
        """advection tendencies (du, dv, db), projected on the rigid lid, interior arrays"""
        h = self.halo
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        fe, fn, ft = self.mom_flux_2nd_u(u, u, v, w, self.maskU, self.maskV, self.maskW)
        du = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskU
        fe, fn, ft = self.mom_flux_2nd_v(v, u, v, w, self.maskU, self.maskV, self.maskW)
        dv = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskV
        fe, fn, ft = self.tracer_flux_2nd(b, u, v, w, self.maskU, self.maskV, self.maskW)
        db = ((fe - self.rollx(fe, 1))/self.dx + (fn - self.rolly(fn, 1))/self.dy
              + (ft - self.rollz(ft, 1))/self.dz)*self.maskT
        du, dv, _ = self.rigid_project(self.apply_bc(du), self.apply_bc(dv))
        return [np.asarray(a)[h:-h, h:-h, h:-h] for a in (du, dv, db)]

    def band_balanced(self, order=2, forcing=None, forcing_dot=None):
        """
        balanced state of the model state up to the order (0, 1 or 2),
        returns the fields (u, v, w, b, p, p_s)

        forcing      (Fu, Fv, Fb), the slow forcing of the model at the time of
                     the state as tendencies (interior arrays, Fb may be 0),
                     e.g. the wind stress on the top cell. It is slaved like
                     the nonlinear terms: X1 = K [N(X0) + F], where K F is the
                     steady forced response (L X = -F, the Ekman flow of a
                     steady wind), and its slow part P F enters the tendency
                     of X0. Only the part of the forcing that varies slowly
                     against f belongs here (a low-pass of the wind); the
                     response to the rest is near-inertial and is a wave
        forcing_dot  dF/dt at that time (interior arrays), enters the second
                     order as K dF/dt
        """
        h = self.halo
        I = lambda a: np.asarray(a)[h:-h, h:-h, h:-h]
        u, v, b, p = self.model_state()
        Fz = [0.*I(u), 0.*I(v), 0.*I(b)]
        proj = lambda F: (lambda d: [I(d[0]), I(d[1]), np.asarray(F[2]) + Fz[2]])(
            self.rigid_project(self._pad(np.asarray(F[0]) + Fz[0])*self.maskU, self._pad(np.asarray(F[1]) + Fz[1])*self.maskV))
        Fs = proj(forcing) if forcing is not None else Fz             # on the rigid lid, as the model applies it
        Fd = proj(forcing_dot) if forcing_dot is not None else Fz
        # the part of the depth-integrated flow that the walls block (a window
        # cut from a larger domain): the potential flow of the normal transport
        # at the walls. It is the balanced flow the outside imposes on the
        # window (the rigid lid has no barotropic waves) and counts as slow;
        # the projection sees the nondivergent remainder. Zero for a periodic
        # domain or closed basin, where the state is nondivergent
        du, dv, _ = self.rigid_project(u*self.maskU, v*self.maskV)
        Xphi = [I(u) - I(du), I(v) - I(dv), 0.*I(b)]
        X0 = self._band_fields(self._lanczos([I(du), I(dv), I(b)], 'P')[0])
        # the orders are built on the nondivergent X0 alone: the potential
        # flow has its divergence in the cells next to the walls (the blocked
        # inflow), there its vertical velocity is of order u H/dx, and the
        # advection by it through thin surface cells made the first order
        # several m/s at the walls of an icon033 window (Nov 9 2026). The
        # advection of X0 by the throughflow, a slow tendency, is left out
        Xout = self._band_fields([I(a) + c for a, c in zip((X0[0], X0[1], X0[3]), Xphi)])
        self.band_phi_rms = float(np.sqrt(self._band_dot(Xphi, Xphi)))
        out = list(Xout)
        if order >= 1:
            # the expansion term of the T-level treatment of N^2 (zero
            # otherwise) enters with the nonlinear tendencies, as (f - f0) u
            # in warn.py
            NX = lambda u, v, b: [a + c + f for a, c, f in zip(self._band_nonlin(u, v, b), self._band_dL([I(u), I(v), I(b)]), Fs)]
            NX0 = NX(X0[0], X0[1], X0[3])
            Q, PN = self._lanczos(NX0, 'qS')              # K = L q(B); slow tendency of X0 + X1 from N(X0)
            F1 = self._band_L(Q)
            X1 = self._band_fields(F1)
            out = [a + c for a, c in zip(out, X1)]
            if order >= 2:
                N0 = self._band_nonlin(X0[0], X0[1], X0[3])
                N1 = self._band_nonlin(X1[0], X1[1], X1[3])
                N01 = self._band_nonlin(X0[0] + X1[0], X0[1] + X1[1], X0[3] + X1[3])
                dL1 = self._band_dL([I(X1[0]), I(X1[1]), I(X1[3])])
                dF = [a - c - d + e for a, c, d, e in zip(N01, N0, N1, dL1)]
                # time derivative of the first order from model steps: the
                # first order correction K N(P Y) of the state Y at both ends
                # of the step. P K is not zero on the transition band (slow
                # and 1 - slow overlap there), so P(X0 + X1) differs from X0
                # and the correction at the start must be recomputed from
                # P(X0 + X1), as at the end; with F1 of X0 the difference
                # contained the first order content of the band divided by
                # dt, and the second order was worse than the first with
                # waves or a mixed layer (Nov 7 2026)
                if self.band_dt_mode == 'model':
                    Y = [X0[0] + X1[0], X0[1] + X1[1], X0[3] + X1[3]]
                    Xn = self._run(Y[0], Y[1], Y[2], X0[5] + X1[5], self.bal_nsteps)
                    Z0 = self._band_fields(self._lanczos([I(a) for a in Y], 'P')[0])
                    Z = self._band_fields(self._lanczos([I(Xn[0]), I(Xn[1]), I(Xn[2])], 'P')[0])
                    F10 = self.band_inverse(NX(Z0[0], Z0[1], Z0[3]))
                    F1n = self.band_inverse(NX(Z[0], Z[1], Z[3]))
                    dt = self.bal_nsteps*self.dt
                    dF1 = [(f1n - f10)/dt for f1n, f10 in zip(F1n, F10)]
                    if forcing_dot is not None:
                        dF1 = [a + c for a, c in zip(dF1, self.band_inverse(Fd))]
                else:
                    # the slow tendency of X0 + X1, L X0 + P [N(X0) + L X1] =
                    # L X0 + slow(B)^2 N(X0) (P L X1 is not zero on the
                    # transition band, where it is as large as P N(X0)), and
                    # the derivative of the quadratic N along it (exact:
                    # N(a + b) - N(a) - N(b) is the bilinear cross term),
                    # scaled to the size of X0 for the round-off of the
                    # difference
                    LX0 = self._band_L([I(X0[0]), I(X0[1]), I(X0[3])])
                    DX0 = [I(a) for a in self.damp_tend(X0[0], X0[1], X0[3])]   # the model damps X0 too
                    T0 = [a + c + d for a, c, d in zip(LX0, PN, DX0)]
                    sc = np.sqrt(self._band_dot([I(X0[0]), I(X0[1]), I(X0[3])], [I(X0[0]), I(X0[1]), I(X0[3])])
                                 /max(self._band_dot(T0, T0), 1e-300))
                    Td = self._band_fields([sc*a for a in T0])
                    Nd = self._band_nonlin(Td[0], Td[1], Td[3])
                    N0d = self._band_nonlin(X0[0] + Td[0], X0[1] + Td[1], X0[3] + Td[3])
                    dLd = self._band_dL([I(Td[0]), I(Td[1]), I(Td[3])])
                    dF1 = self.band_inverse([(a - c - d + e)/sc + f for a, c, d, e, f in zip(N0d, N0, Nd, dLd, Fd)])
                D = [I(a) for a in self.damp_tend(X1[0], X1[1], X1[3])]
                G = [d - df1 + dd for d, df1, dd in zip(dF, dF1, D)]
                X2 = self._band_fields(self.band_inverse(G))
                out = [a + c for a, c in zip(out, X2)]
        return out


class band_pe_realistic(_band, _grid, _pe_realistic):

    # ------------------------------------------------------------------
    # the background and the helpers
    def init_balance(self, b_back=None, f0=None, N2_floor=1e-8, verbose=True):
        """
        the background of the decomposition:
        b_back    buoyancy b_back(z) at the T levels (interior, length nz),
                  default the horizontal mean of b over the wet cells of each
                  level. It is horizontally uniform: a horizontal gradient of
                  the background would put u grad b_back into L, which is not
                  skew-adjoint in a diagonal energy. With an equation of state
                  (eq_of_state > 0) the background is built from the means of
                  temp and salt (see below) and b_back is ignored
        f0        the reference Coriolis frequency, default the mean of
                  |coriolis_t| over the wet surface cells; it only scales the
                  band (om_cut, om_width are in units of f0) and the Lanczos
                  problem, L itself has the local f
        N2_floor  the least N^2 of the background (the weight 1/N^2)

        With an equation of state the buoyancy of the decomposition is
          b = -g (rho(T, S, p0) - rho_back(z))/rho_0,
        rho_back the horizontal mean of rho per level, and N^2 at the w levels
        is that of the potential density of the means, -g/rho_0 (drho/dT dT/dz
        + drho/dS dS/dz) with the derivatives of the EOS at the mean T, S and
        the level's p0 (the model's linearized tracer dynamics give db/dt =
        -w N^2 with this N^2). band_apply_TS puts a balanced b back into T at
        fixed S. The nonlinear tendency of b is the advection of b (the
        separate advection of T and S and the nonlinearity of the EOS are
        the same to first order).
        """
        h = self.halo
        mT = np.asarray(self.maskT)
        wet = mT[h:-h, h:-h, h:-h] > 0
        n = wet.sum(axis=(1, 2))
        def hmean(a):
            a = np.asarray(a)[h:-h, h:-h, h:-h]
            m = np.where(n > 0, (a*wet).sum(axis=(1, 2))/np.maximum(n, 1), 0.)
            for k in range(self.nz - 2, -1, -1):             # a dry level keeps the value above it
                if n[k] == 0:
                    m[k] = m[k + 1]
            return m
        dzw = np.asarray(self.dzw)[h:-h]
        if self.eq_of_state == 0:
            if b_back is None:
                b_back = hmean(self.temp)
            b_back = np.asarray(b_back, dtype='f8')
            N2w = np.zeros(self.nz)
            N2w[:-1] = (b_back[1:] - b_back[:-1])/dzw[:-1]
            self.bal_rho_back = None
        else:
            Tm, Sm = hmean(self.temp), hmean(self.salt)
            rho = get_rho(self.salt, self.temp, self.p0, self.eq_of_state)
            rhom = hmean(rho)
            p0 = np.asarray(self.p0)[h:-h, h, h] if self.p0 is not None else np.zeros(self.nz)
            rT = np.asarray(get_drhodT(Sm, Tm, p0, self.eq_of_state)) + np.zeros(self.nz)
            rS = np.asarray(get_drhodS(Sm, Tm, p0, self.eq_of_state)) + np.zeros(self.nz)
            N2w = np.zeros(self.nz)
            N2w[:-1] = -self.grav/self.rho_0*(0.5*(rT[1:] + rT[:-1])*(Tm[1:] - Tm[:-1])
                                              + 0.5*(rS[1:] + rS[:-1])*(Sm[1:] - Sm[:-1]))/dzw[:-1]
            N2w[-1] = N2w[-2]
            N2w = np.maximum(N2w, N2_floor)
            b_back = np.zeros(self.nz)                        # only its vertical differences enter
            b_back[1:] = np.cumsum(N2w[:-1]*dzw[:-1])
            # a fresh array per field: with the numpy backend asarray does not
            # copy, and the stored field would follow a later overwrite
            full = np.zeros((self.Nz, self.Ny, self.Nx)); full[h:-h] = rhom[:, None, None]
            self.bal_rho_back = self.apply_bc(OM.np.asarray(full))
            full = np.zeros((self.Nz, self.Ny, self.Nx)); full[h:-h] = rT[:, None, None]
            self.bal_rhoT = self.apply_bc(OM.np.asarray(full))
            self.bal_T_mean, self.bal_S_mean = Tm, Sm
        N2w[-1] = N2w[-2]
        N2w = np.maximum(N2w, N2_floor)
        self.bal_b_back_z = b_back
        self.bal_N2w = N2w
        self.bal_N0 = float(np.sqrt(np.mean(N2w)))
        f = np.asarray(self.coriolis_t)[h:-h, h:-h]
        wet_s = mT[-(h + 1), h:-h, h:-h] > 0
        self.bal_f0 = float(f0) if f0 is not None else float(np.mean(np.abs(f[wet_s])))
        full = np.zeros((self.Nz, self.Ny, self.Nx))
        full[h:-h] = b_back[:, None, None]
        self.bal_b_back = self.apply_bc(OM.np.asarray(full))*self.maskT
        self.bal_tau = getattr(self, 'bal_tau', 0)
        if verbose and self.my_pe == 0:
            print('balance: f0 %.3e 1/s, N0 %.3e 1/s, N^2 %.2e to %.2e' % (self.bal_f0, self.bal_N0, N2w.min(), N2w.max()))
        return

    def band_buoyancy(self):
        """the buoyancy perturbation of the model state (with halo)"""
        if self.eq_of_state == 0:
            return (self.temp - self.bal_b_back)*self.maskT
        rho = get_rho(self.salt, self.temp, self.p0, self.eq_of_state)
        return -self.grav/self.rho_0*(rho - self.bal_rho_back)*self.maskT

    def band_apply_TS(self, B):
        """
        the balanced state B of band_balanced as model fields (u, v, temp,
        salt, p_s with halo): the change of the buoyancy goes into temp at
        fixed salt (delta T = delta b rho_0/(-g drho/dT) with the EOS
        derivative at the mean state of the level), salt is unchanged; with
        eq_of_state 0 temp is the buoyancy itself
        """
        db = B[3] - self.band_buoyancy()
        if self.eq_of_state == 0:
            temp = self.temp + db
        else:
            rT = OM.np.where(self.bal_rhoT != 0., self.bal_rhoT, -1.)      # dry cells: no division by zero (masked anyway)
            temp = self.temp + db*self.rho_0/(-self.grav*rT)*self.maskT
        return B[0], B[1], temp, self.salt, B[5]

    def _int_dev(self, a):
        h = self.halo
        return OM.np.asarray(a)[h:-h, h:-h, h:-h]

    def _pad_dev(self, a):
        h = self.halo
        return self.apply_bc(OM.np.pad(OM.np.asarray(a), h))

    def model_state(self):
        """u, v, b, p of the model state, b the buoyancy perturbation"""
        b = self.band_buoyancy()
        p = self.band_ps()[None, :, :] + self.hydrostatic(b)
        return self.u, self.v, b, p

    def band_ps(self):
        """the surface pressure of the model state: one 2-D field (the split-explicit
        solver carries it so, the implicit one with its time levels, bal_tau)"""
        return self.p_s if self.p_s.ndim == 2 else self.p_s[self.bal_tau]

    def init_band(self, om_cut=0.5, om_width=0.3, m_lanczos=300, tol=1e-9, reorth=False, verbose=True, check=True):
        """
        the band and the inner product, after init_balance
        om_cut     centre of the transition between the slow and the fast band,
                   in units of f0
        om_width   half width of the transition (a smooth step in omega, see
                   _band_slow); the band om_cut -+ om_width must stay between
                   about 0.2 f0 and the lowest wave frequency 0.8 f0 (below,
                   the band inverse amplifies the near-zero topographic modes,
                   above, the waves count partly as slow). Defaults 0.5, 0.3
        m_lanczos  maximal number of Lanczos steps (each is two applications
                   of L); the cost is set by omega_max/d omega with omega_max
                   = c1 pi/dx and d omega the width of the transition
        tol        breakdown tolerance of the Lanczos recursion
        reorth     full reorthogonalization (slower, rarely needed)
        check      test the skew-adjointness of L on random fields
        """
        if not hasattr(self, 'bal_N2w'):
            raise RuntimeError('band: call init_balance first')
        if om_width > 0 and om_cut - om_width < 0.1 - 1e-6:
            raise RuntimeError('band: the transition reaches below 0.1 f0 (om_cut - om_width = %.2f), '
                               'the band inverse is unbounded there' % (om_cut - om_width))
        self.band_om_cut = om_cut*abs(self.bal_f0)
        self.band_om_width = om_width*abs(self.bal_f0)
        self.band_m, self.band_tol, self.band_verbose = m_lanczos, tol, verbose
        self.band_reorth = reorth
        h = self.halo
        I = self._interior
        # the rigid-lid pressure solver also with the implicit free surface:
        # the model's cf carries the term -1/(g dt^2) (make_coeff_surf_press is
        # jitted with the flag baked in, so the term is removed by hand)
        # (on a copy: with the numpy backend add_to_array works in place, and
        # the model's own free-surface solver would get the rigid-lid operator
        # - the emission runs of the icon033 window blew up by that, Nov 9 2026)
        if getattr(self, 'implicit_free_surface', False):
            self.band_cf = OM.add_to_array(OM.np.array(self.cf, copy=True), (1, 1, slice(None), slice(None)),
                                           1./(self.grav*self.dt**2)*self.maskT[-h - 1])
        else:
            self.band_cf = self.cf
        # the direct solver of the rigid-lid problem (numpy backend, one PE):
        # the 5-point operator of cg as a sparse matrix, factored once; the
        # iterative cg on the small 2-D problem took two thirds of the time of
        # the method (490 iterations per solve, 600 solves per Lanczos run)
        self.band_lu = None
        if self.n_pes_x*self.n_pes_y == 1:     # one PE, numpy or jax (OM.mpi is set whenever mpi4py is found, also for one PE);
                                               # on jax the factorization lives on the host and the 2-D right-hand side makes the round trip
            try:
                import scipy.sparse as sp
                import scipy.sparse.linalg as spl
                cf = np.asarray(self.band_cf)[:, :, h:-h, h:-h]
                ny, nx = cf.shape[2:]
                idx = np.arange(ny*nx).reshape(ny, nx)
                rows, cols, vals = [idx.ravel()], [idx.ravel()], [cf[1, 1].ravel()]
                for (dj, di, cj, ci) in ((1, 0, 2, 1), (-1, 0, 0, 1), (0, 1, 1, 2), (0, -1, 1, 0)):
                    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
                    jn, iin = jj + dj, ii + di
                    ok = (jn >= 0) & (jn < ny) & (iin >= 0) & (iin < nx)       # the halo is zero (walls)
                    rows.append(idx[ok]); cols.append(idx[jn[ok], iin[ok]]); vals.append(cf[cj, ci][ok])
                A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny*nx, ny*nx)).tolil()
                diag = np.asarray(A.diagonal())
                dry = np.where(diag == 0.)[0]
                for k in dry:                                   # dry cells: identity rows
                    A.rows[k] = [int(k)]; A.data[k] = [1.]
                wet = np.where(diag != 0.)[0]
                if len(wet):                                    # the constant is in the null space: pin one wet cell
                    k = int(wet[0]); A.rows[k] = [k]; A.data[k] = [1.]
                    self.band_lu_pin = k
                self.band_lu = spl.splu(A.tocsc())
                if verbose:
                    print('band: direct rigid-lid solver, %d x %d, %d dry cells' % (ny, nx, len(dry)))
            except Exception as e:
                if verbose:
                    print('band: no direct rigid-lid solver (%s), cg is used' % e)
                self.band_lu = None
        # the energy inner product: the cell volumes of the U, V and T points
        # and the weight 1/N^2 at the T levels, N^2 the mean of the two faces
        # of the cell (the open one at the top and bottom cell)
        vol_u = np.asarray(self.area_u*self.dzt_z + 0.*self.maskU)[h:-h, h:-h, h:-h]
        vol_v = np.asarray(self.area_v*self.dzt_z + 0.*self.maskV)[h:-h, h:-h, h:-h]
        vol_t = np.asarray(self.area_t*self.dzt_z + 0.*self.maskT)[h:-h, h:-h, h:-h]
        N2w = self.bal_N2w[:, None, None] + np.zeros((self.nz, self.ny, self.nx))
        N2lo = np.concatenate([N2w[:1], N2w[:-1]], axis=0)
        mW = I(self.maskW)
        mWlo = np.concatenate([np.zeros_like(mW[:1]), mW[:-1]], axis=0)
        open_hi = mW > 0; open_lo = mWlo > 0
        N2T = np.where(open_hi & open_lo, 0.5*(N2w + N2lo), np.where(open_hi, N2w, N2lo))
        mT = I(self.maskT)
        self.band_N2T = np.where(mT > 0, N2T, self.bal_N0**2)
        self.band_w = (I(self.maskU)*vol_u, I(self.maskV)*vol_v, mT*vol_t/self.band_N2T)
        self.band_w_dev = [OM.np.asarray(w) for w in self.band_w]
        self.band_its = 0
        if check:
            rng = np.random.default_rng(0)
            X = self._band_random(rng); Y = self._band_random(rng)
            LX, LY = self._band_L_np(X), self._band_L_np(Y)
            s = abs(self._band_dot(Y, LX) + self._band_dot(LY, X))/np.sqrt(self._band_dot(LX, LX)*self._band_dot(Y, Y))
            self.band_skew = s
            if verbose and self.my_pe == 0:
                print('band: cutoff %.2f f, width %.2f f, <Y,LX> + <LY,X> = %.1e of |LX||Y|' % (om_cut, om_width, s))
        return

    def _band_dot(self, X, Y):
        return float(sum(OM.np.sum(w*OM.np.asarray(x)*OM.np.asarray(y)) for w, x, y in zip(self.band_w_dev, X, Y)))

    def _div_t(self, fe, fn, ft):
        """divergence of fluxes at a T point (fn with cosu, as the flux routines)"""
        return ((fe - self.rollx(fe, 1))/(self.cost_y*self.dxt_x)
                + (fn - self.rolly(fn, 1))/(self.cost_y*self.dyt_y)
                + (ft - self.rollz(ft, 1))/self.dzt_z)

    @partial(OM.jaxjit, static_argnums=0)
    def _rigid_rhs(self, du, dv):
        """the divergence of the depth-integrated tendency (the right-hand side of the rigid-lid problem)"""
        h = self.halo
        fpx = OM.np.sum(du*self.maskU*self.dzt_z, axis=0)
        fpy = OM.np.sum(dv*self.maskV*self.dzt_z, axis=0)
        fpx, fpy = self.apply_bc(fpx), self.apply_bc(fpy)
        fpyc = fpy*self.cosu_y[0]
        return ((fpx - self.rollx(fpx, 1))/(self.cost_y[0]*self.dxt_x[0])
                + (fpyc - self.rolly(fpyc, 1))/(self.cost_y[0]*self.dyt_y[0]))*self.maskT[-h - 1]

    @partial(OM.jaxjit, static_argnums=0)
    def _rigid_apply(self, du, dv, p_s):
        """the tendencies without the gradient of the surface pressure p_s"""
        du = du - (self.rollx(p_s[None], -1) - p_s[None])/(self.cost_y*self.dxu_x)*self.maskU
        dv = dv - (self.rolly(p_s[None], -1) - p_s[None])/self.dyu_y*self.maskV
        return self.apply_bc(du), self.apply_bc(dv)

    def _rigid_solve(self, fc):
        """the surface pressure of the rigid-lid problem for the right-hand side fc: the direct solver when there is one, else cg"""
        h = self.halo
        if getattr(self, 'band_lu', None) is not None:
            rhs = np.asarray(fc)[h:-h, h:-h].ravel().copy()
            rhs[self.band_lu_pin] = 0.
            p_s = np.zeros(fc.shape)
            p_s[h:-h, h:-h] = self.band_lu.solve(rhs).reshape(fc.shape[0] - 2*h, fc.shape[1] - 2*h)
            self.band_cg_its = 0
            return self.apply_bc(OM.np.asarray(p_s))
        # the solver stops at an absolute change of the solution (epsilon), so
        # the problem is scaled to a solution of order one (the gravest mode)
        Lx = float(OM.np.sum(self.dxt[h:-h])); H = float(OM.np.sum(self.dzt[h:-h]))
        sc = (float(OM.np.max(OM.np.abs(fc))) or 1.)*Lx**2/(4*np.pi**2*H)
        p_s, err, its = self.cg(fc/sc, OM.np.zeros_like(fc), self.band_cf)
        self.band_cg_its = its
        return self.apply_bc(p_s)*sc

    def rigid_project(self, du, dv):
        """
        removes the gradient of the surface pressure from the tendencies
        (fields with halo), so that the depth integrated tendency is
        nondivergent, returns du, dv, p_s
        """
        p_s = self._rigid_solve(self._rigid_rhs(du, dv))
        du, dv = self._rigid_apply(du, dv, p_s)
        return du, dv, p_s

    def _coriolis(self, u, v):
        """the linear part of coriolis_and_metric (the grid-weighted averaging of pyOM2)"""
        f = self.coriolis_t[None]
        _fv = f*(v + self.rolly(v, 1))*self.dxt_x
        du = self.maskU*(_fv + self.rollx(_fv, -1))*0.25/self.dxu_x
        _fu = f*(u + self.rollx(u, 1))*self.dyt_y*self.cost_y
        dv = -self.maskV*(_fu + self.rolly(_fu, -1))*0.25/(self.dyu_y*self.cosu_y)
        return du, dv

    def lin_tend(self, u, v, b):
        """
        the linear tendencies (du, dv, db) of the fields (with halo): Coriolis,
        hydrostatic and surface pressure gradient, advection of the background
        buoyancy by the vertical velocity of continuity
        """
        du, dv, db, fc = self._lin_pre(u, v, b)
        p_s = self._rigid_solve(fc)
        du, dv = self._rigid_apply(du, dv, p_s)
        return du, dv, db, p_s

    @partial(OM.jaxjit, static_argnums=0)
    def _lin_pre(self, u, v, b):
        """the part of lin_tend before the rigid-lid solve, one jitted call on jax"""
        du, dv = self._coriolis(u, v)
        p_h = self.hydrostatic(b)
        du = du - (self.rollx(p_h, -1) - p_h)/(self.cost_y*self.dxu_x)*self.maskU
        dv = dv - (self.rolly(p_h, -1) - p_h)/self.dyu_y*self.maskV
        fc = self._rigid_rhs(du, dv)
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        fe, fn, ft = self.tracer_flux_2nd(self.bal_b_back, u, v, w, self.maskU, self.maskV, self.maskW)
        db = self._div_t(fe, fn, ft)*self.maskT
        return du, dv, self.apply_bc(db), fc

    def _band_L_np(self, X):
        """L applied to interior arrays, numpy result"""
        return [np.asarray(a) for a in self._band_L(X)]

    def _band_L(self, X):
        """L applied to interior arrays (of the backend: on jax the result stays on the device)"""
        I = self._int_dev
        u = self._pad_dev(X[0])*self.maskU; v = self._pad_dev(X[1])*self.maskV; b = self._pad_dev(X[2])
        du, dv, db, _ = self.lin_tend(u, v, b)
        return [I(du), I(dv), I(db)]

    def damp_tend(self, u, v, b):
        """the horizontal biharmonic damping tendencies of the model (fields with halo)"""
        z = lambda a: OM.np.zeros_like(a)
        du, dv, db = z(u), z(v), z(b)
        if hasattr(self, 'Ahbi'):
            un, vn = self.biharmonic_momentum(u, v, self.Ahbi, self.maskU, self.maskV)
            du, dv = (un - u)/self.dt, (vn - v)/self.dt
        if hasattr(self, 'Khbi'):
            db = (self.biharmonic_horizontal(b, self.Khbi, self.maskT) - b)/self.dt
        return du, dv, db

    def _band_nonlin(self, u, v, b):
        """advection tendencies and the metric terms (du, dv, db), projected on the rigid lid, interior arrays"""
        I = self._interior
        w = self.vertical_velocity(u, v, OM.np.zeros_like(u), self.maskW)
        fe, fn, ft = self.mom_flux_2nd_u(u, u, v, w, self.maskU, self.maskV, self.maskW)
        du = ((fe - self.rollx(fe, 1)) + (fn - self.rolly(fn, 1)) + (ft - self.rollz(ft, 1)))*self.maskU/(self.area_u*self.dzt_z)
        fe, fn, ft = self.mom_flux_2nd_v(v, u, v, w, self.maskU, self.maskV, self.maskW)
        dv = ((fe - self.rollx(fe, 1)) + (fn - self.rolly(fn, 1)) + (ft - self.rollz(ft, 1)))*self.maskV/(self.area_v*self.dzt_z)
        if self.coord_degree:                              # the metric terms, quadratic
            cu, cv = self.coriolis_and_metric(u, v, self.coriolis_t, self.maskU, self.maskV)
            lu, lv = self._coriolis(u, v)
            du = du + (cu - lu); dv = dv + (cv - lv)
        fe, fn, ft = self.tracer_flux_2nd(b, u, v, w, self.maskU, self.maskV, self.maskW)
        db = self._div_t(fe, fn, ft)*self.maskT
        du, dv, _ = self.rigid_project(self.apply_bc(du), self.apply_bc(dv))
        return [I(du), I(dv), I(db)]

    def band_fast_tendency(self, u, v, b, forcing=None):
        """
        the fast tendency of a state (fields with halo, b against b_back):
          r = (1 - P) [L X + N(X) + F]
        as interior arrays (du, dv, db). It is zero for a perfectly balanced
        state (the slaved parts cancel the fast tendency), and what remains
        is the rate at which the state generates waves: r/f is the wave
        velocity per inertial period/2 pi, the one-step form of the emission
        test. The forcing F (interior tendencies) is the slow wind as in
        band_balanced. One tendency evaluation and one Lanczos run.
        """
        I = self._interior
        # the throughflow (the potential flow of the transport the walls
        # block) is taken out first, as in band_balanced: its divergence sits
        # in the wall cells, where its vertical velocity through the thin
        # surface cells makes N(X) large and meaningless (on an icon033
        # window it was the whole fast tendency of the top 100 m, Nov 9 2026)
        du, dv, _ = self.rigid_project(u*self.maskU, v*self.maskV)
        LX = self._band_L_np([I(du), I(dv), I(b)])
        NX = self._band_nonlin(du, dv, b)
        T = [a + c for a, c in zip(LX, NX)]
        if forcing is not None:
            T = [a + np.asarray(f) for a, f in zip(T, forcing)]
        PT, = self._lanczos(T, 'P')
        return [a - c for a, c in zip(T, PT)]

    def band_quality(self, B, forcing=None, state=None):
        """
        quality of a balanced state B (band_balanced output) in three numbers
        and two fields:
          fast      the fast tendency of B (band_fast_tendency), rms over the
                    wet cells of u, v and b (b weighted with 1/N^2 as the
                    energy), as a fraction of the fast tendency of the model
                    state (or of `state` = (u, v, b) with halo): 1 means
                    nothing gained, 0 a perfectly slaved state
          fast_abs  the same in absolute terms, the rms wave velocity per
                    inertial period/2 pi, m/s (|r_u, r_v|/f)
          field     |r|/f as a 3-D interior field (m/s), where the balance
                    fails
          ratio     None here; the convergence of the orders is
                    |X2 - X1|/|X1 - X0| from the orders themselves (driver)
        """
        I = self._interior
        if state is None:
            u, v, b, _ = self.model_state()
        else:
            u, v, b = state
        r0 = self.band_fast_tendency(u, v, b, forcing)
        r1 = self.band_fast_tendency(B[0], B[1], B[3], forcing)
        f = np.abs(np.asarray(self.coriolis_t))[self.halo:-self.halo, self.halo:-self.halo]
        f = np.maximum(f, 1e-6)[None]
        nrm = lambda r: np.sqrt(self._band_dot(r, r))
        field = np.hypot(r1[0], r1[1])/f*I(self.maskT)
        field0 = np.hypot(r0[0], r0[1])/f*I(self.maskT)
        wet = I(self.maskT) > 0
        return dict(fast=nrm(r1)/max(nrm(r0), 1e-300), fast_abs=float(np.sqrt(np.mean(field[wet]**2))),
                    fast_abs_state=float(np.sqrt(np.mean(field0[wet]**2))), field=field, field_state=field0)

    def band_balanced(self, order=2, forcing=None, forcing_dot=None, forcing_mode='slab'):
        """
        balanced state of the model state up to the order (0, 1 or 2),
        returns the fields (u, v, w, b, p, p_s), b against b_back

        forcing      (Fu, Fv, Fb), the slow forcing of the model at the time of
                     the state as tendencies (interior arrays, Fb may be 0),
                     e.g. the wind stress spread over the mixed layer. It is
                     slaved like the nonlinear terms: X1 = K [N(X0) + F], K F
                     the steady forced response (the Ekman flow of a steady
                     wind), P F enters the tendency of X0. Only the part of the
                     forcing that varies slowly against f belongs here (a
                     low-pass of the wind); the response to the rest is
                     near-inertial and is a wave
        forcing_dot  dF/dt at that time (interior arrays), enters the second
                     order as K dF/dt
        forcing_mode 'band': the slaved response is K F, the band inverse
                     applied to the forcing projected on the rigid lid.
                     'slab' (default): the slaved response is the Ekman slab
                     u_E = (F_v, -F_u)/f cell by cell, the exact steady
                     response of the Coriolis term to a large-scale forcing
                     (what K F is for a horizontally uniform stress), and
                     the wind enters the slow tendency as P [F + L u_E] (the
                     Ekman pumping). In a window the slab is the choice: the
                     rigid-lid projection of F removes the Ekman transport
                     through the walls and spreads a barotropic return flow
                     over the window (a potential flow from the walls), and
                     K F then carries the wall artefact everywhere; the slab
                     has none, its blocked transport is the throughflow kept
                     as slow (Nov 9 2026)
        """
        I = self._interior
        u, v, b, p = self.model_state()
        Fz = [0.*I(u), 0.*I(v), 0.*I(b)]
        proj = lambda F: (lambda d: [I(d[0]), I(d[1]), np.asarray(F[2]) + Fz[2]])(
            self.rigid_project(self._pad(np.asarray(F[0]) + Fz[0])*self.maskU, self._pad(np.asarray(F[1]) + Fz[1])*self.maskV))
        slab = forcing is not None and forcing_mode == 'slab'
        if slab:
            f = np.asarray(self.coriolis_t)[self.halo:-self.halo, self.halo:-self.halo]
            fU = 0.5*(f + np.roll(f, -1, axis=1)); fV = 0.5*(f + np.roll(f, -1, axis=0))   # f at the U and V points
            ekman = lambda F: [np.asarray(F[1])/fU[None]*I(self.maskU), -np.asarray(F[0])/fV[None]*I(self.maskV), Fz[2]]
            XE = ekman(forcing)
            XEd = ekman(forcing_dot) if forcing_dot is not None else Fz
            Fs, Fd = Fz, Fz
        else:
            Fs = proj(forcing) if forcing is not None else Fz
            Fd = proj(forcing_dot) if forcing_dot is not None else Fz
        # the part of the depth-integrated flow that the walls block (a window
        # cut from a larger domain): the potential flow of the normal
        # transport at the walls, the balanced flow the outside imposes on the
        # window, counts as slow; the projection sees the nondivergent rest
        du, dv, _ = self.rigid_project(u*self.maskU, v*self.maskV)
        Xphi = [I(u) - I(du), I(v) - I(dv), 0.*I(b)]
        X0 = self._band_fields(self._lanczos([I(du), I(dv), I(b)], 'P')[0])
        # the orders are built on the nondivergent X0 alone: the potential
        # flow has its divergence in the cells next to the walls (the blocked
        # inflow), there its vertical velocity is of order u H/dx, and the
        # advection by it through thin surface cells made the first order
        # several m/s at the walls of an icon033 window (Nov 9 2026). The
        # advection of X0 by the throughflow, a slow tendency, is left out
        Xout = self._band_fields([I(a) + c for a, c in zip((X0[0], X0[1], X0[3]), Xphi)])
        Xall = [I(u), I(v), I(b)]
        self.band_phi_rms = float(np.sqrt(self._band_dot(Xphi, Xphi)/max(self._band_dot(Xall, Xall), 1e-300)))   # relative to the state
        out = list(Xout)
        if order >= 1:
            NX = lambda u, v, b: [a + f for a, f in zip(self._band_nonlin(u, v, b), Fs)]
            NX0 = NX(X0[0], X0[1], X0[3])
            Q, PN = self._lanczos(NX0, 'qS')              # K = L q(B); slow tendency of X0 + X1 from N(X0)
            F1 = self._band_L_np(Q)
            if slab:
                # the Ekman slab as the slaved response, and the slow tendency
                # of the wind P [F + L u_E] (F and the slab's Coriolis term
                # cancel, the slab's divergence pumps the stratification)
                F1 = [a + c for a, c in zip(F1, XE)]
                TF = [np.asarray(a) + c for a, c in zip(forcing, self._band_L_np(XE))]
                PF, = self._lanczos(TF, 'P')
                PN = [a + c for a, c in zip(PN, PF)]
            X1 = self._band_fields(F1)
            out = [a + c for a, c in zip(out, X1)]
            if order >= 2:
                N0 = self._band_nonlin(X0[0], X0[1], X0[3])
                N1 = self._band_nonlin(X1[0], X1[1], X1[3])
                N01 = self._band_nonlin(X0[0] + X1[0], X0[1] + X1[1], X0[3] + X1[3])
                dF = [a - c - d for a, c, d in zip(N01, N0, N1)]
                # the slow tendency of X0 + X1, L X0 + P [N(X0) + L X1] + D X0,
                # and the derivative of the quadratic N along it (exact: the
                # bilinear cross term), scaled to the size of X0 for round-off
                LX0 = self._band_L_np([I(X0[0]), I(X0[1]), I(X0[3])])
                DX0 = [I(a) for a in self.damp_tend(X0[0], X0[1], X0[3])]
                T0 = [a + c + d for a, c, d in zip(LX0, PN, DX0)]
                sc = np.sqrt(self._band_dot([I(X0[0]), I(X0[1]), I(X0[3])], [I(X0[0]), I(X0[1]), I(X0[3])])
                             /max(self._band_dot(T0, T0), 1e-300))
                Td = self._band_fields([sc*a for a in T0])
                Nd = self._band_nonlin(Td[0], Td[1], Td[3])
                N0d = self._band_nonlin(X0[0] + Td[0], X0[1] + Td[1], X0[3] + Td[3])
                dF1 = self.band_inverse([(a - c - d)/sc + f for a, c, d, f in zip(N0d, N0, Nd, Fd)])
                if slab and forcing_dot is not None:
                    dF1 = [a + c for a, c in zip(dF1, XEd)]       # d u_E/dt of the slab
                D = [I(a) for a in self.damp_tend(X1[0], X1[1], X1[3])]
                G = [d - df1 + dd for d, df1, dd in zip(dF, dF1, D)]
                X2 = self._band_fields(self.band_inverse(G))
                out = [a + c for a, c in zip(out, X2)]
        return out
