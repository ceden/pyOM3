"""
NEAR-INERTIAL ENERGY FLUX OUT OF THE MIXED LAYER, online (Rimac, von Storch
and Eden 2016, JPO 46). The quantity is the near-inertial pressure work
through the base of the mixed layer, the base being the SLOPING surface
z = -h(x,y,t): the flux vector p'u' through its normal, that is p'w' plus the
horizontal components p'u' dh/dx + p'v' dh/dy, as in Rimac et al. Together
with it come the wind work on the near-inertial currents it is paid from, the
near-inertial kinetic energy, and the vertical part at every level so the
depth dependence is kept. Rimac et al. formed it offline from hourly output. Here the band is
isolated ONLINE by complex demodulation at the local inertial frequency, and
the products go into the ordinary online means (exp_ocean.register_average),
so they come out per averaging window (a month in production) and need no
high-frequency output.

THE DEMODULATION. A signal x(t) in the band around f is x = Re(A e^{-ift})
with a slowly varying complex amplitude A. Multiplying by e^{ift} gives
A/2 + (A*/2) e^{2ift}, and a low pass with time scale T_w leaves Z = A/2. The
low pass is two first-order filters in series,
    z1 <- z1 + a (x e^{ift} - z1) ,  z2 <- z2 + a (z1 - z2) ,  a = dt/T_w
with T_w = 1/(NI_BAND |f|), so the pass band is |omega - f| < NI_BAND |f| at
half power and the mesoscale (omega = 0) is damped by (NI_BAND)^4 in power.
For the horizontal velocity the complex signal u + iv is demodulated with the
SIGNED f, which keeps only the inertial sense of rotation (clockwise on the
northern hemisphere); for w, p and the stress the sign is immaterial. |f| is
held at its value at NI_LAT_MIN towards the equator, where inertial motion
has no meaning and T_w would diverge. The filter is causal and needs a few
T_w to settle: after a start from a restart without its state (a branch from
another run) the first days of the first window carry that transient.

THE PRODUCTS, every step (A = 2 Z for the real signals w and p, A = Z for the
complex signals u + iv and tau_x + i tau_y, whose demodulation returns the
inertial rotation sense directly):
    ni_pw   = 1/2 Re(P W*) rho_0        near-inertial pressure work at the W faces [W/m2]
    pw      = w p rho_0                 the total pressure work at the W faces [W/m2]
    ni_ke   = 1/2 |U|^2                 near-inertial kinetic energy, T points [m2/s2]
    ni_wind = Re(T U_s*) rho_0          wind work on the near-inertial surface current [W/m2]
    wind    = tau . u_L rho_0           total wind work (exp_ocean.stress_work_field) [W/m2]
    p       = p_h + p_s (+ p_SS)        pressure / rho_0 at the T points [m2/s2]
    ni_pu, ni_pv = Re(P U*), -Im(P U*) rho_0   near-inertial horizontal pressure work, T points
    ni_pw_mld, pw_mld                   ni_pw and pw interpolated to the mixed layer base
    ni_pu_mld, ni_pv_mld                ni_pu, ni_pv interpolated to the mixed layer base
    ni_ph_mld, ph_mld                   the horizontal part of the flux through the
                                        sloping base, (p u dh/dx + p v dh/dy) at z = -h,
                                        near-inertial and total, POSITIVE UP like pw
    mld_ni                              the mixed layer depth those used [m]
The flux of energy out of the mixed layer is -(ni_pw_mld + ni_ph_mld): the
normal of z = -h points along (dh/dx, dh/dy, 1), so the flux through it per
unit horizontal area is p w + p u dh/dx + p v dh/dy. The slope is the centred
difference of h on the sphere, one-sided next to land. P at the T points is a
demodulation of its own (state pT).
p is the model's hydrostatic pressure anomaly plus the surface pressure,
both kinematic (m2/s2); at a W face it is the mean of the two T points, at
the surface face the surface value. The window mean of pw contains the mean
flow, <w><p> can be subtracted afterwards from the means of w and p, which is
why p is in the list. The mixed layer depth is the 0.03 kg/m3 criterion of
exp_ocean.mixed_layer_depth, evaluated every NI_MLD_EVERY steps (one
equation of state call each) and held in between.

RAW DUMPS (NI_DUMP_EVERY > 0): every so many steps each rank writes u, v, w,
p of the NI_LEVELS top levels and the mixed layer depth as float32 into
<snapshot dir>_ni/pe=<rank>/n=<step>/, the layout of snap_raw, for the
offline analysis of Rimac et al. as a check of the online one.

RESTART: the filter states and the held mixed layer depth are part of the
raw ocean restart (coupler_io, keys ni_*); the gathered restart does not
carry them and the filters start from zero.

cfg names: NI_DIAG, NI_BAND, NI_LEVELS, NI_LAT_MIN, NI_MLD_EVERY,
NI_DUMP_EVERY, omega_e, rho_0 (exp_common.Cfg.DEFAULTS has them all).
"""
import os
import numpy as np


class ni_common:

    # --- set up: states, averages, dump directory ---------------------------
    def ni_setup(self):
        """Called by exp_ocean.prepare_for_run after the model is built and
        BEFORE average_setup, because the products are registered as online
        means here."""
        cfg = self.cfg
        self._ni = bool(cfg.NI_DIAG)
        if not self._ni:
            return
        h = self.halo
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        nz = Nz - 2*h
        K = int(cfg.NI_LEVELS)
        K = nz if K <= 0 else min(K, nz)
        self._ni_K  = K
        self._ni_k0 = Nz - h - K                 # padded index of the deepest level kept
        # the signed inertial frequency, |f| floored at NI_LAT_MIN
        f    = np.asarray(self.coriolis_t)
        fmin = 2.0*cfg.omega_e*np.sin(np.radians(float(cfg.NI_LAT_MIN)))
        fa   = np.maximum(np.abs(f), fmin)
        self._ni_f = np.where(f >= 0.0, fa, -fa)
        self._ni_a = self.dt*float(cfg.NI_BAND)*fa        # dt/T_w, (Ny, Nx)
        # filter states, two stages each: w and p at the W faces, u + iv at
        # the T points (3-D over the K levels), the stress at the T points
        z3 = lambda: [np.zeros((K, Ny, Nx), np.complex128) for _ in range(2)]
        z2 = lambda: [np.zeros((Ny, Nx), np.complex128) for _ in range(2)]
        self._ni_z = {'w': z3(), 'p': z3(), 'pT': z3(), 'uv': z3(), 'tau': z2()}
        self._ni_mld = np.zeros((Ny, Nx))
        self._ni_have_mld = False
        # face depths of the kept levels, surface first, for the interpolation
        zw = np.asarray(self.zw)
        D  = (zw[-h-1] - zw)[self._ni_k0:Nz-h][::-1]
        self._ni_Dface = np.maximum(D, 0.0)
        # products of the current step, filled by ni_step, read by the means
        self._ni_out = {}
        r = cfg.rho_0
        T3, W3 = ('time', 'zt', 'yt', 'xt'), ('time', 'zu', 'yt', 'xt')
        T2 = ('time', 'yt', 'xt')
        get = lambda nme: (lambda tau, nme=nme: self._ni_out[nme])
        for nme, dims, units, ln in (
                ('ni_pw',   W3, 'W/m2',  'near-inertial pressure work 1/2 Re(P W*) rho_0 at the W faces'),
                ('pw',      W3, 'W/m2',  'pressure work w p rho_0 at the W faces'),
                ('ni_ke',   T3, 'm2/s2', 'near-inertial kinetic energy 1/2 |U|^2'),
                ('p',       T3, 'm2/s2', 'pressure / rho_0, hydrostatic anomaly + surface'),
                ('ni_wind', T2, 'W/m2',  'wind work on the near-inertial surface current Re(T U*) rho_0'),
                ('wind',    T2, 'W/m2',  'wind work tau . u_L rho_0 on the surface velocity'),
                ('ni_pw_mld', T2, 'W/m2', 'near-inertial pressure work p w at the mixed layer base, positive up'),
                ('pw_mld',  T2, 'W/m2',  'pressure work p w at the mixed layer base, positive up'),
                ('ni_pu_mld', T2, 'W/m2', 'near-inertial horizontal pressure work Re(P U*) rho_0 at the mixed layer base'),
                ('ni_pv_mld', T2, 'W/m2', 'near-inertial horizontal pressure work -Im(P U*) rho_0 at the mixed layer base'),
                ('ni_ph_mld', T2, 'W/m2', 'near-inertial p u dh/dx + p v dh/dy at the sloping mixed layer base, positive up'),
                ('ph_mld',  T2, 'W/m2',  'p u dh/dx + p v dh/dy at the sloping mixed layer base, positive up'),
                ('mld_ni',  T2, 'm',     'mixed layer depth used for the fluxes above '
                                         '(0.03 kg/m3, held between evaluations)')):
            self.register_average(nme, dims, units, ln, get(nme))
        # raw dumps
        self._ni_dump = int(cfg.NI_DUMP_EVERY)
        if self._ni_dump > 0:
            base = self.snap_file_name
            if base.endswith('.cdf'):
                base = base[:-4]
            d = base + '_ni'
            self._ni_dump_dir = d
            if self.my_pe == 0:
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, 'zt.npy'), np.asarray(self.zt)[self._ni_k0:Nz-h])
                np.save(os.path.join(d, 'zw.npy'), zw[self._ni_k0:Nz-h])
                np.save(os.path.join(d, 'decomposition.npy'),
                        np.array([self.n_pes_x, self.n_pes_y]))
                for pe in range(self.n_pes):
                    os.makedirs(os.path.join(d, 'pe=%d' % pe), exist_ok=True)
            self.mpi_barrier()
        if self.my_pe == 0:
            Tw_mid = 1.0/(float(cfg.NI_BAND)*2.0*cfg.omega_e*np.sin(np.radians(45.0)))
            print('CONFIG near-inertial diagnostic: band +-%.2f f (T_w = %.1f d at 45 deg), '
                  '%d levels to %.0f m, mixed layer depth every %d steps%s'
                  % (float(cfg.NI_BAND), Tw_mid/86400.0, K, self._ni_Dface[-1],
                     int(cfg.NI_MLD_EVERY),
                     (', raw dumps every %d steps -> %s/' % (self._ni_dump, self._ni_dump_dir))
                     if self._ni_dump > 0 else ''), flush=True)
        return

    # --- one step -----------------------------------------------------------
    def ni_step(self, n, tau):
        """Advance the demodulation with the state after step n and form the
        products the means pick up. Called from diagnose before average."""
        if not getattr(self, '_ni', False):
            return
        h  = self.halo
        Nz = self.Nz
        k0, K = self._ni_k0, self._ni_K
        ks = Nz - h - 1
        sl3 = slice(k0, Nz - h)
        r = self.cfg.rho_0
        t = n*self.dt
        E = np.exp(1j*self._ni_f*t)                        # e^{ift}, (Ny, Nx)
        a = self._ni_a

        mT = np.asarray(self.maskT)[sl3]
        mW = np.asarray(self.maskW)[sl3]
        u  = np.asarray(self.u)[sl3]
        v  = np.asarray(self.v)[sl3]
        w  = np.asarray(self.w)[sl3]*mW
        ps = np.asarray(self.p_s if self.explicit_free_surface else self.p_s[tau])
        p  = (np.asarray(self.p_h)[sl3] + ps[None])*mT
        # STOKES_SHEAR: the vertical component of the vortex force sits in the
        # hydrostatic relation as the pressure p_SS (exp_ocean.stokes_shear_force),
        # formed at the same time level as p_h; the pressure work carries it
        if hasattr(self, 'p_SS'):
            p = p + np.asarray(self.p_SS)[sl3]*mT
        # p at the W faces: face k lies between the T points k and k+1 (the
        # vertical index counts upward), the surface face takes the surface value
        pf = np.empty_like(p)
        pf[:-1] = 0.5*(p[:-1] + p[1:])
        pf[-1]  = p[-1]
        pf = pf*mW
        # velocity at the T points: U point i is the east face of T cell i
        uT = 0.5*(u + np.roll(u, 1, axis=2))*mT
        vT = 0.5*(v + np.roll(v, 1, axis=1))*mT
        # the stress the step applied, U/V -> T points
        tx, ty = np.asarray(self.taux), np.asarray(self.tauy)
        if hasattr(self, '_tau_dir_x'):
            tx = tx + np.asarray(self._tau_dir_x); ty = ty + np.asarray(self._tau_dir_y)
        txT = 0.5*(tx + np.roll(tx, 1, axis=1))*mT[-1]
        tyT = 0.5*(ty + np.roll(ty, 1, axis=0))*mT[-1]

        def upd(z, x):
            z[0] += a*(x - z[0])
            z[1] += a*(z[0] - z[1])
            return z[1]

        Zw = upd(self._ni_z['w'],  w*E)
        Zp = upd(self._ni_z['p'],  pf*E)
        ZpT = upd(self._ni_z['pT'], p*E)
        Zu = upd(self._ni_z['uv'], (uT + 1j*vT)*E)
        Zt = upd(self._ni_z['tau'], (txT + 1j*tyT)*E)

        # mixed layer depth, every NI_MLD_EVERY steps
        if (n % max(int(self.cfg.NI_MLD_EVERY), 1) == 0) or not self._ni_have_mld:
            self._ni_mld = np.asarray(self.mixed_layer_depth())
            self._ni_have_mld = True
            self._ni_hx, self._ni_hy = self._ni_slope(self._ni_mld, mT[-1])

        # the products. For the real signals w and p the amplitude is A = 2 Z,
        # so 1/2 Re(A_p A_w*) = 2 Re(Z_p Z_w*). For the complex signals u + iv
        # and tau_x + i tau_y the demodulation returns the amplitude of the
        # inertial rotation sense itself, A = Z, so 1/2 |A|^2 = 1/2 |Z|^2, and
        # the work of the rotating stress on the rotating current is
        # tau . u = Re(Z_tau Z_u*) with no factor 1/2, the two vectors keep
        # their angle.
        ni_pw = 2.0*r*(Zp*np.conj(Zw)).real
        pw    = r*w*pf
        ni_ke = 0.5*(Zu.real**2 + Zu.imag**2)
        out = self._ni_out
        for nme, val in (('ni_pw', ni_pw), ('pw', pw), ('ni_ke', ni_ke), ('p', p)):
            full = np.zeros((Nz, self.Ny, self.Nx))
            full[sl3] = val
            out[nme] = full
        out['ni_wind'] = r*(Zt*np.conj(Zu[-1])).real
        out['wind']    = np.asarray(self.stress_work_field())
        out['ni_pw_mld'] = self._ni_at_depth(ni_pw, self._ni_mld)
        out['pw_mld']    = self._ni_at_depth(pw, self._ni_mld)
        # the horizontal part through the sloping base, T-point fields of
        # the K levels interpolated to the base like the W-face fields (the
        # T points are half a cell above the faces, which the interpolation
        # of fields this smooth does not see)
        PU = ZpT*np.conj(Zu)
        ni_pu, ni_pv = r*PU.real, -r*PU.imag
        pu, pv = r*p*uT, r*p*vT
        hx, hy = self._ni_hx, self._ni_hy
        out['ni_pu_mld'] = self._ni_at_depth(ni_pu, self._ni_mld)
        out['ni_pv_mld'] = self._ni_at_depth(ni_pv, self._ni_mld)
        out['ni_ph_mld'] = out['ni_pu_mld']*hx + out['ni_pv_mld']*hy
        out['ph_mld']    = self._ni_at_depth(pu, self._ni_mld)*hx + self._ni_at_depth(pv, self._ni_mld)*hy
        out['mld_ni']    = self._ni_mld

        if self._ni_dump > 0 and n % self._ni_dump == 0:
            self._ni_write_dump(n, u, v, w, p)
        return

    def _ni_slope(self, h, m):
        """dh/dx, dh/dy of the mixed layer depth on the sphere, centred
        differences of the T-point field, one-sided where a neighbour is
        land, zero where both are. m is the surface T mask."""
        dxu = np.asarray(self.dxu_x).reshape(-1)           # xt[i+1] - xt[i]
        dyu = np.asarray(self.dyu_y).reshape(-1)           # yt[j+1] - yt[j]
        cost = np.asarray(self.cost_y).reshape(-1)
        rx = lambda a, s: np.roll(a, s, axis=1)
        ry = lambda a, s: np.roll(a, s, axis=0)
        mE, mW = rx(m, -1)*m, rx(m, 1)*m                    # neighbour wet and self wet
        mN, mS = ry(m, -1)*m, ry(m, 1)*m
        # east neighbour at distance dxu[i], west at dxu[i-1], north dyu[j], south dyu[j-1]
        dxE = dxu[None, :]*cost[:, None]; dxW = np.roll(dxu, 1)[None, :]*cost[:, None]
        dyN = dyu[:, None]*np.ones_like(m); dyS = np.roll(dyu, 1)[:, None]*np.ones_like(m)
        num = (rx(h, -1) - h)*mE + (h - rx(h, 1))*mW
        den = dxE*mE + dxW*mW
        hx = np.where(den > 0.0, num/np.where(den > 0.0, den, 1.0), 0.0)
        num = (ry(h, -1) - h)*mN + (h - ry(h, 1))*mS
        den = dyN*mN + dyS*mS
        hy = np.where(den > 0.0, num/np.where(den > 0.0, den, 1.0), 0.0)
        return hx, hy

    def _ni_at_depth(self, F, depth):
        """Linear interpolation of a W-face field F (K levels, padded
        horizontally, vertical index upward) to the depth (Ny, Nx) [m]. Below
        the deepest kept face the deepest value is used, which is zero below
        the bottom."""
        D  = self._ni_Dface                        # surface first, increasing
        Fs = F[::-1]                               # surface first
        K  = len(D)
        idx = np.searchsorted(D, depth, side='right')
        idx = np.clip(idx, 1, K - 1)
        D0, D1 = D[idx - 1], D[idx]
        wgt = np.clip((depth - D0)/np.maximum(D1 - D0, 1e-12), 0.0, 1.0)
        F0 = np.take_along_axis(Fs, (idx - 1)[None], 0)[0]
        F1 = np.take_along_axis(Fs, idx[None], 0)[0]
        return (1.0 - wgt)*F0 + wgt*F1

    def _ni_write_dump(self, n, u, v, w, p):
        d = os.path.join(self._ni_dump_dir, 'pe=%d' % self.my_pe, 'n=%d' % n)
        os.makedirs(d, exist_ok=True)
        h = self.halo
        s = (slice(None), slice(h, -h), slice(h, -h))
        np.save(os.path.join(d, 't.npy'), np.array(n*self.dt))
        for nme, val in (('u', u), ('v', v), ('w', w), ('p', p)):
            np.save(os.path.join(d, nme + '.npy'), val[s].astype(np.float32))
        np.save(os.path.join(d, 'mld.npy'), self._ni_mld[h:-h, h:-h].astype(np.float32))
        return

    # --- restart ------------------------------------------------------------
    def ni_state(self):
        """name -> array of everything the filters hold, for the raw restart;
        None when the diagnostic is off."""
        if not getattr(self, '_ni', False):
            return None
        out = {}
        for nme, z in self._ni_z.items():
            out[nme + '0'], out[nme + '1'] = z[0], z[1]
        out['mld'] = self._ni_mld
        return out

    def ni_restore(self, d):
        if not getattr(self, '_ni', False):
            return
        missing = []
        for nme, z in self._ni_z.items():
            try:
                z0, z1 = np.asarray(d[nme + '0']), np.asarray(d[nme + '1'])
                if z0.shape != z[0].shape:
                    raise ValueError('shape')
                z[0][...], z[1][...] = z0, z1
            except (KeyError, ValueError):
                missing.append(nme)
        if 'mld' in d:
            self._ni_mld = np.asarray(d['mld'], np.float64)
            self._ni_have_mld = True
            self._ni_hx, self._ni_hy = self._ni_slope(
                self._ni_mld, np.asarray(self.maskT)[self.Nz - self.halo - 1])
        if self.my_pe == 0:
            if not missing:
                print('near-inertial diagnostic: filter states restored', flush=True)
            else:
                print('near-inertial diagnostic: filter states restored except %s, which start '
                      'from zero (transient of a few T_w)' % ', '.join(missing), flush=True)
        return
