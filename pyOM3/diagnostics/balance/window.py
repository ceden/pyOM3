"""
 A window of a pe_realistic raw snapshot (snap_raw) as a band_pe_realistic model
 (Nov 2026; moved here from pe_realistic/band_window.py, the code
unchanged): reads the blocks of the ranks, cuts the window, builds the model
 on the window's part of the grid with walls around it, the topography from
 the land mask of the snapshot and f from the latitude, and sets the state.

   snap = read_raw('raw_snap', n, window=(i0, i1, j0, j1))   # the step n, only the window's blocks
   M = window_model(snap, i0, i1, j0, j1, par)                # T cells i0 <= i < i1, j0 <= j < j1
   M.init_balance(); M.init_band()
   B = M.band_balanced(2)
   u, v, temp, salt, p_s = M.band_apply_TS(B)

 par are the model parameters of the run that the window must share with it:
 eq_of_state, rho_0, dt (the implicit free surface and the emission test),
 Ahbi, Khbi (the damping of the second order), and the free surface flags;
 the window runs its own, simpler physics in an emission test (second order
 advection, no mixing) unless more is passed.

 write_run_params / read_run_params: run_params.json, the parameters of the
 run beside its raw snapshots (written by setups/coupled_common/run.py, read
 by the driver run_window.py).
"""
import os
import numpy as np
import pyOM3 as OM
from pyOM3.diagnostics.balance.band import band_pe_realistic


def read_raw(path, n, variables=('u', 'v', 'temp', 'salt', 'p_s', 'p_h'), window=None):
    """
    the raw snapshot of step n as a dict of interior arrays, with the
    coordinates xt, yt, zt, xu, yu, zu (xt, yt, xu, yu in degrees when the
    run had coord_degree, zt, zu the model's vertical coordinate), the time t
    and the land mask 'wet': the hydrostatic pressure p_h is zero exactly in
    the land cells (it carries maskT) and nowhere else (temp and salt are
    not masked in the model).
    window = (i0, i1, j0, j1): only the T cells i0 <= i < i1, j0 <= j < j1 of
    the global interior are read (the rank blocks that overlap it, memory
    mapped), and the field arrays are those of the window; the coordinates
    stay global. Without it the whole domain is assembled, which is slow
    for a global run (every block of every rank).
    """
    out = {}
    for c in ('xt', 'yt', 'zt', 'xu', 'yu', 'zu'):
        out[c] = np.load(os.path.join(path, c + '.npy'))
    npx, npy = [int(a) for a in np.load(os.path.join(path, 'decomposition.npy'))]
    nx, ny, nz = len(out['xt']), len(out['yt']), len(out['zt'])
    x_blk, y_blk = (nx - 1)//npx + 1, (ny - 1)//npy + 1
    i0, i1, j0, j1 = window if window is not None else (0, nx, 0, ny)
    out['window'] = (i0, i1, j0, j1)
    for var in variables:
        full = np.zeros((nz, j1 - j0, i1 - i0)) if var != 'p_s' else np.zeros((j1 - j0, i1 - i0))
        for pe in range(npx*npy):
            bx, by = pe % npx, pe//npx
            xs, ys = bx*x_blk, by*y_blk
            xe, ye = min(xs + x_blk, nx), min(ys + y_blk, ny)
            if xe <= i0 or xs >= i1 or ye <= j0 or ys >= j1:
                continue
            a = np.load(os.path.join(path, 'pe=%d' % pe, 'n=%d' % n, var + '.npy'), mmap_mode='r')
            ia, ib = max(i0, xs), min(i1, xe); ja, jb = max(j0, ys), min(j1, ye)
            if var == 'p_s':
                full[ja - j0:jb - j0, ia - i0:ib - i0] = a[ja - ys:jb - ys, ia - xs:ib - xs]
            else:
                full[:, ja - j0:jb - j0, ia - i0:ib - i0] = a[:, ja - ys:jb - ys, ia - xs:ib - xs]
        out[var] = full
    out['t'] = float(np.load(os.path.join(path, 'pe=0', 'n=%d' % n, 't.npy')))
    out['wet'] = out['p_h'] != 0.
    return out


def read_coords(path):
    """the coordinates of a raw snapshot directory alone (xt, yt, zt, xu, yu, zu)"""
    return {c: np.load(os.path.join(path, c + '.npy')) for c in ('xt', 'yt', 'zt', 'xu', 'yu', 'zu')}


class _window(band_pe_realistic):

    def set_coriolis(self, f0=None):
        lat = np.asarray(self.yt)
        f = 2*7.2921e-5*np.sin(np.deg2rad(lat))
        self.coriolis_t = OM.np.asarray(f[:, None] + np.zeros((self.Ny, self.Nx)))

    def set_topography(self):
        kb = np.zeros((self.Ny, self.Nx), int)
        h = self.halo
        kb[h:-h, h:-h] = self.win_kbot
        self.kbot = OM.np.asarray(kb)

    def set_forcing(self, tau):
        """the steady body force of the emission test (emission(..., forcing)),
        added to du[tau], dv[tau] as the model's forcing hook; nothing without
        it"""
        F = getattr(self, 'win_forcing', None)
        if F is None:
            return
        self.du = OM.modify_array(self.du, (tau, slice(None), slice(None), slice(None)), self.du[tau] + F[0])
        self.dv = OM.modify_array(self.dv, (tau, slice(None), slice(None), slice(None)), self.dv[tau] + F[1])


def window_model(snap, i0, i1, j0, j1, par, halo_points=1, rim=0):
    """
    the band_pe_realistic model of the window i0 <= i < i1, j0 <= j < j1 (T cell
    indices of the global interior) with the state of the snapshot. par: the
    parameters the window shares with the run (see the module docstring);
    nx, ny, nz, the grid and the origin are set here. The window is closed by
    walls; the land of the snapshot is the topography (kbot from the wet
    cells, counted from the bottom).
    rim: the state is tapered to the background over this many cells from
    the walls (u, v, p_s and the anomalies of temp and salt against their
    horizontal means in the window, multiplied by sin^2(pi/2 d/rim) with d
    the distance from the wall in cells). Without it the flow of the global
    model through the wall faces is blocked: the barotropic part becomes the
    potential flow that band_balanced keeps, the baroclinic part a divergence
    of the first column with a vertical velocity of order u dz/dx there, and
    the band inverse amplifies the wall-trapped modes of the band to several
    m/s (the icon033 window of Nov 9 2026). The balanced state is meaningful
    beyond the rim only (balance_cmp1/RESULTS.md, the rim test: 5 to 7 first
    Rossby radii from the wall for the orders 0 and 1).
    """
    nx, ny = i1 - i0, j1 - j0
    nz = len(snap['zt'])
    # the cell sizes from the coordinates, the u-centred grid of pyOM2
    # (grid.py u_centered_grid): xu_i = xu_i-1 + dxt_i, xt_0 = xu_0 - dxt_0/2
    def sizes(xu, xt):
        d = np.empty(len(xu))
        d[0] = 2*(xu[0] - xt[0]); d[1:] = xu[1:] - xu[:-1]
        return d
    dxt = sizes(snap['xu'], snap['xt'])[i0:i1]
    dyt = sizes(snap['yu'], snap['yt'])[j0:j1]
    dzt = np.abs(sizes(snap['zu'], snap['zt']))
    xu, yu = snap['xu'], snap['yu']
    p = dict(par)
    p.update(nx=nx, ny=ny, nz=nz, dxt=dxt, dyt=dyt, dzt=dzt, coord_degree=True,
             x_origin=float(xu[i0]), y_origin=float(yu[j0]),
             periodic_in_x=False, periodic_in_y=False)
    p.setdefault('epsilon', 1e-14); p.setdefault('max_iterations', 5000)
    M = _window(p, halo_points=halo_points)
    # the fields of the snapshot: the window of the global ones, or already the window
    w0, w1, v0, v1 = snap.get('window', (0, len(snap['xt']), 0, len(snap['yt'])))
    if (w0, w1, v0, v1) != (i0, i1, j0, j1) and not (w0 <= i0 and i1 <= w1 and v0 <= j0 and j1 <= v1):
        raise ValueError('window_model: the snapshot was read for the window %s, which does not contain %s' % ((w0, w1, v0, v1), (i0, i1, j0, j1)))
    cut = lambda a: a[:, j0 - v0:j1 - v0, i0 - w0:i1 - w0] if a.ndim == 3 else a[j0 - v0:j1 - v0, i0 - w0:i1 - w0]
    wet = cut(snap['wet'])
    nwet = wet.sum(axis=0)                                   # wet cells from the top (stair topography)
    M.win_kbot = np.where(nwet > 0, nz - nwet + 1, 0)
    M.set_coriolis(); M.set_topography(); M.prepare_for_run()
    u, v, temp, salt, p_s = [cut(snap[k]) for k in ('u', 'v', 'temp', 'salt', 'p_s')]
    if rim > 0:
        u, v, temp, salt, p_s = rim_taper(wet, u, v, temp, salt, p_s, rim)
        M.win_rim = rim
    M.set_initial_conditions(u, v, temp, salt, p_s)
    M.win_t = snap['t']
    return M


def rim_taper(wet, u, v, temp, salt, p_s, rim):
    """the state tapered to the background over `rim` cells from the walls
    (interior arrays): u, v, p_s and the anomalies of temp and salt from
    their level means over the wet cells are multiplied with sin^2 of the
    distance to the wall, 0 at the wall, 1 from `rim` cells on"""
    nz, ny, nx = wet.shape
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    dist = np.minimum.reduce([i + 1, nx - i, j + 1, ny - j])             # 1 at the wall
    r = np.sin(0.5*np.pi*np.minimum(dist/float(rim), 1.))**2
    nw = np.maximum(wet.sum(axis=(1, 2)), 1)
    Tm = (temp*wet).sum(axis=(1, 2))/nw; Sm = (salt*wet).sum(axis=(1, 2))/nw
    temp = np.where(wet, Tm[:, None, None] + r*(temp - Tm[:, None, None]), temp)
    salt = np.where(wet, Sm[:, None, None] + r*(salt - Sm[:, None, None]), salt)
    return u*r, v*r, temp, salt, p_s*r


def emission(M, u, v, temp, salt, p_s, periods=4, levels=(0.8, 0.5, 0.15), forcing=None, rim=0, edge=0, sponge=0):
    """
    the emission test of a state of the window model (interior arrays, the
    tracers as the model carries them): the model runs `periods` inertial
    periods of f0 with its own physics and with the steady body force
    `forcing` = (Fu, Fv[, Fb]) (interior tendencies, the slow wind of the
    balance spread over the mixed layer, wind_body_force) or without
    forcing, u and v at the
    levels (fractions of the depth from the bottom) are sampled 32 times per
    period, a cubic trend in time is removed per point, and the rms velocity
    at frequencies >= 0.9 f0 relative to the rms velocity of the state at
    these levels is returned. A better balanced state emits less. The model
    state is changed. `rim` > 0 tapers the state to the background over rim
    cells from the walls before the run (rim_taper; the same taper for every
    state compared, so that the walls, which cut the state, do not drive the
    run: on the icon033 window the jet cut by the northern wall doubled
    max|u| within 500 steps and the run blew up on Levante, Nov 9 2026);
    `edge` cells along the walls are left out of the sampling. `sponge` > 0
    relaxes u, v, temp, salt and p_s in the sponge cells from the walls to
    their initial values after every step, with the fraction
    (1 - (d - 1)/sponge)^2 / 2 of the difference (d the distance from the wall
    in cells, 1 at the wall: the wall cells are relaxed over two steps, the
    relaxation fades to zero at `sponge` cells),
    so that the state is untouched where it is measured, the cut of the state
    at the walls does not evolve, and waves reaching the rim are absorbed
    (the taper `rim` breaks the balance at the rim instead and its adjustment
    radiates into the window).
    """
    if rim > 0:
        wet = np.asarray(M._interior(M.maskT)) > 0
        u, v, temp, salt, p_s = rim_taper(wet, np.asarray(u), np.asarray(v), np.asarray(temp), np.asarray(salt), np.asarray(p_s), rim)
    M.set_initial_conditions(u, v, temp, salt, p_s)
    if hasattr(M, 'U_bt'):            # the split-explicit solver carries the barotropic transports: a cold start from the new state
        del M.U_bt, M.V_bt
    M.win_forcing = None if forcing is None else (M._pad(np.asarray(forcing[0]))*M.maskU, M._pad(np.asarray(forcing[1]))*M.maskV)
    if sponge > 0:
        ny, nx = M.ny, M.nx
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        dist = np.minimum.reduce([i + 1, nx - i, j + 1, ny - j])
        h = M.halo
        al = np.zeros((M.Ny, M.Nx)); al[h:-h, h:-h] = 0.5*np.maximum(1. - (dist - 1.)/float(sponge), 0.)**2   # 1/2 at the wall (a full reset there upsets the free-surface solver), 0 from sponge cells on
        al = OM.np.asarray(al); al3 = al[None]
        sp0 = [a.copy() if hasattr(a, 'copy') else a for a in (M.u, M.v, M.temp, M.salt, M.band_ps())]
    f0 = M.bal_f0
    spp = int(round(2*np.pi/f0/M.dt)); every = max(spp//32, 1)
    nz = M.nz
    lev = [int(l*nz) for l in levels]
    I = M._interior
    e = edge
    cut = (lambda a: a[:, e:-e, e:-e]) if e > 0 else (lambda a: a)
    U = []
    samp = lambda: U.append(np.stack([cut(np.array(I(M.u))[lev]), cut(np.array(I(M.v))[lev])]))
    samp()
    A, B, C = 1., 0., 0.
    tau, taum1, taum2 = 2, 1, 0
    for n in range(int(round(periods*spp))):
        if M.explicit_free_surface:
            M.time_step_explicit_pressure(A, B, C, tau, taum1, taum2)
        else:
            M.time_step_implicit_pressure(A, B, C, tau, taum1, taum2)
        if sponge > 0:
            M.u = M.u - al3*(M.u - sp0[0]); M.v = M.v - al3*(M.v - sp0[1])
            M.temp = M.temp - al3*(M.temp - sp0[2]); M.salt = M.salt - al3*(M.salt - sp0[3])
            if M.p_s.ndim == 2:                                  # the split-explicit solver: p_s and the barotropic transports it carries
                M.p_s = M.p_s - al*(M.p_s - sp0[4])              # (left alone, the trimming of u to the old U_bt kicks the sponge columns every step)
                M.U_bt, M.V_bt = M.init_barotropic_state(M.u, M.v, M.maskU, M.maskV)
            else:                                                # the implicit one, its current level
                M.p_s = OM.modify_array(M.p_s, (0, slice(None), slice(None)), M.p_s[0] - al*(M.p_s[0] - sp0[4]))
        if (n + 1) % every == 0:
            samp()
        tau, taum1, taum2 = (tau + 1) % 3, (taum1 + 1) % 3, (taum2 + 1) % 3
        if A == 1.:
            A, B, C = 1.5, -0.5, 0.
        elif A == 1.5:
            A, B, C = 23./12, -16./12, 5./12
    M.win_forcing = None
    U = np.array(U); nt = U.shape[0]
    t = np.arange(nt)*every*M.dt; tn = (t - t.mean())/t.std()
    V = np.vander(tn, 4)
    R = U.reshape(nt, -1) - V @ np.linalg.lstsq(V, U.reshape(nt, -1), rcond=None)[0]
    w = np.hanning(nt)[:, None]
    F = np.fft.rfft(R*w, axis=0)
    om = 2*np.pi*np.fft.rfftfreq(nt, every*M.dt)
    Ef = np.sum(np.abs(F[om >= 0.9*f0])**2); Et = np.sum(np.abs(F)**2)
    return float(np.sqrt(np.mean(R**2)*Ef/max(Et, 1e-300))/np.sqrt(np.mean(U[0]**2)*2))


# ----------------------------------------------------------------------
# the slow wind as the forcing term of the band method
def era5_stress_lowpass(directory, tag, start, cycle_days, t, xt, yt, half_width_days=2.0):
    """
    the low-pass of the ERA5 turbulent surface stress at model time t on the
    T points (xt, yt in degrees) of a window, and its time derivative:
    taux, tauy, dtaux/dt, dtauy/dt [N/m^2, N/m^2/s]. The files are those of
    the run, era5_<tag>_YYYY-MM.nc on the ERA5 grid (prep_era5_native.py:
    hourly means over the hour that starts at the record time, lat north to
    south, land filled with the nearest ocean value), with the run's
    calendar (forcing_base: model time 0 = start, a cycle of cycle_days
    repeats). The low-pass is a Hann window of half width half_width_days
    around t, its derivative the window's derivative applied to the series
    (d/dt of the convolution), both bilinearly interpolated to the window
    as forcing_era5_native does. Only the slow part of the wind belongs in
    the forcing term of band_balanced: the response to the rest is
    near-inertial and is a wave.
    """
    import datetime, calendar
    from netCDF4 import Dataset
    start = datetime.datetime.strptime(str(start).strip(), '%Y-%m-%d') if not isinstance(start, datetime.datetime) else start
    cyc = float(cycle_days)*86400.
    W = half_width_days*86400.
    # the records: hourly means centred at (k + 1/2) h since start (cycled)
    s = float(t)
    k0 = int(np.floor((s - W)/3600.)); k1 = int(np.ceil((s + W)/3600.))
    ks = np.arange(k0, k1 + 1)
    tc = (ks + 0.5)*3600.                                   # centre of record k
    sk = tc - s
    inside = np.abs(sk) < W
    w = np.where(inside, np.cos(0.5*np.pi*sk/W)**2, 0.)
    dw = np.where(inside, -np.pi/(2*W)*np.sin(np.pi*sk/W), 0.)      # dw/ds, 1/s
    N = w.sum()
    # tau_bar(t) = sum_k w_k tau(t + s_k)/N, and d/dt of the convolution is
    # -int w'(s) tau(t + s) ds / int w ds = -sum_k w'_k tau_k / N
    w = w/N; wd = -dw/N
    files = {}
    def record(k):
        kk = k % int(round(cyc/3600.)) if cyc else k
        d = start + datetime.timedelta(hours=int(kk))
        y, m = d.year, d.month
        r = int((d - datetime.datetime(y, m, 1)).total_seconds())//3600
        if (y, m) not in files:
            fn = os.path.join(directory, 'era5_%s_%04d-%02d.nc' % (tag, y, m))
            f = Dataset(fn); f.set_auto_mask(False); files[(y, m)] = f
        f = files[(y, m)]
        return np.asarray(f.variables['taux'][r], np.float64), np.asarray(f.variables['tauy'][r], np.float64), f
    # the stencil from the first file
    tx, ty, f = record(ks[0])
    lat = np.asarray(f.variables['lat'][:], np.float64); lon = np.asarray(f.variables['lon'][:], np.float64)
    flip = lat[0] > lat[-1]
    lat = lat[::-1] if flip else lat
    dlon = lon[1] - lon[0]
    p = np.mod(np.asarray(xt) - lon[0], 360.)/dlon
    i0 = np.floor(p).astype(int); wx = (p - i0)[None, :]
    i0, i1 = np.mod(i0, len(lon)), np.mod(i0 + 1, len(lon))
    yq = np.clip(np.asarray(yt), lat[0], lat[-1])
    j0 = np.clip(np.searchsorted(lat, yq) - 1, 0, len(lat) - 2); j1 = j0 + 1
    wy = ((yq - lat[j0])/(lat[j1] - lat[j0]))[:, None]
    def interp(a):
        if flip:
            a = a[::-1]
        return ((1 - wy)*((1 - wx)*a[np.ix_(j0, i0)] + wx*a[np.ix_(j0, i1)])
                + wy*((1 - wx)*a[np.ix_(j1, i0)] + wx*a[np.ix_(j1, i1)]))
    out = [np.zeros((len(yt), len(xt))) for _ in range(4)]
    for k, wk, wdk in zip(ks, w, wd):
        if wk == 0. and wdk == 0.:
            continue
        tx, ty, _ = record(k)
        ax, ay = interp(tx), interp(ty)
        out[0] += wk*ax; out[1] += wk*ay; out[2] += wdk*ax; out[3] += wdk*ay
    for f in files.values():
        f.close()
    return out


def mixed_layer_depth(M, drho=0.03, zref=10.):
    """
    the mixed layer depth of every column of the model state (interior,
    metres): the depth where the potential density (the EOS at the surface
    pressure) exceeds its value at zref by drho, linearly interpolated
    between the levels; the whole column where it never does
    """
    from pyOM3.density import get_rho
    h = M.halo; I = M._interior
    if M.eq_of_state == 0:
        rho = -M.rho_0/M.grav*np.asarray(I(M.temp))
    else:
        rho = np.asarray(I(get_rho(M.salt, M.temp, 0.*M.p0 if M.p0 is not None else None, M.eq_of_state)))
    mT = I(M.maskT) > 0
    zt = np.asarray(M.zt)[h:-h]; ztop = float(np.asarray(M.zw)[-(h + 1)])
    depth = ztop - zt                                        # of the T levels, k = 0 the deepest
    nz, ny, nx = rho.shape
    kref = int(np.argmin(np.abs(depth - zref)))
    rref = rho[kref]
    hml = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            if not mT[-1, j, i]:
                continue
            col = rho[:, j, i]; wet = mT[:, j, i]
            dz_ = depth[wet]; dr = col[wet] - rref[j, i]         # from the bottom up
            dz_ = dz_[::-1]; dr = dr[::-1]                       # from the surface down
            ex = np.where(dr > drho)[0]
            if len(ex) == 0:
                hml[j, i] = dz_[-1]
            else:
                k = ex[0]
                if k == 0:
                    hml[j, i] = dz_[0]
                else:
                    hml[j, i] = dz_[k - 1] + (drho - dr[k - 1])/(dr[k] - dr[k - 1])*(dz_[k] - dz_[k - 1])
    return hml


def wind_body_force(M, taux, tauy, hml):
    """
    the wind stress (N/m^2, T points of the window interior) spread over the
    mixed layer of depth hml (m) as the body force of the band method:
    (Fu, Fv, Fb) interior tendencies, Fu = tau_x/(rho_0 hml) in the cells of
    the mixed layer (the part of the cell inside it), averaged from the T
    points to the U and V points as the model does with the stress
    """
    h = M.halo; I = M._interior
    zw = np.asarray(M.zw)[h:-h]; ztop = float(zw[-1])
    dzt = np.asarray(M.dzt)[h:-h]
    top_face = ztop - zw                                     # depth of the top face of cell k (k = 0 the deepest)
    frac = np.clip((hml[None] - top_face[:, None, None])/dzt[:, None, None], 0., 1.)   # the part of the cell inside the mixed layer
    mT = I(M.maskT)
    fx = taux[None]/(M.rho_0*np.maximum(hml, 1e-3)[None])*frac*mT
    fy = tauy[None]/(M.rho_0*np.maximum(hml, 1e-3)[None])*frac*mT
    FX = M._pad(fx); FY = M._pad(fy)
    Fu = 0.5*(FX + M.rollx(FX, -1))*M.maskU
    Fv = 0.5*(FY + M.rolly(FY, -1))*M.maskV
    return [I(Fu), I(Fv), 0.*fx]


# ----------------------------------------------------------------------
# the parameters of the run beside its raw snapshots
RUN_PARAMS = 'run_params.json'
# the model parameters a window shares with the run (window_model's par), read
# from the model itself where the configuration leaves them to the defaults
WINDOW_KEYS = ('eq_of_state', 'rho_0', 'dt', 'Ahbi', 'Khbi', 'biha_friction_cosPower',
               'implicit_free_surface', 'explicit_free_surface', 'sub_cycling', 'fb_theta',
               'superbee_advection', 'free_surface_tracer_flux')


def _plain(v):
    """v as a JSON value, or None for what has no place in the file (arrays
    of the grid, objects): scalars, strings, and lists and dicts of them"""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)) and len(v) <= 16:
        p = [_plain(a) for a in v]
        return None if any(a is None for a in p) else p
    if isinstance(v, dict):
        p = {str(k): _plain(a) for k, a in v.items()}
        return {k: a for k, a in p.items() if a is not None}
    if getattr(v, 'ndim', None) == 0 or getattr(v, 'shape', None) == ():
        try:
            return _plain(np.asarray(v).item())
        except Exception:
            return None
    return None


def write_run_params(directory, model=None, par=None, **info):
    """
    <directory>/run_params.json: the model parameters of the run (the scalar
    entries of the parameter dict par, with the model's own values where it
    has them, and WINDOW_KEYS from the model) under 'model', and the further
    entries of info (exp_name, segment, the forcing settings, ...) as they
    are. The window driver (run_window.py) takes its parameters from it.
    Written to a temporary file first and renamed, so that a reader never
    sees half a file.
    """
    import json, datetime
    m = {}
    for k, v in (par or {}).items():
        p = _plain(getattr(model, k, v)) if model is not None else _plain(v)
        if p is None and model is not None:
            p = _plain(v)
        if p is not None:
            m[k] = p
    if model is not None:
        for k in WINDOW_KEYS:
            if k not in m and hasattr(model, k):
                p = _plain(getattr(model, k))
                if p is not None:
                    m[k] = p
    out = dict(written=datetime.datetime.now().isoformat(timespec='seconds'))
    out.update({k: _plain(v) for k, v in info.items()})
    out['model'] = m
    fn = os.path.join(directory, RUN_PARAMS)
    with open(fn + '.tmp', 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(fn + '.tmp', fn)
    return fn


def read_run_params(path):
    """the content of <path>/run_params.json, or None if the run did not
    write one (runs before Nov 2026)"""
    import json
    fn = os.path.join(path, RUN_PARAMS)
    if not os.path.isfile(fn):
        return None
    with open(fn) as f:
        return json.load(f)
