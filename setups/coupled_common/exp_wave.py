"""
The WAVE model of the coupled experiments, the part every configuration
shares: the land mask from the ocean's topography (coarsened with the wave
grid), the wind from a forcing object of forcing_common, and the two energy
diagnostics.

    class wave(wave_common, wave_base):
        cfg = Cfg(sys.modules[__name__])

cfg names read here: rho_air, kbot_global(), forcing_settings(),
RAW_SNAPSHOTS, wave_coarsen and, only with wave_coarsen > 1, _je_w, _ie_w,
dxt_deg, dyt_deg, x_origin, y_origin.

The wind forcing is the surface stress tau_a of the forcing source, the same
tau_a the ocean's direct branch uses, converted to an air-side friction
velocity u*_air = sqrt(|tau| / rho_air). The direction convention is wim's:
theta = 0 -> north, theta = pi/2 -> east, so udir = atan2(tau_east, tau_north).
"""
import numpy as np
import pyOM3 as OM
from mpi4py import MPI
from pyOM3.wim_realistic.grid import u_centered_grid
from block_io import local_block
from forcing_base import make as make_forcing


class wave_common:

    def setup(self, snap_file_name='raw_wave', show_each_time_step=True):
        """RAW snapshots: run.py hands a name ending in '.cdf' (it splices
        the segment tag in front of it), and snap_raw writes a DIRECTORY, so
        the suffix is stripped here and each segment gets its own."""
        if self.cfg.RAW_SNAPSHOTS and snap_file_name.endswith('.cdf'):
            snap_file_name = snap_file_name[:-4]
        return super().setup(snap_file_name, show_each_time_step)

    def set_topography(self):
        """Land/sea mask from the OCEAN's topography: wim's kbot is 1 in
        ocean cells and 0 on land, so the wave model sees exactly the
        coastline the ocean does."""
        kb = self.cfg.kbot_global()
        if self.cfg.wave_coarsen > 1:
            # coarser wave grid: a wave cell is sea if ANY ocean cell of its
            # group is wet, so every wet ocean cell lies in a wave cell that
            # has waves. (With 'at least half wet' 0.2 to 0.8 % of the wet
            # ocean cells have no wave cell above them; the large land
            # bridges stay closed under both rules on this topography.)
            kb = np.maximum.reduceat(np.maximum.reduceat(kb, self.cfg._je_w[:-1], axis=0),
                                     self.cfg._ie_w[:-1], axis=1)
        j0, i0, ny_loc, nx_loc = local_block(self)
        ocean_cell = np.where(kb[j0:j0+ny_loc, i0:i0+nx_loc] > 0, 1.0, 0.0)
        kbot = np.zeros((self.Ny, self.Nx), np.float64)
        sl = slice(self.halo, -self.halo)
        kbot[sl, sl] = ocean_cell
        self.kbot = self.apply_bc(OM.np.asarray(kbot))
        return

    def set_initial(self):
        """Build the forcing object (forcing_common, the source is
        cfg.forcing_settings()['source']), set u*/udir at t = 0, and seed a
        small spectrum where there is wind (unforced cells stay exactly 0).

        The ERA5 files are read on rank 0 of the WAVE communicator and
        broadcast, then interpolated to this rank's block
        (forcing_era5_native.py). prep_era5_native.py has already replaced the land
        points by their nearest ocean value and checked every month, so the
        fill-value sanitising of the model-grid reader is not repeated here;
        the guard on u* below stays, since a fill value that survives to
        sqrt(|tau|/rho_air) detonates the wave field in a few steps."""
        h  = self.halo
        j0, i0, ny_loc, nx_loc = local_block(self)
        wet = self.cfg.kbot_global() > 0
        if self.cfg.wave_coarsen > 1:
            # COARSER WAVE GRID: the wind of a wave cell is the AREA MEAN of
            # tau_a over the ocean cells of its group, not the forcing sampled
            # at the wave cell's centre. A wave cell is up to 1.3 degrees at
            # 1/3 degree and ERA5 is 0.28, so a point sample would alias the
            # stress; and the ocean's direct branch tau_a - tau_in takes tau_a
            # on the OCEAN grid, so the two components only see the same
            # momentum input if the wave model's tau_a is the mean of the
            # ocean's. The forcing object is therefore built for the OCEAN
            # cells under this rank's block, exactly as the ocean's is, and
            # update_forcing averages the stress VECTOR before u* is formed.
            je, ie = self.cfg._je_w, self.cfg._ie_w
            ja, jb = int(je[j0]), int(je[j0 + ny_loc])
            ia, ib = int(ie[i0]), int(ie[i0 + nx_loc])
            xu_f, xt_f, _ = u_centered_grid(self.cfg.dxt_deg)
            yu_f, yt_f, _ = u_centered_grid(self.cfg.dyt_deg)
            xt_f = xt_f + self.cfg.x_origin - xu_f[0]
            yt_f = yt_f + self.cfg.y_origin - yu_f[0]
            self._fw  = (self.cfg.dxt_deg[None, ia:ib]
                         *(np.cos(np.deg2rad(yt_f[ja:jb]))*self.cfg.dyt_deg[ja:jb])[:, None])
            self._foj = (je[j0:j0 + ny_loc] - ja).astype(np.intp)
            self._foi = (ie[i0:i0 + nx_loc] - ia).astype(np.intp)
            self._fws = self._group_sum(self._fw)
            geom = {'xt': xt_f[ia:ib], 'yt': yt_f[ja:jb],
                    'block': (ja, ia, jb - ja, ib - ia),
                    'xt_gl': xt_f, 'yt_gl': yt_f, 'wet': wet[ja:jb, ia:ib]}
        else:
            geom = {'xt': np.asarray(self.xt)[h:-h], 'yt': np.asarray(self.yt)[h:-h],
                    'block': (j0, i0, ny_loc, nx_loc),
                    'xt_gl': self.xt_gl, 'yt_gl': self.yt_gl,
                    'wet': wet[j0:j0+ny_loc, i0:i0+nx_loc]}
        geom['comm'] = getattr(self, 'comm', None)
        geom['own_block'] = (j0, i0, ny_loc, nx_loc)      # the WAVE block (a coupled source works on it)
        fs = self.cfg.forcing_settings()
        if str(self.cfg.STRESS_MODE) == 'relwind':
            # the ocean forms the stress from the relative wind and forwards it
            # (coupler_loop, tags 9, 10): the waves grow on the same stress
            fs = {'source': 'coupled', 'start': fs.get('start', '2000-01-01'),
                  'cycle_days': fs.get('cycle_days', 0.0)}
        self._forcing = make_forcing(fs, geom, ('taux', 'tauy'), 'wave')
        self.ustar = OM.np.zeros((self.Ny, self.Nx), np.float64)
        self.udir  = OM.np.zeros((self.Ny, self.Nx), np.float64)
        self.update_forcing(0.0)
        us_max = float(np.max(np.asarray(self.ustar)))
        if not np.isfinite(us_max) or us_max > 5.0:
            raise ValueError(
                'wave forcing pe %d: u* reaches %.4g m/s. Above ~2 m/s this '
                'is not wind, it is a missing value that survived the read; '
                'the WAMDI growth rate is linear in u* and limit_growth '
                'scales with it, so the spectrum detonates within a few time '
                'steps.' % (self.my_pe, us_max))
        if self.my_pe == 0:
            for line in self._forcing.describe():
                print('CONFIG ' + line, flush=True)
            print('wave forcing: u* at t=0 spans %.4g .. %.4g m/s'
                  % (float(np.min(np.asarray(self.ustar))), us_max), flush=True)
        # a 'coupled' source has no wind before the first exchange: seed every
        # wet cell then, the atmosphere's wind arrives one coupling step later
        if getattr(self._forcing, 'own_grid', False):
            seed = 1e-6*OM.np.ones_like(self.ustar)[None, None, :, :]
        else:
            seed = OM.np.where(self.ustar[None, None, :, :] > 0.0, 1e-6, 0.0)
        # keep the full (Nd,Nf,Ny,Nx) shape: seed is (1,1,Ny,Nx)
        self.E = self.E*0.0 + seed*self.maskTp[None, None, :, :]
        self.A = self.E/self.omega_t[None, :, None, None]
        return

    def _group_sum(self, a):
        """Sum of an ocean-grid field under this rank's block over the ocean
        cells of each wave cell (coarser wave grid only)."""
        return np.add.reduceat(np.add.reduceat(a, self._foj, axis=0),
                               self._foi, axis=1)

    def update_forcing(self, t):
        """u*/wind direction from the ERA5 stress at model time t [s]. Called
        every wave sub-step; the ocean reads the same files on the same
        calendar."""
        h  = self.halo
        sl = slice(h, -h)
        F  = self._forcing.get(t)
        tx, ty = F['taux'], F['tauy']                        # N/m^2, east/north
        # (a 'coupled' source hands the stress on the WAVE grid already, the
        # ocean did the area mean: no group sum then)
        if self.cfg.wave_coarsen > 1 and not getattr(self._forcing, 'own_grid', False):
            tx = self._group_sum(self._fw*tx)/self._fws
            ty = self._group_sum(self._fw*ty)/self._fws
        us = np.sqrt(np.hypot(tx, ty)/self.cfg.rho_air)
        ud = np.arctan2(tx, ty)                    # theta=0 north, pi/2 east
        self.ustar = OM.modify_array(self.ustar, (sl, sl), us)
        self.udir  = OM.modify_array(self.udir,  (sl, sl), ud)
        self.ustar = self.apply_bc(self.ustar)*self.maskTp
        self.udir  = self.apply_bc(self.udir)
        return

    def wind_input_field(self):
        """Local wind input to the waves p_in [W/m^2] on the interior block
        (T points), resolved part plus the analytic tail: the integrand of
        wind_work() times rho_0, for the interface dissipation q_int of the
        coupled atmosphere (energy_cycle.tex Eq. eq_qint)."""
        h  = self.halo
        sl = slice(h, -h)
        fld = self.grav*np.sum(np.sum((self.gamma_in*self.E)[sl, sl, sl, sl]
                                      *self.dd, axis=0)
                               *self.dft[sl, None, None], axis=0)
        if hasattr(self, '_tail_input_factors'):
            _, fac_phi, _, _ = self._tail_input_factors()
            Ecg = self.E[:, -h, :, :]
            fld = fld + np.sum((Ecg*fac_phi*self.dd)[sl, sl, sl], axis=0)
        return self.cfg.rho_0*np.asarray(fld)

    # --- diagnostic: wind -> wave energy input ------------------------------
    def wind_work(self):
        """Domain-integrated kinematic energy input rate of the wind to the
        wave field, P_in = g int gamma_in E dtheta df dA [m^5/s^3 per unit
        rho]. Uses the gamma_in of the LAST time_step. Area weighted with the
        spherical cell area. Collective over the wave communicator."""
        h  = self.halo
        sl = slice(h, -h)
        fld = self.grav*np.sum(np.sum((self.gamma_in*self.E)[sl, sl, sl, sl]
                                      *self.dd, axis=0)
                               *self.dft[sl, None, None], axis=0)
        p = float(np.sum(fld*self.area_t[sl, sl]))
        # analytic beyond-grid tail input (tail_flux='input'): the wind pumps
        # energy directly into f > f_grid and Phi_oc carries it, so the input
        # book must too, or 'dissipation' persistently exceeds 'wind work'.
        if hasattr(self, '_tail_input_factors'):
            _, fac_phi, _, _ = self._tail_input_factors()
            Ecg = self.E[:, -h, :, :]
            p += float(np.sum(np.sum((Ecg*fac_phi*self.dd)[sl, sl, sl], axis=0)
                              *self.area_t[sl, sl]))
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            p = comm.allreduce(p, op=MPI.SUM)
        return p

    # --- diagnostic: reversible exchange int U.k S_nl dk (Eq. 11) -----------
    def snl_work(self):
        """Domain-integrated reversible exchange between wave energy and mean
        Lagrangian KE by the nonlinear transfer working against the sheared
        Doppler velocity, W_nl = int U.k S_nl dk dA [m^5/s^3 per unit rho],
        positive into the wave field. This is THE term the energy cycle does
        not close over; it vanishes for depth-independent u_L over the wave
        penetration depth. Resolved range only. Collective."""
        if not (hasattr(self, 'S_nl') and hasattr(self, 'ocean_U')):
            return 0.0
        h  = self.halo
        sl = slice(h, -h)
        Uk = (self.ocean_U[None, :, :, :]*self.sinth[:, None, None, None]
            + self.ocean_V[None, :, :, :]*self.costh[:, None, None, None]) \
            * self.omega_t[None, :, None, None]
        fld = np.sum(np.sum((Uk*self.S_nl)[sl, sl, sl, sl]*self.dd, axis=0)
                     *self.dft[sl, None, None], axis=0)
        p = float(np.sum(fld*self.area_t[sl, sl]))
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            p = comm.allreduce(p, op=MPI.SUM)
        return p
