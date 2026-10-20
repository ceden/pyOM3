"""
The OCEAN of the coupled experiments, the part every configuration shares:
surface forcing (from a forcing object of forcing_common, the source is the
experiment's choice), the u_L coupling terms, the TKE closure's wave hooks,
the energy diagnostics and the online time means.

    class ocean(ocean_common, tke_mixin, ocean_base):
        cfg = Cfg(sys.modules[__name__])

in the experiment module. NOTHING HERE READS THE ENVIRONMENT OR NAMES AN
EXPERIMENT: every number comes from the experiment module through self.cfg
(exp_common.Cfg). What stays in the experiment module is what differs between
configurations: the grid source, the parameters (get_ocean_parameter), the
initial conditions, the snapshot form (prepare_for_run) and the lateral and
bottom friction.

cfg names read here (ni_diag.py adds NI_*): rho_0, cp_0, t_rest, omega_e, nz, exp_name,
kbot_global(), dzt_global(), zw_pyom2(), forcing_settings(), SSS_PISTON,
SW_BOTTOM, direct_stress_branch, STOKES_SHEAR, KE_BUDGET, AVERAGE_MLD, AVERAGE_EDDY,
AVERAGE_TRACER_FLUX, AVERAGE_FIELDS,
HS_FACTOR, MXL_CHOICE, MXL_BETA, MXL_SURF_FACTOR, VADV_SUB_CYCLING,
STRESS_MODE, STRESS_UO, STRESS_U10_MIN, STRESS_CD_MAX, rho_air,
FORCING_START,
FORCING_CYCLE_DAYS.
"""
import os
import shutil
import numpy as np
import pyOM3 as OM
from netCDF4 import Dataset
from mpi4py  import MPI
from pyOM3.density import get_rho
from block_io import local_block
from forcing_base import make as make_forcing
from ni_diag import ni_common


class ocean_common(ni_common):

    # the KE budget (ke_budget below, cfg.KE_BUDGET 1 full | 2 light): the
    # columns of the energy text file and the accumulation interval. An
    # experiment module may override them (icon033_setup does, identically).
    KE_COLUMNS = ('day', 'KE_adv', 'KE_cor', 'KE_ph', 'KE_ps', 'W_tau', 'W_dir', 'W_vortex',
                  'C', 'D_h', 'D_v', 'D_v_num', 'K_diss', 'PEs_dot', 'dE_sp', 'dE_fd', 'LEAK',
                  'KE', 'PEs', 'KE_div', 'D_bot')
    KE_LEGEND = ('KE budget legend [W/m2]: KE_adv u.(advection) | KE_cor u.(Coriolis) | '
                 'KE_ph u.(-grad p_h) = conversion | KE_ps u.(-grad p_s) | W_tau wave stress '
                 'work | W_dir direct-branch work | W_vortex vortex-force work (round-off; with '
                 'STOKES_SHEAR the whole force, and W_ss in the log is the work of its horizontal '
                 'Stokes shear part, cancelled by the pressure p_SS) | '
                 'C exchange int u_L.d(u_S)/dt | D_h horizontal friction | D_bot bottom friction | '
                 'D_v vertical friction '
                 '= -K_diss - D_v_num | dE_sp sum of terms | dE_fd (E(t1)-E(t0))/T | '
                 'LEAK = dE_fd - dE_sp (time stepping)')
    eb_every_days = 1.0

    # TKE closure fields added to the snapshot, alongside the package's
    # u, v, w, temp, salt, p_h, p_s
    _extra_snap = ('tke', 'K_diss_E', 'K_diss_L')

    def prepare_for_run(self, *a, **k):
        out = super().prepare_for_run(*a, **k)
        # the near-inertial diagnostic (ni_diag.py, cfg.NI_DIAG) registers its
        # products as online means, so it is set up before average_setup
        self.ni_setup()
        return out

    def diagnose(self, n, tau):
        super().diagnose(n, tau)
        if n > 0 and self.cfg.KE_BUDGET:
            self.ke_budget(n, tau)
        self.ni_step(n, tau)
        self.average(n, tau)
        if not (n == 0 or np.mod(n, int(self.snapint/self.dt)) == 0.):
            return
        if os.path.isdir(self.snap_file_name):
            # RAW snapshots: the parent has just created this directory and
            # written its fields
            d = self.snap_file_name + '/pe=%i/n=%i/' % (self.my_pe, n)
            for v in self._extra_snap:
                if hasattr(self, v):
                    OM.np.save(d + '%s.npy' % v, self.unpad(getattr(self, v)))
            return
        # netCDF snapshots: the variables are created by the experiment's
        # prepare_for_run, the record is the one the parent has just written
        self.mpi_barrier()
        m = 0
        if self.my_pe == 0:
            fid = Dataset(self.snap_file_name, mode='r')
            m = len(fid.dimensions['time']) - 1        # record the parent wrote
            fid.close()
        m = self.pe0_bcast(m)
        j0, i0, ny_loc, nx_loc = local_block(self)
        for pe in range(self.n_pes):
            self.mpi_barrier()
            if self.my_pe == pe:
                fid = Dataset(self.snap_file_name, mode='r+')
                sl = (m, slice(None), slice(j0, j0+ny_loc), slice(i0, i0+nx_loc))
                for v in self._extra_snap:
                    if hasattr(self, v):
                        fid.variables[v][sl] = self.unpad(getattr(self, v))
                fid.close()
        return

    # --- online time means ---------------------------------------------------
    # Hourly forcing aliases into daily or monthly snapshots (inertial and
    # diurnal signals). The means below are sums over EVERY ocean step, taken
    # after the step in diagnose, divided by the number of steps. A window
    # closes when the step end crosses a boundary: a calendar month of the
    # forcing date ('month') or a multiple of a fixed interval. The partial
    # sums are part of the ocean restart (coupler_io), so a window may span
    # job segments. Each segment writes its own file; a window goes into the
    # file of the segment in which it ends.
    # ------------------------------------------------------------------
    # REGISTERING A FIELD TO BE AVERAGED, pyOM2's register_average
    # (diag_averages.f90) in this machinery. The experiment module calls it
    # for anything the standard list below does not hold, BEFORE run.py calls
    # average_setup -- its own prepare_for_run, after super(), is the place:
    #
    #     def prepare_for_run(self, *a, **k):
    #         out = super().prepare_for_run(*a, **k)
    #         self.register_average('rho', T3, 'kg/m3', 'in-situ density',
    #                               lambda tau: self.rho)
    #         return out
    #
    # where the dimension tuples are the ones _average_spec uses (T3, U3, V3,
    # W3, T2, U2, V2 below). The getter takes the time level and returns the
    # PADDED field; the sum runs over every ocean step of the window, as for
    # every other field, and the restart carries it. pyOM2 registers a
    # POINTER to an array, which is the same thing where the array is
    # recomputed in place; a closure also covers fields that are formed only
    # for the diagnostic (a product of two states, say).
    # ------------------------------------------------------------------
    def register_average(self, name, dims, units, long_name, get):
        if not hasattr(self, '_avg_extra'):
            self._avg_extra = []
        if len(dims) not in (3, 4):
            raise ValueError('register_average %s: dims must be the 3-D or '
                             '2-D tuple, got %r' % (name, (dims,)))
        if any(name == nme for nme, *_ in self._avg_extra):
            raise ValueError('register_average: the name %s is already in use'
                             % name)
        self._avg_extra.append((name, tuple(dims), units, long_name, get))
        if self.my_pe == 0:
            print('ocean means: registered %s [%s] %s'
                  % (name, units, long_name), flush=True)
        return

    def _average_spec(self):
        """(name, dims, units, long_name, getter(tau)) of every mean field.
        Decided once from the configuration, identical on all PEs."""
        r, g = self.cfg.rho_0, self.grav
        T3, U3 = ('time', 'zt', 'yt', 'xt'), ('time', 'zt', 'yt', 'xu')
        V3, W3 = ('time', 'zt', 'yu', 'xt'), ('time', 'zu', 'yt', 'xt')
        T2, U2, V2 = ('time', 'yt', 'xt'), ('time', 'yt', 'xu'), ('time', 'yu', 'xt')
        if self.explicit_free_surface:
            ps = lambda tau: self.p_s
        else:
            ps = lambda tau: self.p_s[tau]
        z2 = lambda: OM.np.zeros((self.Ny, self.Nx))
        spec = [
            ('u',    U3, 'm/s', 'zonal velocity (u_L when coupled)', lambda tau: self.u),
            ('v',    V3, 'm/s', 'meridional velocity (v_L when coupled)', lambda tau: self.v),
            ('w',    W3, 'm/s', 'vertical velocity', lambda tau: self.w),
            ('temp', T3, 'degC', 'temperature', lambda tau: self.temp),
            ('salt', T3, '', 'salinity', lambda tau: self.salt),
            ('u2',   U3, 'm2/s2', 'u*u', lambda tau: self.u*self.u),
            ('v2',   V3, 'm2/s2', 'v*v', lambda tau: self.v*self.v),
            ('ssh',  T2, 'm', 'sea surface height p_s/g', lambda tau: ps(tau)/g),
            ('sst',  T2, 'degC', 'temperature of the top cell', lambda tau: self.temp[-self.halo-1]),
            ('sss',  T2, '', 'salinity of the top cell', lambda tau: self.salt[-self.halo-1]),
            ('ssh2', T2, 'm2', 'ssh*ssh', lambda tau: (ps(tau)/g)**2),
            ('qnet', T2, 'W/m2', 'surface heat flux applied (ERA5 + restoring, zero under ice)',
             lambda tau: self.forc_temp_surface*self.cfg.cp_0*r),
            ('salt_flux', T2, 'm/s', 'surface salt flux applied (restoring + E-P)',
             lambda tau: self.forc_salt_surface),
            ('taux', U2, 'N/m2', 'surface stress on the ocean (tau_a alone, tau_o coupled)',
             lambda tau: self.taux*r),
            ('tauy', V2, 'N/m2', 'surface stress on the ocean (tau_a alone, tau_o coupled)',
             lambda tau: self.tauy*r),
        ]
        # 'mld' only where the experiment asks for it (AVERAGE_MLD):
        # mixed_layer_depth evaluates the equation of state on the whole 3-D
        # field and the mean is formed every step, a full density call per
        # step. icon15 and global4 have it, icon033 does not.
        if self.cfg.AVERAGE_MLD:
            spec.append(('mld', T2, 'm', 'mixed layer depth, potential density 0.03 kg/m3 above '
                         'its value at 10 m (de Boyer Montegut et al. 2004), every step',
                         lambda tau: self.mixed_layer_depth()))
        if getattr(self, 'enable_tke', False):
            for nm, u_, ln in (('tke', 'm2/s2', 'turbulent kinetic energy'),
                               ('K_diss_E', 'm2/s3', 'Eulerian shear production'),
                               ('K_diss_L', 'm2/s3', 'Stokes shear production'),
                               ('kappaH_tke', 'm2/s', 'vertical diffusivity'),
                               ('kappaM_tke', 'm2/s', 'vertical viscosity')):
                spec.append((nm, W3, u_, ln, lambda tau, nm=nm: getattr(self, nm)))
            spec.append(('tke_flux', T2, 'm3/s3', 'surface TKE flux',
                         lambda tau: self.forc_tke_surface))
        if self.coupled:
            spec += [
                ('u_S', U3, 'm/s', 'Stokes drift', lambda tau: self.u_S),
                ('v_S', V3, 'm/s', 'Stokes drift', lambda tau: self.v_S),
                ('phi_oc', T2, 'm3/s3', 'wave dissipation energy flux into the ocean / rho_0',
                 lambda tau: self.phi_oc),
                ('hs', T2, 'm', 'significant wave height from the wave model',
                 lambda tau: getattr(self, 'hs', z2())),
                ('tau_in_x', T2, 'N/m2', 'wind input stress into the waves',
                 lambda tau: self.tau_in_x*r),
                ('tau_in_y', T2, 'N/m2', 'wind input stress into the waves',
                 lambda tau: self.tau_in_y*r)]
            if self.cfg.direct_stress_branch:
                spec += [
                    ('tau_dir_x', U2, 'N/m2', 'direct branch tau_a - tau_in on the ocean',
                     lambda tau: getattr(self, '_tau_dir_x', z2())*r),
                    ('tau_dir_y', V2, 'N/m2', 'direct branch tau_a - tau_in on the ocean',
                     lambda tau: getattr(self, '_tau_dir_y', z2())*r)]
        # the relative-wind stress (cfg.STRESS_MODE 'relwind'): the wind it
        # used, the drag it kept and the dissipation of the interface
        if str(self.cfg.STRESS_MODE) == 'relwind':
            spec += [
                ('u10',    T2, 'm/s',   '10 m wind (ERA5) the stress was formed with',
                 lambda tau: getattr(self, '_u10', z2())),
                ('v10',    T2, 'm/s',   '10 m wind (ERA5) the stress was formed with',
                 lambda tau: getattr(self, '_v10', z2())),
                ('cd_rho', T2, 'kg/m3', 'C_D rho_a = |tau_ERA5|/|u10|^2 kept from ERA5',
                 lambda tau: getattr(self, '_cd_rho', z2())),
                ('d_int',  T2, 'W/m2',  'interface dissipation (u10 - u_o).tau of the relative-wind stress',
                 lambda tau: getattr(self, '_d_int', z2()))]
        # the ice mask that switched the momentum down, so a mean file made
        # with cfg.ICE_TAU_FRAC < 1 says where and how often it acted
        if float(self.cfg.ICE_TAU_FRAC) < 1.0:
            spec.append(('ice', T2, '1', 'fraction of the window with T(surface) '
                         'at or below the freezing point',
                         lambda tau: getattr(self, '_ice_mask', z2())))

        # EDDY FLUXES (cfg.AVERAGE_EDDY), pyOM2's diag_variance.f90. What is
        # accumulated is the density and the three ADVECTIVE FLUXES OF IT,
        # formed with the model's own face interpolation:
        #     urho = u (rho_i + rho_i+1)/2 , vrho = v (rho_j + rho_j+1)/2 ,
        #     wrho = w (rho_k + rho_k+1)/2 .
        # The eddy part is then a subtraction OF MEANS, done where the file is
        # read rather than here, because the face interpolation is linear and
        # therefore commutes with the time mean:
        #     urho' = <urho> - <u> (<rho>_i + <rho>_i+1)/2    (and v, w)
        # is exactly pyOM2's diag_variance_write, to round-off. For the same
        # reason the eddy kinetic energy needs nothing new: u2 and v2 are in
        # the list already, so
        #     EKE = ( <u2> - <u><u> + <v2> - <v><v> )/2
        # on the T points after averaging the two to them, as pyOM2 does.
        # Keeping the products rather than the differences is what makes the
        # window restartable and the segments composable: sums add, a
        # difference of means does not.
        if self.cfg.AVERAGE_EDDY:
            spec += [
                ('rho',  T3, 'kg/m3', 'density as the model carries it',
                 lambda tau: self.rho),
                ('urho', U3, 'kg/m2/s', 'zonal density flux u (rho_i + rho_i+1)/2',
                 lambda tau: self.u*0.5*(self.rho + self.rollx(self.rho, -1))*self.maskU),
                ('vrho', V3, 'kg/m2/s', 'meridional density flux v (rho_j + rho_j+1)/2',
                 lambda tau: self.v*0.5*(self.rho + self.rolly(self.rho, -1))*self.maskV),
                ('wrho', W3, 'kg/m2/s', 'vertical density flux w (rho_k + rho_k+1)/2',
                 lambda tau: self.w*0.5*(self.rho + self.rollz(self.rho, -1))*self.maskW)]
        # TRACER FLUXES (cfg.AVERAGE_TRACER_FLUX), the same form for T and S:
        # the advective flux with the model's centred face value, so that the
        # transient part is <vT> - <v><T>_face after reading. The meridional
        # heat transport is rho_0 c_p sum_x sum_z <vtemp> dx dz per basin
        # (heat_transport.py), the freshwater transport follows from vsalt.
        if self.cfg.AVERAGE_TRACER_FLUX:
            for tr, un in (('temp', 'degC'), ('salt', '')):
                get = lambda tau, tr=tr: getattr(self, tr)
                spec += [
                    ('u' + tr, U3, 'm/s %s' % un, 'zonal flux u (%s_i + %s_i+1)/2' % (tr, tr),
                     lambda tau, g=get: self.u*0.5*(g(tau) + self.rollx(g(tau), -1))*self.maskU),
                    ('v' + tr, V3, 'm/s %s' % un, 'meridional flux v (%s_j + %s_j+1)/2' % (tr, tr),
                     lambda tau, g=get: self.v*0.5*(g(tau) + self.rolly(g(tau), -1))*self.maskV),
                    ('w' + tr, W3, 'm/s %s' % un, 'vertical flux w (%s_k + %s_k+1)/2' % (tr, tr),
                     lambda tau, g=get: self.w*0.5*(g(tau) + self.rollz(g(tau), -1))*self.maskW)]
                # the STOKES part of the horizontal flux (coupled runs): u is
                # u_L = u_E + u_S here, so <vtemp> holds the Stokes advection
                # of heat; <vstemp> isolates it, with the same face value and
                # u_S, v_S (T points) averaged to the faces as the vortex force does
                if self.coupled:
                    spec += [
                        ('us' + tr, U3, 'm/s %s' % un, 'zonal Stokes flux u_S (%s_i + %s_i+1)/2' % (tr, tr),
                         lambda tau, g=get: 0.5*(self.u_S + self.rollx(self.u_S, -1))
                         * 0.5*(g(tau) + self.rollx(g(tau), -1))*self.maskU),
                        ('vs' + tr, V3, 'm/s %s' % un, 'meridional Stokes flux v_S (%s_j + %s_j+1)/2' % (tr, tr),
                         lambda tau, g=get: 0.5*(self.v_S + self.rolly(self.v_S, -1))
                         * 0.5*(g(tau) + self.rolly(g(tau), -1))*self.maskV)]
        # whatever the experiment registered (register_average above)
        spec += list(getattr(self, '_avg_extra', []))
        return spec

    def set_wave_roughness(self):
        """Surface roughness of the coupled run from the wave model's
        significant wave height (NEMO ln_mxhsw): z0 = HS_FACTOR * Hs, so
        tke.py's surface_mxl returns l0 = kappa z0 instead of Charnock's
        stress form. Called by the coupler after hs has been delivered and
        after a restart read, i.e. always BEFORE the closure runs."""
        if not (self.coupled and self.mxl_choice == 2 and self.cfg.HS_FACTOR > 0.0
                and hasattr(self, 'hs')):
            return
        ks = -self.halo - 1
        self.z0_surf = OM.np.maximum(0.0, self.cfg.HS_FACTOR*self.hs)*self.maskT[ks]
        return

    def mixed_layer_depth(self, dsig=0.03, zref=10.0):
        """Mixed layer depth [m] on this PE, padded (Ny, Nx), 0 on land.
        The definition of the de Boyer Montegut et al. (2004) climatology:
        the shallowest depth where potential density exceeds its value at
        zref = 10 m by dsig = 0.03 kg/m^3. Potential density is the model's
        equation of state at zero pressure. The density at 10 m and the
        crossing depth are interpolated linearly between T points. Where
        the criterion is never met the MLD is the depth of the column."""
        h  = self.halo
        sl = slice(h, -h)
        rho = np.asarray(get_rho(self.salt, self.temp, self.p0*0.0,
                                 self.eq_of_state))[sl, sl, sl][::-1]     # surface first
        wet = np.asarray(self.maskT)[sl, sl, sl][::-1] > 0
        d   = np.asarray(self.zw[-h-1] - self.zt)[sl][::-1]              # T-point depths
        dzw = np.asarray(self.dzt)[sl][::-1]
        nk  = len(d)
        # density at zref
        k1 = max(0, min(int(np.searchsorted(d, zref)) - 1, nk - 2))
        w  = min(max((zref - d[k1])/(d[k1+1] - d[k1]), 0.0), 1.0)
        r10 = np.where(wet[k1+1], (1.0 - w)*rho[k1] + w*rho[k1+1], rho[k1])
        thr = r10 + dsig
        below = d > zref
        cross = (rho > thr[None]) & wet & below[:, None, None]
        has = cross.any(axis=0)
        kc  = np.argmax(cross, axis=0)                                     # first crossing
        # upper point of the interpolation: the level above, or (zref, r10)
        ku  = np.maximum(kc - 1, 0)
        up_is_ref = ~below[ku]
        d_up = np.where(up_is_ref, zref, d[ku])
        r_up = np.where(up_is_ref, r10, np.take_along_axis(rho, ku[None], 0)[0])
        r_c  = np.take_along_axis(rho, kc[None], 0)[0]
        frac = np.clip((thr - r_up)/np.maximum(r_c - r_up, 1e-12), 0.0, 1.0)
        mld  = d_up + frac*(d[kc] - d_up)
        col  = np.sum(wet*dzw[:, None, None], axis=0)                      # column depth
        mld  = np.where(has, mld, col)*wet[0]
        out = np.zeros((self.Ny, self.Nx))
        out[sl, sl] = mld
        return out

    def average_setup(self, file_name, mode):
        """Create the mean file and zero the sums. mode from parse_average;
        None switches the means off."""
        self._avg = mode
        if mode is None:
            return
        self._avg_spec = self._average_spec()
        if self.cfg.AVERAGE_FIELDS:
            known = [nme for nme, *_ in self._avg_spec]
            bad = [f for f in self.cfg.AVERAGE_FIELDS if f not in known]
            if bad:
                raise ValueError('AVERAGE_FIELDS: unknown %s; available: %s'
                                 % (bad, ', '.join(known)))
            self._avg_spec = [sp for sp in self._avg_spec if sp[0] in self.cfg.AVERAGE_FIELDS]
        # RAW means (cfg.RAW_SNAPSHOTS): the same per-rank .npy directory the
        # snapshots use, for the same reason. The netCDF form below walks the
        # PEs one at a time through ONE shared file -- open, write, close, a
        # barrier, next rank -- which is 936 sequential opens of a Lustre file
        # at 1/3 degree, minutes per record when it works at all (HDF5 file
        # locking on a shared file system blocks it outright unless
        # HDF5_USE_FILE_LOCKING is off). Raw means write one small file per
        # rank and share nothing.
        self._avg_raw  = bool(self.cfg.RAW_SNAPSHOTS)
        self._avg_file = file_name
        self.average_reset()
        if self._avg_raw:
            return self._average_setup_raw(file_name, mode)
        # get_full_domain_extent is COLLECTIVE (an allreduce): every PE must
        # call it, not only the one that writes the file header
        nx_, ny_ = self.get_full_domain_extent()
        nz_ = self.Nz - 2*self.halo
        if self.my_pe == 0:
            f = Dataset(file_name, mode='w')
            for d, nn in (('xt', nx_), ('xu', nx_), ('yt', ny_), ('yu', ny_),
                          ('zt', nz_), ('zu', nz_), ('nb', 2)):
                f.createDimension(d, nn)
            f.createDimension('time', None)
            for nme, val in zip(('xt', 'yt', 'zt', 'xu', 'yu', 'zu'),
                                self._snap_coords(nx_, ny_, nz_)):
                f.createVariable(nme, 'f8', (nme,))[:] = np.asarray(val)
            v = f.createVariable('time', 'f8', ('time',))
            v.units = 's since model start'; v.long_name = 'middle of the averaging window'
            f.createVariable('time_bnds', 'f8', ('time', 'nb')).units = 's since model start'
            f.createVariable('nsteps', 'i4', ('time',)).long_name = 'ocean steps in the mean'
            for nme in ('date_start', 'date_end'):
                f.createVariable(nme, 'i4', ('time',)).long_name = 'forcing date YYYYMMDDHH'
            for nme, dims, units, ln, _ in self._avg_spec:
                v = f.createVariable(nme, 'f8', dims)
                v.units, v.long_name = units, 'time mean of ' + ln
            label = 'calendar month' if mode[0] == 'month' else '%g h' % (mode[1]/3600.0)
            f.averaging = label
            f.forcing_start, f.forcing_cycle_days = self.cfg.FORCING_START, self.cfg.FORCING_CYCLE_DAYS
            f.waves = int(self.coupled)
            f.alpha_CB = float(self.alpha_CB) if not self.coupled else -1.0
            f.r_bot = float(getattr(self, 'r_bot', 0.0)) if getattr(self, 'enable_bottom_friction', False) else 0.0
            f.kappaH_min = float(getattr(self, 'kappaH_min', 0.0))
            f.alpha_tke = float(getattr(self, 'alpha_tke', np.nan))
            f.mxl_choice = int(self.mxl_choice)
            f.mxl_beta = float(self.mxl_beta)
            f.hs_factor = float(self.cfg.HS_FACTOR) if self.coupled else 0.0
            f.mxl_surf = float(self.surface_mixing_length())
            f.mxl_surf_factor = float(getattr(self, 'mxl_surf_factor', 1.0))
            f.close()
            print('ocean means: %s -> %s, %d fields'
                  % (label, file_name, len(self._avg_spec)), flush=True)
        self.mpi_barrier()
        return

    def _average_setup_raw(self, file_name, mode):
        """Directory form of average_setup: <name>/pe=<rank>/n=<step>/<field>.npy,
        one directory per window, beside the axes and the decomposition. The
        layout is the snapshots' (pe_realistic snap_raw), so the same reader
        assembles both."""
        d = file_name[:-4] if file_name.endswith('.cdf') else file_name
        self._avg_dir = d
        nx_, ny_ = self.get_full_domain_extent()          # COLLECTIVE
        nz_ = self.Nz - 2*self.halo
        if self.my_pe == 0:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
            for nme, val in zip(('xt', 'yt', 'zt', 'xu', 'yu', 'zu'),
                                self._snap_coords(nx_, ny_, nz_)):
                np.save(os.path.join(d, nme + '.npy'), np.asarray(val))
            np.save(os.path.join(d, 'decomposition.npy'),
                    np.array([self.n_pes_x, self.n_pes_y]))
            for pe in range(self.n_pes):
                os.mkdir(os.path.join(d, 'pe=%d' % pe))
            label = 'calendar month' if mode[0] == 'month' else '%g h' % (mode[1]/3600.0)
            print('ocean means: %s -> %s/ (raw, %d fields, %d blocks)'
                  % (label, d, len(self._avg_spec), self.n_pes), flush=True)
        # rank 0 creates every PE's directory, the others mkdir inside them at
        # the first write: the barrier must stay
        self.mpi_barrier()
        return

    def _average_write_raw(self, t_end):
        """Each rank writes its own block of the window that just closed. The
        window is named by the step it ends at, as the snapshots are."""
        nsteps, t0 = self._avg_n, self._avg_t0
        step = int(round(t_end/self.dt))
        d = os.path.join(self._avg_dir, 'pe=%d' % self.my_pe, 'n=%d' % step)
        os.makedirs(d, exist_ok=True)
        for nme, *_ in self._avg_spec:
            np.save(os.path.join(d, nme + '.npy'), self._avg_sum[nme]/nsteps)
        if self.my_pe == 0:
            d0, d1 = self._forcing.date(t0), self._forcing.date(t_end)
            # what the netCDF form kept as time / time_bnds / nsteps / dates
            np.save(os.path.join(self._avg_dir, 'window_n=%d.npy' % step),
                    np.array([0.5*(t0 + t_end), t0, t_end, nsteps,
                              int(d0.strftime('%Y%m%d%H')),
                              int(d1.strftime('%Y%m%d%H'))], np.float64))
            print('ocean means: %s .. %s (%d steps) -> %s/*/n=%d'
                  % (d0.strftime('%Y-%m-%d %H:%M'), d1.strftime('%Y-%m-%d %H:%M'),
                     nsteps, self._avg_dir, step), flush=True)
        return

    def average_reset(self):
        if getattr(self, '_avg', None) is None:
            return
        h = self.halo
        nz_, ny_, nx_ = self.Nz - 2*h, self.Ny - 2*h, self.Nx - 2*h
        self._avg_sum = {nme: np.zeros((nz_, ny_, nx_) if len(dims) == 4 else (ny_, nx_))
                         for nme, dims, *_ in self._avg_spec}
        self._avg_n, self._avg_t0 = 0, -1.0
        return

    def average_state(self):
        """(list of (name, local sum), n, t0) for the restart, None when off."""
        if getattr(self, '_avg', None) is None:
            return None
        return ([(nme, self._avg_sum[nme]) for nme, *_ in self._avg_spec],
                self._avg_n, self._avg_t0)

    def average_restore(self, sums, n, t0):
        """Partial sums from the restart. A different field list (the run
        was changed between segments) drops them: that window is then
        shorter, and nsteps in the file says so."""
        if getattr(self, '_avg', None) is None:
            return
        if sorted(sums) != sorted(nme for nme, *_ in self._avg_spec):
            if self.my_pe == 0:
                print('ocean means: the restart holds a different field list, '
                      'the partial window is dropped', flush=True)
            return
        for nme in sums:
            self._avg_sum[nme] = np.array(sums[nme], np.float64)
        self._avg_n, self._avg_t0 = int(n), float(t0)
        return

    def _average_key(self, t):
        kind, L = self._avg
        if kind == 'month':
            d = self._forcing.date(t)
            return (d.year, d.month)
        return int(np.floor(t/L))

    def average(self, n, tau):
        """Add the state after step n; write and restart the window at its end."""
        if getattr(self, '_avg', None) is None:
            return
        h  = self.halo
        s3 = (slice(h, -h),)*3
        for nme, dims, _, _, get in self._avg_spec:
            a = np.asarray(get(tau))
            self._avg_sum[nme] += a[s3] if len(dims) == 4 else a[s3[1:]]
        if self._avg_n == 0:
            self._avg_t0 = n*self.dt
        self._avg_n += 1
        t_end = (n + 1)*self.dt
        if self._average_key(t_end - 0.5*self.dt) != self._average_key(t_end + 0.5*self.dt):
            self._average_write(t_end)
            self.average_reset()
        return

    def _average_write(self, t_end):
        if getattr(self, '_avg_raw', False):
            return self._average_write_raw(t_end)
        nsteps, t0 = self._avg_n, self._avg_t0
        self.mpi_barrier()
        m = 0
        if self.my_pe == 0:
            d0, d1 = self._forcing.date(t0), self._forcing.date(t_end)
            f = Dataset(self._avg_file, mode='r+')
            m = len(f.dimensions['time'])
            f['time'][m] = 0.5*(t0 + t_end)
            f['time_bnds'][m] = [t0, t_end]
            f['nsteps'][m] = nsteps
            f['date_start'][m] = int(d0.strftime('%Y%m%d%H'))
            f['date_end'][m] = int(d1.strftime('%Y%m%d%H'))
            f.close()
        m = self.pe0_bcast(m)
        j0, i0, ny_loc, nx_loc = local_block(self)
        for pe in range(self.n_pes):
            self.mpi_barrier()
            if self.my_pe == pe:
                f = Dataset(self._avg_file, mode='r+')
                for nme, dims, *_ in self._avg_spec:
                    sl = (m,) + ((slice(None),) if len(dims) == 4 else ()) + \
                         (slice(j0, j0+ny_loc), slice(i0, i0+nx_loc))
                    f.variables[nme][sl] = self._avg_sum[nme]/nsteps
                f.close()
        if self.my_pe == 0:
            print('ocean means: %s .. %s (%d steps) -> %s'
                  % (d0.strftime('%Y-%m-%d %H:%M'), d1.strftime('%Y-%m-%d %H:%M'),
                     nsteps, self._avg_file), flush=True)
        return

    # --- spherical Coriolis --------------------------------------------------
    def set_coriolis(self, f0=None):
        yt = np.asarray(self.yt)                       # degrees, local+halo
        f  = 2*self.cfg.omega_e*np.sin(yt/180.0*np.pi)
        self.coriolis_t = OM.np.asarray(
            np.broadcast_to(f[:, None], (self.Ny, self.Nx)).copy())
        return

    # --- topography ---------------------------------------------------------
    def set_topography(self):
        kb = self.cfg.kbot_global()
        j0, i0, ny_loc, nx_loc = local_block(self)
        kbot = np.zeros((self.Ny, self.Nx), np.float64)
        sl = slice(self.halo, -self.halo)
        kbot[sl, sl] = kb[j0:j0+ny_loc, i0:i0+nx_loc]
        self.kbot = self.apply_bc(OM.np.asarray(kbot))
        return

    # --- forcing: ERA5 + the two climatologies + the shortwave profile ------
    def localise_forcing(self):
        """Build the forcing object of this rank (forcing_common: the source
        is cfg.forcing_settings()['source']) and the shortwave penetration
        profile. WHERE THE FORCING COMES FROM IS NOT KNOWN HERE."""
        h  = self.halo
        sl = slice(h, -h)
        ks = -h-1
        geom = {'xt': np.asarray(self.xt)[sl], 'yt': np.asarray(self.yt)[sl],
                'block': local_block(self), 'comm': getattr(self, 'comm', None),
                'xt_gl': self.xt_gl, 'yt_gl': self.yt_gl}
        self._relwind = (str(self.cfg.STRESS_MODE) == 'relwind')
        want = ('taux', 'tauy', 'qnet', 'qsol', 'sst_star', 'dqdt', 'emp', 'sss_star')
        if self._relwind:
            want = want + ('u10', 'v10')
        self._forcing = make_forcing(self.cfg.forcing_settings(), geom, want, 'ocean')
        self._dz_surf = float(np.asarray(self.dzt)[ks])
        if self.my_pe == 0:
            for line in self._forcing.describe():
                print('CONFIG ' + line, flush=True)
            if self._relwind:
                print('CONFIG stress: RELATIVE WIND -- tau = C_D rho_a |u10 - u_o| (u10 - u_o), '
                      'C_D rho_a = min(|tau_ERA5| / max(|u10|, %.1f m/s)^2, %.1e rho_a), u_o %s '
                      'of the top cell (energy_cycle.tex Sec. 6.2); the waves take this stress '
                      'from the ocean'
                      % (float(self.cfg.STRESS_U10_MIN), float(self.cfg.STRESS_CD_MAX),
                         self.cfg.STRESS_UO), flush=True)
            piston = (self.cfg.SSS_PISTON if self.cfg.SSS_PISTON > 0.0
                      else self._dz_surf/self.cfg.t_rest)
            print('CONFIG forcing: SSS restoring piston velocity %.4f m/day'
                  % (piston*86400.0), flush=True)
            if float(self.cfg.ICE_TAU_FRAC) < 1.0:
                print('CONFIG forcing: under ice (T <= %.2f C) the surface '
                      'stress is scaled by %.2f, the heat and salt fluxes are '
                      'zero as before'
                      % (float(self.cfg.ICE_T_FREEZE),
                         float(self.cfg.ICE_TAU_FRAC)), flush=True)

        # penetrating shortwave: the fortran double-exponential divpen profile
        dzt_gl = self.cfg.dzt_global()
        zw = self.cfg.zw_pyom2()
        rpart, efold1, efold2 = 0.58, 0.35, 23.0
        pen = np.zeros(self.cfg.nz+1)                              # pen[0] = pen[nz] = 0
        for k in range(1, self.cfg.nz):                            # faces 1..nz-1
            pen[k] = (rpart*np.exp(zw[k-1]/efold1)
                      + (1.0-rpart)*np.exp(zw[k-1]/efold2))
        divpen = (pen[1:] - pen[:-1])/dzt_gl              # (nz,), bottom -> top
        dp = np.zeros(self.Nz); dp[sl] = divpen
        # kept as (Nz, 1, 1) so the bottom term below can be added to it and
        # set_forcing does not care which of the two forms it holds
        self._divpen = OM.np.asarray(dp)[:, None, None]
        if self.cfg.SW_BOTTOM:
            # pen[k] is the fraction still travelling downward through the
            # BOTTOM face of cell k, so pen at the deepest wet cell of a
            # column is exactly what that column loses. Give it to that cell.
            pp = np.zeros(self.Nz); pp[sl] = pen[:self.cfg.nz]
            mT = np.asarray(self.maskT)
            below = np.concatenate([np.zeros((1,) + mT.shape[1:]), mT[:-1]], axis=0)
            deepest = (mT > 0.0) & (below == 0.0)      # one cell per wet column
            dztp = np.asarray(self.dzt)
            add = deepest*(pp/np.maximum(dztp, 1e-30))[:, None, None]
            self._divpen = OM.np.asarray(np.asarray(self._divpen) + add)
            if self.my_pe == 0:
                print('CONFIG forcing: the solar radiation reaching the sea floor is '
                      'absorbed in the deepest wet cell (SW_BOTTOM)', flush=True)
        return

    # ----------------------------------------------------------------------
    # energy diagnostics (all area weighted on the spherical grid; kinematic,
    # i.e. per unit rho -- multiply by rho_w for Watts)
    # ----------------------------------------------------------------------
    # ------------------------------------------------------------------
    # FREE-SURFACE TRACER FLUX, booked for the conservation identity.
    # With free_surface_tracer_flux = True (thermodynamics.py) the surface
    # face carries ft = -w_surf var_surf instead of being closed, so the
    # column sums of the advective tendency no longer telescope to zero: the
    # tracer content of the domain changes by
    #     F_fs = -sum_cells w_surf var_surf A_t
    # per step, on top of the surface forcing. That is not a loss -- it is
    # the tracer that crosses the moving surface, the term whose ABSENCE in
    # pyOM2's form leaves the top cell with -var dEta/dz every step -- but
    # the identity below compares the content change with the FORCING alone,
    # so without this term the residual is that flux (5 to 10 % of the
    # surface heat flux at 1.5 degrees, 500 % on the first day when
    # d(eta)/dt is largest).
    # It is booked HERE, at the moment of application and from the same
    # state and time level the advection used, so the identity stays a
    # round-off test rather than acquiring a one-step lag.
    # ------------------------------------------------------------------
    def tracer_advection(self, var, dvar, u, v, w, maskU, maskV, maskW, maskT, tau):
        out = super().tracer_advection(var, dvar, u, v, w, maskU, maskV, maskW, maskT, tau)
        if getattr(self, 'free_surface_tracer_flux', False) and int(self.cfg.KE_BUDGET) == 1:
            h = self.halo; ks = -h-1
            I2 = (slice(h, -h), slice(h, -h))
            f = -float(np.sum((np.asarray(w)[ks]*np.asarray(var)[ks]
                               *np.asarray(maskW)[ks]*np.asarray(maskT)[ks]
                               *np.asarray(self.area_t)[0])[I2]))
            # temp is advected first, salt second (pe_realistic/model.py), and
            # the object itself says which one this is
            if var is self.salt:
                self._fs_flux_S = f
            else:
                self._fs_flux_T = f
        return out

    def vadv_check(self):
        """cfg.VADV_SUB_CYCLING -> the model option, before the package's own
        check in prepare_for_run (pe_realistic/vadv_split.py)"""
        self.vadv_sub_cycling = int(self.cfg.VADV_SUB_CYCLING)
        super().vadv_check()
        if self.my_pe == 0 and self.vadv_sub_cycling > 0:
            print('CONFIG ocean: tracer advection outside the AB step, forward '
                  'superbee sweeps, %d vertical sub-step(s): w dt/dz <= %d '
                  '(VADV_SUB_CYCLING)' % (self.vadv_sub_cycling, self.vadv_sub_cycling), flush=True)
        return

    def _reduce(self, w):
        comm = getattr(self, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            return comm.allreduce(float(w), op=MPI.SUM)
        return float(w)

    # --- what the coupled ATMOSPHERE needs from the ocean (energy_cycle.tex Sec. 6) --
    def surface_velocity(self, mode=None):
        """(u_o, v_o): the surface velocity the wind sees, T points, padded and
        masked. mode (default cfg.ATM_UO) = 'eulerian': u_L - u_S of the top
        cell (the wave orbital motion is the wave model's business),
        'lagrangian': u_L."""
        h = self.halo; ks = -h-1
        mode = mode or self.cfg.ATM_UO
        u, v = np.asarray(self.u)[ks], np.asarray(self.v)[ks]
        uT = 0.5*(u + np.roll(u, 1, axis=1)); vT = 0.5*(v + np.roll(v, 1, axis=0))
        if mode == 'eulerian' and hasattr(self, 'u_S'):
            uT = uT - np.asarray(self.u_S)[ks]; vT = vT - np.asarray(self.v_S)[ks]
        mT = np.asarray(self.maskT)[ks]
        return uT*mT, vT*mT

    def stress_work_field(self):
        """Local work w_o [W/m^2] of the surface momentum forcing on u_L at the
        T points (padded): tau_o plus the direct branch, i.e. exactly what the
        step applied, U/V -> T with the metric weights so that
        sum w_o A_t = rho_0 x stress_work_diag."""
        h = self.halo; ks = -h-1
        Au, Av, At = (np.asarray(x)[0] for x in (self.area_u, self.area_v, self.area_t))
        mU, mV = np.asarray(self.maskU), np.asarray(self.maskV)
        if self.use_stress_work:
            # depth-distributed stress: u(z) tau(z) level by level, as
            # stress_work_diag sums it (the stress on U/V points as applied)
            u, v = np.asarray(self.u), np.asarray(self.v)
            twx, twy = np.asarray(self.tau_work_x), np.asarray(self.tau_work_y)
            dz = np.asarray(self.dzt_z)
            tx = 0.5*(twx + np.roll(twx, -1, axis=2)); ty = 0.5*(twy + np.roll(twy, -1, axis=1))
            pu = np.sum(u*tx*mU*dz, axis=0)*Au
            pv = np.sum(v*ty*mV*dz, axis=0)*Av
            u, v = u[ks], v[ks]
        else:
            u, v = np.asarray(self.u)[ks], np.asarray(self.v)[ks]
            pu = u*np.asarray(self.taux)*mU[ks]*Au
            pv = v*np.asarray(self.tauy)*mV[ks]*Av
        if hasattr(self, '_tau_dir_x'):
            pu = pu + u*np.asarray(self._tau_dir_x)*mU[ks]*Au
            pv = pv + v*np.asarray(self._tau_dir_y)*mV[ks]*Av
        return self.cfg.rho_0*(0.5*(pu + np.roll(pu, 1, axis=1)) + 0.5*(pv + np.roll(pv, 1, axis=0)))/At

    def vortex_work(self):
        """u_L-work of the vortex force, booked BY set_forcing at the moment
        the force is formed -- with the velocities the pairing actually used.
        With the area-weighted corner pairing the total then vanishes
        IDENTICALLY by summation by parts, walls and land masks included: a
        pure round-off monitor, so the budget can state that the vortex force
        does no work on u_L.

        (Booking it here rather than re-evaluating u.F after the step
        matters: by the time the coupler asks, the AB update, the pressure
        solve and the vertical friction have all moved u, and the stored
        force dotted with the moved velocity no longer cancels -- that
        version reads O(1e-5) of the stress work, which is a time-lag
        artefact of the diagnostic and not a property of the force.)
        Collective."""
        return self._reduce(getattr(self, '_vortex_step', 0.0))

    def stokes_shear_work(self):
        """u_L-work of the HORIZONTAL Stokes shear force alone (the w dz(u_S)
        terms), booked by set_forcing when cfg.STOKES_SHEAR is on. The
        pressure p_SS does the same work with the opposite sign, so this is
        the size of a cancelling pair and not an energy source. Collective."""
        return self._reduce(getattr(self, '_ss_step', 0.0))

    def stokes_shear_force(self):
        """THE FULL VORTEX FORCE in the hydrostatic model (cfg.STOKES_SHEAR).

        The force -u_L x (nabla x u_S) with u_S = (u_S, v_S, 0) is
            F_x = -v zeta_S + w dz(u_S)
            F_y = +u zeta_S + w dz(v_S)
            F_z = -u dz(u_S) - v dz(v_S)
        set_forcing applies the zeta_S part. This method applies the rest:
        the horizontal Stokes shear terms w dz(u_S), w dz(v_S) and the
        vertical component F_z, which the hydrostatic model carries in its
        hydrostatic relation
            dz(p) = -g rho/rho_0 + F_z .
        Integrated down from the surface, where it is zero, F_z gives the
        extra pressure
            p_SS(z) = int_z^0 (u dz(u_S) + v dz(v_S)) dz'
        and its horizontal gradient is added to the momentum tendency
        exactly as the gradient of p_h is (the model's p_h itself is not
        touched, so the KE budget's conversion KE_ph and ni_diag's p_h keep
        their meaning; ni_diag adds p_SS to its pressure).

        ENERGETICS. In the continuum u_L.F = 0 pointwise. Discretely the
        work of the horizontal terms,
            W_h = sum (u F_x A_u + v F_y A_v) dz ,
        and the work of the pressure gradient,
            W_p = -sum (u dx(p_SS) A_u + v dy(p_SS) A_v) dz
                = sum_T p_SS div_h(u) A_T dz = sum_W w (p_SS(k+1) - p_SS(k)) A_T
        (summation by parts with the divergence that defines the model's w,
        no boundary terms because p_SS = 0 at the surface T point and w = 0
        at the bottom face) cancel IDENTICALLY, W_h + W_p = 0, when the
        averages are chosen as transposes of each other:
          * the Stokes shear S^u = (u_S(k+1) - u_S(k))/dzw(k) at the W face
            k (T point horizontally, between the T levels k and k+1), zero at
            the surface face and masked by maskW;
          * the face product w S^u, averaged to the T level k as
            H^u(k) = (w S^u(k) + w S^u(k-1))/2, then to the U point with the
            T areas: F_x = (A_T H^u(i) + A_T H^u(i+1))/(2 A_u);
          * the velocity at the T point is the plain two-point average
            u^x = (u(i-1) + u(i))/2 (the transpose of the previous step),
            and at the W face the THICKNESS-WEIGHTED average
            u^xz(k) = (dzt(k) u^x(k) + dzt(k+1) u^x(k+1))/(2 dzw(k)) (the
            transpose of the H average on the stretched grid), so that
            p_SS(k) = p_SS(k+1) + dzw(k) (u^xz S^u + v^yz S^v)(k), p_SS(ks) = 0.
        Then W_h = sum_T (u^x H^u + v^y H^v) A_T dz = sum_W w dzw (u^xz S^u
        + v^yz S^v) A_T = -W_p. Both works are booked here with the
        velocities the force used (as vortex_work explains, booking after
        the step would not cancel). Returns (F_x, F_y, W_h, W_p), local sums.

        STABILITY. The paper's App. B argument (a rotation at the rate
        |dz(u_S)|) does not apply here, because w is not prognostic: the
        linear operator u -> F is anti-symmetric in the energy norm, so it
        is purely oscillatory, and its eigenfrequencies were measured
        offline (Oct 2026: exponential u_S profile, 2 m top cell, doubly periodic) at about
            omega_max = 10 u_S(0)/dx ,
        nearly independent of the decay depth and of the direction of u_S,
        i.e. a Doppler-like wave of the Stokes-drift scale and not a shear
        rate. On icon033 (dx = 7.4 km at the narrowest, dt = 216 s) that is
        omega dt = 0.09 at u_S(0) = 0.3 m/s and 0.17 at 0.6 m/s, against
        the AB3 limit 0.72: no time step change is needed.
        """
        h  = self.halo
        ks = -h-1
        si = slice(h, -h)
        rx, ry, rz = self.rollx, self.rolly, self.rollz
        u, v, w = self.u, self.v, self.w
        mT, mU, mV, mW = self.maskT, self.maskU, self.maskV, self.maskW
        dzt, dzw = self.dzt_z, self.dzw_z
        At, Au, Av = self.area_t, self.area_u, self.area_v

        # the Stokes shear at the W faces, zero at the surface face (and in
        # the halo above it, where u_S is not defined)
        face = (OM.np.arange(self.Nz) < self.Nz - h - 1).astype(float)[:, None, None]
        Su = (rz(self.u_S, -1) - self.u_S)/dzw*mW*face
        Sv = (rz(self.v_S, -1) - self.v_S)/dzw*mW*face

        # --- horizontal part: w dz(u_S), w dz(v_S) ----------------------------
        wSu, wSv = w*mW*Su, w*mW*Sv                           # at the W faces
        Hu = 0.5*(wSu + rz(wSu, 1))*mT                         # T level k: faces k and k-1
        Hv = 0.5*(wSv + rz(wSv, 1))*mT
        AHu, AHv = self.apply_bc(At*Hu), self.apply_bc(At*Hv)
        Fu = 0.5*(AHu + rx(AHu, -1))/Au*mU
        Fv = 0.5*(AHv + ry(AHv, -1))/Av*mV
        W_h = float(OM.np.sum((u*Fu*Au + v*Fv*Av)[si, si, si]*dzt[si]))

        # --- vertical part, as the pressure p_SS ------------------------------
        uT = 0.5*(u + rx(u, 1))*mT                             # the transposes of the averages above
        vT = 0.5*(v + ry(v, 1))*mT
        duT, dvT = dzt*uT, dzt*vT
        uW = 0.5*(duT + rz(duT, -1))/dzw                       # thickness-weighted, at the W faces
        vW = 0.5*(dvT + rz(dvT, -1))/dzw
        G  = dzw*(uW*Su + vW*Sv)*mW                            # p_SS(k) - p_SS(k+1)
        # p_SS(k) = sum over the faces k' >= k (the surface face carries 0)
        p_SS = OM.np.cumsum(G[::-1], axis=0)[::-1]*mT
        p_SS = self.apply_bc(p_SS)
        self.p_SS = p_SS
        Fpu = -(rx(p_SS, -1) - p_SS)/(self.cost_y*self.dxu_x)*mU
        Fpv = -(ry(p_SS, -1) - p_SS)/self.dyu_y*mV
        W_p = float(OM.np.sum((u*Fpu*Au + v*Fpv*Av)[si, si, si]*dzt[si]))
        return Fu + Fpu, Fv + Fpv, W_h, W_p

    def _direct_work(self, u, v):
        """Surface work of the direct branch tau_a - tau_in (zero when the
        branch is off), against the velocity (u, v) handed in."""
        if not hasattr(self, '_tau_dir_x'):
            return 0.0
        h  = self.halo
        sl = slice(h, -h)
        ks = -h-1
        return OM.np.sum((u[ks]*self._tau_dir_x*self.maskU[ks]*self.area_u[0]
                        + v[ks]*self._tau_dir_y*self.maskV[ks]*self.area_v[0])[sl, sl])

    def stress_work_diag(self):
        """Work of the SURFACE MOMENTUM FORCING on the model velocity u_L, the
        u_L.dz(tau) channel of Eq. (6), exactly as applied in set_forcing:
        the wave stress tau_o plus the direct branch tau_a - tau_in when
        direct_stress_branch is on (before Sep 8 2026 the branch's work was
        not booked here, so the channel read only the tau_o part)."""
        h  = self.halo
        sl = slice(h, -h)
        ks = -h-1
        if self.use_stress_work:
            Fu = 0.5*(self.tau_work_x + self.rollx(self.tau_work_x, -1))*self.maskU
            Fv = 0.5*(self.tau_work_y + self.rolly(self.tau_work_y, -1))*self.maskV
            w = OM.np.sum((self.u*Fu*self.area_u + self.v*Fv*self.area_v)[sl, sl, sl]
                          *self.dzt_z[sl])
        else:
            w = OM.np.sum((self.u[ks]*self.taux*self.maskU[ks]*self.area_u[0]
                         + self.v[ks]*self.tauy*self.maskV[ks]*self.area_v[0])[sl, sl])
        w = w + self._direct_work(self.u, self.v)
        return self._reduce(w)

    def stress_work_eul_diag(self):
        """The same forcing dotted with the EULERIAN velocity u = u_L - u_S.
        NOT the stress channel of the consistent u_L cycle (Eq. 6 wants
        u_L.dz(tau)) -- kept for comparison with the old formulation."""
        h  = self.halo
        sl = slice(h, -h)
        ks = -h-1
        uSU = 0.5*(self.u_S + self.rollx(self.u_S, -1))*self.maskU
        vSV = 0.5*(self.v_S + self.rolly(self.v_S, -1))*self.maskV
        uE, vE = self.u - uSU, self.v - vSV
        if self.use_stress_work:
            Fu = 0.5*(self.tau_work_x + self.rollx(self.tau_work_x, -1))*self.maskU
            Fv = 0.5*(self.tau_work_y + self.rolly(self.tau_work_y, -1))*self.maskV
            w = OM.np.sum((uE*Fu*self.area_u + vE*Fv*self.area_v)[sl, sl, sl]
                          *self.dzt_z[sl])
        else:
            w = OM.np.sum((uE[ks]*self.taux*self.maskU[ks]*self.area_u[0]
                         + vE[ks]*self.tauy*self.maskV[ks]*self.area_v[0])[sl, sl])
        w = w + self._direct_work(uE, vE)
        return self._reduce(w)

    def exchange_work_C(self):
        """Exchange term C = int u_L d(u_S)/dt dV of the wave-averaged energy
        cycle, booked EXACTLY at its application point by
        set_forcing_direct."""
        return self._reduce(getattr(self, '_C_step', 0.0))

    def vertical_friction(self):
        """Book the exact KE change of the implicit friction (and its
        backward-Euler part |u+ - u|^2/(2 dt)) at the moment it is applied.
        Only with the KE budget on: without it the two copies and the four
        full-field reductions per step buy nothing."""
        if not self.cfg.KE_BUDGET:
            return super().vertical_friction()
        u0, v0 = self.u*1.0, self.v*1.0
        k0 = self._ke_of(u0, v0)
        super().vertical_friction()
        self._ke_vfric = self._ke_of(self.u, self.v) - k0
        self._ke_vnum  = self._ke_of(self.u - u0, self.v - v0)
        return

    def init_tke(self):
        """Package init_tke, then the mixing-length settings.
        EXP_MXL_CHOICE = 1 (pyOM2 limits): EXP_MXL_SURF_FACTOR acts
        through the hook z0_surf, since the surface W point is limited to
        mxl_min + z0_surf + dzt(top) after the downward sweep (tke.f90 order),
        so z0_surf = (f - 1) dzt(top).
        EXP_MXL_CHOICE = 2 (NEMO nn_mxl 3): the surface length is
        Charnock's kappa beta |tau|/(g rho_0); z0_surf would be read as a
        roughness length there, so the factor is not applied."""
        super().init_tke()
        self.mxl_choice = self.cfg.MXL_CHOICE
        self.mxl_surf_factor = self.cfg.MXL_SURF_FACTOR
        self.mxl_beta = self.cfg.MXL_BETA
        self.z0_surf = (self.cfg.MXL_SURF_FACTOR - 1.0)*self._dzt_top() if self.cfg.MXL_CHOICE == 1 else 0.0
        return

    def _dzt_top(self):
        return float(np.asarray(self.dzt)[-self.halo-1])

    def surface_mixing_length(self):
        """A single number for the surface mixing length [m]: the limit at
        the surface W point (choice 1), or Charnock's l0 AT |tau| = 0.1 Pa
        (choice 2, where l0 follows the stress and has no fixed value)."""
        if self.mxl_choice == 2:
            return float(self.mxl_kappa*self.mxl_beta*0.1/(self.grav*self.cfg.rho_0))
        return (float(getattr(self, 'mxl_min', 0.0)) + float(getattr(self, 'z0_surf', 0.0))
                + self._dzt_top())

    def _surface_forcing(self):
        """Surface TKE flux [m^3/s^3], the flux boundary condition of the TKE
        equation (tke.py puts it into the top half W cell, as tke.f90).
        Coupled: the wave dissipation flux phi_oc (package behaviour).
        Ocean alone: the wave-breaking flux of Craig and Banner (1994),
            F = alpha_CB |tau_a/rho_0|^{3/2} = alpha_CB u*_w^3 ,
        with the stress first averaged to T points as pyOM2's set_forcing
        does for forc_tke_surface. alpha_CB = 1 is the pyOM2 boundary
        condition, alpha_CB ~ 100 the observed breaking-wave flux, which the
        coupled wave model delivers as phi_oc."""
        if self.coupled:
            return super()._surface_forcing()
        ks = -self.halo - 1
        tx = 0.5*(self.taux + self.rollx(self.taux, 1))       # U -> T
        ty = 0.5*(self.tauy + self.rolly(self.tauy, 1))       # V -> T
        F = self.alpha_CB*OM.np.hypot(tx, ty)**1.5*self.maskT[ks]
        return self.apply_bc(F)

    def integrate_tke(self):
        """TKE budget identity, booked around the package's integrate_tke:
        sum V (tke+ - tke)/dt = sum V (K_diss_v - P_diss_v) + A forc_surf
        + A tke_surf_corr - sum V c_eps sqrt(tke)/mxl tke+ ; V = dzw A_t
        (0.5 dzw at the surface level), diffusion telescopes to zero."""
        if int(self.cfg.KE_BUDGET) != 1:
            return super().integrate_tke()
        tke0 = self.tke*1.0
        super().integrate_tke()
        h = self.halo; ks = -h-1; sl = slice(h, -h)
        V = self.dzw_z*self.area_t*self.maskW
        V = OM.modify_array(V, (ks, slice(None), slice(None)), 0.5*V[ks])
        # the solve's linearised sink acted on tke+ BEFORE the negative-surface
        # clipping; reconstruct it from tke_surf_corr = -neg*(0.5 dzw)/dt
        tke_pre = OM.modify_array(self.tke*1.0, (ks, slice(None), slice(None)),
                                  self.tke[ks] - self.tke_surf_corr*self.dt/(0.5*self.dzw[ks]))
        lhs = float(OM.np.sum((V*(self.tke - tke0))[sl, sl, sl]))/self.dt
        rhs = float(OM.np.sum((V*(self.K_diss_v - self.P_diss_v))[sl, sl, sl])) \
            + float(OM.np.sum(((self.forc_tke_surface + self.tke_surf_corr)
                               *self.area_t[0]*self.maskW[ks])[sl, sl])) \
            - float(OM.np.sum((V*self.c_eps/self.mxl_eps*self.sqrttke*tke_pre)[sl, sl, sl]))
        self._tke_lhs, self._tke_rhs = lhs, rhs
        return

    def _ke_of(self, u, v):
        """local int |u|^2/2 dV (no reduction)"""
        h = self.halo; sl = slice(h, -h)
        return 0.5*float(OM.np.sum((u*u*self.area_u + v*v*self.area_v)[sl, sl, sl]
                                   *self.dzt_z[sl]))

    def ke_budget(self, n, tau, efile=None):
        # run.py sets energy_file, one file per segment
        efile = getattr(self, 'energy_file', None) or efile or self.cfg.exp_name + '_energy.txt'
        h = self.halo; ks = -h-1
        I  = (slice(h, -h), slice(h, -h), slice(h, -h)); I2 = (slice(h, -h), slice(h, -h))
        gs = lambda x: self._reduce(float(np.sum(x)))     # global sum over the ocean PEs
        rx = lambda a, s: np.roll(a, s, axis=-1); ry = lambda a, s: np.roll(a, s, axis=-2)
        rz = lambda a, s: np.roll(a, s, axis=0)
        if not hasattr(self, 'eb_area'):
            At, Au, Av = (np.asarray(x) for x in (self.area_t, self.area_u, self.area_v))
            dz = np.asarray(self.dzt_z)
            self.eb_mu = (np.asarray(self.maskU)*Au*dz)[I]
            self.eb_mv = (np.asarray(self.maskV)*Av*dz)[I]
            self.eb_A2 = (np.asarray(self.maskT)[ks]*At[0])[I2]
            self.eb_area = gs(self.eb_A2)
            self.eb_last = None; self.eb_acc = np.zeros(15); self.eb_chk = np.zeros(5); self.eb_n = 0
            self.eb_fs = 0.0; self.eb_ss = 0.0
            self.eb_steps = max(1, int(round(self.eb_every_days*86400.0/self.dt)))
            if self.my_pe == 0:
                print(self.KE_LEGEND, flush=True)
                with open(efile, 'w') as fh: fh.write('# ' + ' '.join(self.KE_COLUMNS) + '\n')
        # KE_BUDGET = 2 is the LIGHT form: the energy pathways (wind work,
        # C, the three dissipations, K_diss, the conversion, PEs) and the
        # finite-difference leak, WITHOUT the identity checks that re-evaluate
        # the momentum tendencies and the tracer contents every step. The
        # conversion then uses the model's own p_h of the step (one time level
        # behind the state, which the daily mean does not see), and the
        # advection and Coriolis terms, round-off in the full form, are
        # counted into LEAK. About a tenth of the full form's cost.
        light = int(self.cfg.KE_BUDGET) == 2
        u, v, w = (np.asarray(x) for x in (self.u, self.v, self.w))
        mU, mV, mT, mW = (np.asarray(x) for x in (self.maskU, self.maskV, self.maskT, self.maskW))
        p_s  = np.asarray(self.p_s[tau])
        dz3  = np.asarray(self.dzt_z)
        cost, cosu = np.asarray(self.cost_y), np.asarray(self.cosu_y)
        dxt, dxu, dyt, dyu = (np.asarray(x) for x in (self.dxt_x, self.dxu_x, self.dyt_y, self.dyu_y))
        Au3, Av3, At3 = (np.asarray(x) for x in (self.area_u, self.area_v, self.area_t))
        KEu = lambda a, b: gs((u*a)[I]*self.eb_mu) + gs((v*b)[I]*self.eb_mv)
        # reservoirs
        KE  = 0.5*gs((u[I]**2)*self.eb_mu) + 0.5*gs((v[I]**2)*self.eb_mv)
        PEs = 0.5*gs((p_s*p_s/self.grav)[I2]*self.eb_A2)
        if light:
            p_h = np.asarray(self.p_h)
            du_ph = -(rx(p_h, -1) - p_h)/(cost*dxu)*mU;  dv_ph = -(ry(p_h, -1) - p_h)/dyu*mV
            du_ps = -(rx(p_s, -1) - p_s)[None]/(cost*dxu)*mU;  dv_ps = -(ry(p_s, -1) - p_s)[None]/dyu*mV
            KE_ph, KE_ps = KEu(du_ph, dv_ph), KEu(du_ps, dv_ps)
            KE_adv = KE_cor = KE_div = CONV_chk = 0.0
        else:
            # tendencies as the model forms them (momentum_tendency_impl)
            fe, fn, ft = (np.asarray(x) for x in self.mom_flux_2nd_u(self.u, self.u, self.v, self.w,
                                                                    self.maskU, self.maskV, self.maskW))
            du_adv = ((fe-rx(fe, 1)) + (fn-ry(fn, 1)) + (ft-rz(ft, 1)))*mU/(Au3*dz3)
            fe, fn, ft = (np.asarray(x) for x in self.mom_flux_2nd_v(self.v, self.u, self.v, self.w,
                                                                    self.maskU, self.maskV, self.maskW))
            dv_adv = ((fe-rx(fe, 1)) + (fn-ry(fn, 1)) + (ft-rz(ft, 1)))*mV/(Av3*dz3)
            du_cor, dv_cor = (np.asarray(x) for x in self.coriolis_and_metric(self.u, self.v, self.coriolis_t,
                                                                            self.maskU, self.maskV))
            p_h = np.asarray(self.hydrostatic_pressure(self.rho, OM.np.zeros_like(self.rho), self.maskT))
            du_ph = -(rx(p_h, -1) - p_h)/(cost*dxu)*mU;  dv_ph = -(ry(p_h, -1) - p_h)/dyu*mV
            du_ps = -(rx(p_s, -1) - p_s)[None]/(cost*dxu)*mU;  dv_ps = -(ry(p_s, -1) - p_s)[None]/dyu*mV
            KE_adv, KE_cor, KE_ph, KE_ps = KEu(du_adv, dv_adv), KEu(du_cor, dv_cor), KEu(du_ph, dv_ph), KEu(du_ps, dv_ps)
            # CONVERSION IDENTITY: the pressure-gradient work of u_L equals, by
            # summation by parts with the same divergence that defines w, the
            # buoyancy flux through the interior faces plus the work of the
            # surface face (linear free surface): KE_ph = CONV_int + CONV_top,
            #   CONV_int = sum_faces w_k (p_h(k+1) - p_h(k)) A_t  (k < ks),
            #   CONV_top = -sum w_ks p_h(ks) A_t.
            # KE_ph - (CONV_int + CONV_top) must be round-off; CONV_int written
            # with p_h differences equals -(g/rho_0) sum w 0.5(rho+rho_up) dzw A_t.
            wk = w*mW
            CONV_int = gs((wk*(rz(p_h, -1) - p_h)*At3)[h:ks, h:-h, h:-h])
            CONV_top = -gs((wk[ks]*p_h[ks]*At3[0])[I2])
            CONV_chk = KE_ph - (CONV_int + CONV_top)
            # CHECK of the advection: the centred flux form telescopes exactly to
            # -1/2 sum u^2 (div T) with T the face transports the fluxes use
            # (mom_flux_2nd_u/v: 1/2(utr_i + utr_i+1) etc., top face closed), so
            # KE_adv - KE_div must be round-off; the remainder KE_div itself is the
            # divergence of the transports, i.e. the free-surface top layer.
            utr = u*mU*dz3*dyt; vtr = v*mV*dz3*cosu*dxt; wtr = w*mW*At3
            def divT(tr_x, tr_y, tr_z):
              d = (tr_x - rx(tr_x, 1)) + (tr_y - ry(tr_y, 1)) + (tr_z - rz(tr_z, 1))
              return d
            Tx = 0.5*(utr + rx(utr, -1)); Ty = 0.5*(vtr + rx(vtr, -1)); Tz = 0.5*(wtr + rx(wtr, -1))
            Tz[ks] = 0.0
            KE_div = -0.5*gs((u*u*divT(Tx, Ty, Tz)*mU)[I])
            Tx = 0.5*(utr + ry(utr, -1)); Ty = 0.5*(vtr + ry(vtr, -1)); Tz = 0.5*(wtr + ry(wtr, -1))
            Tz[ks] = 0.0
            KE_div += -0.5*gs((v*v*divT(Tx, Ty, Tz)*mV)[I])
        # surface forcing as applied in set_forcing (surface-stress mode)
        W_tau = gs((u[ks]*np.asarray(self.taux)*mU[ks]*Au3[0])[I2]) \
              + gs((v[ks]*np.asarray(self.tauy)*mV[ks]*Av3[0])[I2])
        W_dir = gs(self._direct_work(self.u, self.v))
        # exact bookings at the moment of application
        W_vortex = gs(getattr(self, '_vortex_step', 0.0))
        W_ss     = gs(getattr(self, '_ss_step', 0.0))        # STOKES_SHEAR: the cancelling pair's size
        C        = gs(getattr(self, '_C_step', 0.0))
        D_h      = gs(getattr(self, '_ke_hfric', 0.0))/self.dt
        D_bot    = gs(getattr(self, '_ke_bfric', 0.0))/self.dt
        D_v      = gs(getattr(self, '_ke_vfric', 0.0))/self.dt
        D_v_num  = gs(getattr(self, '_ke_vnum', 0.0))/self.dt
        K_diss   = gs(np.sum((np.asarray(self.K_diss_v)*At3*np.asarray(self.dzw_z))[I])) \
                   if hasattr(self, 'K_diss_v') else 0.0
        # surface potential energy tendency by the divergence of the transport
        divU = np.sum(((u*mU - rx(u*mU, 1))/(cost*dxt)
                       + ((v*mV*cosu) - ry(v*mV*cosu, 1))/(cost*dyt))*dz3*mT, axis=0)
        PEs_dot = -gs((p_s*divU)[I2]*self.eb_A2)
        if self.eb_last is None: self.eb_last = KE + PEs
        # TRACER CONSERVATION: the content change per step must equal the
        # AB-weighted surface flux (all interior operators are flux-form);
        # the penetrating shortwave integrates to zero over the column.
        if light:
            Tc = Sc = FT = FS = 0.0
        else:
            Tc = gs((np.asarray(self.temp)*At3*dz3*mT)[I]); Sc = gs((np.asarray(self.salt)*At3*dz3*mT)[I])
            FT = gs(((np.asarray(self.forc_temp_surface) + np.asarray(getattr(self, '_sw_col', 0.0)))*At3[0]*mT[ks])[I2])
            FS = gs((np.asarray(self.forc_salt_surface)*At3[0]*mT[ks])[I2])
        # the tracer carried through the moving surface, booked in
        # tracer_advection above (zero unless free_surface_tracer_flux)
        F_fs_T = self._reduce(getattr(self, '_fs_flux_T', 0.0))
        F_fs_S = self._reduce(getattr(self, '_fs_flux_S', 0.0))
        FT = FT + F_fs_T; FS = FS + F_fs_S
        # vadv_sub_cycling > 0: tracer_advection above is never called. The
        # surface face is applied inside the vertical sub-steps, WITHOUT AB
        # weights, and the model reports its sub-step mean (vadv_fs_flux,
        # var m/s, already masked): it enters the identity unweighted.
        U_fs_T = U_fs_S = 0.0
        if getattr(self, 'vadv_sub_cycling', 0) > 0:
            U_fs_T = gs((np.asarray(self.vadv_fs_flux['temp'])*At3[0]*mT[ks])[I2])
            U_fs_S = gs((np.asarray(self.vadv_fs_flux['salt'])*At3[0]*mT[ks])[I2])
            F_fs_T = U_fs_T                      # for the 'free surface' size in the line
        # the weights of THIS step (coupler_loop sets _ab; AB3 has three) and
        # the fluxes of the two steps before. After a restart the old levels
        # are not known, so the residual is formed once they are.
        A_, B_, C_ = getattr(self, '_ab', (1.6, -0.6, 0.0))
        if not hasattr(self, 'eb_Tc'):
            self.eb_trc = np.zeros(2); self.eb_Fold = []
        else:
            need = 2 if C_ != 0.0 else 1
            if len(self.eb_Fold) >= need:
                o = self.eb_Fold + [(0.0, 0.0)]
                self.eb_trc += np.abs([ (Tc - self.eb_Tc)/self.dt - (A_*FT + B_*o[0][0] + C_*o[1][0]) - U_fs_T,
                                        (Sc - self.eb_Sc)/self.dt - (A_*FS + B_*o[0][1] + C_*o[1][1]) - U_fs_S ])
        self.eb_Tc, self.eb_Sc = Tc, Sc
        self.eb_Fold = ([(FT, FS)] + self.eb_Fold)[:2]
        # HARMONIC FRICTION and TKE identities, booked at application
        D_h_chk = getattr(self, '_ke_hfric', 0.0)/self.dt - (getattr(self, '_ke_hstencil', 0.0)
                                                             + getattr(self, '_ke_hnum', 0.0))
        TKE_chk = getattr(self, '_tke_lhs', 0.0) - getattr(self, '_tke_rhs', 0.0)
        self.eb_acc += self.dt*np.array([KE_adv, KE_cor, KE_ph, KE_ps, W_tau, W_dir, W_vortex, C,
                                         D_h, D_v, D_v_num, K_diss, PEs_dot, KE_div, D_bot])
        self.eb_chk += self.dt*np.abs([CONV_chk, gs(D_h_chk), gs(TKE_chk), gs(getattr(self, '_ke_hnum', 0.0)),
                                       gs(getattr(self, '_sw_col', 0.0)*At3[0]*mT[ks])])
        self.eb_fs += self.dt*abs(F_fs_T)
        self.eb_ss += self.dt*W_ss
        self.eb_n += 1
        if self.eb_n >= self.eb_steps:
            T = self.eb_n*self.dt; A = self.eb_area; r = self.rho_0
            a = self.eb_acc/T/A*r
            (KE_adv, KE_cor, KE_ph, KE_ps, W_tau, W_dir, W_vortex, C, D_h, D_v, D_v_num,
             K_diss, PEs_dot, KE_div, D_bot) = a[:15]
            dE_sp = (KE_adv + KE_cor + KE_ph + KE_ps + W_tau + W_dir + W_vortex + C + D_h + D_bot
                     + D_v + PEs_dot)
            E = KE + PEs; dE_fd = (E - self.eb_last)/T/A*r
            LEAK = dE_fd - dE_sp
            chk = self.eb_chk/T/A*r                      # mean |identity residual| [W/m2]
            trc = self.eb_trc/max(self.eb_n - 1, 1)/A     # mean |content residual| [K m/s, psu m/s]
            if self.my_pe == 0:
                print('KE-budget ocn day %8.2f [W/m2] W_tau=%8.4f W_dir=%8.4f C=%8.1e D_h=%8.4f D_bot=%8.1e '
                      'D_v=%8.4f (K_diss=%8.4f num=%8.4f) CONV=%8.4f KE_ps=%8.1e PEs_dot=%8.1e '
                      'KE_adv=%8.1e (adv-div=%8.1e) KE_cor=%8.1e W_vort=%8.1e%s | dE/dt=%8.4f sum=%8.4f LEAK=%8.4f'
                      % (n*self.dt/86400., W_tau, W_dir, C, D_h, D_bot, D_v, K_diss, D_v_num, KE_ph,
                         KE_ps, PEs_dot, KE_adv, KE_adv - KE_div, KE_cor, W_vortex,
                         (' W_ss=%8.1e' % (self.eb_ss/T/A*r)) if self.cfg.STOKES_SHEAR else '',
                         dE_fd, dE_sp, LEAK), flush=True)
                print('identities ocn day %8.2f: |KE_ph - CONV| %8.1e | |D_h - (stencil + fwd-Euler)| %8.1e '
                      '(fwd-Euler part %8.1e) | |TKE budget| %8.1e [W/m2] | heat %8.1e salt %8.1e '
                      '[flux units, vs |F_T| %8.1e; SW leak %8.1e; free surface %8.1e]'
                      % (n*self.dt/86400., chk[0], chk[1], chk[3], chk[2], trc[0], trc[1],
                         abs(FT)/A, chk[4]/r, self.eb_fs/T/A), flush=True)
                with open(efile, 'a') as fh:
                    fh.write(' '.join('%.10e' % x for x in
                             [n*self.dt/86400., KE_adv, KE_cor, KE_ph, KE_ps, W_tau, W_dir, W_vortex, C,
                              D_h, D_v, D_v_num, K_diss, PEs_dot, dE_sp, dE_fd, LEAK,
                              KE/A*r, PEs/A*r, KE_div, D_bot]) + '\n')
            self.eb_last = E; self.eb_acc[:] = 0.0; self.eb_chk[:] = 0.0; self.eb_trc[:] = 0.0
            self.eb_fs = 0.0; self.eb_ss = 0.0; self.eb_n = 0
        return

    def set_forcing_direct(self):
        """Plain forward increment u_L += Delta u_S over this ocean step (the
        d(u_S)/dt forcing). Called by the core BETWEEN the AB update and the
        pressure solve: it must bypass the AB weights (they would break the
        exact telescoping u = u_L - u_S) and precede the projection (the
        increment has divergence the surface pressure must answer).

        u_S is piecewise constant between couplings and the layout is
        CONCURRENT, so the increment is the difference of the RECEIVED
        timeline, u_S - u_S_old (kept by the coupler loop) -- one coupling
        interval behind the wave clock, like every other coupling field.
        Books the exact local work sum for exchange_work_C, which is the
        identity (|u_L + d|^2 - |u_L|^2)/(2 dt) per point."""
        if not hasattr(self, 'u_S_old'):
            self._C_step = 0.0
            return
        dus = self.u_S - self.u_S_old
        dvs = self.v_S - self.v_S_old
        incu = 0.5*(dus + self.rollx(dus, -1))*self.maskU
        incv = 0.5*(dvs + self.rolly(dvs, -1))*self.maskV
        h  = self.halo
        sl = slice(h, -h)
        self._C_step = float(OM.np.sum(
            ((self.u + 0.5*incu)*incu*self.area_u
           + (self.v + 0.5*incv)*incv*self.area_v)[sl, sl, sl]
            *self.dzt_z[sl])/self.dt)
        self.u = self.apply_bc(self.u + incu)
        self.v = self.apply_bc(self.v + incv)
        return

    # ----------------------------------------------------------------------
    def surface_forcing_fields(self, t):
        """The forcing at model time t on this PE, padded and masked:
        tau_a averaged to U/V points and divided by rho_0 [m^2/s^2], qnet and
        qsol [W/m^2, positive into the ocean], the restoring SST [deg C] and
        dQ/dT [W/m^2/K], P - E [m/s] and the SSS target [g/kg]. A field the
        source does not deliver comes back as None and its term is off.
        The U/V average is the one set_forcing applies to tau_in."""
        h  = self.halo
        sl = slice(h, -h)
        ks = -h-1
        F  = self._forcing.get(t)

        def pad(a):
            b = np.zeros((self.Ny, self.Nx))
            b[sl, sl] = a
            return self.apply_bc(OM.np.asarray(b))
        tx, ty = pad(F['taux']), pad(F['tauy'])
        mT  = self.maskT[ks]
        if getattr(self, '_relwind', False):
            tx, ty = self._relative_wind_stress(tx, ty, pad(F['u10']), pad(F['v10']), mT)
        tax = 0.5*(tx + self.rollx(tx, -1))*self.maskU[ks]/self.cfg.rho_0
        tay = 0.5*(ty + self.rolly(ty, -1))*self.maskV[ks]/self.cfg.rho_0
        g = lambda k: pad(F[k])*mT if k in F else None
        return (tax, tay, g('qnet'), g('qsol'), g('sst_star'), g('emp'),
                g('dqdt'), g('sss_star'))

    # ------------------------------------------------------------------
    # THE RELATIVE-WIND STRESS (cfg.STRESS_MODE = 'relwind'), the forced
    # counterpart of the coupled model's bulk formula (energy_cycle.tex Sec.
    # 6.2, 6.5). ERA5's stress is the IFS bulk formula on the ABSOLUTE wind,
    # tau_E = rho_a C_D |u10| u10 with the IFS's own stability and sea-state
    # dependent C_D. That drag coefficient is kept and only the wind is
    # replaced by the wind relative to the surface current,
    #     C_D rho_a = |tau_E| / |u10|^2 ,
    #     tau      = C_D rho_a |u10 - u_o| (u10 - u_o) ,
    # so tau = tau_E where u_o = 0 and no constant of our own enters. |u10| is
    # floored at STRESS_U10_MIN (1 m/s) in the denominator, where C_D is not
    # defined by the data anyway and tau_E is tiny. u_o is the Eulerian or
    # the Lagrangian velocity of the top cell (STRESS_UO, as the coupled
    # model's ATM_UO). Without this the interface can do work on the ocean:
    # with tau from the absolute wind, (u_a - u_o).tau < 0 where the current
    # runs with the wind and exceeds it (Sec. 6.5); with it the interface is
    # a sink, d_int = (u10 - u_o).tau >= 0, which is booked below and goes
    # into the means as d_int. The ocean forms tau on its T points every
    # step and, coupled, hands it to the waves (coupler_loop, tags 9, 10), so
    # the waves grow on the same stress.
    # ------------------------------------------------------------------
    def _relative_wind_stress(self, tx, ty, u10, v10, mT):
        """tx, ty: ERA5 stress [N/m2], u10, v10 [m/s], all padded T-point
        fields -> the relative-wind stress [N/m2] on the T points."""
        uo, vo = self.surface_velocity(self.cfg.STRESS_UO)
        du, dv = (u10 - uo)*mT, (v10 - vo)*mT
        s10 = OM.np.maximum(OM.np.hypot(u10, v10), float(self.cfg.STRESS_U10_MIN))
        # C_D rho_a [kg/m3], capped: the ERA5 stress is the mean over the hour
        # and the wind the analysis at the stamp, so where the wind turns
        # within the hour |u10| can be near zero while |tau| is not, and the
        # quotient would give C_D of order 0.1 and a spurious stress at a calm
        # spot. The IFS itself does not exceed about 2.5e-3.
        cdr = OM.np.minimum(OM.np.hypot(tx, ty)/(s10*s10),
                            float(self.cfg.STRESS_CD_MAX)*float(self.cfg.rho_air))
        sd  = OM.np.hypot(du, dv)
        trx, try_ = cdr*sd*du, cdr*sd*dv
        self._d_int = (du*trx + dv*try_)*mT                   # interface sink [W/m2], >= 0
        self._cd_rho = cdr*mT
        self._u10, self._v10 = u10*mT, v10*mT
        h = self.halo; hb = slice(h, -h)
        self._tau_rel_T = (np.asarray(trx)[hb, hb], np.asarray(try_)[hb, hb])   # for the waves
        # halo exchange: surface_velocity's np.roll wraps at the first row and
        # column, so the halo of trx/try_ is wrong there; the U/V average
        # below and the Craig-Banner flux (U/V -> T) read it at the first
        # interior row/column of every PE block (test/coupled, Oct 20 2026)
        return self.apply_bc(trx), self.apply_bc(try_)

    def set_forcing(self, tau):
        h  = self.halo
        ks = -h-1                                    # surface (top) level index
        sl_srf = (tau, ks, slice(None), slice(None))
        sl_all = (tau, slice(None), slice(None), slice(None))

        t = self._gitt*self.dt                       # model time of this step
        (tax, tay, qqnet, qsol, tstar, fw,
         qqnec, sstar) = self.surface_forcing_fields(t)

        # --- THE ICE MASK ---------------------------------------------------
        # A cell at or below the freezing point is taken to be ice covered.
        # The heat and salt fluxes have always been switched off there (see
        # the buoyancy block below); cfg.ICE_TAU_FRAC switches the MOMENTUM
        # off as well, which matters because this configuration has no sea
        # ice: without it the full ERA5 stress is applied where in reality
        # the ice pack carries most of it and delivers a small, rough
        # residual to the water. At 4 degrees that asymmetry is invisible; on
        # 2 km cells over an Antarctic shelf it drives a cross-slope flow
        # whose w = u.grad(H) reached 2 cm/s in the icon01 cold start.
        #     1.0   the momentum is not masked (the behaviour of every run
        #           before this switch existed, and the default)
        #     0.0   no stress at all under ice, the free-drift upper bound
        #     ~0.3  a crude allowance for the stress an ice pack does pass on
        # The test is on the FREEZING POINT ALONE, not on the sign of the
        # heat flux the buoyancy block adds: there is no such thing as a
        # stress that only acts when the ocean is warming.
        # The factor is formed on T points and averaged to U and V exactly as
        # the stress itself is, so a cell at the ice edge sees the mean of
        # its two neighbours rather than a step.
        ice_tau = float(self.cfg.ICE_TAU_FRAC)
        if ice_tau < 1.0:
            frz = float(self.cfg.ICE_T_FREEZE)
            fT  = OM.np.where(self.temp[ks]*self.maskT[ks] <= frz, ice_tau, 1.0)
            f_u = 0.5*(fT + self.rollx(fT, -1))
            f_v = 0.5*(fT + self.rolly(fT, -1))
            self._ice_mask = OM.np.where(self.temp[ks]*self.maskT[ks] <= frz,
                                         1.0, 0.0)*self.maskT[ks]
            tax, tay = tax*f_u, tay*f_v
        else:
            f_u = f_v = 1.0

        # --- surface momentum -------------------------------------------------
        # coupled: the stress the WAVES release to the ocean (taux/tauy are
        # delivered by the coupler). Ocean alone: tau_a itself; stored in
        # taux/tauy so the TKE surface flux, the KE budget and the restart
        # see the stress that was applied -- which is why the ice factor is
        # applied BEFORE the store: under ice the TKE surface flux, the KE
        # budget and the diagnostics must all see the reduced stress too.
        if not self.coupled:
            self.taux, self.tauy = tax, tay
        elif ice_tau < 1.0:
            self.taux, self.tauy = self.taux*f_u, self.tauy*f_v
        if self.use_stress_work:
            self.du = OM.add_to_array(self.du, sl_all,
                0.5*(self.tau_work_x + self.rollx(self.tau_work_x, -1))
                * (f_u if ice_tau == 1.0 else f_u[None, :, :])*self.maskU)
            self.dv = OM.add_to_array(self.dv, sl_all,
                0.5*(self.tau_work_y + self.rolly(self.tau_work_y, -1))
                * (f_v if ice_tau == 1.0 else f_v[None, :, :])*self.maskV)
        else:
            self.du = OM.add_to_array(self.du, sl_srf,
                self.taux/self._dz_surf*self.maskU[ks])
            self.dv = OM.add_to_array(self.dv, sl_srf,
                self.tauy/self._dz_surf*self.maskV[ks])

        # --- OPTIONAL direct branch tau_a - tau_in --------------------------
        # The wave model delivers only the momentum that passed through the
        # wave field. Switching this on hands the remainder of the prescribed
        # stress to the ocean directly, so the total surface momentum flux is
        # tau_a again (never letting the remainder go negative, which happens
        # where the sea is young and tau_in exceeds tau_a).
        # (tax, tay are already the ice-masked stress, so the branch carries
        # the same factor and tau_o + branch is still the applied tau_a)
        if self.coupled and self.cfg.direct_stress_branch and hasattr(self, 'tau_in_x'):
            ta  = OM.np.hypot(tax, tay)
            tin = OM.np.hypot(0.5*(self.tau_in_x + self.rollx(self.tau_in_x, -1)),
                              0.5*(self.tau_in_y + self.rolly(self.tau_in_y, -1)))
            frac = OM.np.maximum(0.0, 1.0 - tin/OM.np.maximum(ta, 1e-20))
            self.du = OM.add_to_array(self.du, sl_srf,
                tax*frac/self._dz_surf*self.maskU[ks])
            self.dv = OM.add_to_array(self.dv, sl_srf,
                tay*frac/self._dz_surf*self.maskV[ks])
            # kept for stress_work_diag / stress_work_eul_diag: the surface
            # momentum forcing of u_L is tau_o PLUS this branch
            self._tau_dir_x, self._tau_dir_y = tax*frac, tay*frac

        # --- vortex force, u_L formulation (Eq. 5), pseudo-spherical --------
        # The model velocity is the Lagrangian mean u_L; its flux-form
        # advection provides u_L.grad(u_L), so the remaining piece of the
        # rotational term is -u_L x (nabla x u_S). HYDROSTATIC truncation:
        # only the vertical-vorticity part is kept,
        #     F_x = -v_L zeta_S ,  F_y = +u_L zeta_S ,
        #     zeta_S = (dx(v_S) - dy(cost u_S))/(cosu) ,
        # which is exactly work-free on u_L. The w_L dz(u_S) terms that
        # complete the curl are omitted BY DEFAULT (the paper's App. C):
        # without a vertical force component they would do uncancelled net
        # work. cfg.STOKES_SHEAR adds them together with the vertical
        # component, carried by the hydrostatic relation (stokes_shear_force
        # below). The Coriolis-Stokes block is GONE (the core Coriolis
        # acts on u_L, which IS f zhat x u_L), and tracers are advected by
        # u_L with its own continuity w.
        if self.coupled and hasattr(self, 'u_S'):
            uSU = 0.5*(self.u_S + self.rollx(self.u_S, -1))*self.maskU
            vSV = 0.5*(self.v_S + self.rolly(self.v_S, -1))*self.maskV
            zeta_S = ((self.rollx(vSV, -1) - vSV)/(self.cosu_y*self.dxu_x)
                    - (self.rolly(uSU*self.cost_y, -1) - uSU*self.cost_y)
                      /(self.cosu_y*self.dyu_y))
            zeta_S = self.apply_bc(zeta_S)
            # ENERGY-CONSERVING corner pairing (Sadourny type) with the AREA
            # weights of the spherical grid: form the products at the zeta
            # corners first, weighted by area_z, then average to U/V and
            # divide by area_u/area_v. The discrete work
            #   sum (u F_x area_u + v F_y area_v) dz
            # then telescopes to zero IDENTICALLY, walls and land masks
            # included -- vortex_work() is a pure round-off monitor of it.
            # (Dropping the area weights, as on the uniform acc1 grid, would
            # leave a spurious O(dy/R) source here.)
            P = self.area_z*zeta_S*0.5*(self.v + self.rollx(self.v, -1))
            Q = self.area_z*zeta_S*0.5*(self.u + self.rolly(self.u, -1))
            Fu = -0.5*(P + self.rolly(P, 1))/self.area_u*self.maskU
            Fv =  0.5*(Q + self.rollx(Q, 1))/self.area_v*self.maskV
            # book the work HERE, against the velocities the pairing used
            h_ = self.halo
            si = slice(h_, -h_)
            self._vortex_step = float(OM.np.sum(
                (self.u*Fu*self.area_u + self.v*Fv*self.area_v)[si, si, si]
                *self.dzt_z[si]))
            self.du = OM.add_to_array(self.du, sl_all, Fu)
            self.dv = OM.add_to_array(self.dv, sl_all, Fv)

            # --- OPTIONAL full force: the Stokes shear force as well --------
            # cfg.STOKES_SHEAR = 1 adds the w dz(u_S) terms omitted above AND
            # the vertical component -u dz(u_S) - v dz(v_S) of the force,
            # the latter through the hydrostatic relation as the extra
            # pressure p_SS (stokes_shear_force). The two do no net work on
            # u_L, exactly, so W_vortex stays the round-off monitor of the
            # whole force; W_ss in the log is the size of the cancelling pair.
            if self.cfg.STOKES_SHEAR:
                Fu, Fv, w_h, w_p = self.stokes_shear_force()
                self._ss_step = w_h
                self._vortex_step += w_h + w_p
                self.du = OM.add_to_array(self.du, sl_all, Fu)
                self.dv = OM.add_to_array(self.dv, sl_all, Fv)

        # --- buoyancy forcing: structure of global4_pe.py, ERA5 fluxes -------
        # Q = Q_ERA5 + dQ/dT (SST_ERA5 - SST): the linearisation of the ERA5
        # bulk fluxes around the SST they were computed with
        # (with an ERA5 source; a source without dqdt/sst_star has no
        # restoring, one without sss_star no salinity restoring)
        if qqnec is not None and tstar is not None:
            f_t = ((qqnet + qqnec*(tstar - self.temp[ks]))
                   * self.maskT[ks]/self.cfg.cp_0/self.cfg.rho_0)
        else:
            f_t = qqnet*self.maskT[ks]/self.cfg.cp_0/self.cfg.rho_0
        # restoring as a piston velocity, plus the virtual salt flux of the
        # freshwater flux, S (E - P), with the LOCAL surface salinity
        piston = self.cfg.SSS_PISTON if self.cfg.SSS_PISTON > 0.0 else self._dz_surf/self.cfg.t_rest
        if sstar is not None:
            f_s = piston*(sstar - self.salt[ks])*self.maskT[ks]
        else:
            f_s = 0.0*self.salt[ks]
        if fw is not None:
            f_s = f_s - self.salt[ks]*fw*self.maskT[ks]

        sst = self.temp[ks]*self.maskT[ks]
        ice = OM.np.logical_and(sst <= -1.8, f_t <= 0.0)
        f_t = OM.np.where(ice, 0.0, f_t)
        f_s = OM.np.where(ice, 0.0, f_s)
        # surface tracer FLUXES for the TKE closure's surface buoyancy-flux
        # term (pyOM2 forc_temp_surface / forc_salt_surface, K m/s, psu m/s)
        self.forc_temp_surface, self.forc_salt_surface = f_t, f_s
        ice_f = OM.np.where(ice, 0.0, 1.0)

        self.dtemp = OM.add_to_array(self.dtemp, sl_srf, f_t/self._dz_surf)
        self.dsalt = OM.add_to_array(self.dsalt, sl_srf, f_s/self._dz_surf)

        # penetrating shortwave: 3-D temperature source, zeroed under ice.
        # Q_ERA5 already contains ssr at the surface, the source moves the
        # penetrating part down (its column integral is zero)
        if qsol is None:                  # the source has no solar part
            self._sw_col = 0.0*f_t
            return
        src = (qsol[None, :, :]*self._divpen
               * ice_f[None, :, :]*self.maskT/self.cfg.cp_0/self.cfg.rho_0)
        self.dtemp = OM.add_to_array(self.dtemp, sl_all, src)
        # column integral of the applied shortwave source [K m/s]: zero over a
        # full column (pen(0) = pen(nz) = 0), NONZERO where the column is
        # shallower than the light penetration -- the light absorbed below the
        # bottom is lost (pyOM2 setup1 does the same); booked for the heat
        # conservation identity and printed as "SW leak". With EXP_SW_BOTTOM
        # the remainder goes into the deepest wet cell and this is round-off
        self._sw_col = OM.np.sum(src*self.dzt_z, axis=0)
        return
