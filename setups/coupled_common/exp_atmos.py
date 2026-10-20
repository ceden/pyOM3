"""
THE ATMOSPHERE of the coupled model: the Held-Suarez atmosphere of
energy_cycle.tex (pe_realistic upside down in pressure coordinates) with the
surface exchange of Sec. 6 -- the relative wind in the bulk formulae, the
K-profile boundary layer, the dissipation heating and the interface
dissipation q_int -- as a mixin on a pe_realistic snapshot class, the way
ocean_common and wave_common are. Everything configuration specific comes
from the experiment module through self.cfg: the grid, the HS constants,
the vertical grid, the switches (Cfg.DEFAULTS ATM_*).

The atmosphere runs on ITS OWN GRID, the ocean grid coarsened by
cfg.atmos_coarsen (an integer, a multiple of wave_coarsen), and exchanges
with the OCEAN ONLY (coupler_loop.run_atmos): it hands down tau_a and Q_s,
copied to every ocean cell of its group, and receives area sums over the wet
ocean cells of its group -- SST and u_o as means, the stress work w_o, the
wave input p_in and the applied heat as sums per atmosphere cell area. The
land fraction of a cell takes the stress like a rough surface and no heat.

What the experiment module must define for this class:
    p_bot, alpha_s, hs_theta_eq(p, phi), k_a, k_s, k_f, sigma_b, p_ref,
    R_gas, cp_air, kappa, rho_air, C_H, z0_bulk, GROUND_FIX, atmos_dt,
    atmos_snapint, energy_file
Physical constants that are not in the module fall back to the values below.
"""
import numpy as np
import pyOM3 as OM
from functools import partial
from mpi4py import MPI

jnp = OM.np
g0 = 9.81
day = 86400.0
kappa_v = 0.4                             # von Karman
Ri_c    = 0.5                             # bulk Richardson number of the boundary layer top


def c_drag(u10):
    """Large and Pond (1981) neutral drag coefficient."""
    return np.where(u10 < 11.0, 1.2e-3, (0.49 + 0.065*u10)*1e-3)


ENERGY_COLUMNS = ['day', 'KE_adv', 'KE_cor', 'KE_ph', 'KE_ps', 'D_ray', 'D_bih', 'H_hadv', 'H_vadv',
                  'H_fix', 'Q', 'PEs', 'CONV', 'CONV_eos', 'CONV_gnd', 'LEAK_sp', 'dE_sp', 'dE_fd',
                  'LEAK_dt', 'W_tau', 'Q_s', 'D_bl', 'D_vnum', 'H_vdif', 'H_diss', 'Q_int', 'E_K', 'E_H', 'E_s']
ENERGY_LEGEND = """
 Atmosphere energy budget (hs_energy.txt, interval means in W/m2, see energy_cycle.tex):
   KE_adv/KE_cor/KE_ph/KE_ps  KE tendencies: momentum advection (ground-cell leak), Coriolis (0),
                              hydrostatic work, surface-pressure work;  D_ray, D_bih friction
   H_hadv/H_vadv/H_fix/Q      enthalpy: horizontal (0) and vertical advection, GROUND_FIX leak, HS heating
   PEs  external-mode energy tendency (= -KE_ps)
   CONV = KE_ph+H_vadv = CONV_eos + CONV_gnd  (centered advection: no limiter part)
   LEAK_sp total spatial leak, dE_sp/dE_fd tendency sum vs finite difference, LEAK_dt time stepping
   W_tau = -sum u_a.tau  KE lost to the surface stress;  Q_s = sum of the sensible heat flux
           received from the ocean.
   D_bl  KE change by the boundary layer diffusion, D_vnum = D_bl + int eps its part the pairing
         does not return (<= 0);  H_vdif enthalpy change by the vertical diffusion of theta;
   H_diss heating returned to theta (int eps - D_bih, or 0);  Q_int interface dissipation
         q_int = u_a.tau - w_o - p_in applied to the ground cell.  E_K, E_H, E_s in J/m2."""


class atmosphere_common:
    """HS94 atmosphere with the surface exchange of the coupled setup, the
    relative wind, the K-profile boundary layer and the dissipation heating.
    Mixin: class atmosphere(atmosphere_common, snap_nc4) in the experiment module."""

    # switches and constants of the experiment module, with defaults
    def _c(self, name, default=None):
        m = self.cfg.__dict__["_m"]
        if hasattr(m, name): return getattr(m, name)
        if default is None: raise AttributeError("the experiment module does not define %s" % name)
        return default

    def set_mean_pressure(self):
        super().set_mean_pressure()
        self.p0 = jnp.maximum(self.p0, 1.0)

    def set_coriolis(self, f0=None):
        f = 2*self.cfg.omega_e*jnp.sin(jnp.deg2rad(self.yt))[:, None]*jnp.ones_like(self.coriolis_t)
        self.coriolis_t = OM.modify_array(self.coriolis_t, (slice(None), slice(None)), f,
                                          out_sharding=self.sharding_2D)

    def prepare_hs(self):
        """HS coefficient fields and the coupling fields on the local grid. The
        ocean fraction f_wet of every cell arrives with the first exchange
        (set_ocean_fraction); until then the whole cell is ocean."""
        sigma = self.zt/self._c('p_bot')
        phi   = jnp.deg2rad(self.yt)
        wgt   = jnp.maximum(0., (sigma - self._c('sigma_b'))/(1. - self._c('sigma_b')))
        self.hs_kv   = (self._c('k_f')*wgt)[:, None, None]
        if self.cfg.ATM_BL == 'kprofile':
            self.hs_kv = self.hs_kv*0.0                     # Rayleigh friction off
        self.hs_kT   = self._c('k_a') + (self._c('k_s') - self._c('k_a'))*wgt[:, None, None]*jnp.cos(phi)[None, :, None]**4
        self.hs_theq = jnp.asarray(self._c('hs_theta_eq')(np.asarray(self.zt)[:, None, None],
                                               np.deg2rad(np.asarray(self.yt))[None, :, None]))
        h = self.halo; kg = -h-1
        # coupling fields on the local grid (T points)
        z2 = lambda: np.zeros((self.Ny, self.Nx))
        self.f_wet = np.ones((self.Ny, self.Nx))*np.asarray(self.maskT)[kg]   # ocean fraction of the cell
        self.q_ocn = z2()                                           # heat the ocean applied, W/m2 of cell area
        self.f_ice = z2()                                           # fraction of the cell under the freezing floor
        self.sst   = np.full((self.Ny, self.Nx), 288.0)             # K
        self.tau_ax, self.tau_ay = z2(), z2()                       # Pa
        self.q_s   = z2()                                           # W/m2, ocean -> air
        self.u10, self.v10, self.t10 = z2(), z2(), z2()             # RELATIVE wind at 10 m, T at 10 m
        self.u_o, self.v_o = z2(), z2()                             # ocean surface velocity received
        self.w_o   = z2()                                           # stress work on the ocean, W/m2
        self.p_in  = z2()                                           # wind input to the waves, W/m2
        self.w_a   = z2()                                           # -u_a.tau, W/m2
        self.q_int = z2()                                           # interface dissipation, W/m2
        self.Pi_g = float((max(float(self.zt[kg]), 1.0)/self._c('p_ref'))**self._c('kappa', 0.287))
        self.dp_g = float(self.dzt[kg])
        # ground-cell centre height above the ground, for the log law
        self.z_L  = 0.5*self.dp_g*self._c('alpha_s')/g0
        self.log_fac = np.log(10.0/self._c('z0_bulk', 1e-4))/np.log(self.z_L/self._c('z0_bulk', 1e-4))
        # booked per-step scalars (W) and fields for the budget and the heating
        self._wa_step = self._qint_step = self._eps_step = self._dbl_step = 0.0
        self._bih_step = self._hvdif_step = self._heat_step = 0.0
        self._eps_T = OM.np.zeros((self.Nz, self.Ny, self.Nx), np.float64)
        self._bih_T = OM.np.zeros((self.Nz, self.Ny, self.Nx), np.float64)
        self._Kp    = OM.np.zeros((self.Nz, self.Ny, self.Nx), np.float64)
        self.bl_h   = z2()                                          # boundary layer height, m
        if self.my_pe == 0:
            print('CONFIG atmosphere: boundary layer %s | u_o %s x %.2f | dissipation heating %s | '
                  'biharmonic heating %s | %d levels, ground cell %.0f hPa (z_L = %.1f m), dt %g s'
                  % (self.cfg.ATM_BL, self.cfg.ATM_UO, self.cfg.ATM_RWIND, bool(self.cfg.ATM_DISS_HEAT),
                     bool(self.cfg.ATM_BIH_HEAT and self.cfg.ATM_DISS_HEAT), self.Nz - 2*h,
                     self.dp_g/100.0, self.z_L, self.dt), flush=True)

    def set_ocean_fraction(self, fwet_block):
        """Interior block (ny_loc, nx_loc) of the ocean fraction of the cells,
        sum(area x mask of the wet ocean cells of the group) / cell area, from
        the static weights the ocean sends once."""
        h = self.halo; hb = slice(h, -h)
        f = np.zeros((self.Ny, self.Nx)); f[hb, hb] = np.clip(fwet_block, 0.0, 1.0)
        self.f_wet = np.asarray(self.apply_bc(OM.np.asarray(f)))*np.asarray(self.maskT)[-h-1]
        if self.my_pe == 0:
            print('CONFIG atmosphere: ocean fraction of the cells %.3f .. %.3f (interior of rank 0)'
                  % (float(f[hb, hb].min()), float(f[hb, hb].max())), flush=True)

    # --- averaging helpers (metric weights, as pe_realistic/tke.py) ------------
    def _u2t(self, p): return 0.5*(p*self.area_u + self.rollx(p*self.area_u, 1))/self.area_t
    def _v2t(self, p): return 0.5*(p*self.area_v + self.rolly(p*self.area_v, 1))/self.area_t

    # --- bulk fluxes from the current ground-cell state, the received SST and u_o
    def bulk_fluxes(self):
        h = self.halo; kg = -h-1
        u, v, th = (np.asarray(x) for x in (self.u, self.v, self.temp))
        uT = 0.5*(u[kg] + np.roll(u[kg], 1, axis=1))                 # to T points
        vT = 0.5*(v[kg] + np.roll(v[kg], 1, axis=0))
        T_L  = th[kg]*self.Pi_g                                        # K
        rho_a = max(float(self.zt[kg]), 1.0)/(self._c('R_gas', 287.0)*np.maximum(T_L, 150.0))
        # RELATIVE wind: the log law is a statement about u - u_o (tex Eq. eq_loglaw)
        du = uT - self.cfg.ATM_RWIND*self.u_o; dv = vT - self.cfg.ATM_RWIND*self.v_o
        self.u10, self.v10 = du*self.log_fac, dv*self.log_fac
        self.t10 = T_L + 0.0098*(self.z_L - 10.0)
        U10 = np.hypot(self.u10, self.v10)
        cd  = c_drag(U10)
        self.tau_ax = rho_a*cd*U10*self.u10                          # Pa, T points
        self.tau_ay = rho_a*cd*U10*self.v10
        # sensible heat over the OCEAN FRACTION of the cell only (the land takes none):
        # q_s is per unit CELL area, what the ground cell loses; the ocean receives
        # q_s/f_wet per unit wet area, so the two sides exchange the same heat
        dT = np.where(self.f_wet > 0.0, self.sst - self.t10, 0.0)
        self.q_s = rho_a*self._c('cp_air', 1000.0)*self._c('C_H', 1e-3)*U10*dT*self.f_wet   # W/m2, positive ocean->air
        # stress work on the atmosphere, w_a = -u_a.tau, U/V points -> T points with the
        # metric weights so that sum w_a A_t = -(sum u txU A_u + sum v tyV A_v) exactly
        mU, mV = np.asarray(self.maskU)[kg], np.asarray(self.maskV)[kg]
        Au, Av, At = (np.asarray(x)[0] for x in (self.area_u, self.area_v, self.area_t))
        txU = 0.5*(self.tau_ax + np.roll(self.tau_ax, -1, axis=1))
        tyV = 0.5*(self.tau_ay + np.roll(self.tau_ay, -1, axis=0))
        pu = u[kg]*txU*mU*Au; pv = v[kg]*tyV*mV*Av
        self.w_a = -(0.5*(pu + np.roll(pu, 1, axis=1)) + 0.5*(pv + np.roll(pv, 1, axis=0)))/At
        # interface dissipation (tex Eq. eq_qint), with the w_o and p_in of the last interval
        self.q_int = (-self.w_a - self.w_o - self.p_in)*np.asarray(self.maskT)[kg]        # u_a.tau - w_o - p_in >= 0 (mostly)
        return

    @partial(OM.jaxjit, static_argnums=0)
    def tracer_advection(self, var, dvar, u, v, w, maskU, maskV, maskW, maskT, tau):
        dvar = super().tracer_advection(var, dvar, u, v, w, maskU, maskV, maskW, maskT, tau)
        if not self._c('GROUND_FIX', True): return dvar
        k = -self.halo-1
        return OM.add_to_array(dvar, (tau, k, slice(None), slice(None)),
                               -var[k]*w[k]/self.dzt[k]*maskT[k])

    def set_forcing(self, tau):
        """HS forcing plus the surface exchange in the ground cell."""
        a = slice(None); kg = -self.halo-1
        self.du = OM.add_to_array(self.du, (tau, a, a, a), -self.hs_kv*self.u*self.maskU)
        self.dv = OM.add_to_array(self.dv, (tau, a, a, a), -self.hs_kv*self.v*self.maskV)
        self.dtemp = OM.add_to_array(self.dtemp, (tau, a, a, a),
                                     -self.hs_kT*(self.temp - self.hs_theq)*self.maskT)
        # stress (T points) to U/V points; the atmosphere loses tau_a
        txU = 0.5*(self.tau_ax + jnp.roll(self.tau_ax, -1, axis=1))
        tyV = 0.5*(self.tau_ay + jnp.roll(self.tau_ay, -1, axis=0))
        fac = g0/self.dp_g
        self.du = OM.add_to_array(self.du, (tau, kg, a, a), -txU*fac*self.maskU[kg])
        self.dv = OM.add_to_array(self.dv, (tau, kg, a, a), -tyV*fac*self.maskV[kg])
        # sensible heat and, when returned, the interface dissipation heat the ground cell
        heat = self.q_s + (self.q_int if self.cfg.ATM_DISS_HEAT else 0.0)
        self.dtemp = OM.add_to_array(self.dtemp, (tau, kg, a, a),
                                     heat*fac/(self._c('cp_air', 1000.0)*self.Pi_g)*self.maskT[kg])
        h = self.halo; I2 = (slice(h, -h), slice(h, -h))
        At = np.asarray(self.area_t)[0]; mT = np.asarray(self.maskT)[kg]
        self._wa_step   = float(np.sum((self.w_a*mT*At)[I2]))
        self._qint_step = float(np.sum((self.q_int*mT*At)[I2])) if self.cfg.ATM_DISS_HEAT else 0.0
        self._qint_diag = float(np.sum((self.q_int*mT*At)[I2]))
        self._qint_neg  = float(np.sum((np.where(self.q_int < 0.0, 1.0, 0.0)*mT*At)[I2]))
        return

    # --- K-profile boundary layer (tex Sec. 6.6) --------------------------------
    def bl_diffusivity(self):
        """K_p = (rho g)^2 K at the W faces (T points), K = self._c('kappa', 0.287) u* z (1 - z/h)^2
        below the boundary layer height h from the bulk Richardson number Ri_c."""
        h = self.halo; kg = -h-1
        u, v, th = (np.asarray(x) for x in (self.u, self.v, self.temp))
        zT = np.asarray(self.p_h)/g0                                   # height of the cell centres, m
        uT = 0.5*(u + np.roll(u, 1, axis=2)); vT = 0.5*(v + np.roll(v, 1, axis=1))
        th_a = th[kg]
        du2 = np.maximum((uT - self.u_o[None])**2 + (vT - self.v_o[None])**2, 1e-2)
        Ri = g0*(th - th_a[None])*zT/(np.maximum(th_a, 100.0)[None]*du2)
        # first level above the ground with Ri >= Ri_c, interpolated; ground first
        R  = Ri[h:kg][::-1]; Z = zT[h:kg][::-1]                          # index 0: first level above the ground
        Rg = np.concatenate([np.zeros((1,) + R.shape[1:]), R], axis=0)   # Ri = 0 at the ground cell
        Zg = np.concatenate([zT[kg][None], Z], axis=0)
        above = Rg >= Ri_c
        k1 = np.argmax(above, axis=0)                                    # first True; 0 if none
        found = above.any(axis=0)
        k1 = np.where(found, k1, Rg.shape[0] - 1)
        k0 = np.maximum(k1 - 1, 0)
        take = lambda A, k: np.take_along_axis(A, k[None], axis=0)[0]
        R1, R0, Z1, Z0 = take(Rg, k1), take(Rg, k0), take(Zg, k1), take(Zg, k0)
        frac = np.clip((Ri_c - R0)/np.maximum(R1 - R0, 1e-12), 0.0, 1.0)
        hbl = np.where(found, Z0 + frac*(Z1 - Z0), take(Zg, k1))
        hbl = np.maximum(hbl, np.maximum(1.5*Zg[1], 1.0))                # at least the first face (halo: 1 m)
        self.bl_h = hbl
        ustar = np.sqrt(np.hypot(self.tau_ax, self.tau_ay)/self._c('rho_air', 1.225))
        # faces: face k lies between cell k and cell k+1 (below), height 0.5 (z_k + z_k+1)
        zf = 0.5*(zT + np.roll(zT, -1, axis=0))
        K  = kappa_v*ustar[None]*zf*np.maximum(1.0 - zf/hbl[None], 0.0)**2
        K  = np.where(zf < hbl[None], K, 0.0)
        pT = np.maximum(np.asarray(self.zt), 1.0)[:, None, None]
        TT = th*(pT/self._c('p_ref'))**self._c('kappa', 0.287)
        pf = 0.5*(pT + np.roll(pT, -1, axis=0)); Tf = 0.5*(TT + np.roll(TT, -1, axis=0))
        rho_f = pf/(self._c('R_gas', 287.0)*np.maximum(Tf, 150.0))
        Kp = (rho_f*g0)**2*K*np.asarray(self.maskW)
        Kp[kg:] = 0.0; Kp[:h] = 0.0                                      # ground face closed, halo
        return OM.np.asarray(Kp)

    def vertical_friction(self):
        if self.cfg.ATM_BL != 'kprofile':
            return super().vertical_friction()
        self._Kp = self.bl_diffusivity()
        kmu = 0.5*(self._Kp + self.rollx(self._Kp, -1))*self.maskU*self.rollz(self.maskU, -1)
        kmv = 0.5*(self._Kp + self.rolly(self._Kp, -1))*self.maskV*self.rollz(self.maskV, -1)
        u_new = self.implicit_vdiff(self.u, kmu, self.maskU)
        v_new = self.implicit_vdiff(self.v, kmv, self.maskV)
        # exact pairing (tex Eq. eq_vfric_energy): flux x shear of the UPDATED velocities at
        # the faces, U/V -> T with the metric weights, then half of each face to its two cells
        shu = (self.rollz(u_new, -1) - u_new)/self.dzw_z
        shv = (self.rollz(v_new, -1) - v_new)/self.dzw_z
        F = (self._u2t(kmu*shu*shu) + self._v2t(kmv*shv*shv))*self.dzw_z     # per face, x dp_w
        eps = 0.5*(F + self.rollz(F, 1))/self.dzt_z*self.maskT               # per unit mass, T cells
        self._eps_T = eps
        h = self.halo; I = (slice(h, -h),)*3
        mass_t = np.asarray(self.maskT*self.area_t*self.dzt_z)/g0
        mass_u = np.asarray(self.maskU*self.area_u*self.dzt_z)/g0
        mass_v = np.asarray(self.maskV*self.area_v*self.dzt_z)/g0
        self._eps_step = float(np.sum((np.asarray(eps)*mass_t)[I]))
        dku = 0.5*(np.asarray(u_new)**2 - np.asarray(self.u)**2)/self.dt
        dkv = 0.5*(np.asarray(v_new)**2 - np.asarray(self.v)**2)/self.dt
        self._dbl_step = float(np.sum((dku*mass_u)[I]) + np.sum((dkv*mass_v)[I]))
        self.u, self.v = self.apply_bc(u_new), self.apply_bc(v_new)
        return

    def biharmonic_momentum(self, u, v, Ahbi, maskU, maskV):
        u2, v2 = super().biharmonic_momentum(u, v, Ahbi, maskU, maskV)
        dku = 0.5*(u2*u2 - u*u)/self.dt*maskU
        dkv = 0.5*(v2*v2 - v*v)/self.dt*maskV
        self._bih_T = -(self._u2t(dku) + self._v2t(dkv))*self.maskT           # heating per mass, T cells
        h = self.halo; I = (slice(h, -h),)*3
        mass_u = np.asarray(self.maskU*self.area_u*self.dzt_z)/g0
        mass_v = np.asarray(self.maskV*self.area_v*self.dzt_z)/g0
        self._bih_step = float(np.sum((np.asarray(dku)*mass_u)[I]) + np.sum((np.asarray(dkv)*mass_v)[I]))
        return u2, v2

    def vertical_mixing(self):
        """Vertical diffusion of theta (convective adjustment as before, plus the
        boundary layer K_p with Prandtl number 1), then the dissipation heating."""
        scheme, conv, tkeo = self._vmix_options()
        if self.cfg.ATM_BL != 'kprofile':
            return super().vertical_mixing()
        # pe_realistic's default is the explicit scheme with kappa_T = 0, i.e. no vertical
        # mixing of theta at all; the boundary layer diffusion is always implicit
        if scheme != 'explicit' and conv:
            kappaH = self.convective_kappa(self.temp, self.salt, self.rho, self.maskW) + self._Kp
        else:
            kappaH = getattr(self, 'kappa_T', 0.0)*self.maskW + self._Kp
        h = self.halo; I = (slice(h, -h),)*3
        Pi3 = OM.np.asarray(((np.maximum(np.asarray(self.zt), 1.0)/self._c('p_ref'))**self._c('kappa', 0.287))[:, None, None])
        mass_t = np.asarray(self.maskT*self.area_t*self.dzt_z)/g0
        th_old = self.temp
        self.temp = self.implicit_vdiff(self.temp, kappaH, self.maskT)
        self._hvdif_step = self._c('cp_air', 1000.0)*float(np.sum((np.asarray((self.temp - th_old)*Pi3)*mass_t)[I]))/self.dt
        q = 0.0
        if self.cfg.ATM_DISS_HEAT:
            q = self._eps_T if self.cfg.ATM_BL == 'kprofile' else 0.0
            if self.cfg.ATM_BIH_HEAT and hasattr(self, 'Ahbi'): q = q + self._bih_T
        if self.cfg.ATM_DISS_HEAT and not isinstance(q, float):
            self.temp = self.apply_bc(self.temp + self.dt*q/(self._c('cp_air', 1000.0)*Pi3)*self.maskT)
            self._heat_step = float(np.sum((np.asarray(q)*mass_t)[I]))
        else:
            self._heat_step = 0.0
        return

    # --- energy budget (hs_atmos.py, reduced to the terms of the centered scheme) --
    def energy_budget(self, n, tau, efile=None):
        efile = efile or getattr(self, 'energy_file', 'atmos_energy.txt')
        h = self.halo; kg = -h-1
        I  = (slice(h,-h), slice(h,-h), slice(h,-h)); I2 = (slice(h,-h), slice(h,-h))
        # NOT OM.global_sum: that is hardwired to COMM_WORLD and would block on the other component
        comm = self.comm
        gs = (lambda x: float(comm.allreduce(float(np.sum(x)), op=MPI.SUM))) if (comm is not None and comm.Get_size() > 1) \
             else (lambda x: float(np.sum(x)))
        rx = lambda a, s: np.roll(a, s, axis=-1); ry = lambda a, s: np.roll(a, s, axis=-2)
        rz = lambda a, s: np.roll(a, s, axis=0)
        NB = 23
        if not hasattr(self, 'eb_area'):
            dp = np.asarray(self.dzt)[:, None, None]
            At = np.asarray(self.area_t); Au = np.asarray(self.area_u); Av = np.asarray(self.area_v)
            self.eb_mt = (np.asarray(self.maskT)*At*dp/g0)[I]
            self.eb_mu = (np.asarray(self.maskU)*Au*dp/g0)[I]
            self.eb_mv = (np.asarray(self.maskV)*Av*dp/g0)[I]
            self.eb_A2 = (np.asarray(self.maskT)[kg]*At[0])[I2]/g0
            self.eb_area = gs(np.asarray(self.maskT)[kg][I2]*At[0][I2])
            p = np.maximum(np.asarray(self.zt), 1.0)
            self.eb_Pi = (p/1e5)**self._c('kappa', 0.287)
            self.eb_last = None; self.eb_acc = np.zeros(NB); self.eb_n = 0
            if self.my_pe == 0:
                print(ENERGY_LEGEND, flush=True)
                with open(efile, 'w') as fh: fh.write('# ' + ' '.join(ENERGY_COLUMNS) + '\n')
        u, v, w, th = (np.asarray(x) for x in (self.u, self.v, self.w, self.temp))
        mU, mV, mT = (np.asarray(x) for x in (self.maskU, self.maskV, self.maskT))
        p_s = np.asarray(self.p_s[tau])
        Pi3 = self.eb_Pi[:, None, None]; dp3 = np.asarray(self.dzt)[:, None, None]
        cost, cosu = np.asarray(self.cost_y), np.asarray(self.cosu_y)
        dxt, dxu, dyt, dyu = (np.asarray(x) for x in (self.dxt_x, self.dxu_x, self.dyt_y, self.dyu_y))
        Au3, Av3 = np.asarray(self.area_u), np.asarray(self.area_v)
        KE  = 0.5*gs((u[I]**2)*self.eb_mu) + 0.5*gs((v[I]**2)*self.eb_mv)
        H   = self._c('cp_air', 1000.0)*gs((th*Pi3)[I]*self.eb_mt)
        PEs = 0.5*gs((p_s*p_s/self.grav)[I2]*self.eb_A2)
        E   = KE + H + PEs
        fe, fn, ft = (np.asarray(x) for x in self.mom_flux_2nd_u(self.u,self.u,self.v,self.w,self.maskU,self.maskV,self.maskW))
        du_adv = ((fe-rx(fe,1)) + (fn-ry(fn,1)) + (ft-rz(ft,1)))*mU/(Au3*dp3)
        fe, fn, ft = (np.asarray(x) for x in self.mom_flux_2nd_v(self.v,self.u,self.v,self.w,self.maskU,self.maskV,self.maskW))
        dv_adv = ((fe-rx(fe,1)) + (fn-ry(fn,1)) + (ft-rz(ft,1)))*mV/(Av3*dp3)
        du_cor, dv_cor = (np.asarray(x) for x in self.coriolis_and_metric(self.u,self.v,self.coriolis_t,self.maskU,self.maskV))
        Phi = np.asarray(self.hydrostatic_pressure(self.rho, OM.np.zeros_like(self.rho), self.maskT))
        du_ph = -(rx(Phi,-1)-Phi)/(cost*dxu)*mU;  dv_ph = -(ry(Phi,-1)-Phi)/dyu*mV
        du_ps = -(rx(p_s,-1)-p_s)[None]/(cost*dxu)*mU;  dv_ps = -(ry(p_s,-1)-p_s)[None]/dyu*mV
        KEu = lambda a, b: gs((u*a)[I]*self.eb_mu) + gs((v*b)[I]*self.eb_mv)
        KE_adv, KE_cor, KE_ph, KE_ps = KEu(du_adv,dv_adv), KEu(du_cor,dv_cor), KEu(du_ph,dv_ph), KEu(du_ps,dv_ps)
        kv = np.asarray(self.hs_kv)
        D_ray = -gs((kv*u*u)[I]*self.eb_mu) - gs((kv*v*v)[I]*self.eb_mv)
        D_bih = gs(self._bih_step) if hasattr(self, 'Ahbi') else 0.
        fe, fn, ft = (np.asarray(x) for x in self.tracer_flux_2nd(self.temp,self.u,self.v,self.w,self.maskU,self.maskV,self.maskW))
        dth_h = ((fe-rx(fe,1))/(cost*dxt) + (fn-ry(fn,1))/(cost*dyt))*mT
        dth_v = ((ft-rz(ft,1))/dp3)*mT
        H_hadv = self._c('cp_air', 1000.0)*gs((Pi3*dth_h)[I]*self.eb_mt)
        H_vadv = self._c('cp_air', 1000.0)*gs((Pi3*dth_v)[I]*self.eb_mt)
        H_fix  = -self._c('cp_air', 1000.0)*gs((th[kg]*w[kg]/float(self.dzt[kg])*mT[kg]*self.eb_Pi[kg])[I2]*self.eb_mt[-1]) if self._c('GROUND_FIX', True) else 0.
        kT, theq = np.asarray(self.hs_kT), np.asarray(self.hs_theq)
        Q = self._c('cp_air', 1000.0)*gs((kT*(theq-th)*Pi3)[I]*self.eb_mt)
        divU = np.sum(((u*mU - rx(u*mU,1))/(cost*dxt) + ((v*mV*cosu) - ry(v*mV*cosu,1))/(cost*dyt))*dp3*mT, axis=0)
        PEs_dot = -gs((p_s*divU)[I2]*self.eb_A2)
        # conversion face by face (centered scheme: no limiter part) + ground work
        F = (slice(h, kg), slice(h,-h), slice(h,-h)); Af = np.asarray(self.area_t)[0][I2]/g0
        dPi = (self.eb_Pi[h+1:-h] - self.eb_Pi[h:kg])[:, None, None]
        hyd = -(Phi[h+1:-h, h:-h, h:-h] - Phi[F])
        CONV_gnd = -gs(w[kg][I2]*Phi[kg][I2]*Af)
        CONV_eos = gs(w[F]*(self._c('cp_air', 1000.0)*0.5*(th[F]+th[h+1:-h, h:-h, h:-h])*dPi - hyd)*Af)
        # surface exchange and boundary layer terms, booked by the step
        W_tau = gs(self._wa_step)
        Q_srf = gs((self.q_s*mT[kg])[I2]*np.asarray(self.area_t)[0][I2])
        D_bl  = gs(self._dbl_step); EPS = gs(self._eps_step); D_vnum = D_bl + EPS
        H_vdif = gs(self._hvdif_step); H_diss = gs(self._heat_step); Q_int = gs(self._qint_step)
        if self.eb_last is None: self.eb_last = E
        self.eb_acc += self.dt*np.array([KE_adv, KE_cor, KE_ph, KE_ps, D_ray, D_bih, H_hadv, H_vadv,
                                         H_fix, Q, PEs_dot, CONV_eos, CONV_gnd, W_tau, Q_srf,
                                         D_bl, D_vnum, H_vdif, H_diss, Q_int, 0., 0., 0.])
        self.eb_n += 1
        if n > 0 and n % int(self.snapint/self.dt) == 0:
            T = self.eb_n*self.dt; A = self.eb_area
            a = self.eb_acc/T/A
            (KE_adv, KE_cor, KE_ph, KE_ps, D_ray, D_bih, H_hadv, H_vadv, H_fix, Q, PEs_dot,
             CONV_eos, CONV_gnd, W_tau, Q_srf, D_bl, D_vnum, H_vdif, H_diss, Q_int) = a[:20]
            dE_sp = a[:11].sum() + W_tau + Q_srf + D_bl + H_vdif + H_diss + Q_int
            dE_fd = (E - self.eb_last)/T/A
            CONV = KE_ph + H_vadv; LEAK = KE_adv + KE_cor + CONV + H_hadv + H_fix
            if self.my_pe == 0:
                print('E-budget atm day %7.2f [W/m2]  Q=%9.4f D_ray=%8.4f D_bih=%8.4f D_bl=%8.4f W_tau=%8.4f '
                      'Q_s=%8.4f Q_int=%8.4f H_diss=%8.4f H_vdif=%8.4f | CONV=%9.5f H_fix=%9.5f LEAK_sp=%9.5f '
                      'LEAK_dt=%9.5f | dE/dt=%9.4f'
                      % (n*self.dt/day, Q, D_ray, D_bih, D_bl, W_tau, Q_srf, Q_int, H_diss, H_vdif,
                         CONV, H_fix, LEAK, dE_fd - dE_sp, dE_fd), flush=True)
                with open(efile, 'a') as fh:
                    fh.write(' '.join('%.10e' % x for x in
                             [n*self.dt/day, KE_adv, KE_cor, KE_ph, KE_ps, D_ray, D_bih, H_hadv, H_vadv,
                              H_fix, Q, PEs_dot, CONV, CONV_eos, CONV_gnd, LEAK, dE_sp, dE_fd,
                              dE_fd - dE_sp, W_tau, Q_srf, D_bl, D_vnum, H_vdif, H_diss, Q_int,
                              KE/A, H/A, PEs/A]) + '\n')
            self.eb_last = E; self.eb_acc[:] = 0.; self.eb_n = 0

    def diagnose(self, n, tau):
        super().diagnose(n, tau)
        self.energy_budget(n, tau)


