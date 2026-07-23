"""
acc1 for pyOM3 / pe_simple
==========================

Idealised Southern Ocean channel, after Viebahn and Eden (2010), Ocean Modelling.
Port of pyOM2's for_config/acc1.f90 to the pyOM3 pe_simple hydrostatic solver,
staying as close to the original as the simpler pe_simple physics allow.

Run:
    serial :  python acc1.py
    MPI    :  mpirun -np <n_pes_x*n_pes_y> python acc1.py     (set n_pes_x/n_pes_y below)

"""

# add here pyOM3 code location, here it's $HOME/src/pyOM3/
import os,sys; sys.path.append(os.path.expanduser('~')+'/src/pyOM3/')
import numpy as np
import pyOM3 as OM
from pyOM3.pe_simple.snapshot import snap_nc4
from pyOM3.pe_simple.tke import tke


# ----------------------------------------------------------------------------
# resolution / switches (mirror the config_module block of acc1.f90)
# ----------------------------------------------------------------------------
hRESOLVE = 1.0       # 1 in the original model
vRESOLVE = 1.0       # 1 in the original model

t_rest   = 30 * 86400.0
phi0     = -25.0 / 180.0 * np.pi
N_0      = 0.004                    # target buoyancy frequency for initial stratification [1/s]
omega    = 7.292123516990375e-05    # Earth rotation rate [1/s]
radius   = 6370.0e3                 # Earth radius [m]
grav     = 9.81
rho_0    = 1024.0

nx = int(128 * hRESOLVE)
ny = int(128 * hRESOLVE)
nz = int(18 * vRESOLVE)

dx = 20e3 / hRESOLVE
dy = 20e3 / hRESOLVE
dz = 50.0 / vRESOLVE
dt = 800.0 / hRESOLVE

L_x = nx * dx
L_y = ny * dy

# domain decomposition (set >1 and launch with mpirun for parallel runs)
n_pes_x = 2
n_pes_y = 3
halo    = 1        

runlen  = 365 * 86400.0 * 5.5
snapint = 86400.0 * 3.0


class acc1(snap_nc4,tke):
    """acc1 channel configuration for pe_simple."""

    # -- global cell-centre / face coordinates (Cartesian, as in acc1.f90) ----
    #    xt = (0.5+i)*dx , yt = (0.5+j)*dy ; yu = (1+j)*dy  (matches snapshot.py)
    @property
    def yt_glob(self):
        return (0.5 + np.arange(ny)) * dy

    @property
    def yu_glob(self):
        return (1.0 + np.arange(ny)) * dy

    @property
    def xt_glob(self):
        return (0.5 + np.arange(nx)) * dx

    # ------------------------------------------------------------------------
    # Coriolis:  beta plane, f(y) = 2 Om sin(phi0) + 2 Om cos(phi0)/radius * yt
    # ------------------------------------------------------------------------
    def set_coriolis(self, f0=None):
        f_glob = 2 * omega * np.sin(phi0) + 2 * omega * np.cos(phi0) / radius * self.yt_glob
        # broadcast to (Ny) local slice incl. halos
        fyx = np.zeros((self.Ny, self.Nx), OM.prec)
        j0 = self.ys_pe - 1                      # global index of first interior row
        for jj in range(self.halo, self.Ny - self.halo):
            jg = j0 + (jj - self.halo)
            jg = min(max(jg, 0), ny - 1)
            fyx[jj, :] = f_glob[jg]
        self.coriolis_t = OM.modify_array(self.coriolis_t, (slice(None), slice(None)),
                                          OM.np.asarray(fyx, OM.prec),
                                          out_sharding=self.sharding_2D)
        self.coriolis_t = self.apply_bc(self.coriolis_t)
        return

    # ------------------------------------------------------------------------
    # Topography: ridge -> land where y > L_y/2 and (x > 0.75 L_x or x < 0.25 L_x)
    # kbot = k means bottom is below k-th level.  Flat floor -> kbot = halo.
    # ------------------------------------------------------------------------
    def set_topography(self):
        kbot = np.full((self.Ny, self.Nx), float(self.halo), OM.prec)
        j0 = self.ys_pe - 1
        i0 = self.xs_pe - 1
        for jj in range(self.halo, self.Ny - self.halo):
            jg = j0 + (jj - self.halo)
            if jg < 0 or jg >= ny:
                continue
            y = self.yt_glob[jg]
            for ii in range(self.halo, self.Nx - self.halo):
                ig = i0 + (ii - self.halo)
                if ig < 0 or ig >= nx:
                    continue
                x = self.xt_glob[ig]
                if (y > L_y / 2.0) and (x > L_x * 0.75 or x < L_x * 0.25):
                    kbot[jj, ii] = 0.0
        self.kbot = OM.modify_array(self.kbot, (slice(None), slice(None)),
                                    OM.np.asarray(kbot, OM.prec),
                                    out_sharding=self.sharding_2D)
        return

    # ------------------------------------------------------------------------
    # b_surface(y): target surface buoyancy profile from acc1.f90
    # ------------------------------------------------------------------------
    @staticmethod
    def _b_surface(y):
        db = 30e-3
        bstar = np.full_like(y, db)
        south = y < L_y / 2.0
        bstar = np.where(south, db * y / (L_y / 2.0), bstar)
        y2 = L_y * 0.75
        north = y > y2
        bstar = np.where(north, db * (1.0 - (y - y2) / (L_y - y2)), bstar)
        return bstar


    # ------------------------------------------------------------------------
    # initial stratification + zonal wind stress field.
    # Builds FULL-domain fields; set_initial_conditions slices them per PE.
    # ------------------------------------------------------------------------
    def make_initial_fields(self):
        # z of T points: k=0 is bottom, k=nz-1 is surface (pe_simple convention)
        zt = (0.5 + np.arange(nz)) * dz - nz * dz          # depths, negative, surface ~ 0

        # Stable initial stratification b(z) = b_mid + N_0^2 * z  (z <= 0, so b
        # decreases downward -> statically stable). The surface value equals the
        # target buoyancy in the MIDDLE of the domain, which is the maximum of
        # _b_surface (= db, the lightest / warmest target). With this choice the
        # surface restoring only cools (lowers b) to the north and south of the
        # middle, where _b_surface drops below b_mid; in the middle the surface
        # already sits at its target, so there is no surface flux there.
        b_mid = self._b_surface(np.array([L_y / 2.0]))[0]  # target b at mid-channel
        bz = b_mid + N_0**2 * zt                           # (nz,) stable profile

        temp = np.zeros((nz, ny, nx), OM.prec)             # holds buoyancy b
        salt = np.zeros((nz, ny, nx), OM.prec)             # inactive for eq_of_state = 0
        #for k in range(nz):
        #    temp[k, :, :] = bz[k]

        u = np.zeros((nz, ny, nx), OM.prec)
        v = np.zeros((nz, ny, nx), OM.prec)
        p_s = np.zeros((ny, nx), OM.prec)

        # zonal wind stress on southern half: tau_x = 1e-4 * sin(2 pi yu/L_y)
        y = self.yu_glob
        taux = np.zeros((ny, nx), OM.prec)
        south = self.yt_glob < L_y / 2.0
        prof = 0.1e-3 * np.sin(2 * np.pi * y / L_y)
        for j in range(ny):
            if south[j]:
                taux[j, :] = prof[j]
        self._taux_glob = taux

        return u, v, temp, salt, p_s


    # ------------------------------------------------------------------------
    # slice global wind stress to the local (haloed) T grid, once, for forcing
    # ------------------------------------------------------------------------
    def _localise_forcing(self):
        j0 = self.ys_pe - 1
        i0 = self.xs_pe - 1
        taux = np.zeros((self.Ny, self.Nx), OM.prec)
        for jj in range(self.halo, self.Ny - self.halo):
            jg = min(max(j0 + (jj - self.halo), 0), ny - 1)
            for ii in range(self.halo, self.Nx - self.halo):
                ig = min(max(i0 + (ii - self.halo), 0), nx - 1)
                taux[jj, ii] = self._taux_glob[jg, ig]
        self.taux = OM.np.asarray(taux, OM.prec)
        self.tauy = OM.np.zeros_like(taux)
        
        # target surface buoyancy per local row -> temperature restoring target
        yloc = np.zeros(self.Ny, OM.prec)
        for jj in range(self.halo, self.Ny - self.halo):
            jg = min(max(j0 + (jj - self.halo), 0), ny - 1)
            yloc[jj] = self.yt_glob[jg]
        # eq_of_state = 0: prognostic variable is buoyancy, so the restoring
        # target IS b_surface directly (no alpha = drho/dT conversion needed).
        self._bstar = self._b_surface(yloc)               # target surface buoyancy per row
        return

    # ------------------------------------------------------------------------
    # set_forcing(tau): add wind stress, surface T restoring, bottom drag and
    # a constant background vertical friction. Called after momentum tendency,
    # adds into du[tau]/dv[tau] (momentum) and dtemp[tau] (tracer).
    # ------------------------------------------------------------------------
    def set_forcing(self, tau):
        h = self.halo
        ks = -h - 1                     # surface (top) T/U/V level index

        # --- vertical friction with wind stress as the surface-face flux ------
        # ft[k] is the stress on the TOP face of cell k:  ft = A_v*(u[k+1]-u[k])/dz.
        # The top wet cell's top face is the surface, where the flux must equal
        # the wind stress tau_x/rho_0 (acc1: enable_implicit_vert_friction with
        # taux as the top BC).  We build ft, overwrite the surface face,
        # take the divergence and add it as a proper AB3 tendency to du[tau].
        ftu, ftv = OM.np.zeros_like(self.u), OM.np.zeros_like(self.v)
        ftu = OM.modify_array(ftu, (ks, slice(None), slice(None)), self.taux  * self.maskU[ks])
        ftv = OM.modify_array(ftv, (ks, slice(None), slice(None)), self.tauy  * self.maskV[ks])
        
        dvisc_u = (ftu - self.rollz(ftu, 1)) / dz * self.maskU
        dvisc_v = (ftv - self.rollz(ftv, 1)) / dz * self.maskV

        # --- linear bottom friction:  -r_bot * u_bottom / dz -----------------
        if getattr(self, 'r_bot', 0.0) > 0:
            bot_u = self.maskU * (1.0 - self.rollz(self.maskU, 1))     # 1 at bottom wet cell
            bot_v = self.maskV * (1.0 - self.rollz(self.maskV, 1))
            dvisc_u = dvisc_u - self.r_bot  * bot_u * self.u
            dvisc_v = dvisc_v - self.r_bot  * bot_v * self.v

        self.du = OM.modify_array(self.du, (tau, slice(None), slice(None), slice(None)),
                                  self.du[tau] + dvisc_u, out_sharding=self.sharding_4D)
        self.dv = OM.modify_array(self.dv, (tau, slice(None), slice(None), slice(None)),
                                  self.dv[tau] + dvisc_v, out_sharding=self.sharding_4D)

        # --- surface buoyancy restoring (acc1 b_surface -> forc) -------------
        # eq_of_state = 0: "temp" is buoyancy b, restore toward b_surface(y).
        bstar2 = OM.np.asarray(self._bstar[:, None] * OM.np.ones((1, self.Nx)), OM.prec)
        restore = OM.np.zeros_like(self.dtemp[tau])
        surf_b = self.temp[ks, :, :]
        forc = (1.0 / t_rest) * (bstar2 - surf_b)            # tendency [m/s^3] at surface cell
        restore = OM.modify_array(restore, (ks, slice(None), slice(None)),
                                  forc * self.maskT[ks])
        self.dtemp = OM.modify_array(self.dtemp, (tau, slice(None), slice(None), slice(None)),
                                     self.dtemp[tau] + restore, out_sharding=self.sharding_4D)
        return


# ----------------------------------------------------------------------------
# build, initialise and run
# ----------------------------------------------------------------------------
def main():
    parameter = dict(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz, dt=dt,
        eq_of_state=0,                       # buoyancy formulation: temp == buoyancy b
        periodic_in_x=True,                  # zonally periodic channel
        periodic_in_y=False,                 # closed north/south walls
        implicit_free_surface=True,          # rigid-lid / CG surface pressure solver
        explicit_free_surface=False,
        superbee_advection=True,             # enable_superbee_advection
        Ahbi=5e11 / hRESOLVE**4,             # A_hbi, resolution-scaled (eddy-resolving line in acc1)
        #Khbi=5e11 / hRESOLVE**4,             # biharmonic tracer mixing (kept = A_hbi)
        #kappa_T=1e-4,# / vRESOLVE,             # background vertical diffusivity
        enable_implicit_vmix=True,           # use backward-Euler vertical mixing
        #kappa_conv=1.0,                      # convective diffusivity where Nsqr<0 (pyOM2 value)
        #A_v=1e-4, #/ vRESOLVE,                 # background vertical viscosity (implicit-fric stand-in)        
        r_bot=1e-5 * vRESOLVE,               # linear bottom drag
        epsilon=1e-8,                        # congr_epsilon
        max_iterations=5000,                 # congr_max_iterations
        snapint=snapint, enable_tke = True,
    )

    M = acc1(parameter, n_pes_x=n_pes_x, n_pes_y=n_pes_y, halo_points=halo)

    M.set_coriolis()
    M.set_topography()
    M.make_topo_masks()

    u0, v0, t0, s0, ps0 = M.make_initial_fields()
    M.set_initial_conditions(u0, v0, t0, s0, ps0)
    M._localise_forcing()

    M.alpha_tke = 30.
    M.mxl_min = 1e-8
    M.tke_surf_from_waves = False
    M.init_tke()
    M.prepare_for_run(snap_file_name='acc1.cdf', show_each_time_step=True)
    
    nsteps = int(runlen / dt)
    M.loop(nsteps)


if __name__ == '__main__':
    main()
