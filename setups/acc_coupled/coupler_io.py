"""
Coupler-level I/O: the coupling-diagnostics netCDF writer and the netCDF
restart mechanism (global gathered fields; restart across a different PE
decomposition is allowed; atomic writes). All grid sizes and options are
passed as arguments -- this module holds no experiment state.
"""
import os
import numpy as np
import pyOM3 as OM
from coupler_mpi import gather_unpadded_mpi, scatter_array_mpi


class CouplingWriter:
    """Write the wave->ocean coupling fields to a netCDF file, one record per
    coupling (ocean) time step. Runs only on wave-rank-0, which holds the
    gathered full-domain fields. Variables:
      ustar_x, ustar_y : friction-velocity vector u* (cos/sin of udir)  (t,y,x)
      u_stokes, v_stokes : SURFACE Stokes drift velocity                (t,y,x)
                           (profile dropped: dominated file size; the ocean
                            gets the full profile via MPI, and the restart
                            keeps 3D u_S -- only the archive is surface-only)
      stress_x, stress_y : surface wave stress / force                  (t,y,x)
                           (or (t,z,y,x) if stress_work / depth-resolved)
    """
    def __init__(self, fname, nx, ny, nz, dx, dy, stress_is_3d, dt_cpl=0.0):
        from netCDF4 import Dataset
        self.f = Dataset(fname, 'w', format='NETCDF4')
        self.f.createDimension('time', None)
        try:
            self.f.setncattr('dt_cpl', float(dt_cpl))
        except Exception:
            pass      # unlimited
        self.f.createDimension('x', nx)
        self.f.createDimension('y', ny)
        self.f.createDimension('z', nz)
        self.f.createVariable('time', 'f8', ('time',))
        xv = self.f.createVariable('x', 'f8', ('x',)); xv[:] = (0.5 + np.arange(nx)) * dx
        yv = self.f.createVariable('y', 'f8', ('y',)); yv[:] = (0.5 + np.arange(ny)) * dy
        self.f.createVariable('z', 'f8', ('z',))[:] = np.arange(nz)
        self.f.createVariable('ustar_x', 'f4', ('time', 'y', 'x'))
        self.f.createVariable('ustar_y', 'f4', ('time', 'y', 'x'))
        v = self.f.createVariable('u_stokes', 'f4', ('time', 'y', 'x'))
        v.long_name = 'surface Stokes drift (z=0 level only)'
        v = self.f.createVariable('v_stokes', 'f4', ('time', 'y', 'x'))
        v.long_name = 'surface Stokes drift (z=0 level only)'
        sdims = ('time', 'z', 'y', 'x') if stress_is_3d else ('time', 'y', 'x')
        self.f.createVariable('stress_x', 'f4', sdims)
        self.f.createVariable('stress_y', 'f4', sdims)
        # wave->ocean dissipation energy flux (WAM: PHIOC), positive into the
        # ocean; the local field that will feed the TKE surface BC.
        # wind -> wave momentum input (delivered stress TO the wave model),
        # int gamma_in E omega (sin,cos) dOmega, resolved + analytic tail
        v = self.f.createVariable('stress_in_x', 'f4', ('time', 'y', 'x'))
        v.units = 'm^2 s^-2'; v.long_name = 'wind to wave input stress, x'
        v = self.f.createVariable('stress_in_y', 'f4', ('time', 'y', 'x'))
        v.units = 'm^2 s^-2'; v.long_name = 'wind to wave input stress, y'
        v = self.f.createVariable('phi_oc', 'f4', ('time', 'y', 'x'))
        v.units = 'm^3 s^-3'; v.long_name = 'wave-to-ocean energy flux -g int S_diss dOmega'
        # domain-integrated energy diagnostics, one scalar per coupling step,
        # kinematic (per unit rho): multiply by rho_w for Watts.
        for name, long_name in (
                ('wind_work',      'wind -> wave energy input g*int gamma_in E'),
                ('diss_work',      'wave -> ocean dissipation energy flux, domain integral'),
                ('mom_in_x',       'wind -> wave momentum input, domain integral [m^4/s^2]'),
                ('mom_in_y',       'wind -> wave momentum input, y'),
                ('mom_del_x',      'wave -> ocean delivered stress, domain integral'),
                ('mom_del_y',      'wave -> ocean delivered stress, y'),
                ('mom_wave_x',     'wave field momentum content, domain integral [m^4/s]'),
                ('mom_wave_y',     'wave field momentum content, y'),
                ('stress_work',    'wave-stress work on ocean u_L (Eq. 6)'),
                ('stress_work_eul','wave-stress work against Eulerian u = u_L - u_S (comparison, not the Eq. 6 channel)'),
                ('snl_work',       'reversible exchange int U.k S_nl dk (Eq. 11 non-closure term, positive into waves)'),
                ('exchange_C',     'exchange work C: exact applied-increment work of d(u_S)/dt on u_L'),
                ('vortex_residual','vortex-force u_L-work residual (O(dx^2))')):
            v = self.f.createVariable(name, 'f8', ('time',))
            v.units = 'm^5 s^-3'
            v.long_name = long_name
        self.stress_is_3d = stress_is_3d
        self.n = 0

    def write(self, t, ustar_x, ustar_y, u_S, stress, phioc=None, ediag=None,
              mdiag=None, stress_in=None):
        k = self.n
        self.f['time'][k]    = t
        self.f['ustar_x'][k] = ustar_x
        self.f['ustar_y'][k] = ustar_y
        self.f['u_stokes'][k] = u_S[0][-1]     # surface level only
        self.f['v_stokes'][k] = u_S[1][-1]
        self.f['stress_x'][k] = stress[0]
        self.f['stress_y'][k] = stress[1]
        if phioc is not None:
            self.f['phi_oc'][k] = phioc
        if stress_in is not None:
            self.f['stress_in_x'][k] = stress_in[0]
            self.f['stress_in_y'][k] = stress_in[1]
        if ediag is not None:
            pw, wst, wste, cxc, wvf, pdiss, wnl = ediag
            self.f['wind_work'][k]       = pw
            self.f['diss_work'][k]       = pdiss
            self.f['stress_work_eul'][k] = wste
            self.f['snl_work'][k]        = wnl
        if mdiag is not None:
            for nm, v in zip(('mom_in_x','mom_in_y','mom_del_x',
                              'mom_del_y','mom_wave_x','mom_wave_y'), mdiag):
                self.f[nm][k] = v
            self.f['stress_work'][k]     = wst
            self.f['exchange_C'][k]      = cxc
            self.f['vortex_residual'][k] = wvf
        self.f.sync()
        self.n += 1

    def close(self):
        self.f.close()


# ===========================================================================
# CONFIGURATION
# ===========================================================================
# PE distribution -- wave and ocean decompositions are INDEPENDENT.
# Total ranks must equal n_pes_x_wave*n_pes_y_wave + n_pes_x_ocean*n_pes_y_ocean.
n_pes_x_wave,  n_pes_y_wave  = 4, 4      # wave model decomposition (6 PEs)
n_pes_x_ocean, n_pes_y_ocean = 1, 1      # ocean model decomposition (2 PEs)


def _nc_create(fname, dims):
    from netCDF4 import Dataset
    f = Dataset(fname, 'w', format='NETCDF4')
    for dname, dlen in dims:
        f.createDimension(dname, dlen)
    return f

def wave_restart_write(M, fname, ny, nx, n_cpl, pre, tau, taum1, taum2, tA, tB, tC):
    sl = slice(M.halo, -M.halo)
    A_g  = gather_unpadded_mpi(M, M.A[sl, sl, sl, sl], ny, nx)
    dA_g = gather_unpadded_mpi(M, M.dA[:, sl, sl, sl, sl], ny, nx)
    # HIGH-FREQUENCY GHOST SLAB (as in the global4 coupler): imphftail keeps
    # the imposed f^-5 continuation in the frequency ghost bins every source
    # step, and the f-advection boundary flux and the tail diagnostics read
    # exactly those values. Storing only the interior bins leaves them zero
    # after the read -> the first post-restart step advects against a zero
    # anchor (~1e-8 seed in A, amplified by the ocean's convective flips).
    h_   = M.halo
    gh_g = gather_unpadded_mpi(M, M.A[:, -h_:, sl, sl], ny, nx)
    has_cur = int(hasattr(M, 'ocean_U'))
    if has_cur:
        oU  = gather_unpadded_mpi(M, M.ocean_U[sl, sl, sl],   ny, nx)
        oV  = gather_unpadded_mpi(M, M.ocean_V[sl, sl, sl],   ny, nx)
        oUf = gather_unpadded_mpi(M, M.ocean_Udf[sl, sl, sl], ny, nx)
        oVf = gather_unpadded_mpi(M, M.ocean_Vdf[sl, sl, sl], ny, nx)
    if M.my_pe == 0:
        nd_, nf_ = A_g.shape[0], A_g.shape[1]
        tmp = fname + '.tmp'
        f = _nc_create(tmp, [('three', 3), ('d', nd_), ('f', nf_),
                             ('y', ny), ('x', nx),
                             ('dpad', gh_g.shape[0]), ('fh', gh_g.shape[1])])
        f.createVariable('A',  'f8', ('d', 'f', 'y', 'x'))[:]           = A_g
        f.createVariable('dA', 'f8', ('three', 'd', 'f', 'y', 'x'))[:]  = dA_g
        f.createVariable('A_fghost', 'f8', ('dpad', 'fh', 'y', 'x'))[:] = gh_g
        if has_cur:
            f.createVariable('ocean_U',   'f8', ('f', 'y', 'x'))[:] = oU
            f.createVariable('ocean_V',   'f8', ('f', 'y', 'x'))[:] = oV
            f.createVariable('ocean_Udf', 'f8', ('f', 'y', 'x'))[:] = oUf
            f.createVariable('ocean_Vdf', 'f8', ('f', 'y', 'x'))[:] = oVf
        f.setncatts(dict(n_cpl=n_cpl, pre=pre, tau=tau, taum1=taum1,
                         taum2=taum2, tA=tA, tB=tB, tC=tC, has_cur=has_cur))
        f.close()
        if os.path.exists(fname):
            os.replace(fname, fname + '.prev')   # rolling backup: previous
                                                 # pair survives one interval
        os.replace(tmp, fname)
        print('wrote wave restart %s at coupling step %d' % (fname, n_cpl),
              flush=True)
    return

def wave_restart_read(M, fname, ny, nx, comm):
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in
               ('n_cpl', 'pre', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC',
                'has_cur')}
        A_g  = np.array(f['A'][:]);  dA_g = np.array(f['dA'][:])
        gh_g = np.array(f['A_fghost'][:]) if 'A_fghost' in f.variables else None
        att['_has_ghost'] = gh_g is not None
        cur  = [np.array(f[v][:]) for v in
                ('ocean_U', 'ocean_V', 'ocean_Udf', 'ocean_Vdf')]                if att['has_cur'] else None
        f.close()
    else:
        att, A_g, dA_g, gh_g, cur = None, None, None, None, None
    att = comm.bcast(att, root=0)
    sl = slice(M.halo, -M.halo)
    nd_ = M.A.shape[0] - 2 * M.halo
    nf_ = M.A.shape[1] - 2 * M.halo
    blk = scatter_array_mpi(M, A_g, (nd_, nf_))
    M.A = OM.modify_array(M.A, (sl, sl, sl, sl), blk)
    M.A = M.apply_bc(M.A)
    blk = scatter_array_mpi(M, dA_g, (3, nd_, nf_))
    M.dA = OM.modify_array(M.dA, (slice(None), sl, sl, sl, sl), blk)
    if att.get('_has_ghost', False):
        # AFTER apply_bc, which zero-fills the frequency halo by design
        h_  = M.halo
        blk = scatter_array_mpi(M, gh_g, (M.A.shape[0], h_))
        M.A = OM.modify_array(M.A, (slice(None), slice(-h_, None), sl, sl), blk)
    elif M.my_pe == 0:
        print('wave restart has no A_fghost: the f^-5 tail anchor starts at '
              'zero and is rebuilt by the first imphftail call (one-step '
              'transient in the tail fluxes)', flush=True)
    if att['has_cur']:
        for name, full in zip(('ocean_U', 'ocean_V', 'ocean_Udf', 'ocean_Vdf'),
                              cur if M.my_pe == 0 else [None]*4):
            blk = scatter_array_mpi(M, full, (nf_,))
            setattr(M, name, M.apply_bc(np.pad(blk, M.halo)))
    M.E = M.A * M.omega_t[None, :, None, None]
    if M.my_pe == 0:
        print('read wave restart %s: resuming at coupling step %d'
              % (fname, att['n_cpl']), flush=True)
    return att

def ocean_restart_write(M, fname, ny, nx, use_stress_work, n_cpl, tau, taum1, taum2, tA, tB, tC):
    h = M.halo
    sl = slice(h, -h)
    g3 = lambda a: gather_unpadded_mpi(M, a[sl, sl, sl], ny, nx)
    g4 = lambda a: gather_unpadded_mpi(M, a[:, sl, sl, sl], ny, nx)
    g2 = lambda a: gather_unpadded_mpi(M, a[sl, sl], ny, nx)
    u_g, v_g, t_g = g3(M.u), g3(M.v), g3(M.temp)
    du_g, dv_g, dt_g = g4(M.du), g4(M.dv), g4(M.dtemp)
    # p_s carries THREE time levels (3,Ny,Nx), like pyOM2's psi(:,:,tau..):
    # gather with the time axis as a leading dim, not as y.
    ps_g = gather_unpadded_mpi(M, M.p_s[:, sl, sl], ny, nx)
    uS_g, vS_g = g3(M.u_S), g3(M.v_S)
    # d(u_S)/dt forcing state of the u_L formulation: the PREVIOUS coupling's
    # Stokes drift. Without it, the first post-restart step applies no
    # d(u_S)/dt increment (a one-interval error in u_L and in C). Presence is
    # uniform across PEs (set in the coupler loop after the first coupling),
    # so the hasattr guard is collective-safe.
    has_uSo = int(hasattr(M, 'u_S_old'))
    if has_uSo:
        uSo_g, vSo_g = g3(M.u_S_old), g3(M.v_S_old)
    tke_g = g3(M.tke) if hasattr(M, 'tke') else None
    # K_diss_v (shear production of the LAST step) is STATE: the next step's
    # set_tke_diffusivities runs first and builds the Richardson number, hence
    # kappaH, from it; without it the first post-restart step has Pr = 10.
    kdv_g = g3(M.K_diss_v) if hasattr(M, 'K_diss_v') else None
    if use_stress_work:
        sx_g, sy_g = g3(M.tau_work_x), g3(M.tau_work_y)
    else:
        sx_g, sy_g = g2(M.taux), g2(M.tauy)
    phi_g = g2(M.phi_oc)          # genuinely 2D (unlike p_s!)
    if M.my_pe == 0:
        nz_ = u_g.shape[0]
        tmp = fname + '.tmp'
        f = _nc_create(tmp, [('three', 3), ('z', nz_), ('y', ny), ('x', nx)])
        for nme, arr, dims in (
                ('u', u_g, ('z','y','x')), ('v', v_g, ('z','y','x')),
                ('temp', t_g, ('z','y','x')),
                ('du', du_g, ('three','z','y','x')),
                ('dv', dv_g, ('three','z','y','x')),
                ('dtemp', dt_g, ('three','z','y','x')),
                ('p_s', ps_g, ('three','y','x')),
                ('u_S', uS_g, ('z','y','x')), ('v_S', vS_g, ('z','y','x')),
                *((('u_S_old', uSo_g, ('z','y','x')),
                   ('v_S_old', vSo_g, ('z','y','x'))) if has_uSo else ()),
                *((('tke', tke_g, ('z','y','x')),) if tke_g is not None else ()),
                *((('K_diss_v', kdv_g, ('z','y','x')),) if kdv_g is not None else ()),
                ('stress_x', sx_g, ('z','y','x') if use_stress_work else ('y','x')),
                ('stress_y', sy_g, ('z','y','x') if use_stress_work else ('y','x')),
                ('phi_oc', phi_g, ('y','x'))):
            f.createVariable(nme, 'f8', dims)[:] = arr
        f.setncatts(dict(n_cpl=n_cpl, tau=tau, taum1=taum1, taum2=taum2,
                         tA=tA, tB=tB, tC=tC,
                         use_stress_work=int(use_stress_work)))
        f.close()
        if os.path.exists(fname):
            os.replace(fname, fname + '.prev')   # rolling backup: previous
                                                 # pair survives one interval
        os.replace(tmp, fname)
        print('wrote ocean restart %s at coupling step %d' % (fname, n_cpl),
              flush=True)
    return

def ocean_restart_read(M, fname, ny, nx, use_stress_work, comm):
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in
               ('n_cpl', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC')}
        # phi_oc: tolerate older restart files without it
        names = ['u', 'v', 'temp', 'du', 'dv', 'dtemp', 'p_s',
                 'u_S', 'v_S', 'stress_x', 'stress_y']
        if 'phi_oc' in f.variables: names.append('phi_oc')
        if 'tke' in f.variables: names.append('tke')
        if 'K_diss_v' in f.variables: names.append('K_diss_v')
        if 'u_S_old' in f.variables: names += ['u_S_old', 'v_S_old']
        dat = {v: np.array(f[v][:]) for v in names}
        if 'phi_oc' not in dat: dat['phi_oc'] = None
        # tke presence is a COLLECTIVE decision: it must be broadcast, not
        # inferred rank-locally (non-root dat has no 'tke' -> ranks would
        # skip the scatter root participates in -> MPI_ERR_TRUNCATE when
        # root's tke blocks land in the next exchange's smaller buffers)
        att['_has_tke'] = 'tke' in dat
        att['_has_kdv'] = 'K_diss_v' in dat
        att['_has_uSo'] = 'u_S_old' in dat     # same collectivity argument
        f.close()
    else:
        att, dat = None, {v: None for v in
               ('u', 'v', 'temp', 'du', 'dv', 'dtemp', 'p_s',
                'u_S', 'v_S', 'u_S_old', 'v_S_old',
                'stress_x', 'stress_y', 'phi_oc', 'tke', 'K_diss_v')}
    att = comm.bcast(att, root=0)
    h = M.halo
    sl = slice(h, -h)
    nz_ = M.u.shape[0] - 2 * h
    def r3(name, tgt):
        blk = scatter_array_mpi(M, dat[name], (nz_,))
        tgt = OM.modify_array(tgt, (sl, sl, sl), blk)
        return M.apply_bc(tgt)
    def r4(name, tgt):
        blk = scatter_array_mpi(M, dat[name], (3, nz_))
        return OM.modify_array(tgt, (slice(None), sl, sl, sl), blk)
    M.u, M.v, M.temp = r3('u', M.u), r3('v', M.v), r3('temp', M.temp)
    M.du, M.dv, M.dtemp = r4('du', M.du), r4('dv', M.dv), r4('dtemp', M.dtemp)
    # p_s: three time levels (3,Ny,Nx). Must NOT go through apply_bc as one
    # 3-axis array: apply_bc_mpi sizes its recv buffer as (Nz,Nx) for ANY
    # 3-axis field (the serial roll path is generic, hence 1-PE restarts
    # worked). Exchange each time level separately as a 2D field.
    blk = scatter_array_mpi(M, dat['p_s'], (3,))
    for k in range(3):
        M.p_s = OM.modify_array(M.p_s, (k, sl, sl), blk[k])
        M.p_s = OM.modify_array(M.p_s, (k, slice(None), slice(None)),
                                M.apply_bc(M.p_s[k]))

    M.u_S, M.v_S = r3('u_S', M.u_S), r3('v_S', M.v_S)
    if att.get('_has_uSo', False):
        # d(u_S)/dt forcing state (u_L formulation): restore the previous
        # coupling's Stokes drift so the first post-restart step applies the
        # correct increment; absent in pre-u_L restarts -> first step has no
        # forcing (one-interval transient, loud in the log via C = 0)
        M.u_S_old = r3('u_S_old', OM.np.zeros_like(M.u_S))
        M.v_S_old = r3('v_S_old', OM.np.zeros_like(M.v_S))
    elif M.my_pe == 0:
        print('ocean restart has no u_S_old: first step applies no d(u_S)/dt '
              'forcing (one coupling-interval transient)', flush=True)
    if att.get('_has_tke', False) and hasattr(M, 'tke'):
        M.tke = r3('tke', M.tke)     # backward-compat: absent in old restarts
                                     # -> keep the init_tke seed (respins in
                                     # hours; loud in no log, benign)
    if att.get('_has_kdv', False) and hasattr(M, 'K_diss_v'):
        M.K_diss_v = r3('K_diss_v', M.K_diss_v)
    elif M.my_pe == 0 and hasattr(M, 'K_diss_v'):
        print('ocean restart has no K_diss_v: first step uses Pr = 10 in the '
              'TKE diffusivities (one-step transient)', flush=True)
    if use_stress_work:
        M.tau_work_x = r3('stress_x', M.tau_work_x)
        M.tau_work_y = r3('stress_y', M.tau_work_y)
    else:
        blk = scatter_array_mpi(M, dat['stress_x'], ())
        M.taux = M.apply_bc(OM.modify_array(M.taux, (sl, sl), blk))
        blk = scatter_array_mpi(M, dat['stress_y'], ())
        M.tauy = M.apply_bc(OM.modify_array(M.tauy, (sl, sl), blk))
    # phi_oc (2D): present in restarts written after WP2; skip if absent
    # (rank-0 knowledge of absence must reach all PEs before the collective)
    has_phi = comm.bcast(dat['phi_oc'] is not None if M.my_pe == 0 else None, root=0)
    if has_phi and hasattr(M, 'phi_oc'):
        blk = scatter_array_mpi(M, dat['phi_oc'], ())
        M.phi_oc = M.apply_bc(OM.modify_array(M.phi_oc, (sl, sl), blk))
    # rho and w are exact functions of the restored state and are READ at the
    # start of the next step (N^2 for the diffusivities, p_h, tracer
    # advection) -- without this they keep their initial values for one step.
    from pyOM3.density import get_rho
    M.rho = get_rho(M.salt, M.temp, M.p0, M.eq_of_state)
    M.w   = M.vertical_velocity(M.u, M.v, M.w, M.maskW)
    if M.my_pe == 0:
        print('read ocean restart %s: resuming at coupling step %d'
              % (fname, att['n_cpl']), flush=True)
    return att

