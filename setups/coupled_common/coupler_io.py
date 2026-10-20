"""
Coupler-level I/O: the coupling-diagnostics netCDF writer and the netCDF
restart mechanism (global gathered fields; restarting on a different PE
decomposition is allowed; atomic writes). All grid sizes and options are
passed as arguments -- this module holds no experiment state.

Differences to the acc1 version:
  * the horizontal coordinates written into the coupling file are the true
    grid coordinates (degrees on the pseudo-spherical grid), passed in from
    the model rather than reconstructed from a scalar dx/dy;
  * the ocean restart carries SALT and its tendency (the global setup has a
    prognostic salinity); rho and w are recomputed after the read, both
    being exact functions of the restored state;
  * the wave restart carries A, dA, the f^-5 ghost slab, the four Doppler
    fields and, with input_scheme='janssen', tau_w and z0 (has_jan).
"""
import os
import numpy as np
import pyOM3 as OM
from coupler_mpi import gather_unpadded_mpi, scatter_array_mpi

# ---------------------------------------------------------------------------
# RAW (per-rank) OCEAN RESTART, EXP_RESTART_RAW=1. Default off, so every
# existing experiment keeps the gathered netCDF file.
#
# The gathered form sends every field block by block to rank 0, holds the
# whole domain there and writes it serially. At 1/3 degree that is about 20
# three-dimensional fields of 762 MB plus the partial sums of the online
# means: 15 to 25 GB through one rank, minutes per restart. The raw form is
# per rank: every rank writes its own padded block to <base>_pe/<rank>.npz,
# nothing is gathered, and rank 0 writes only the small header file that
# carries the time levels and the step counter -- the SAME file name the
# gathered form uses, so run.py's existence check and its segment tag work
# unchanged.
#
# PRICE: a raw restart belongs to its decomposition. Resuming, and branching
# from it with EXP_INIT_OCEAN, needs the same n_pes_x x n_pes_y. The header
# records the layout and the read stops with that message if it differs.
#
# READING dispatches on the FILE, not on the switch (see _is_raw), so the
# switch controls writing only: a run that writes raw restarts can branch
# from a gathered EXP_INIT_OCEAN written on another layout, and a run that
# writes gathered ones can resume a raw restart of its own layout.
# ---------------------------------------------------------------------------
RESTART_RAW = os.environ.get('EXP_RESTART_RAW', '0') not in ('0', '', 'no')


def _raw_dir(fname):
    return (fname[:-3] if fname.endswith('.nc') else fname) + '_pe'


class CouplingWriter:
    """Write the wave<->ocean coupling fields to a netCDF file, one record
    per written coupling step (the loop gates the cadence). Runs only on
    wave-rank-0, which holds the gathered full-domain fields.
      ustar_x, ustar_y   : friction-velocity vector u* (air side)     (t,y,x)
      u_stokes, v_stokes : SURFACE Stokes drift                       (t,y,x)
                           (the profile is dropped -- it dominated the file
                            size; the ocean gets it via MPI and the restart
                            keeps the 3-D u_S)
      stress_x, stress_y : wave -> ocean stress tau_o                 (t,y,x)
                           (or (t,z,y,x) with transfer_mode='stress_work')
      stress_in_x/y      : wind -> wave input stress tau_in           (t,y,x)
      phi_oc             : wave -> ocean energy flux                  (t,y,x)
      plus one scalar per record for each domain-integrated diagnostic.
    """
    def __init__(self, fname, nx, ny, nz, xt, yt, stress_is_3d, dt_cpl=0.0,
                 zt=None):
        from netCDF4 import Dataset
        self.f = Dataset(fname, 'w', format='NETCDF4')
        self.f.createDimension('time', None)
        try:
            self.f.setncattr('dt_cpl', float(dt_cpl))
        except Exception:
            pass
        self.f.createDimension('x', nx)
        self.f.createDimension('y', ny)
        self.f.createDimension('z', nz)
        self.f.createVariable('time', 'f8', ('time',))
        self.f.createVariable('x', 'f8', ('x',))[:] = np.asarray(xt)
        self.f.createVariable('y', 'f8', ('y',))[:] = np.asarray(yt)
        self.f.createVariable('z', 'f8', ('z',))[:] = (np.arange(nz) if zt is None
                                                       else np.asarray(zt))
        self.f.createVariable('ustar_x', 'f4', ('time', 'y', 'x'))
        self.f.createVariable('ustar_y', 'f4', ('time', 'y', 'x'))
        v = self.f.createVariable('u_stokes', 'f4', ('time', 'y', 'x'))
        v.long_name = 'surface Stokes drift (top level only)'
        v = self.f.createVariable('v_stokes', 'f4', ('time', 'y', 'x'))
        v.long_name = 'surface Stokes drift (top level only)'
        sdims = ('time', 'z', 'y', 'x') if stress_is_3d else ('time', 'y', 'x')
        self.f.createVariable('stress_x', 'f4', sdims)
        self.f.createVariable('stress_y', 'f4', sdims)
        v = self.f.createVariable('stress_in_x', 'f4', ('time', 'y', 'x'))
        v.units = 'm^2 s^-2'; v.long_name = 'wind to wave input stress, x'
        v = self.f.createVariable('stress_in_y', 'f4', ('time', 'y', 'x'))
        v.units = 'm^2 s^-2'; v.long_name = 'wind to wave input stress, y'
        v = self.f.createVariable('phi_oc', 'f4', ('time', 'y', 'x'))
        v.units = 'm^3 s^-3'
        v.long_name = 'wave-to-ocean energy flux -g int S_diss dOmega'
        # domain-integrated diagnostics, one scalar per record, kinematic
        # (per unit rho): multiply by rho_w for Watts.
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
                ('stress_work_eul','wave-stress work against Eulerian u = u_L - u_S (comparison)'),
                ('snl_work',       'reversible exchange int U.k S_nl dk (Eq. 11 non-closure term)'),
                ('exchange_C',     'exchange work C: applied-increment work of d(u_S)/dt on u_L'),
                ('vortex_residual','vortex-force u_L-work residual (round-off monitor)'),
                ('dtA_work',       'wave-side exchange int U.k dA/dt dk dA (compare exchange_C)'),
                ('dke_dt',         'd/dt of the ocean KE int |u_L|^2/2 dV, this step'),
                ('vfric_work',     'KE change rate by vertical friction (must equal -kdiss_int)'),
                ('kdiss_int',      'int K_diss_v dV: shear production handed to the TKE'),
                ('hfric_work',     'KE change rate by horizontal friction'),
                ('vfric_num',      'numerical dissipation |u+ - u|^2/(2 dt) of the implicit friction: vfric_work = -kdiss_int - vfric_num')):
            v = self.f.createVariable(name, 'f8', ('time',))
            v.units = 'm^5 s^-3'
            v.long_name = long_name
        v = self.f.createVariable('ke_ocean', 'f8', ('time',))
        v.units = 'm^5 s^-2'; v.long_name = 'ocean kinetic energy int |u_L|^2/2 dV'
        self.stress_is_3d = stress_is_3d
        self.n = 0

    def write(self, t, ustar_x, ustar_y, u_S, stress, phioc=None, ediag=None,
              mdiag=None, stress_in=None, kediag=None):
        k = self.n
        self.f['time'][k]     = t
        self.f['ustar_x'][k]  = ustar_x
        self.f['ustar_y'][k]  = ustar_y
        self.f['u_stokes'][k] = u_S[0][-1]              # surface level only
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
            self.f['stress_work'][k]     = wst
            self.f['stress_work_eul'][k] = wste
            self.f['exchange_C'][k]      = cxc
            self.f['vortex_residual'][k] = wvf
            self.f['snl_work'][k]        = wnl
        if kediag is not None:
            for nm, v in zip(('ke_ocean', 'dke_dt', 'vfric_work', 'kdiss_int',
                              'hfric_work', 'dtA_work', 'vfric_num'), kediag):
                self.f[nm][k] = v
        if mdiag is not None:
            for nm, v in zip(('mom_in_x', 'mom_in_y', 'mom_del_x',
                              'mom_del_y', 'mom_wave_x', 'mom_wave_y'), mdiag):
                self.f[nm][k] = v
        self.f.sync()
        self.n += 1

    def close(self):
        self.f.close()


# ===========================================================================
# restarts
# ===========================================================================
def _nc_create(fname, dims):
    from netCDF4 import Dataset
    f = Dataset(fname, 'w', format='NETCDF4')
    for dname, dlen in dims:
        f.createDimension(dname, dlen)
    return f


def _replace(fname, tmp, what, n_cpl):
    if os.path.exists(fname):
        os.replace(fname, fname + '.prev')   # rolling backup: the previous
                                             # pair survives one interval
    os.replace(tmp, fname)
    print('wrote %s restart %s at coupling step %d' % (what, fname, n_cpl),
          flush=True)


# --------------------------------------------------------------------- wave
# RAW wave restart, same switch (EXP_RESTART_RAW) as the ocean one. The
# gathered form pulls A (nd*nf*ny*nx) and its three tendency levels onto wave
# rank 0: at 1/3 degree with nf=30, nd=20 that is 3.6 GB plus 10.7 GB, block
# by block, then written serially. The raw form writes each rank's PADDED
# arrays to <base>_pe/<rank>.npz.
#
# Storing A padded is why there is no separate A_fghost here: the padded
# array already contains the high-frequency ghost slab imphftail maintains,
# and the theta and x/y halos with it, so the read restores exactly the state
# the write saw and no apply_bc is needed.
# ---------------------------------------------------------------------------
def _wave_restart_write_raw(M, fname, n_cpl, pre, tau, taum1, taum2,
                            tA, tB, tC):
    d = _raw_dir(fname)
    if M.my_pe == 0:
        os.makedirs(d, exist_ok=True)
    M.comm.Barrier()
    dat = {'A': np.asarray(M.A), 'dA': np.asarray(M.dA)}
    has_cur = int(hasattr(M, 'ocean_U'))
    if has_cur:
        dat['ocean_U'], dat['ocean_V'] = np.asarray(M.ocean_U), np.asarray(M.ocean_V)
        # ocean_Udf/Vdf are formed from ocean_U/V at the NEXT coupling step, so
        # after exactly one step they do not exist yet: store zeros (their
        # state), the restarted run rebuilds them from ocean_U as it would have
        if hasattr(M, 'ocean_Udf'):
            dat['ocean_Udf'] = np.asarray(M.ocean_Udf)
            dat['ocean_Vdf'] = np.asarray(M.ocean_Vdf)
        else:
            dat['ocean_Udf'] = np.zeros_like(dat['ocean_U'])
            dat['ocean_Vdf'] = np.zeros_like(dat['ocean_U'])
    # JANSSEN INPUT STATE (wim_realistic/source.py, input_scheme='janssen'):
    # tau_w and z0 of the previous step seed the z0 <-> tau_w iteration of the
    # next one. Without them a restart begins from Charnock z0 = alpha u*^2/g.
    has_jan = int(hasattr(M, 'tau_w') and hasattr(M, 'z0'))
    if has_jan:
        dat['tau_w'], dat['z0'] = np.asarray(M.tau_w), np.asarray(M.z0)
    tmp = os.path.join(d, '%05d.tmp.npz' % M.my_pe)
    np.savez(tmp, **dat)
    os.replace(tmp, os.path.join(d, '%05d.npz' % M.my_pe))
    M.comm.Barrier()
    if M.my_pe == 0:
        t = fname + '.tmp'
        f = _nc_create(t, [('one', 1)])
        f.setncatts(dict(n_cpl=n_cpl, pre=pre, tau=tau, taum1=taum1,
                         taum2=taum2, tA=tA, tB=tB, tC=tC, has_cur=has_cur,
                         has_jan=has_jan, raw=1, n_pes=M.n_pes,
                         n_pes_x=M.n_pes_x, n_pes_y=M.n_pes_y))
        f.close()
        _replace(fname, t, 'wave (raw, %d blocks in %s)' % (M.n_pes, d), n_cpl)
    return


def _wave_restart_read_raw(M, fname, comm):
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in
               ('n_cpl', 'pre', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC',
                'has_cur')}
        att['has_jan'] = int(getattr(f, 'has_jan', 0))
        att['raw'] = int(getattr(f, 'raw', 0))
        att['layout'] = (int(getattr(f, 'n_pes', -1)), int(getattr(f, 'n_pes_x', -1)),
                         int(getattr(f, 'n_pes_y', -1)))
        f.close()
    else:
        att = None
    att = comm.bcast(att, root=0)
    if att['layout'] != (M.n_pes, M.n_pes_x, M.n_pes_y):
        raise RuntimeError(
            'raw wave restart %s was written on %d ranks (%d x %d), this run '
            'has %d (%d x %d). A raw restart belongs to its decomposition: run '
            'the same wave layout, or write the restart gathered '
            '(EXP_RESTART_RAW=0).'
            % ((fname,) + att['layout'] + (M.n_pes, M.n_pes_x, M.n_pes_y)))
    R = np.load(os.path.join(_raw_dir(fname), '%05d.npz' % M.my_pe))
    M.A  = OM.np.asarray(R['A'])         # padded: ghost slab and halos included
    M.dA = OM.np.asarray(R['dA'])
    if att['has_cur']:
        for name in ('ocean_U', 'ocean_V', 'ocean_Udf', 'ocean_Vdf'):
            setattr(M, name, OM.np.asarray(R[name]))
    if att['has_jan']:
        M.tau_w, M.z0 = OM.np.asarray(R['tau_w']), OM.np.asarray(R['z0'])
    elif getattr(M, 'input_scheme', 'wamdi') == 'janssen' and M.my_pe == 0:
        print('wave restart has no tau_w/z0: the Janssen roughness starts '
              'from Charnock (one-step transient in gamma_in)', flush=True)
    M.E = M.A*M.omega_t[None, :, None, None]
    if M.my_pe == 0:
        print('read raw wave restart %s (%d blocks): resuming at coupling '
              'step %d' % (fname, M.n_pes, att['n_cpl']), flush=True)
    return att


def wave_restart_write(M, fname, ny, nx, n_cpl, pre, tau, taum1, taum2,
                       tA, tB, tC):
    if RESTART_RAW:
        return _wave_restart_write_raw(M, fname, n_cpl, pre, tau, taum1,
                                       taum2, tA, tB, tC)
    h_  = M.halo
    sl  = slice(h_, -h_)
    A_g  = gather_unpadded_mpi(M, M.A[sl, sl, sl, sl], ny, nx)
    dA_g = gather_unpadded_mpi(M, M.dA[:, sl, sl, sl, sl], ny, nx)
    # HIGH-FREQUENCY GHOST SLAB. imphftail maintains the imposed f^-5
    # continuation in the frequency ghost bins every source step BY DESIGN,
    # and both the f-advection boundary flux and every tail diagnostic
    # (calc_stress/_phioc/_tail_input_factors anchor at bin -halo) consume
    # exactly those values. Storing only the interior bins -- as the acc1
    # restart does -- leaves them zero after the read, so the first
    # post-restart step advects against a zero anchor. Measured cost of
    # omitting it here: a ~1e-8 relative seed in A which convective branch
    # flips near the shelf amplify to ~1e-5 in the ocean within six hours.
    # The ghost's own x/y halo is NOT stored: it is never consumed (the
    # interior f-flux reads the ghost at its own column).
    gh_g = gather_unpadded_mpi(M, M.A[:, -h_:, sl, sl], ny, nx)
    has_cur = int(hasattr(M, 'ocean_U'))
    if has_cur:
        oU  = gather_unpadded_mpi(M, M.ocean_U[sl, sl, sl],   ny, nx)
        oV  = gather_unpadded_mpi(M, M.ocean_V[sl, sl, sl],   ny, nx)
        # ocean_Udf/Vdf are formed from ocean_U/V at the NEXT coupling step, so
        # after exactly one step they do not exist yet: store zeros (their
        # state), the restarted run rebuilds them from ocean_U as it would have
        if hasattr(M, 'ocean_Udf'):
            oUf = gather_unpadded_mpi(M, M.ocean_Udf[sl, sl, sl], ny, nx)
            oVf = gather_unpadded_mpi(M, M.ocean_Vdf[sl, sl, sl], ny, nx)
        else:
            z = OM.np.zeros_like(M.ocean_U[sl, sl, sl])
            oUf = gather_unpadded_mpi(M, z, ny, nx)
            oVf = gather_unpadded_mpi(M, z, ny, nx)
    # JANSSEN INPUT STATE (wim_realistic/source.py, input_scheme='janssen'):
    # tau_w and z0 of the previous step seed the z0 <-> tau_w iteration of the
    # next one. Without them a restart begins from Charnock z0 = alpha u*^2/g,
    # a one-step transient in gamma_in; with them the restart is exact.
    has_jan = int(hasattr(M, 'tau_w') and hasattr(M, 'z0'))
    if has_jan:
        tw_g = gather_unpadded_mpi(M, M.tau_w[sl, sl], ny, nx)
        z0_g = gather_unpadded_mpi(M, M.z0[sl, sl],    ny, nx)
    if M.my_pe == 0:
        nd_, nf_ = A_g.shape[0], A_g.shape[1]
        tmp = fname + '.tmp'
        f = _nc_create(tmp, [('three', 3), ('d', nd_), ('f', nf_),
                             ('y', ny), ('x', nx),
                             ('dpad', gh_g.shape[0]), ('fh', gh_g.shape[1])])
        f.createVariable('A',  'f8', ('d', 'f', 'y', 'x'))[:]          = A_g
        f.createVariable('dA', 'f8', ('three', 'd', 'f', 'y', 'x'))[:] = dA_g
        f.createVariable('A_fghost', 'f8', ('dpad', 'fh', 'y', 'x'))[:] = gh_g
        if has_cur:
            f.createVariable('ocean_U',   'f8', ('f', 'y', 'x'))[:] = oU
            f.createVariable('ocean_V',   'f8', ('f', 'y', 'x'))[:] = oV
            f.createVariable('ocean_Udf', 'f8', ('f', 'y', 'x'))[:] = oUf
            f.createVariable('ocean_Vdf', 'f8', ('f', 'y', 'x'))[:] = oVf
        if has_jan:
            f.createVariable('tau_w', 'f8', ('y', 'x'))[:] = tw_g
            f.createVariable('z0',    'f8', ('y', 'x'))[:] = z0_g
        f.setncatts(dict(n_cpl=n_cpl, pre=pre, tau=tau, taum1=taum1,
                         taum2=taum2, tA=tA, tB=tB, tC=tC, has_cur=has_cur,
                         has_jan=has_jan))
        f.close()
        _replace(fname, tmp, 'wave', n_cpl)
    return


def _is_raw(fname, comm, my_pe):
    """Which form a restart file on disk is, decided by the file itself and
    not by EXP_RESTART_RAW: the env switch controls WRITING only. So a run
    that writes raw restarts can still branch from a gathered EXP_INIT_OCEAN
    of another decomposition, and vice versa. Collective."""
    raw = None
    if my_pe == 0:
        from netCDF4 import Dataset
        with Dataset(fname) as f:
            raw = int(getattr(f, 'raw', 0))
    return bool(comm.bcast(raw, root=0))


def wave_restart_read(M, fname, ny, nx, comm):
    if _is_raw(fname, comm, M.my_pe):
        return _wave_restart_read_raw(M, fname, comm)
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in
               ('n_cpl', 'pre', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC',
                'has_cur')}
        A_g  = np.array(f['A'][:]);  dA_g = np.array(f['dA'][:])
        gh_g = np.array(f['A_fghost'][:]) if 'A_fghost' in f.variables else None
        att['_has_ghost'] = gh_g is not None
        cur  = ([np.array(f[v][:]) for v in
                 ('ocean_U', 'ocean_V', 'ocean_Udf', 'ocean_Vdf')]
                if att['has_cur'] else None)
        att['has_jan'] = int(f.getncattr('has_jan')) if 'has_jan' in f.ncattrs() else 0
        jan  = ([np.array(f[v][:]) for v in ('tau_w', 'z0')]
                if att['has_jan'] else None)
        f.close()
    else:
        att, A_g, dA_g, gh_g, cur, jan = None, None, None, None, None, None
    att = comm.bcast(att, root=0)
    sl  = slice(M.halo, -M.halo)
    nd_ = M.A.shape[0] - 2*M.halo
    nf_ = M.A.shape[1] - 2*M.halo
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
    if att['has_jan']:
        for name, full in zip(('tau_w', 'z0'), jan if M.my_pe == 0 else [None]*2):
            blk = scatter_array_mpi(M, full, ())
            setattr(M, name, M.apply_bc(np.pad(blk, M.halo)))
    elif getattr(M, 'input_scheme', 'wamdi') == 'janssen' and M.my_pe == 0:
        print('wave restart has no tau_w/z0: the Janssen roughness starts '
              'from Charnock (one-step transient in gamma_in)', flush=True)
    M.E = M.A*M.omega_t[None, :, None, None]
    if M.my_pe == 0:
        print('read wave restart %s: resuming at coupling step %d'
              % (fname, att['n_cpl']), flush=True)
    return att


# -------------------------------------------------------------------- ocean
_OCEAN_3D = ('u', 'v', 'temp', 'salt')
_OCEAN_4D = ('du', 'dv', 'dtemp', 'dsalt')


def _vadv_switch(M, att):
    """dtemp, dsalt hold advection + forcing in the unsplit model and the
    forcing only with vadv_sub_cycling > 0 (pe_realistic/vadv_split.py). When
    the restart was written in the other mode the old tendency levels do not
    belong to this run: drop them. The first step then misses the -0.6
    weighted old FORCING once (1e-3 K in the top cell), instead of applying a
    -0.6 weighted old ADVECTION to every cell."""
    was, now = att.get('vadv', 0) > 0, getattr(M, 'vadv_sub_cycling', 0) > 0
    if was != now:
        M.dtemp, M.dsalt = 0.*M.dtemp, 0.*M.dsalt
        if M.my_pe == 0:
            print('ocean restart was written with vadv_sub_cycling %s, this run has '
                  'it %s: old tracer tendency levels dropped (one-step transient)'
                  % ('on' if was else 'off', 'on' if now else 'off'), flush=True)
    return


def _ocean_fields_raw(M, use_stress_work):
    """name -> padded local array of everything the ocean restart carries.
    Same content as the gathered form, block by block instead of assembled."""
    out = {n: np.asarray(getattr(M, n)) for n in _OCEAN_3D + _OCEAN_4D}
    out['p_s'] = np.asarray(M.p_s)
    out['u_S'], out['v_S'] = np.asarray(M.u_S), np.asarray(M.v_S)
    for n in ('u_S_old', 'v_S_old', 'tke', 'K_diss_v', 'phi_oc',
              'tau_in_x', 'tau_in_y', 'hs'):
        if hasattr(M, n):
            out[n] = np.asarray(getattr(M, n))
    if use_stress_work:
        out['stress_x'], out['stress_y'] = np.asarray(M.tau_work_x), np.asarray(M.tau_work_y)
    else:
        out['stress_x'], out['stress_y'] = np.asarray(M.taux), np.asarray(M.tauy)
    return out


def _ocean_restart_write_raw(M, fname, use_stress_work, n_cpl,
                             tau, taum1, taum2, tA, tB, tC):
    """Every rank writes its own block; rank 0 writes the header LAST, so a
    header that exists is always backed by a complete set of blocks."""
    d = _raw_dir(fname)
    if M.my_pe == 0:
        os.makedirs(d, exist_ok=True)
    M.mpi_barrier()
    dat = _ocean_fields_raw(M, use_stress_work)
    avg = M.average_state() if hasattr(M, 'average_state') else None
    if avg is not None:
        for nme, a in avg[0]:
            dat['avg_' + nme] = np.asarray(a)
    # the near-inertial diagnostic's filter states (ni_diag.py)
    ni = M.ni_state() if hasattr(M, 'ni_state') else None
    if ni is not None:
        for nme, a in ni.items():
            dat['ni_' + nme] = np.asarray(a)
    # np.savez appends '.npz' to a name that has no such suffix, so the
    # temporary must already carry it
    tmp = os.path.join(d, '%05d.tmp.npz' % M.my_pe)
    np.savez(tmp, **dat)
    os.replace(tmp, os.path.join(d, '%05d.npz' % M.my_pe))
    M.mpi_barrier()
    if M.my_pe == 0:
        t = fname + '.tmp'
        f = _nc_create(t, [('one', 1)])
        f.setncatts(dict(n_cpl=n_cpl, tau=tau, taum1=taum1, taum2=taum2,
                         tA=tA, tB=tB, tC=tC,
                         use_stress_work=int(use_stress_work),
                         vadv=int(getattr(M, 'vadv_sub_cycling', 0)),
                         raw=1, n_pes=M.n_pes,
                         n_pes_x=M.n_pes_x, n_pes_y=M.n_pes_y))
        if avg is not None:
            f.setncatts(dict(avg_n=avg[1], avg_t0=avg[2]))
        f.close()
        _replace(fname, t, 'ocean (raw, %d blocks in %s)' % (M.n_pes, d), n_cpl)
    return


def _ocean_restart_read_raw(M, fname, use_stress_work, comm, restore_average=True):
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in
               ('n_cpl', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC')}
        att['raw'] = int(getattr(f, 'raw', 0))
        att['vadv'] = int(getattr(f, 'vadv', 0))
        att['layout'] = (int(getattr(f, 'n_pes', -1)), int(getattr(f, 'n_pes_x', -1)),
                         int(getattr(f, 'n_pes_y', -1)))
        att['avg_n'] = int(getattr(f, 'avg_n', 0))
        att['avg_t0'] = float(getattr(f, 'avg_t0', -1.0))
        f.close()
    else:
        att = None
    att = comm.bcast(att, root=0)
    if att['layout'] != (M.n_pes, M.n_pes_x, M.n_pes_y):
        raise RuntimeError(
            'raw ocean restart %s was written on %d ranks (%d x %d), this run '
            'has %d (%d x %d). A raw restart belongs to its decomposition: run '
            'the same ocean layout, or write the restart gathered '
            '(EXP_RESTART_RAW=0).'
            % ((fname,) + att['layout'] + (M.n_pes, M.n_pes_x, M.n_pes_y)))
    R = np.load(os.path.join(_raw_dir(fname), '%05d.npz' % M.my_pe))
    have = set(R.files)
    for n in _OCEAN_3D + _OCEAN_4D + ('p_s', 'u_S', 'v_S'):
        setattr(M, n, OM.np.asarray(R[n]))
    for n in ('u_S_old', 'v_S_old', 'tke', 'K_diss_v', 'phi_oc',
              'tau_in_x', 'tau_in_y', 'hs'):
        if n in have and (n in ('u_S_old', 'v_S_old') or hasattr(M, n)):
            setattr(M, n, OM.np.asarray(R[n]))
    if use_stress_work:
        M.tau_work_x, M.tau_work_y = OM.np.asarray(R['stress_x']), OM.np.asarray(R['stress_y'])
    else:
        M.taux, M.tauy = OM.np.asarray(R['stress_x']), OM.np.asarray(R['stress_y'])
    if 'K_diss_v' not in have and M.my_pe == 0 and hasattr(M, 'K_diss_v'):
        print('ocean restart has no K_diss_v: first step uses Pr = 10 in the '
              'TKE diffusivities (one-step transient)', flush=True)
    sums = {n[4:]: R[n] for n in R.files if n.startswith('avg_')}
    if sums and restore_average and hasattr(M, 'average_restore'):
        M.average_restore(sums, att['avg_n'], att['avg_t0'])
    ni = {n[3:]: R[n] for n in R.files if n.startswith('ni_')}
    if ni and hasattr(M, 'ni_restore'):
        M.ni_restore(ni)
    _vadv_switch(M, att)
    from pyOM3.density import get_rho
    M.rho = get_rho(M.salt, M.temp, M.p0, M.eq_of_state)
    M.w   = M.vertical_velocity(M.u, M.v, M.w, M.maskW)
    if M.my_pe == 0:
        print('read raw ocean restart %s (%d blocks): resuming at coupling '
              'step %d' % (fname, M.n_pes, att['n_cpl']), flush=True)
    return att


def ocean_restart_write(M, fname, ny, nx, use_stress_work, n_cpl,
                        tau, taum1, taum2, tA, tB, tC):
    if RESTART_RAW:
        return _ocean_restart_write_raw(M, fname, use_stress_work, n_cpl,
                                        tau, taum1, taum2, tA, tB, tC)
    h  = M.halo
    sl = slice(h, -h)
    g3 = lambda a: gather_unpadded_mpi(M, a[sl, sl, sl], ny, nx)
    g4 = lambda a: gather_unpadded_mpi(M, a[:, sl, sl, sl], ny, nx)
    g2 = lambda a: gather_unpadded_mpi(M, a[sl, sl], ny, nx)
    dat = {n: g3(getattr(M, n)) for n in _OCEAN_3D}
    dat.update({n: g4(getattr(M, n)) for n in _OCEAN_4D})
    # p_s carries THREE time levels (3,Ny,Nx), like pyOM2's psi(:,:,tau..):
    # gather with the time axis as a leading dim, not as y.
    ps_g = gather_unpadded_mpi(M, M.p_s[:, sl, sl], ny, nx)
    dat['u_S'], dat['v_S'] = g3(M.u_S), g3(M.v_S)
    # d(u_S)/dt forcing state: the PREVIOUS coupling's Stokes drift. Without
    # it the first post-restart step applies no d(u_S)/dt increment (a
    # one-interval error in u_L and in C). Presence is uniform across PEs
    # (set in the coupler loop after the first coupling), so the hasattr
    # guard is collective-safe.
    has_uSo = int(hasattr(M, 'u_S_old'))
    if has_uSo:
        dat['u_S_old'], dat['v_S_old'] = g3(M.u_S_old), g3(M.v_S_old)
    has_tke = int(hasattr(M, 'tke'))
    if has_tke:
        dat['tke'] = g3(M.tke)
    # K_diss_v (shear production of the LAST step) is STATE: the next step's
    # set_tke_diffusivities runs first and builds the Richardson number, hence
    # kappaH, from it. Without it the first post-restart step has Ri -> inf,
    # Pr = 10, kappaH = kappaM/10 everywhere (6e-2 K in temp after one step).
    has_kdv = int(hasattr(M, 'K_diss_v'))
    if has_kdv:
        dat['K_diss_v'] = g3(M.K_diss_v)
    if use_stress_work:
        dat['stress_x'], dat['stress_y'] = g3(M.tau_work_x), g3(M.tau_work_y)
    else:
        dat['stress_x'], dat['stress_y'] = g2(M.taux), g2(M.tauy)
    dat['phi_oc'] = g2(M.phi_oc)                  # genuinely 2-D (unlike p_s!)
    dat['tau_in_x'], dat['tau_in_y'] = g2(M.tau_in_x), g2(M.tau_in_y)
    if hasattr(M, 'hs'):
        dat['hs'] = g2(M.hs)                      # significant wave height
    # partial sums of the online time means (ocean.average_state; None when
    # the means are off). Same configuration on all PEs -> collective-safe.
    avg = M.average_state() if hasattr(M, 'average_state') else None
    if avg is not None:
        for nme, a in avg[0]:
            dat['avg_' + nme] = gather_unpadded_mpi(M, a, ny, nx)
    if M.my_pe == 0:
        nz_ = dat['u'].shape[0]
        tmp = fname + '.tmp'
        f = _nc_create(tmp, [('three', 3), ('z', nz_), ('y', ny), ('x', nx)])
        d3, d4, d2 = ('z', 'y', 'x'), ('three', 'z', 'y', 'x'), ('y', 'x')
        dims = {}
        for n in _OCEAN_3D: dims[n] = d3
        for n in _OCEAN_4D: dims[n] = d4
        for n in ('u_S', 'v_S', 'u_S_old', 'v_S_old', 'tke', 'K_diss_v'):
            dims[n] = d3
        for n in ('phi_oc', 'tau_in_x', 'tau_in_y'): dims[n] = d2
        dims['stress_x'] = dims['stress_y'] = d3 if use_stress_work else d2
        dims['hs'] = d2
        for nme in dat:
            if nme.startswith('avg_'):
                dims[nme] = d3 if dat[nme].ndim == 3 else d2
        for nme, arr in dat.items():
            f.createVariable(nme, 'f8', dims[nme])[:] = arr
        f.createVariable('p_s', 'f8', ('three', 'y', 'x'))[:] = ps_g
        f.setncatts(dict(n_cpl=n_cpl, tau=tau, taum1=taum1, taum2=taum2,
                         tA=tA, tB=tB, tC=tC,
                         use_stress_work=int(use_stress_work),
                         vadv=int(getattr(M, 'vadv_sub_cycling', 0))))
        if avg is not None:
            f.setncatts(dict(avg_n=avg[1], avg_t0=avg[2]))
        f.close()
        _replace(fname, tmp, 'ocean', n_cpl)
    return


def ocean_restart_read(M, fname, ny, nx, use_stress_work, comm,
                       restore_average=True):
    if _is_raw(fname, comm, M.my_pe):
        return _ocean_restart_read_raw(M, fname, use_stress_work, comm,
                                       restore_average)
    names = list(_OCEAN_3D) + list(_OCEAN_4D) + \
            ['p_s', 'u_S', 'v_S', 'stress_x', 'stress_y', 'phi_oc',
             'tau_in_x', 'tau_in_y']
    optional = ['tke', 'K_diss_v', 'u_S_old', 'v_S_old', 'hs']
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in
               ('n_cpl', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC')}
        dat = {v: np.array(f[v][:]) for v in names}
        for v in optional:
            if v in f.variables:
                dat[v] = np.array(f[v][:])
        # presence of the optional fields is a COLLECTIVE decision: it must
        # be broadcast, not inferred rank-locally (otherwise non-root ranks
        # skip a scatter the root participates in -> MPI_ERR_TRUNCATE when
        # root's blocks land in the next exchange's smaller buffers)
        att['vadv'] = int(getattr(f, 'vadv', 0))
        att['_has_tke'] = 'tke' in dat
        att['_has_kdv'] = 'K_diss_v' in dat
        att['_has_uSo'] = 'u_S_old' in dat
        att['_has_hs']  = 'hs' in dat
        # partial sums of the online time means, if the file has them
        att['_avg'] = {v: f[v].ndim for v in f.variables if v.startswith('avg_')}
        for v in att['_avg']:
            dat[v] = np.array(f[v][:])
        if att['_avg']:
            att['avg_n'], att['avg_t0'] = int(f.getncattr('avg_n')), float(f.getncattr('avg_t0'))
        f.close()
    else:
        att = None
        dat = {v: None for v in names + optional}
    att = comm.bcast(att, root=0)
    for v in att['_avg']:
        dat.setdefault(v, None)
    h   = M.halo
    sl  = slice(h, -h)
    nz_ = M.u.shape[0] - 2*h

    def r3(name, tgt):
        blk = scatter_array_mpi(M, dat[name], (nz_,))
        return M.apply_bc(OM.modify_array(tgt, (sl, sl, sl), blk))

    def r4(name, tgt):
        blk = scatter_array_mpi(M, dat[name], (3, nz_))
        return OM.modify_array(tgt, (slice(None), sl, sl, sl), blk)

    def r2(name, tgt):
        blk = scatter_array_mpi(M, dat[name], ())
        return M.apply_bc(OM.modify_array(tgt, (sl, sl), blk))

    for n in _OCEAN_3D:  setattr(M, n, r3(n, getattr(M, n)))
    for n in _OCEAN_4D:  setattr(M, n, r4(n, getattr(M, n)))
    # p_s: three time levels (3,Ny,Nx). Must NOT go through apply_bc as one
    # 3-axis array: apply_bc_mpi sizes its recv buffer as (Nz,Nx) for ANY
    # 3-axis field. Exchange each time level separately as a 2-D field.
    blk = scatter_array_mpi(M, dat['p_s'], (3,))
    for k in range(3):
        M.p_s = OM.modify_array(M.p_s, (k, sl, sl), blk[k])
        M.p_s = OM.modify_array(M.p_s, (k, slice(None), slice(None)),
                                M.apply_bc(M.p_s[k]))
    M.u_S, M.v_S = r3('u_S', M.u_S), r3('v_S', M.v_S)
    if att.get('_has_uSo', False):
        M.u_S_old = r3('u_S_old', OM.np.zeros_like(M.u_S))
        M.v_S_old = r3('v_S_old', OM.np.zeros_like(M.v_S))
    elif M.my_pe == 0 and getattr(M, 'coupled', True):
        print('ocean restart has no u_S_old: first step applies no d(u_S)/dt '
              'forcing (one coupling-interval transient)', flush=True)
    if att.get('_has_tke', False) and hasattr(M, 'tke'):
        M.tke = r3('tke', M.tke)     # absent in older restarts -> keep the
                                     # init_tke seed (respins in hours)
    if att.get('_has_kdv', False) and hasattr(M, 'K_diss_v'):
        M.K_diss_v = r3('K_diss_v', M.K_diss_v)
    elif M.my_pe == 0 and hasattr(M, 'K_diss_v'):
        print('ocean restart has no K_diss_v: first step uses Pr = 10 in the '
              'TKE diffusivities (one-step transient)', flush=True)
    if use_stress_work:
        M.tau_work_x = r3('stress_x', M.tau_work_x)
        M.tau_work_y = r3('stress_y', M.tau_work_y)
    else:
        M.taux, M.tauy = r2('stress_x', M.taux), r2('stress_y', M.tauy)
    M.phi_oc = r2('phi_oc', M.phi_oc)
    M.tau_in_x, M.tau_in_y = r2('tau_in_x', M.tau_in_x), r2('tau_in_y', M.tau_in_y)
    if hasattr(M, 'hs') and att.get('_has_hs', False):
        M.hs = r2('hs', M.hs)
    if att['_avg']:
        # every PE takes part in the scatter, also when this run averages
        # nothing (average_restore then ignores the sums)
        sums = {v[4:]: scatter_array_mpi(M, dat[v], (nz_,) if nd == 3 else ())
                for v, nd in sorted(att['_avg'].items())}
        if restore_average and hasattr(M, 'average_restore'):
            M.average_restore(sums, att['avg_n'], att['avg_t0'])
    # rho and w are exact functions of the restored state; recomputing them
    # is cheaper and safer than carrying them in the file.
    _vadv_switch(M, att)
    from pyOM3.density import get_rho
    M.rho = get_rho(M.salt, M.temp, M.p0, M.eq_of_state)
    M.w   = M.vertical_velocity(M.u, M.v, M.w, M.maskW)
    if M.my_pe == 0:
        print('read ocean restart %s: resuming at coupling step %d'
              % (fname, att['n_cpl']), flush=True)
    return att


# ======================= the ATMOSPHERE of the three-component coupler =======================
# pe_realistic restart of the atmosphere (gathered netCDF, the atmosphere is
# small), and its coupling record. From the HS coupler, unchanged in content.

_PE_3D = ('u', 'v', 'temp', 'salt')
_PE_4D = ('du', 'dv', 'dtemp', 'dsalt')


def atmos_restart_write(M, fname, ny, nx, n_cpl, tau, taum1, taum2, ab, fields2d, what, fields3d=()):
    """fields2d: names of 2-D (Ny,Nx) attributes of M to store (coupling
    fields); ab = (A,B,C) Adams-Bashforth weights in force."""
    h = M.halo; sl = slice(h, -h)
    g3 = lambda a: gather_unpadded_mpi(M, np.asarray(a)[sl, sl, sl], ny, nx)
    g4 = lambda a: gather_unpadded_mpi(M, np.asarray(a)[:, sl, sl, sl], ny, nx)
    g2 = lambda a: gather_unpadded_mpi(M, np.asarray(a)[sl, sl], ny, nx)
    dat = {n: g3(getattr(M, n)) for n in _PE_3D}
    dat.update({n: g4(getattr(M, n)) for n in _PE_4D})
    ps_g = gather_unpadded_mpi(M, np.asarray(M.p_s)[:, sl, sl], ny, nx)
    has_tke = hasattr(M, 'tke')
    if has_tke: dat['tke'] = g3(M.tke)
    for n in fields2d: dat[n] = g2(getattr(M, n))
    for n in fields3d: dat[n] = g3(getattr(M, n))
    if M.my_pe == 0:
        nz_ = dat['u'].shape[0]
        tmp = fname + '.tmp'
        f = _nc_create(tmp, [('three', 3), ('z', nz_), ('y', ny), ('x', nx)])
        d3, d4, d2 = ('z', 'y', 'x'), ('three', 'z', 'y', 'x'), ('y', 'x')
        for n in _PE_3D: f.createVariable(n, 'f8', d3)[:] = dat[n]
        for n in _PE_4D: f.createVariable(n, 'f8', d4)[:] = dat[n]
        if has_tke: f.createVariable('tke', 'f8', d3)[:] = dat['tke']
        for n in fields2d: f.createVariable(n, 'f8', d2)[:] = dat[n]
        for n in fields3d: f.createVariable(n, 'f8', d3)[:] = dat[n]
        f.createVariable('p_s', 'f8', ('three', 'y', 'x'))[:] = ps_g
        f.setncatts(dict(n_cpl=n_cpl, tau=tau, taum1=taum1, taum2=taum2,
                         tA=ab[0], tB=ab[1], tC=ab[2], fields2d=' '.join(fields2d), fields3d=' '.join(fields3d)))
        f.close()
        _replace(fname, tmp, what, n_cpl)


def atmos_restart_read(M, fname, ny, nx, comm, fields2d, what, fields3d=()):
    names = list(_PE_3D) + list(_PE_4D) + ['p_s'] + list(fields2d) + list(fields3d)
    if M.my_pe == 0:
        from netCDF4 import Dataset
        f = Dataset(fname)
        att = {k: f.getncattr(k) for k in ('n_cpl', 'tau', 'taum1', 'taum2', 'tA', 'tB', 'tC')}
        # coupling fields absent from an older restart are skipped (collective
        # decision: the list of missing names is broadcast with att)
        att['_missing'] = [v for v in names if v not in f.variables]
        dat = {v: np.array(f[v][:]) for v in names if v not in att['_missing']}
        att['_has_tke'] = 'tke' in f.variables
        if att['_has_tke']: dat['tke'] = np.array(f['tke'][:])
        f.close()
    else:
        att = None; dat = {v: None for v in names + ['tke']}
    att = comm.bcast(att, root=0)
    missing = set(att['_missing'])
    if missing and M.my_pe == 0:
        print('%s restart lacks %s: kept at their initial values (one-step transient)'
              % (what, ' '.join(sorted(missing))), flush=True)
    fields2d = [v for v in fields2d if v not in missing]
    fields3d = [v for v in fields3d if v not in missing]
    h = M.halo; sl = slice(h, -h)
    nz_ = M.u.shape[0] - 2*h

    def r3(name, tgt):
        blk = scatter_array_mpi(M, dat[name], (nz_,))
        return M.apply_bc(OM.modify_array(tgt, (sl, sl, sl), blk))

    def r4(name, tgt):
        blk = scatter_array_mpi(M, dat[name], (3, nz_))
        return OM.modify_array(tgt, (slice(None), sl, sl, sl), blk)

    def r2(name, tgt):
        blk = scatter_array_mpi(M, dat[name], ())
        return np.asarray(M.apply_bc(OM.modify_array(OM.np.asarray(tgt), (sl, sl), blk)))

    for n in _PE_3D: setattr(M, n, r3(n, getattr(M, n)))
    for n in _PE_4D: setattr(M, n, r4(n, getattr(M, n)))
    blk = scatter_array_mpi(M, dat['p_s'], (3,))
    for k in range(3):
        M.p_s = OM.modify_array(M.p_s, (k, sl, sl), blk[k])
        M.p_s = OM.modify_array(M.p_s, (k, slice(None), slice(None)), M.apply_bc(M.p_s[k]))
    if att['_has_tke'] and hasattr(M, 'tke'):
        M.tke = r3('tke', M.tke)
    for n in fields2d: setattr(M, n, r2(n, getattr(M, n)))
    for n in fields3d: setattr(M, n, r3(n, getattr(M, n)))
    from pyOM3.density import get_rho
    M.rho = get_rho(M.salt, M.temp, M.p0, M.eq_of_state)
    M.w   = M.vertical_velocity(M.u, M.v, M.w, M.maskW)
    if M.my_pe == 0:
        print('read %s restart %s: resuming at coupling step %d' % (what, fname, att['n_cpl']), flush=True)
    return att



class AtmosCouplingWriter:
    """Coupling record (atmosphere rank 0): the interval-mean surface fluxes
    and the states they were computed from, one record per written coupling
    step, plus the domain-integrated exchange scalars."""
    FIELDS = (('tau_x', 'wind stress east, Pa'), ('tau_y', 'wind stress north, Pa'),
              ('q_s', 'sensible heat flux ocean->air, W/m2'),
              ('q_int', 'interface dissipation u_a.tau - w_o - p_in, W/m2'),
              ('u10', 'relative wind at 10 m east, m/s'), ('v10', 'relative wind at 10 m north, m/s'),
              ('t10', 'air temperature at 10 m, K'), ('sst', 'sea surface temperature, K'),
              ('u_o', 'ocean surface velocity east seen by the wind, m/s'),
              ('v_o', 'ocean surface velocity north seen by the wind, m/s'),
              ('w_o', 'stress work on the ocean, W/m2 of cell area'),
              ('p_in', 'wind input to the waves (surface TKE flux without waves), W/m2 of cell area'),
              ('q_ocn', 'heat the ocean applied after the freezing floor, W/m2 of cell area'),
              ('f_ice', 'fraction of the cell under the freezing floor'),
              ('bl_h', 'boundary layer height, m'))
    SCALARS = (('W_tau_atm', 'stress work on the atmosphere W_a = -int u_a.tau, W/m2'),
               ('Q_s_atm', 'sensible heat received by the atmosphere, W/m2'),
               ('Q_int_atm', 'interface dissipation returned to the atmosphere, W/m2'),
               ('neg_area', 'area fraction where q_int < 0'),
               ('D_bl_atm', 'KE change by the boundary layer diffusion, W/m2'),
               ('D_bih_atm', 'KE change by the biharmonic friction, W/m2'),
               ('P_in', 'wind input to the waves, W/m2'),
               ('W_tau_ocn', 'stress work on the ocean W_o, W/m2'),
               ('Q_ocn', 'heat applied to the ocean after the freezing floor, W/m2'),
               ('ice_area', 'area under the freezing floor, fraction'))

    def __init__(self, fname, nx, ny, xt, yt, dt_cpl):
        """nx, ny, xt, yt of the ATMOSPHERE grid."""
        from netCDF4 import Dataset
        self.f = Dataset(fname, 'w', format='NETCDF4')
        self.f.createDimension('time', None); self.f.createDimension('x', nx); self.f.createDimension('y', ny)
        self.f.setncattr('dt_cpl', float(dt_cpl))
        self.f.createVariable('time', 'f8', ('time',))
        self.f.createVariable('x', 'f8', ('x',))[:] = np.asarray(xt)
        self.f.createVariable('y', 'f8', ('y',))[:] = np.asarray(yt)
        for n, ln in self.FIELDS:
            self.f.createVariable(n, 'f4', ('time', 'y', 'x')).long_name = ln
        for n, ln in self.SCALARS:
            self.f.createVariable(n, 'f8', ('time',)).long_name = ln
        self.n = 0

    def write(self, t, fields, scalars):
        self.f.variables['time'][self.n] = t
        for k, v in fields.items():
            self.f.variables[k][self.n] = v
        for k, v in scalars.items():
            self.f.variables[k][self.n] = v
        self.n += 1
        self.f.sync()

    def close(self):
        self.f.close()


