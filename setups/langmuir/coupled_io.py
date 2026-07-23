"""
I/O helpers for the coupled wave-ocean run (coupled.py):

  CouplingWriter       netCDF record writer for the exchanged Stokes
                       velocity profiles and wave stress (wave PE 0)
  segname              per-segment output file names (name_<seg>.ext)
  read_restart_header  (n, seg) from a wave restart file, or None
  write/load_wave_restart, write/load_ocean_restart
                       exact-restart checkpoints of both models
"""
import os, shutil
import numpy as np
import netCDF4
from pyOM3.nh_simple.snapshot import _open_group, _create_array


def segname(name, seg):
    """output file name of segment seg: base_<seg:03d>.ext"""
    base, ext = os.path.splitext(name)
    return '%s_%03i%s' % (base, seg, ext)


def read_restart_header(fname):
    """(n, seg) of the restart, or None if no restart file exists"""
    if not os.path.exists(fname):
        return None
    d = np.load(fname)
    return int(d['n']), int(d['seg'])


# ---------------------------------------------------------------------------
class CouplingWriter:
    """
    Append Stokes velocity profiles and the wave stress to a netCDF file
    with an unlimited time dimension. Written from wave PE rank 0 only;
    one record per call to write(). All quantities are kinematic
    (density-normalized): stress in m^2/s^2, stress work in m/s^2.
    """

    def __init__(self, path, nz, use_stress_work):
        self.n = 0
        self.use_stress_work = use_stress_work
        self.nc = netCDF4.Dataset(path, 'w', format='NETCDF4')
        self.nc.createDimension('time', None)
        self.nc.createDimension('depth', nz)
        v = self.nc.createVariable('depth', 'f8', ('depth',))
        v[:] = np.arange(nz, dtype=np.float64)
        v = self.nc.createVariable('time', 'f8', ('time',))
        v.long_name, v.units = 'model time', 's'
        v = self.nc.createVariable('u_stokes', 'f8', ('time', 'depth'))
        v.long_name, v.units = 'Stokes drift (x)', 'm/s'
        v = self.nc.createVariable('v_stokes', 'f8', ('time', 'depth'))
        v.long_name, v.units = 'Stokes drift (y)', 'm/s'
        v = self.nc.createVariable('dt_us_x', 'f8', ('time', 'depth'))
        v.long_name, v.units = 'Stokes drift tendency (x)', 'm/s^2'
        v = self.nc.createVariable('dt_us_y', 'f8', ('time', 'depth'))
        v.long_name, v.units = 'Stokes drift tendency (y)', 'm/s^2'
        if use_stress_work:
            v = self.nc.createVariable('tau_work_x', 'f8', ('time', 'depth'))
            v.long_name, v.units = 'wave stress work (x)', 'm/s^2'
            v = self.nc.createVariable('tau_work_y', 'f8', ('time', 'depth'))
            v.long_name, v.units = 'wave stress work (y)', 'm/s^2'
        else:
            v = self.nc.createVariable('taux', 'f8', ('time',))
            v.long_name, v.units = 'wave stress (x)', 'm^2/s^2'
            v = self.nc.createVariable('tauy', 'f8', ('time',))
            v.long_name, v.units = 'wave stress (y)', 'm^2/s^2'

    def write(self, time, u_S, stress, dt_us):
        i = self.n
        self.nc['time'][i] = time
        self.nc['u_stokes'][i, :] = np.asarray(u_S[0, :, 0, 0], np.float64)
        self.nc['v_stokes'][i, :] = np.asarray(u_S[1, :, 0, 0], np.float64)
        self.nc['dt_us_x'][i, :] = np.asarray(dt_us[0, :, 0, 0], np.float64)
        self.nc['dt_us_y'][i, :] = np.asarray(dt_us[1, :, 0, 0], np.float64)
        if self.use_stress_work:
            self.nc['tau_work_x'][i, :] = np.asarray(stress[0, :, 0, 0], np.float64)
            self.nc['tau_work_y'][i, :] = np.asarray(stress[1, :, 0, 0], np.float64)
        else:
            self.nc['taux'][i] = float(np.asarray(stress[0, 0, 0]))
            self.nc['tauy'][i] = float(np.asarray(stress[1, 0, 0]))
        self.nc.sync()
        self.n += 1


# ---------------------------------------------------------------------------
# restart I/O
# ---------------------------------------------------------------------------
def write_wave_restart(M, fname, seg, n, tau, taum1, taum2, tA, tB, tC,
                       next_snap):
    """single wave PE: full arrays incl. halos -> exact restart, and the
    coupling products are stored on the ocean side"""
    tmp = fname + '.tmp.npz'
    np.savez(tmp, E=M.E, A=M.A, dA=M.dA, ust_last=M.ust_last,
             ocean_U=M.ocean_U, ocean_V=M.ocean_V,
             ocean_Udf=M.ocean_Udf, ocean_Vdf=M.ocean_Vdf,
             n=n, seg=seg, tau=tau, taum1=taum1, taum2=taum2,
             tA=tA, tB=tB, tC=tC, next_snap=next_snap)
    os.replace(tmp, fname)   # atomic
    print('wave : restart written at n=', n)

def load_wave_restart(M, fname):
    d = np.load(fname)
    M.E[...], M.A = d['E'], d['A']
    M.ust_last = d['ust_last']
    M.ocean_U, M.ocean_V   = d['ocean_U'],   d['ocean_V']
    M.ocean_Udf, M.ocean_Vdf = d['ocean_Udf'], d['ocean_Vdf']
    if 'dA' in d.files:
        # exact restart: the Adams-Bashforth tendency history of the wave
        # model is restored together with the saved AB coefficients
        M.dA[...] = d['dA']
        tA, tB, tC = float(d['tA']), float(d['tB']), float(d['tC'])
    else:
        # restart file from an older format WITHOUT the dA buffers: using
        # the saved AB3 coefficients on zeroed buffers would kick A by
        # 23/12 of the first tendency. Fall back to an Euler first step
        # (the ab_coeffs upgrade rebuilds the AB sequence over two steps,
        # an O(dt^2) seam instead).
        print('WARNING: wave restart without dA buffers (old format), '
              'restarting the wave AB scheme from an Euler step')
        tA, tB, tC = 1., 0., 0.
    return (int(d['tau']), int(d['taum1']), int(d['taum2']),
            tA, tB, tC, float(d['next_snap']))

_OCEAN_RVARS = ('u', 'v', 'w', 'b')
_OCEAN_RBUFS = ('du', 'dv', 'dw', 'db')

def write_ocean_restart(M, fname, n, tau, taum1, taum2, tA, tB, tC,
                        cpl_u_S, cpl_stress, cpl_dt_us):
    """collective over the ocean communicator: fields and AB tendency
    buffers are gathered and written by PE0, together with the coupling
    products of the last exchange (PE0's local variables)"""
    gath = {}
    for k in _OCEAN_RVARS:
        gath[k] = M.gather_array_mpi(getattr(M, k))
    for k in _OCEAN_RBUFS:
        for s in range(3):
            gath['%s%d' % (k, s)] = M.gather_array_mpi(getattr(M, k)[s])
    if M.my_pe == 0:
        tmp = fname + '.tmp'
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        st = _open_group(tmp, mode='w')
        for k, g in gath.items():
            # gather_array_mpi returns (nx,ny,nz) under fortran_ordering:
            # TRANSPOSE to the (nz,ny,nx) restart layout. (A reshape here
            # relabels axes without moving data and swaps x and z -- on
            # cubic grids silently, which corrupted restarted segments.)
            g = np.asarray(g)
            if M.fortran_ordering:
                g = g.transpose(2, 1, 0)
            a = _create_array(st, k, shape=g.shape, chunks=g.shape,
                              dtype=np.float64, attrs={})
            a[:] = g
        for k, g in (('cpl_u_S', cpl_u_S), ('cpl_stress', cpl_stress),
                     ('cpl_dt_us', cpl_dt_us)):
            g = np.asarray(g, np.float64)
            a = _create_array(st, k, shape=g.shape, chunks=g.shape,
                              dtype=np.float64, attrs={})
            a[:] = g
        sc = np.array([n, tau, taum1, taum2, tA, tB, tC], np.float64)
        a = _create_array(st, 'scalars', shape=sc.shape, chunks=sc.shape,
                          dtype=np.float64, attrs={})
        a[:] = sc
        if os.path.exists(fname):
            shutil.rmtree(fname)
        os.rename(tmp, fname)
        print('ocean: restart written at n=', n)
    M.comm.Barrier()

def _put_slab(M, g, target):
    """fill the local interior of target from the global (nz,ny,nx) array
    and refresh the halos"""
    loc = g[M.zs_pe - 1:M.ze_pe, M.ys_pe - 1:M.ye_pe, M.xs_pe - 1:M.xe_pe]
    if M.fortran_ordering:
        loc = loc.transpose(2, 1, 0)
    h = M.halo
    target[h:-h, h:-h, h:-h] = loc
    return M.apply_bc(target)

def load_ocean_restart(M, fname, n_expected):
    """collective: every PE reads its own slab (read-only, parallel-safe)"""
    st = _open_group(fname, mode='r')
    for k in _OCEAN_RVARS:
        setattr(M, k, _put_slab(M, np.asarray(st[k]), getattr(M, k)))
    for k in _OCEAN_RBUFS:
        buf = getattr(M, k)
        for s in range(3):
            tmp = np.array(buf[s])
            buf[s, ...] = _put_slab(M, np.asarray(st['%s%d' % (k, s)]), tmp)
    sc = np.asarray(st['scalars'])
    if int(sc[0]) != n_expected:
        raise RuntimeError('ocean/wave restart step mismatch: %d vs %d'
                           % (int(sc[0]), n_expected))
    cpl = (np.asarray(st['cpl_u_S']), np.asarray(st['cpl_stress']),
           np.asarray(st['cpl_dt_us'])) if M.my_pe == 0 else (0, 0, 0)
    return (int(sc[1]), int(sc[2]), int(sc[3]),
            float(sc[4]), float(sc[5]), float(sc[6])), cpl
