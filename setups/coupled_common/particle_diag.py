"""
LAGRANGIAN PARTICLES of the forced or coupled ocean, online, for the
Lagrangian frequency analysis (draft1, Shakespeare and Hogg 2017): the
particles of pyOM3/diagnostics/particles (particles_pe_realistic) (trilinear interpolation, AB4, every
ocean step, every PE holds every particle) set up from the configuration,
written to one netCDF file per segment, and carried through the restarts.

THE SEEDS, in the model's T cells. A box lon0, lon1, lat0, lat1 [degrees]
takes the T points xt, yt of the global grid inside it, every STRIDE-th in
each direction counted from the global index 0 (so two runs with the same
box and stride get the same points), at every depth of DEPTHS [m below the
surface]. The longitude test is modulo 360, so a box may be given as -45,
-20 or 315, 340. Two such sets exist:
    PARTICLE_BOX, PARTICLE_STRIDE, PARTICLE_DEPTHS   moving particles
    MOORING_BOX,  MOORING_STRIDE,  MOORING_DEPTHS    fixed points (moving = 0):
                                                     the Eulerian time series at
                                                     the same cadence, for the
                                                     filter at fixed points
or PARTICLE_FILE, an npz with lon, lat, depth (and moving, default 1), in
place of both. Seeds in dry cells are dropped (the log says how many).

THE RECORD, at the ocean steps n with n % PARTICLE_OUT_STEPS == 0 (n counted
from step 0 of the run, so a restart keeps the cadence), into
<exp>_particles<segment>.nc on PE 0:
    time(time)             n dt [s]: the time label the snapshot of the same
                           diagnose(n) carries (snap_raw: t = n dt), so a record
                           at a snapshot step samples exactly the snapshot's state.
                           A snapshot step is a record step when
                           PARTICLE_OUT_STEPS divides snapint/dt.
    lon, lat, depth        (time, particle) the position before the particle
                           step of that diagnose (pe_simple's order: record,
                           interpolate, step)
    u, v, w, temp, salt    (time, particle) the values there, from the model
                           state of that diagnose; u, v, w are the velocities the
                           particle moves with
    id, moving, lon0, lat0, depth0   (particle) the seed: id is the index into
                           the seed list (PARTICLE_FILE order, or box order:
                           depth fastest, then lon, then lat; moorings after
                           the particles)
Velocities, temp and salt as float32, the positions as float64.

THE RESTART. With every ocean restart PE 0 writes <restart>_particles.npz
next to it (BEFORE the ocean header, which is written last and makes a
restart valid): positions, the AB4 history, the step counters, the ids and
the refused-step count. The ocean restart of the run itself restores them
(the step of the file must equal the ocean restart's, otherwise the run
stops); a branch (EXP_INIT_OCEAN) or a cold start seeds anew at step 0.

cfg names: PARTICLES, PARTICLE_BOX, PARTICLE_STRIDE, PARTICLE_DEPTHS,
MOORING_BOX, MOORING_STRIDE, MOORING_DEPTHS, PARTICLE_FILE,
PARTICLE_OUT_STEPS (exp_common.Cfg.DEFAULTS has them, PARTICLES off).
"""
import os
import numpy as np
import pyOM3 as OM
from pyOM3.diagnostics.particles.particles import particles_pe_realistic as particles
try:          # HDF5 is not thread safe: the lock of the ERA5 read-ahead thread
    from forcing_era5_native import NC_LOCK
except ImportError:
    import threading
    NC_LOCK = threading.RLock()


def _restart_name(fname):
    return (fname[:-3] if fname.endswith('.nc') else fname) + '_particles.npz'


class particle_common(particles):

    # --- the seeds ------------------------------------------------------------
    def _particle_box(self, box, stride, depths):
        """seeds of one box: x, y, z (model coordinates), depth fastest"""
        if box is None or len(depths) == 0:
            return np.zeros((3, 0))
        lon0, lon1, lat0, lat1 = [float(b) for b in box]
        stride = max(1, int(stride))
        xt, yt = np.asarray(self.xt_gl, float), np.asarray(self.yt_gl, float)
        if getattr(self, 'coord_degree', False):
            w = (lon1 - lon0) % 360.0
            ix = np.nonzero(((xt - lon0) % 360.0) < w)[0]
        else:
            ix = np.nonzero((xt >= lon0) & (xt < lon1))[0]
        jy = np.nonzero((yt >= lat0) & (yt < lat1))[0]
        ix = ix[ix % stride == 0]; jy = jy[jy % stride == 0]
        z = self.particle_depth_to_z(np.asarray(depths, float))
        Y, X, Z = np.meshgrid(yt[jy], xt[ix], z, indexing='ij')       # lat slowest, depth fastest
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()])

    def _particle_seeds(self):
        cfg = self.cfg
        if cfg.PARTICLE_FILE:
            d = np.load(cfg.PARTICLE_FILE)
            P = np.stack([np.asarray(d['lon'], float), np.asarray(d['lat'], float),
                          self.particle_depth_to_z(np.asarray(d['depth'], float))])
            mv = np.asarray(d['moving'], float) if 'moving' in d.files else np.ones(P.shape[1])
            return P, mv
        A = self._particle_box(cfg.PARTICLE_BOX, cfg.PARTICLE_STRIDE, cfg.PARTICLE_DEPTHS)
        B = self._particle_box(cfg.MOORING_BOX, cfg.MOORING_STRIDE, cfg.MOORING_DEPTHS)
        return np.concatenate([A, B], axis=1), np.concatenate([np.ones(A.shape[1]), np.zeros(B.shape[1])])

    # --- set up (run.py, after average_setup) ----------------------------------
    def particle_setup(self, fname):
        """seed the particles and open the record file of this segment.
        A restart of the run itself replaces the seeds (particle_restart_read)."""
        cfg = self.cfg
        self._part = bool(cfg.PARTICLES)
        if not self._part:
            return
        self._part_every = max(1, int(cfg.PARTICLE_OUT_STEPS))
        P, mv = self._particle_seeds()
        self._part_seed = (P, mv)
        self.init_particles(P[0], P[1], P[2], moving=mv, quiet=True)
        self._part_file = fname
        self._part_nc = None
        self._part_nrec = 0
        if self.my_pe == 0:
            print('CONFIG particles: %d seeds, %d in wet cells (%d moving, %d fixed); record every %d steps '
                  '(%g s) into %s' % (P.shape[1], self.p_n, int(self.p_moving_np.sum()),
                                      self.p_n - int(self.p_moving_np.sum()), self._part_every,
                                      self._part_every*self.dt, fname), flush=True)
        return

    def _particle_open(self):
        """the record file of this segment, created at the first record (on
        PE 0), with the seeds of the particles the run carries"""
        with NC_LOCK:                       # the ERA5 reader may be reading (HDF5 is not thread safe)
            self._particle_create()
        import atexit
        atexit.register(self._particle_close)
        return

    def _particle_create(self):
        from netCDF4 import Dataset
        P, mv = self._part_seed
        ids = np.asarray(self.p_ids)
        N = self.p_n
        f = Dataset(self._part_file, 'w', format='NETCDF4')
        f.createDimension('particle', N); f.createDimension('time', None)
        f.setncatts(dict(description='Lagrangian particles (setups/coupled_common/particle_diag.py): '
                                     'positions before the particle step of each record, values from the model '
                                     'state of that step (the state a snapshot at the same time holds)',
                         dt=float(self.dt), out_steps=int(self._part_every),
                         forcing_start=str(getattr(self.cfg, 'FORCING_START', ''))))
        v = f.createVariable('time', 'f8', ('time',)); v.units = 's since model time 0 (n dt, as the snapshots)'
        f.createVariable('id', 'i8', ('particle',))[:] = ids
        f.createVariable('moving', 'i1', ('particle',))[:] = np.asarray(self.p_moving_np, np.int8)
        H = float(np.asarray(self.zw_gl)[-1])
        f.createVariable('lon0', 'f8', ('particle',))[:] = P[0, ids]
        f.createVariable('lat0', 'f8', ('particle',))[:] = P[1, ids]
        f.createVariable('depth0', 'f8', ('particle',))[:] = H - P[2, ids]
        ch = (1, N)
        for nme, typ, units in (('lon', 'f8', 'degrees_east'), ('lat', 'f8', 'degrees_north'),
                                ('depth', 'f4', 'm'), ('u', 'f4', 'm/s'), ('v', 'f4', 'm/s'),
                                ('w', 'f4', 'm/s'), ('temp', 'f4', 'deg C'), ('salt', 'f4', 'g/kg')):
            f.createVariable(nme, typ, ('time', 'particle'), chunksizes=ch).units = units
        self._part_nc = f
        return

    def _particle_close(self):
        if getattr(self, '_part_nc', None) is not None:
            try:
                with NC_LOCK:
                    self._part_nc.close()
            except Exception:
                pass
            self._part_nc = None

    # --- every step (exp_ocean.diagnose) ----------------------------------------
    def particle_step(self, n):
        if not getattr(self, '_part', False):
            return
        record = n % self._part_every == 0
        r = self.step_particles(record=record, host=(self.my_pe == 0))
        if not record or self.my_pe != 0:
            return
        if self._part_nc is None:
            self._particle_open()
        f = self._part_nc; k = self._part_nrec
        H = float(np.asarray(self.zw_gl)[-1])
        with NC_LOCK:                       # the ERA5 reader may be reading (HDF5 is not thread safe)
            f.variables['time'][k] = n*float(self.dt)          # snap_raw's label of the same state
            f.variables['lon'][k, :] = r['x']; f.variables['lat'][k, :] = r['y']
            f.variables['depth'][k, :] = H - r['z']
            for nme in ('u', 'v', 'w', 'temp', 'salt'):
                f.variables[nme][k, :] = r[nme]
            if k % 100 == 99:
                f.sync()
        self._part_nrec = k + 1
        return

    # --- restart (coupler_io, coupler_loop) -------------------------------------
    def particle_restart_write(self, fname, n_cpl):
        """<restart>_particles.npz on PE 0, before the ocean restart itself"""
        if not getattr(self, '_part', False):
            return
        if self.my_pe == 0:                              # PE 0's copy of the replicated state
            st = self.particles_state()
            out = _restart_name(fname)
            tmp = out[:-4] + '.tmp.npz'
            np.savez(tmp, n_cpl=np.int64(n_cpl), n_seed=np.int64(self._part_seed[0].shape[1]), **st)
            os.replace(tmp, out)
            if self._part_nc is not None:
                with NC_LOCK:
                    self._part_nc.sync()
            print('particles: restart %s at step %d (%d particles, %d refused steps so far, %d records '
                  'in this segment)' % (out, n_cpl, self.p_n, int(st['p_frozen']), self._part_nrec), flush=True)
        return

    def particle_restart_read(self, fname, n_cpl):
        """the particles of the run's own restart; none there (a run that
        switched them on later): the seeds stay, with a note"""
        if not getattr(self, '_part', False):
            return
        rn = _restart_name(fname)
        if not os.path.exists(rn):
            if self.my_pe == 0:
                print('particles: no %s, the particles start from their seeds at step %d'
                      % (rn, n_cpl), flush=True)
            return
        d = np.load(rn)
        if int(d['n_cpl']) != int(n_cpl):
            raise RuntimeError('particle restart %s is of step %d, the ocean restart of step %d: '
                               'they belong together (a job died between the two writes?)'
                               % (rn, int(d['n_cpl']), int(n_cpl)))
        ns = self._part_seed[0].shape[1]
        if 'n_seed' in d.files and int(d['n_seed']) != ns:
            raise RuntimeError('particle restart %s was seeded with %d positions, the configuration now gives %d: '
                               'the seeds (box, stride, depths, file) must stay as they were'
                               % (rn, int(d['n_seed']), ns))
        self.particles_restore(d)
        if self.my_pe == 0:
            print('particles: read %s (%d particles, step %d)' % (rn, self.p_n, int(n_cpl)), flush=True)
        return
