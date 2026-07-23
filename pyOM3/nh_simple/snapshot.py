"""
Snapshot diagnostics, based on class model.

class snap_nc4         writes to a netCDF file via the netCDF4 module
class snap_zarr_gather gathers fields on PE0 and writes large chunks to a
                       single zarr store (Lustre-friendly: ~6 files per
                       snapshot regardless of the number of PEs)
"""
import pyOM3                 as OM
from   pyOM3.nh_simple.model import model
from   netCDF4               import Dataset as NF
import numpy                 as np__
import zarr, os, shutil, warnings

# Silence the cosmetic zarr warning emitted when writing a v2-format store
# under zarr-python v3: consolidated metadata is not yet part of the v3 spec.
# Matched by message substring so only this specific warning is muted.
# ZarrUserWarning (newer zarr) subclasses UserWarning, so this covers both.
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part",
    category=UserWarning,
)


class snap_nc4(model):

    
      def setup(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          """
          Initialisation of the netcdf file.
          Uses parallel I/O if the netCDF4/HDF5 build supports it,
          otherwise falls back to the original serialized PE-by-PE writing.
          """
          super().setup()
          
          self.snap_file_name      = snap_file_name
          self.show_each_time_step = show_each_time_step

          # --- detect parallel netCDF4 capability ---
          self.parallel_io = False
          if self.n_pes > 1:
              if getattr(NF, '__has_parallel4_support__', False) or \
                 getattr(NF, '__has_pnetcdf_support__', False):
                  self.parallel_io = True

          if self.my_pe == 0:
             print(" snap_nc4: parallel I/O =", self.parallel_io)

          z = np__.arange(self.nz)*self.dz 
          z = z - (z[-1] + self.dz/2) 
          x,y = (0.5+np__.arange(self.nx))*self.dx, (0.5+np__.arange(self.ny))*self.dy 

          def _define(id):
             id.createDimension('xt',self.nx)
             id.createDimension('yt',self.ny)
             id.createDimension('zt',self.nz)
             id.createDimension('xu',self.nx)
             id.createDimension('yu',self.ny)
             id.createDimension('zu',self.nz)
             id.createDimension('time',None)
             id.createVariable('time', OM.prec ,('time',) )
             id.createVariable('xt',   OM.prec ,('xt',) )[:] = x
             id.createVariable('yt',   OM.prec ,('yt',) )[:] = y
             id.createVariable('zt',   OM.prec ,('zt',) )[:] = z
             id.createVariable('xu',   OM.prec ,('xu',) )[:] = x + self.dx/2
             id.createVariable('yu',   OM.prec ,('yu',) )[:] = y + self.dy/2
             id.createVariable('zu',   OM.prec ,('zu',) )[:] = z + self.dz/2
             id.createVariable('b', OM.prec ,('time','zt','yt','xt') )
             id.createVariable('u', OM.prec ,('time','zt','yt','xu') )
             id.createVariable('v', OM.prec ,('time','zt','yu','xt') )
             id.createVariable('w', OM.prec ,('time','zu','yt','xt') )
             id.createVariable('p', OM.prec ,('time','zt','yt','xt') )

          if self.parallel_io:
             # all PEs open collectively; define is identical on every PE
             id = NF(snap_file_name, mode='w', parallel=True, comm=self.comm)
             _define(id)
             id.close()
          else:
             # original behaviour: only PE0 creates the file
             if self.my_pe == 0:
                id = NF(snap_file_name,mode='w')
                _define(id)
                id.close()
          return

    
      def diagnose(self,n,tau):  
          """
          Writing a snapshot into the netcdf file.
          Parallel: every PE writes its subdomain collectively in one open.
          Serial:   original PE-by-PE serialized writing.
          """
          def _show_time(t):
              if     t <=86400.: return "%6.3f s"%t
              else:   return "%8.1f d"%(t/86400.)

          def _print_text():  print ("t=",_show_time(t)," n=",n)

          def _write(a):
              if self.fortran_ordering: 
                 sl = (slice(self.halo,-self.halo),
                       slice(self.halo,-self.halo),
                       slice(self.halo,-self.halo) if len(a.shape)>2 else None )
              else:
                 sl = (slice(self.halo,-self.halo) if len(a.shape)>2 else None ,
                       slice(self.halo,-self.halo),
                       slice(self.halo,-self.halo))                              
              return a[sl].transpose() if self.fortran_ordering else a[sl]

          def _open():
              if self.parallel_io:
                 return NF(self.snap_file_name,mode='r+',parallel=True,comm=self.comm)
              return NF(self.snap_file_name,mode='r+')

          t = n*self.dt 
          if self.my_pe==0 and self.show_each_time_step: _print_text()

          if not (n==0 or np__.mod(n,int(self.snapint/self.dt))==0.): return

          if self.my_pe == 0:
             if not self.show_each_time_step: _print_text()
             print(" --> writing to file ",self.snap_file_name)

          # PE0 extends the time dimension and writes the time coordinate
          self.comm.Barrier()
          if self.my_pe == 0:
             fid = NF(self.snap_file_name,mode='r+')
             m = len(fid.dimensions['time'])
             fid.variables['time'][m] = t
             fid.close()
          else: m = 0
          m = self.comm.bcast(m,root=0)

          sl = (m,slice(self.zs_pe-1,self.ze_pe),
                  slice(self.ys_pe-1,self.ye_pe),
                  slice(self.xs_pe-1,self.xe_pe))

          # parallel: one collective pass (pe=None); serial: loop PE-by-PE
          for pe in (range(self.n_pes) if not self.parallel_io else (None,)):
              self.comm.Barrier()
              if self.parallel_io or self.my_pe == pe:
                 fid = _open()
                 for var in ('u','v','w','b','p'):
                     v = fid.variables[var]
                     if self.parallel_io: v.set_collective(True)
                     v[sl] = _write(getattr(self,var))
                 fid.close()
          return


# ---------------------------------------------------------------------------
# zarr v2 / v3 compatibility helpers
# ---------------------------------------------------------------------------
_zarr_v3 = int(zarr.__version__.split('.')[0]) >= 3


def _create_array(store, name, shape, chunks, dtype, attrs=None):
    """Create a zarr array compatibly with v2 and v3."""
    if _zarr_v3:
        a = store.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    else:
        a = store.require_dataset(name, shape=shape, chunks=chunks,
                                  dtype=dtype, exact=True, overwrite=True)
    if attrs:
        a.attrs.update(attrs)
    return a


def _resize(arr, new_shape):
    """Resize a zarr array to new_shape (tuple). Works for v2 and v3."""
    arr.resize(new_shape)


def _open_group(path, mode):
    """
    Open a zarr group, forcing the v2 ON-DISK format even under zarr v3 so the
    store stays readable by v2 environments (e.g. the default zarr 2.x kernel
    on levante). Under zarr v2 the kwarg is not accepted, so fall back.
    """
    if _zarr_v3:
        return zarr.open_group(path, mode=mode, zarr_format=2)
    return zarr.open_group(path, mode=mode)


class snap_zarr_gather(model):

    def setup(self, snap_file_name='snap.zarr', show_each_time_step=True,
              z_chunks=1):
        """
        Initialise a single global zarr store with arrays (time, nx, ny, nz).

        Only PE0 ever touches the store. Field chunks are large: a full
        horizontal slab (nx, ny) and ``ceil(nz / z_chunks)`` levels, so each
        snapshot writes only a few big chunk files instead of one per PE.

        Parameters
        ----------
        snap_file_name : str
            Path of the .zarr store.
        show_each_time_step : bool
            Print the model time every step.
        z_chunks : int
            Number of chunks along z. 1 (default) => one chunk per field per
            snapshot (largest files, fewest of them). Increase only if a single
            field slab is too large to write comfortably in one go.
        """
        super().setup()

        self.snap_file_name = snap_file_name
        self.show_each_time_step = show_each_time_step

        nx, ny, nz = self.nx, self.ny, self.nz

        z = np__.arange(nz) * self.dz
        z = z - (z[-1] + self.dz / 2)
        x = (0.5 + np__.arange(nx)) * self.dx
        y = (0.5 + np__.arange(ny)) * self.dy

        # large field chunks: full (nx, ny) slab, nz split into z_chunks pieces
        nz_c = max(1, -(-nz // max(1, int(z_chunks))))   # ceil division

        if self.my_pe == 0:
            if os.path.exists(snap_file_name):
                shutil.rmtree(snap_file_name)
            store = _open_group(snap_file_name, mode='w')

            def _axis(name, data, attrs):
                a = _create_array(store, name, shape=data.shape,
                                  chunks=data.shape, dtype=OM.prec, attrs=attrs)
                a[:] = data

            _axis('xt', x,           {'units': 'm', 'long_name': 'x T grid'})
            _axis('yt', y,           {'units': 'm', 'long_name': 'y T grid'})
            _axis('zt', z,           {'units': 'm', 'long_name': 'z T grid'})
            _axis('xu', x + self.dx / 2, {'units': 'm', 'long_name': 'x U grid'})
            _axis('yu', y + self.dy / 2, {'units': 'm', 'long_name': 'y U grid'})
            _axis('zu', z + self.dz / 2, {'units': 'm', 'long_name': 'z U grid'})

            _create_array(store, 'time', shape=(0,), chunks=(1,), dtype=OM.prec,
                          attrs={'units': 'seconds', 'long_name': 'time'})

            # field arrays: (time, nx, ny, nz), x-first; one time slice per chunk
            field_shape = (0, nx, ny, nz)
            field_chunks = (1, nx, ny, nz_c)
            for var, long_name, units in (
                    ('u', 'zonal velocity',      'm/s'),
                    ('v', 'meridional velocity', 'm/s'),
                    ('w', 'vertical velocity',   'm/s'),
                    ('b', 'buoyancy',            'm/s^2'),
                    ('p', 'pressure',            'm^2/s^2')):
                _create_array(store, var, shape=field_shape, chunks=field_chunks,
                              dtype=OM.prec,
                              attrs={'units': units, 'long_name': long_name})

            self._store = store
        else:
            self._store = None

        self.comm.Barrier()
        return

    def diagnose(self, n, tau):
        """
        Append one snapshot. Every PE participates in the gather of each field
        (gather_array_mpi is collective); only PE0 writes to the store.
        """
        def _show_time(t):
            return "%6.3f s" % t if t <= 86400. else "%8.1f d" % (t / 86400.)

        t = n * self.dt
        if self.my_pe == 0 and self.show_each_time_step:
            print("t=", _show_time(t), " n=", n)

        if not (n == 0 or np__.mod(n, int(self.snapint / self.dt)) == 0):
            return

        if self.my_pe == 0:
            if not self.show_each_time_step:
                print("t=", _show_time(t), " n=", n)
            print(" --> writing to zarr store", self.snap_file_name)
            m = self._store['time'].shape[0]
            _resize(self._store['time'], (m + 1,))
            self._store['time'][m] = t
            for var in ('u', 'v', 'w', 'b', 'p'):
                _resize(self._store[var], (m + 1, self.nx, self.ny, self.nz))

        # gather + write one field at a time to bound PE0 memory.
        # gather_array_mpi returns the full de-haloed domain in the model's own
        # axis convention: (nx, ny, nz) when fortran_ordering, else (nz, ny, nx).
        # The store is always x-first (time, nx, ny, nz), so reverse the spatial
        # axes in the C-ordering case to get (nx, ny, nz) before writing. This
        # keeps the on-disk layout identical for both orderings.
        for var in ('u', 'v', 'w', 'b', 'p'):
            full = self.gather_array_mpi(getattr(self, var))   # collective
            if self.my_pe == 0:
                if not self.fortran_ordering:
                    full = full.transpose(*reversed(range(full.ndim)))
                if full.shape != (self.nx, self.ny, self.nz):
                    raise ValueError(
                        "snap_zarr_gather: gathered field '%s' has shape %s, "
                        "expected (nx,ny,nz)=%s. Check fortran_ordering and the "
                        "domain decomposition."
                        % (var, full.shape, (self.nx, self.ny, self.nz)))
                self._store[var][m] = full

        return
