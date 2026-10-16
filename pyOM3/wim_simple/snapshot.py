"""
Snapshot diagnostics, based on class model.

class snap_nc4 is based on the netCDF4 module: one shared file, written in a
serial round robin over the PEs.

class snap_raw writes every PE's own block as .npy files and never gathers or
shares a file, the wave counterpart of pe_realistic's snap_raw. It exists
because snap_nc4 does not scale: its diagnose loops over all PEs in turn,
each opening, writing and closing the one file, and it writes five FOUR-
dimensional fields. On a 1080 x 689 grid with nd=20, nf=30 that is 18 GB per
record pushed through one file handle at a time, which at 2048 PEs takes
longer than the model run it is diagnosing.

snap_raw also writes a CHOICE of fields rather than all of them, because the
four ray rates (xdot, ydot, fdot, ddot) are diagnosable from the spectrum,
the grid and the currents, and are 14 of those 18 GB. What it writes is
self.snap_fields, a tuple of names:

    E_mean, om_mean   the WAMDI moments, already on the instance   (2-D)
    Hs                4 sqrt(E_mean)                               (2-D)
    E_f               the spectrum integrated over DIRECTION       (nf,y,x)
    E_d               the spectrum integrated over FREQUENCY       (nd,y,x)
    E                 the full spectrum                            (nd,nf,y,x)
    any other 2-D or 4-D attribute of the model, ustar and the ray rates
    included, stripped of its halo by its rank

Default: E_mean, om_mean, Hs, E_f, E_d -- 160 MB per record on that grid
against 18 GB, and the two marginals answer where the peak sits, how the tail
behaves and how broad the directional spread is. The full 4-D E has its own
interval, self.snapint_full (0 = never), so it can be written a few times a
year while the marginals come every 30 days.

Written in float32: these are diagnostics, and the restart state (A, dA)
is a separate file in full precision.
"""
import os
import shutil
import pyOM3           as OM
from   pyOM3.wim_simple.model import model
from   netCDF4         import Dataset as NF
import numpy           as np__


class snap_nc4(model):


      def _snap_xy(self):
          """coordinates written to the snapshot file (cell-centre metres;
          wim_realistic overrides with the grid coordinates of calc_grid)"""
          return (0.5+np__.arange(self.nx))*self.dx, (0.5+np__.arange(self.ny))*self.dy

    
      def setup(self, snap_file_name = 'snap.cdf', show_each_time_step = True ):
          """
          Initialisation of the netcdf file. The body lives in the
          module-level _snap_setup so wim_realistic can share it: a
          zero-argument super() inside a shared method binds to the class
          it was defined in, so this chaining line must stay per package.
          """
          if hasattr(super(),'setup'): super().setup()
          return _snap_setup(self, snap_file_name, show_each_time_step)


      def diagnose(self,n,tau):  
          """
          Writing a snapshot into the netcdf file.
          """
          def _show_time(t):
              if     t <=86400.: return "%6.3f s"%t
              else:   return "%8.1f d"%(t/86400.)

          def _print_text():  print ("t=",_show_time(t)," n=",n)

                  
          t = n*self.dt 
          if self.my_pe==0 and self.show_each_time_step: _print_text()
                     
          if n==0 or np__.mod(n,int(self.snapint/self.dt))==0. : # writing to file
             if self.my_pe == 0: 
                 if not self.show_each_time_step: _print_text()
                 print(" --> writing to file ",self.snap_file_name)
             write = True 
          else: write = False

          if write: 
             
             if self.comm is not None: self.comm.Barrier()  
             if self.my_pe == 0:
                fid = NF(self.snap_file_name,mode='r+')
                m = len(fid.dimensions['time'])
                fid.variables['time'][m] = t
                fid.close() 
             else: m=0
                 
             if self.comm is not None: m = self.comm.bcast(m,root=0)
              
             for pe in range(self.n_pes if self.comm is not None else 1): 
                 
                 if self.comm is not None: self.comm.Barrier()  

                 sl1 = (m,slice(self.ys_pe-1,self.ye_pe), 
                          slice(self.xs_pe-1,self.xe_pe))    
                 sl2 = (m,slice(None),slice(None),slice(self.ys_pe-1,self.ye_pe), 
                          slice(self.xs_pe-1,self.xe_pe))  
                 if self.my_pe == pe:
                    fid = NF(self.snap_file_name,mode='r+')
                    fid.variables['E_mean'][sl1]  = _strip(self, self.e_mean)
                    fid.variables['om_mean'][sl1] = _strip(self, self.om_mean)
                    fid.variables['E'][sl2]       = _strip(self, self.E)
                    fid.variables['xdot'][sl2]    = _strip(self, self.xdot)
                    fid.variables['ydot'][sl2]    = _strip(self, self.ydot)
                    fid.variables['fdot'][sl2]    = _strip(self, self.fdot)
                    fid.variables['ddot'][sl2]    = _strip(self, self.ddot)
                    if 'tau_w' in fid.variables:          # janssen state, if
                       if hasattr(self, 'tau_w'):         # already set by a
                          fid.variables['tau_w'][sl1] = _strip(self, self.tau_w)  # source step
                       if hasattr(self, 'z0'):
                          fid.variables['z0'][sl1]    = _strip(self, self.z0)
                    fid.close()


def _strip(self, a):
    """the interior of a model array as numpy: MPI the local block, JAX the
    global interior (every inline shard halo removed), the d,f halo stripped
    from a 4-D array as well"""
    h = self.halo; sl = slice(h, -h)
    if OM.jax:
        return np__.asarray(self.unpad(a, lead = 2 if a.ndim == 4 else 0))
    a = np__.asarray(a)
    return a[sl, sl] if a.ndim == 2 else a[sl, sl, sl, sl]


def _snap_setup(self, snap_file_name, show_each_time_step):
    """netCDF body of snap_nc4.setup, shared with wim_realistic"""
    self.snap_file_name      = snap_file_name
    self.show_each_time_step = show_each_time_step
    
    if self.my_pe == 0:
       id = NF(snap_file_name,mode='w')    
       id.createDimension('x',self.nx)
       id.createDimension('y',self.ny)
       id.createDimension('d',self.nd) 
       id.createDimension('f',self.nf)  
       id.createDimension('time',None)

       x,y = self._snap_xy()
        
       id.createVariable('time', OM.prec ,('time',) )
       id.createVariable('x',   OM.prec ,('x',) )[:] = x
       id.createVariable('y',   OM.prec ,('y',) )[:] = y
       id.createVariable('f',   OM.prec ,('f',) )[:] = self.ft[self.halo:-self.halo] 
       id.createVariable('d',   OM.prec ,('d',) )[:] = self.theta[self.halo:-self.halo] 
       id.createVariable('E_mean', OM.prec ,('time','y','x') )
       id.createVariable('om_mean', OM.prec ,('time','y','x') )
       id.createVariable('E', OM.prec ,('time','d','f','y','x') ) 
       id.createVariable('xdot', OM.prec ,('time','d','f','y','x') )
       id.createVariable('ydot', OM.prec ,('time','d','f','y','x') )
       id.createVariable('fdot', OM.prec ,('time','d','f','y','x') )
       id.createVariable('ddot', OM.prec ,('time','d','f','y','x') )
       # Janssen input state (wim_realistic, input_scheme='janssen'): the
       # wave-supported stress tau_w (kinematic, per rho_a) and the
       # roughness length z0, carried between source steps
       if getattr(self, 'input_scheme', 'wamdi') == 'janssen':
          id.createVariable('tau_w', OM.prec ,('time','y','x') )
          id.createVariable('z0',    OM.prec ,('time','y','x') )
       id.close()    
    return

# ---------------------------------------------------------------------------
# RAW, per-PE snapshots
# ---------------------------------------------------------------------------
SNAP_FIELDS_DEFAULT = ('E_mean', 'om_mean', 'Hs', 'E_f', 'E_d')

# snapshot name -> attribute name, where snap_nc4's variable name differs
# from the instance's (E_mean is self.e_mean)
SNAP_ALIAS = {}

# names formed here rather than read off the instance
_MOMENTS = ('E_mean', 'om_mean', 'Hs', 'E_f', 'E_d')


def _snap_raw_moments(self, want):
    """The moment group, all from the CURRENT spectrum and with the weights
    calc_emean_wamdi uses, so an offline integral of E_f or E_d reproduces
    E_mean exactly and Hs is the 4 sqrt(m0) the ocean received as z0.

    NOTE the difference to snap_nc4, which writes self.e_mean and
    self.om_mean: those are evaluated inside Sdiss and are therefore one
    sub-step behind the spectrum written beside them."""
    sl = slice(self.halo, -self.halo)
    xy = (lambda a: self.unpad(a)) if OM.jax else (lambda a: a[..., sl, sl])
    out = {}
    e_dir = OM.np.sum(self.E[sl, :, :, :]*self.dd, axis=0)        # (f pad, y, x)
    if 'E_f' in want or 'E_mean' in want or 'om_mean' in want or 'Hs' in want:
        E_f = xy(e_dir[sl])
        if 'E_f' in want:
            out['E_f'] = E_f
    if 'E_d' in want:
        out['E_d'] = xy(OM.np.sum(self.E[:, sl, :, :]*self.dft[sl, None, None],
                               axis=1)[sl])
    if 'E_mean' in want or 'om_mean' in want or 'Hs' in want:
        w = self.dft[sl][:, None, None]
        E_mean = OM.np.maximum(OM.np.sum(E_f*w, axis=0), 0.0)
        if 'E_mean' in want:
            out['E_mean'] = E_mean
        if 'Hs' in want:
            out['Hs'] = 4.0*OM.np.sqrt(E_mean)
        if 'om_mean' in want:
            m = OM.np.sum(E_f/self.omega_t[sl][:, None, None]*w, axis=0)
            om = E_mean/OM.np.maximum(1e-32, m)
            out['om_mean'] = OM.np.clip(om, self.omega_t[self.halo],
                                        self.omega_t[-self.halo-1])
    return out


def _snap_raw_field(self, name):
    """The interior (halo stripped) block of one model attribute, float32.
    The moment group is handled by _snap_raw_moments; every other name is an
    attribute and is stripped by its rank."""
    sl = slice(self.halo, -self.halo)
    a = getattr(self, SNAP_ALIAS.get(name, name), None)
    if a is None:
        return None
    a = OM.np.asarray(a)
    if a.ndim in (2, 4):
        return np__.asarray(_strip(self, a), np__.float32)
    raise ValueError('snap_raw: %s has %d dimensions, expected 2 or 4'
                     % (name, a.ndim))


def _snap_raw_setup(self, snap_file_name, show_each_time_step):
    """Create the output directory and write the axes. Collective: PE 0 makes
    every PE's directory, so the barrier at the end must not be removed --
    the others mkdir inside them at the first diagnose."""
    self.snap_file_name      = snap_file_name
    self.show_each_time_step = show_each_time_step
    self.snap_fields  = tuple(getattr(self, 'snap_fields', SNAP_FIELDS_DEFAULT))
    self.snapint_full = float(getattr(self, 'snapint_full', 0.0))
    h = self.halo
    if self.my_pe == 0:
        if os.path.exists(snap_file_name):
            shutil.rmtree(snap_file_name)
        os.makedirs(snap_file_name)
        x, y = self._snap_xy()
        np__.save(snap_file_name + '/x.npy', np__.asarray(x))
        np__.save(snap_file_name + '/y.npy', np__.asarray(y))
        np__.save(snap_file_name + '/f.npy', np__.asarray(self.ft[h:-h]))
        np__.save(snap_file_name + '/d.npy', np__.asarray(self.theta[h:-h]))
        # the integration weights, so offline moments are the model's
        np__.save(snap_file_name + '/dft.npy', np__.asarray(self.dft[h:-h]))
        np__.save(snap_file_name + '/dd.npy', np__.asarray(self.dd))
        # JAX: one process writes the global interior as a single block
        np__.save(snap_file_name + '/decomposition.npy',
                  np__.array([1, 1] if OM.jax else [self.n_pes_x, self.n_pes_y]))
        for pe in range(1 if OM.jax else self.n_pes):
            os.mkdir(snap_file_name + '/pe=%i' % pe)
        print(' --> writing raw output to ', snap_file_name, flush=True)
        print('     fields %s, full spectrum every %s'
              % (','.join(self.snap_fields),
                 ('%g d' % (self.snapint_full/86400.)) if self.snapint_full > 0
                 else 'never'), flush=True)
    if self.comm is not None: self.comm.Barrier()
    return


def _snap_raw_diagnose(self, n, tau):
    """Each PE writes its own block as .npy into <dir>/pe=<pe>/n=<step>/."""
    t = n*self.dt
    if self.my_pe == 0 and self.show_each_time_step:
        print("t=", ("%6.3f s" % t) if t <= 86400. else ("%8.1f d" % (t/86400.)),
              " n=", n, flush=True)
    if not (n == 0 or np__.mod(n, int(self.snapint/self.dt)) == 0.):
        return
    names = list(self.snap_fields)
    if self.snapint_full > 0 and 'E' not in names:
        if n == 0 or np__.mod(n, int(self.snapint_full/self.dt)) == 0.:
            names.append('E')
    d = self.snap_file_name + '/pe=%i/n=%i/' % (self.my_pe, n)
    os.mkdir(d)
    np__.save(d + 't.npy', np__.asarray(t))
    absent = []
    mom = _snap_raw_moments(self, [n_ for n_ in names if n_ in _MOMENTS])
    for nm in names:
        a = mom[nm] if nm in mom else _snap_raw_field(self, nm)
        if a is None:
            absent.append(nm)
        else:
            np__.save(d + '%s.npy' % nm, np__.asarray(a, np__.float32))
    # a requested name that resolves to nothing would otherwise be missing
    # from the output with no trace of why
    if absent and self.my_pe == 0 and not getattr(self, '_snap_warned', False):
        print(' --> snap_raw: no such field on the model, NOT written: %s'
              % ','.join(absent), flush=True)
        self._snap_warned = True
    return


class snap_raw(model):
      """Per-PE raw snapshots. Same interface as snap_nc4: setup() takes the
      output name, which is a DIRECTORY here, and diagnose(n, tau) writes on
      the snapint steps."""

      def _snap_xy(self):
          """coordinates written beside the blocks (cell-centre metres;
          wim_realistic overrides with the grid coordinates of calc_grid)"""
          return (0.5+np__.arange(self.nx))*self.dx, (0.5+np__.arange(self.ny))*self.dy

      def setup(self, snap_file_name = 'raw_snap', show_each_time_step = True):
          if hasattr(super(),'setup'): super().setup()
          return _snap_raw_setup(self, snap_file_name, show_each_time_step)

      diagnose = _snap_raw_diagnose
