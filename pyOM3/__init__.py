
""" 
    Initialize pyOM3 package.
    Define some helper functions and variables to deal with JAX and mpi4py
    Import JAX or NumPy if JAX fails. 
    Also import mpi4py
"""    


def modify_array(arr, where, value, out_sharding = None):
    """
    use this function to change slices in a numpy or jax array
    syntax: a = modify_array(a,(slice(2,3),slice(1,2)),b)
    """  
    if jax: 
        # out_sharding of .at[].set() is only legal on Explicit mesh axes (jax >= 0.7), the
        # model's mesh uses Auto axes (shard_map). with_sharding_constraint works for both.
        res = arr.at[where].set(value)
        if out_sharding is not None: res = jax.lax.with_sharding_constraint(res, out_sharding)
        return res
    #res = arr.copy() # do not do that
    #res[where] = value
    #return res
    arr[where] = value
    return arr

def add_to_array(arr, where, value, out_sharding = None):
    """
    same but adding to old value
    """  
    if jax: 
        res = arr.at[where].add(value)
        if out_sharding is not None: res = jax.lax.with_sharding_constraint(res, out_sharding)
        return res
    #res = arr.copy() # do not do that
    #res[where] += value
    #return res
    arr[where] += value
    return arr

# ---------------------------------------------------------------------------
# LARGE RUN CONSTANTS OF JITTED METHODS. The jitted methods of the models
# take self as a STATIC argument and read the masks and the metric from it
# (self.maskT, self.maskW, ...). A jax.Array read that way while tracing is
# not an input of the compiled function: it is EMBEDDED in the executable as
# a constant, and a sharded one is embedded WHOLE and replicated on every
# device (the HLO holds f64[Nz,Ny,Nx] constant({...}) and dynamic-slices it
# by partition id). On the test grids that is kilobytes; on icon033 one 3-D
# field is 780 MB, _tke_solve alone embedded 3.1 GB and the step has ~50
# such executables -- the GPUs ran out of memory on constants. So jaxjit
# below, for a method with self static, passes every jax.Array attribute of
# self of at least JIT_CONST_BYTES as an extra (non-static) argument and
# binds it to self while the body is traced: the same attribute reads become
# inputs of the compiled function, sharded as the arrays are. Arrays below
# the threshold (1-D grid vectors, small grids) stay embedded as before, so
# small runs are untouched and the timings there do not change. Results are
# identical either way. wim_simple does the same by hand for its named
# constants (shard_consts / bind_consts).
# ---------------------------------------------------------------------------
JIT_CONST_BYTES = 1 << 20          # 1 MB; PYOM3_JIT_CONST_BYTES overrides

def _large_arrays(obj):
    """{name: jax.Array} of obj's attributes of at least JIT_CONST_BYTES"""
    try:    d = vars(obj)
    except TypeError: return None
    # several processes: a jitted function may not close over ANY array that
    # spans other processes' devices, so every jax.Array attribute is passed
    thr = 0 if jax.process_count() > 1 else JIT_CONST_BYTES
    out = {}
    for k, v in d.items():
        if isinstance(v, jax.Array):
            try:    n = v.size*v.dtype.itemsize
            except Exception: continue
            if n >= thr: out[k] = v
    return out or None

def jaxjit(fun: callable, *args, **kwargs) -> callable: 
    """ 
    use this instead of original jaxjit decorator function. For a method
    with self static (0 in static_argnums) the large arrays of self are
    passed as arguments, see above.
    """
    if not jax: return fun
    static = kwargs.get('static_argnums', ())
    if isinstance(static, int): static = (static,)
    if 0 not in static: return jax.jit(fun, *args, **kwargs)
    import functools
    # wraps: jax reads the ORIGINAL signature through __wrapped__, so the
    # positional static_argnums still match keyword calls (pad(a, lead=1))
    @functools.wraps(fun)
    def body(self, *a, _consts=None, **k):
        if not _consts: return fun(self, *a, **k)
        saved = {n: getattr(self, n) for n in _consts}
        for n, v in _consts.items(): setattr(self, n, v)
        try:     return fun(self, *a, **k)
        finally:
            for n, v in saved.items(): setattr(self, n, v)
    inner = jax.jit(body, *args, **kwargs)
    static_names = kwargs.get('static_argnames', ())
    if isinstance(static_names, str): static_names = (static_names,)
    # WHAT EACH CALL PASSES. Passing every large array of self to every call
    # costs ~0.2 ms per call (the collection, the flattening and checking of
    # ~60 extra arguments), ~10 ms per step over the ~60 jitted calls of an
    # ocean step. So the body is traced ONCE (jit's .trace, no compile) with
    # all of them, the jaxpr says which of those inputs the function reads,
    # and from then on only those are passed -- per instance and per set of
    # static argument values (a branch on a static flag may read others).
    # The name list itself is re-collected only when self gains attributes.
    names_cache = {}          # id(self) -> (number of attributes, names)
    usage = {}                # (id(self), static values) -> names used
    def outer(self, *a, **k):
        try:    d = vars(self)
        except TypeError: return inner(self, *a, **k)
        ent = names_cache.get(id(self))
        if ent is None or ent[0] != len(d):
            thr = 0 if jax.process_count() > 1 else JIT_CONST_BYTES
            names = tuple(n for n, v in d.items() if isinstance(v, jax.Array)
                          and v.size*v.dtype.itemsize >= thr)
            names_cache[id(self)] = (len(d), names)
        else:
            names = ent[1]
        skey = (id(self),) + tuple(a[i-1] for i in static if 0 < i <= len(a)) \
                           + tuple(sorted((n, k[n]) for n in static_names if n in k))
        try:    used = usage[skey]
        except (KeyError, TypeError):
            used = _used_consts(inner, self, a, k, names)
            try:    usage[skey] = used
            except TypeError: pass          # an unhashable static: no caching
        consts = {n: d[n] for n in used if n in d} or None
        return inner(self, *a, _consts=consts, **k)
    outer.jitted = inner
    return outer


def _used_consts(inner, self, a, k, names):
    """the names among `names` whose arrays the jitted body reads: one trace
    with all of them, then the inputs of the jaxpr that occur in an equation
    (an input handed to a nested jit counts as read)"""
    if not names: return ()
    consts = {n: getattr(self, n) for n in names}
    try:
        tr = inner.trace(self, *a, _consts=consts, **k)
        tree = tr.in_tree.unflatten(list(range(tr.in_tree.num_leaves)))
        cidx = tree[1]['_consts']                      # name -> leaf index
        jp = tr.jaxpr.jaxpr
        read = set()
        for e in jp.eqns:
            read.update(v for v in e.invars if isinstance(v, jax.core.Var))
        read.update(v for v in jp.outvars if isinstance(v, jax.core.Var))
        return tuple(n for n, i in cidx.items() if jp.invars[i] in read)
    except Exception:
        return names                                   # pass everything, as before


def while_loop(cond_fun, body_fun, init_state):
    """
    while loop for jax
    """
    if jax: return jax.lax.while_loop(cond_fun, body_fun, init_state)  
    else:    
       state = init_state
       while cond_fun(state): state = body_fun(state)
       return state


def for_loop(lower, upper, body_fun, init_val):
    """
    for loop for jax
    """
    if jax: return jax.lax.fori_loop(lower, upper, body_fun, init_val)
    else:    
       val = init_val
       for i in range(lower, upper):
           val = body_fun(i, val)
       return val


def cond(pred,positive_branch,negative_branch,x):
    """
    if-else clause for jax
    """
    if jax: return jax.lax.cond(pred,positive_branch,negative_branch,x)
    else:    
        if pred: return positive_branch(x)
        else:    return negative_branch(x)


def mpi_barrier():
    if mpi:  mpi_comm.Barrier()


def pe0_bcast(x):
    if mpi:  return mpi_comm.bcast(x,root=0)
    return x  


def global_sum(x):
    if mpi:       
       buf =  np.empty_like(x) 
       mpi_comm.Allreduce( np.ascontiguousarray(x),buf, op = mpi.SUM)
       return buf
    return x 


def global_max(x):
    if mpi:
       buf =  np.empty_like(x) 
       mpi_comm.Allreduce( np.ascontiguousarray(x),buf, op = mpi.MAX)        
       return buf
    return x


def send(buf, dest, tag):
    if mpi:  mpi_comm.Send(  np.ascontiguousarray(buf) , dest = dest, tag = tag )


def recv(buf, source, tag):
    if mpi: mpi_comm.Recv(buf,source = source, tag = tag)
    return buf


def _import_jax(which):
    
    if which == 'jax': 
           # XLA's GPU autotuner benchmarks every fused kernel with real-size
           # buffers at compile time; for large wim fields that takes minutes
           # per step for no measurable gain (README_jax_gpu.md). Off, unless
           # the user's XLA_FLAGS already set a level. Must precede import jax.
           if 'xla_gpu_autotune_level' not in _os.environ.get('XLA_FLAGS', ''):
               _os.environ['XLA_FLAGS'] = (_os.environ.get('XLA_FLAGS', '')
                                           + ' --xla_gpu_autotune_level=0').strip()
           import jax             
           try:
              #raise ValueError 
              # several processes: jax finds the cluster itself under SLURM
              # or Open MPI. PYOM3_JAX_COORDINATOR=host:port names the
              # coordinator explicitly (process count and index from the
              # launcher's variables) where that detection fails or hangs --
              # an mpirun on a workstation, say, or a proxied environment.
              _coord = _os.environ.get('PYOM3_JAX_COORDINATOR', '')
              if _coord:
                  _n = int(_os.environ.get('SLURM_NTASKS') or _os.environ.get('OMPI_COMM_WORLD_SIZE') or _os.environ.get('PMI_SIZE') or 1)
                  _r = int(_os.environ.get('SLURM_PROCID') or _os.environ.get('OMPI_COMM_WORLD_RANK') or _os.environ.get('PMI_RANK') or 0)
                  jax.distributed.initialize(coordinator_address=_coord, num_processes=_n, process_id=_r)
              else:
                  jax.distributed.initialize() 
              if jax.process_index() == 0: print('found Jax and initialised multihost Jax environment')    
           except ValueError:              print('found Jax but could not initialise multihost Jax environment') 
               
           jax.config.update("jax_enable_x64", True)     
           import jax.numpy as np

           # protect for old jax versions
           try: 
               from jax import shard_map as shard_map
           except ImportError:    
               from jax.experimental.shard_map import shard_map as shard_map
               
           try: 
             v = np.empty( (4,4) )
             v.at[2,2].set( 0 ,out_sharding = None) 
             out_sharding_present_in_set = True  
           except TypeError:
             out_sharding_present_in_set = False 
               
    else:  #jax not present  
        
           import numpy as np   
           jax = None  
           def shard_map(fun, *args, **kwargs ): 
               return fun           
           out_sharding_present_in_set = False 
        
    return (jax,np,shard_map,out_sharding_present_in_set)


# first try to import mpi4py
try: 
   #raise ImportError 
   from mpi4py import MPI as mpi
   mpi_comm = mpi.COMM_WORLD 
   my_pe, n_pes  = mpi_comm.Get_rank() ,  mpi_comm.Get_size()  
   if my_pe == 0:  print("found mpi4py")        
except ImportError:  
   mpi = None 
   mpi_comm , my_pe , n_pes  = None, 0, 1 


import os as _os
# PYOM3_BACKEND=numpy : numpy although jax is installed; =jax : jax although mpi4py is
# installed (single process). Unset: mpi4py present -> numpy, else jax if importable.
_backend = _os.environ.get('PYOM3_BACKEND', '').lower()
JIT_CONST_BYTES = int(_os.environ.get('PYOM3_JIT_CONST_BYTES', JIT_CONST_BYTES))
if (mpi and _backend != 'jax') or _backend == 'numpy':
   jax, np, shard_map, out_sharding_present_in_set = _import_jax('numpy') 
else:     
   # try to import JAX, fall back to numpy    
   for n in ('jax','numpy'):
       try: 
           jax, np, shard_map, out_sharding_present_in_set = _import_jax(n)
           break
       except RuntimeError: pass
       except ImportError:  pass   
   if jax and mpi:
       # jax forced although mpi4py is importable: the model must not see MPI
       # (every "if OM.mpi" would take the MPI reduction path). Single process.
       mpi, mpi_comm, my_pe, n_pes = None, None, 0, 1


if jax:  
   n_procs, n_devs, devs, my_proc = jax.process_count() , \
                 jax.device_count(), jax.devices(), jax.process_index()
   if my_proc == 0: 
      print('found ',n_procs,' processes and ',n_devs,' devices ')
      print('Device list :',devs)
        
   if out_sharding_present_in_set and my_proc == 0: 
        print("found keyword out_sharding in .at[].set())")
   elif my_proc == 0:                               
        print("found no keyword out_sharding in .at[].set())") 
            
else:
   n_procs, n_devs, devs, my_proc = 1, 1, None, 0 
   if my_pe == 0:  print("Jax not available")
    
prec = np.float64   



