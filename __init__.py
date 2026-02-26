
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
        if out_sharding_present_in_set and out_sharding:
                return arr.at[where].set(value  , out_sharding = out_sharding  )
        else:   return arr.at[where].set(value) 
    res = arr.copy()
    res[where] = value
    return res


def add_to_array(arr, where, value, out_sharding = None):
    """
    same but adding to old value
    """  
    if jax: 
        if out_sharding_present_in_set:
           return arr.at[where].add(value , out_sharding = out_sharding  ) 
        else:    
           return arr.at[where].add(value  )
    res = arr.copy()
    res[where] += value
    return res


def jaxjit(fun: callable, *args, **kwargs) -> callable: 
    """ 
    use this instead of original jaxjit decorator function
    """
    if jax:   return jax.jit(fun, *args, **kwargs) 
    return fun


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
        
           import jax             
           try:
              #raise ValueError 
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


if mpi:
   jax, np, shard_map, out_sharding_present_in_set = _import_jax('numpy') 
else:     
   # try to import JAX, fall back to numpy    
   for n in ('jax','numpy'):
       try: 
           jax, np, shard_map, out_sharding_present_in_set = _import_jax(n)
           break
       except RuntimeError: pass
       except ImportError:  pass   


if jax:  
   n_procs, n_devs, devs, my_proc = jax.process_count() , jax.device_count(), jax.devices(), jax.process_index()
   if my_proc == 0: 
      print('found ',n_procs,' processes and ',n_devs,' devices ')
      print('Device list :',devs)
        
   if out_sharding_present_in_set and my_proc == 0: print("found keyword out_sharding in .at[].set())")
   elif my_proc == 0:                               print("found no keyword out_sharding in .at[].set())") 
            
else:
   n_procs, n_devs, devs, my_proc = 1, 1, None, 0 
   if my_pe == 0:  print("Jax not available")
    
prec = np.float64   



