"""

  Atmosphere as ideal gas after Vallis Section 2.6.1
  k runs here from p=0 to p=p_max, so top-down.

  initial conditions for temp as in Fig. 1 of 
  Snyder, Skamarock, Rotunno 91, JAS, A comparison of  ...

"""


import pyOM3 as OM
from pyOM3.pe_simple.snapshot  import snap_raw,snap_nc4
from pyOM3.density             import get_rho
jnp = OM.np

# set some parameter
Lx, Ly, Lz, f0 = 8000e3, 8000e3, 1e5, 1e-4 # ! 1bar = 10^5 Pa

R_gas, cp, p_C, u0 = 287.0 , 1e3, 1e5, 40.0 
kappa = R_gas/cp
p_T = 1.5e4 
dp_T = 0
N_strato , N_tropo = 0.02,   0.01



n_pes_x, n_pes_y = 1, 1
fac = 1  # scale factor
nx,ny,nz = fac*50, fac*50, fac*20
dx,dy,dz,dt = Lx/nx,Ly/ny,Lz/nz,96./fac

# dictionary with model parameter for intialisation
pa={ 'nx':nx, 'ny':ny, 'nz':nz, 
     'dx':dx, 'dy':dy, 'dz':dz, 
     'dt':dt, 'snapint':dt, 
     'max_iterations':5000, 'epsilon':1e-9,
     'Ahbi': 0.5e14/fac**3,   
     'eq_of_state':100,
     'implicit_free_surface':False,
     'periodic_in_x':True,'periodic_in_y':False,
     'explicit_free_surface':False, 'sub_cycling':10}



# define initial conditions, Jax compatible
class jax_model(snap_nc4): # inheret here from snap_raw instead for raw output

      def my_init(self):  
          """
          set some initial conditions
          """
          x_full = jnp.arange(nx,dtype=jnp.float64)*dx +dx/2.
          y_full = jnp.arange(ny,dtype=jnp.float64)*dy +dy/2.
          z_full = jnp.arange(nz,dtype=jnp.float64)*dz +dz/2

          self.u , self.v, self.temp = self.unpad(self.u), self.unpad(self.v), self.unpad(self.temp)
          uz = jnp.zeros_like(self.u)
          for shard in self.u.addressable_shards:    
              s = shard.index
              z,y,x = z_full[s[0]]  , y_full[s[1]] , x_full[s[2]]                  
              Z,Y,X = jnp.meshgrid(z,y,x, indexing='ij')

              fxb = p_T + dp_T*jnp.tanh( (Y-Ly/2.)/1000e3 )
              fxa = jnp.where(Z< fxb, Z/fxb, 1.-(Z-fxb)/(p_C-fxb))
              u = u0*jnp.exp(-(Y-Ly/2)**2/1000e3**2 )*fxa
              
              # p_z = - g \rho
              # N^2 = g/theta dtheta/dz = g/theta dtheta/dp dp/dz = g^2/(theta*alpha) dtheta/dp
              
              exner = cp*(Z/p_C)**kappa # Exner function 
              fxb = p_T + dp_T*jnp.tanh( (Y-Ly/2.)/1000e3 )
              N2 = jnp.where( Z<fxb, N_strato**2, N_tropo**2)
              theta = jnp.zeros_like(u)  

              theta = OM.modify_array(theta, (-1,slice(None),slice(None)) , 300)
              for k in range(nz-2,-1,-1):
                exner = cp*(Z[k+1,:,:]/p_C)**kappa # Exner function 
                _add = theta[k+1,:,:] +  N2[k,:,:]*kappa*exner*theta[k+1,:,:]**2/Z[k+1,:,:]/self.grav**2*self.dz
                theta = OM.modify_array(theta, (k,slice(None),slice(None)) , _add ) 
                  
              theta +=  0.05*jnp.cos(jnp.pi*Z/p_C) * jnp.sin(2*2*jnp.pi*X/Lx)
              
              _uz = jnp.zeros_like(u)  
              _uz = OM.modify_array( _uz,  (slice(1,-1),slice(None),slice(None)) , (u[2:,:,:]-u[:-2,:,:])/(2*dz) )
              _uz = OM.modify_array( _uz,  (-1,slice(None),slice(None))          , (u[-1,:,:]-u[-2,:,:])/dz )
              _uz = OM.modify_array( _uz,  ( 0,slice(None),slice(None))          , (u[ 1,:,:]-u[ 0,:,:])/dz)
              fxa = f0*Z**(1.-kappa)/(kappa*cp/p_C**kappa) 
              _uz = _uz*dy*fxa 
                            
              v = jnp.zeros_like(u)  
                            
              self.u = OM.modify_array(self.u, s, u, out_sharding = self.sharding_3D)    
              self.v = OM.modify_array(self.v, s, v, out_sharding = self.sharding_3D)
              self.temp = OM.modify_array(self.temp, s, theta, out_sharding = self.sharding_3D)     
              uz = OM.modify_array(uz, s, _uz, out_sharding = self.sharding_3D)   

              
          self.temp += jnp.cumsum(uz,axis=1)
          
          self.u , self.v, self.temp = self.pad(self.u), self.pad(self.v), self.pad(self.temp)    
          self.u, self.v  = self.apply_bc(  self.u) , self.apply_bc(  self.v)
          self.w    = self.vertical_velocity(self.u,self.v,self.w,self.maskW) 
          self.temp = self.apply_bc(  self.temp) 
          self.rho  = get_rho(self.salt,self.temp,self.p0,self.eq_of_state)
          self.p_h = self.hydrostatic_pressure(self.rho,self.p_h,self.maskT)
          return    


M = jax_model(pa, n_pes_x = n_pes_x, n_pes_y = n_pes_y)
M.set_coriolis(f0)
M.set_topography()
M.prepare_for_run( snap_file_name = 'snap.cdf',show_each_time_step = True)
M.my_init()
##M.diagnose(0,0)
M.snapint = 86400.
M.loop(max_steps=int(15*86400/dt)+1)  
