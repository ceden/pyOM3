"""
 Equation of state for ocean models.
"""

import pyOM3 as OM

# TODO: add gsw from fortran code

def get_rho(salt,temp,press,eq_of_state_type): 
   """  
   #-----------------------------------------------------------------------
   # calculate density as a function of temperature, salinity and pressure
   #-----------------------------------------------------------------------
   """
   if eq_of_state_type == 1:
     rho = linear_eq_of_state_rho(salt,temp)
#   elif eq_of_state_type == 5:
#     rho = gsw_rho(salt,temp,press)
   elif eq_of_state_type == 100:
     rho = ideal_gas_eq_of_state_rho(temp,press)  
   elif eq_of_state_type == 0:
     rho = no_eq_of_state_rho(temp)    
   else:     
     raise pyOM3Error    
   return rho


def get_drhodT(salt,temp,press): 
   """ 
   #-----------------------------------------------------------------------
   # calculate drho/dT as a function of temperature, salinity and pressure
   #-----------------------------------------------------------------------
   """
   if eq_of_state_type == 1:
     drhodT = linear_eq_of_state_drhodT()
#   elif eq_of_state_type == 5:
#     drhodT = gsw_drhodT(salt,temp,press)
   elif eq_of_state_type == 100:
     drhodT = 0
   elif eq_of_state_type == 0:
     drhodT = 0 
   else:     
     raise pyOM3Error    
   return drhodT


def get_drhodS(salt,temp,press): 
   """ 
   #-----------------------------------------------------------------------
   # calculate drho/dS as a function of temperature, salinity and pressure
   #-----------------------------------------------------------------------
   """
   if eq_of_state_type == 1:
     drhodS = linear_eq_of_state_drhodS()
#   elif eq_of_state_type == 5:
#     drhodS = gsw_drhodS(salt,temp,press)
   elif eq_of_state_type == 100:
     drhodS = 0
   elif eq_of_state_type == 0:
     drhodS = 0 
   else:     
     raise pyOM3Error    
   return drhodS


@OM.jaxjit
def ideal_gas_eq_of_state_rho(ct,press):
   """ 
   #==========================================================================
   #  equation of state for an ideal gas atmosphere
   #  input is pressure in Pascal = kg/m/s^2 = 10^-4 dbar and  pot. temperature ct in Kelvin
   #  ct = T (p_c/p)^kappa, with in situ temperature T, reference pressure p_c, kappa=R/cp
   #  check section 1.5.2 of Vallis book 
   #  and https://mitgcm.readthedocs.io/en/latest/overview/hydro_prim_eqn.html
   #==========================================================================
   #R_gas = 287.0 # gas constant in J/kg/K
   """
   cp    = 1e3   # specfic heat at const. pressure in J/kg/K 
   p_c   = 1e5   # reference pressure in Pascal, 1 bar = 10^5 Pa
   kappa = 287./1e3 #, gamma = cp/(cp-R_gas) 
   exner = cp*(press/p_c)**kappa # Exner function 
   alpha = kappa*exner*ct/press  # specific volume in m^3/kg
   return alpha*1024.0/9.81 # to account for factors in hydrostatic equation


@OM.jaxjit
def no_eq_of_state_rho(b):
   return -b*1024.0/9.81 # to account for factors in hydrostatic equation


@OM.jaxjit
def linear_eq_of_state_rho(sa,ct):
   """ 
   #!==========================================================================
   #!  linear equation of state 
   #!  input is Salinity sa in g/kg, 
   #!  pot. temperature ct in deg C 
   #!==========================================================================
   # rho0 = 1024.0,theta0 = 283.0-273.15, S0 = 35.0
   # betaT = 1.67d-4, betaS = 0.78d-3 
   # grav = 9.81, z0=0.0 
   #  return  - (betaT*(ct-theta0) -betaS*(sa-S0) )*rho0
   """
   return  - (1.67e-4*(ct-(283.0-273.15)) -0.78e-3 *(sa-35.0) )*1024.0


@OM.jaxjit
def linear_eq_of_state_drhodT():
   return -1.67e-4*1024.0


@OM.jaxjit
def linear_eq_of_state_drhodS():
   return  0.78e-3*1024.0

