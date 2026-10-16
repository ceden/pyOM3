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
   elif eq_of_state_type == 5:
     rho = gsw_rho(salt,temp,press)
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
   elif eq_of_state_type == 5:
     drhodT = gsw_drhodT(salt,temp,press)
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
   elif eq_of_state_type == 5:
     drhodS = gsw_drhodS(salt,temp,press)
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



# ---------------------------------------------------------------------------
# GSW / TEOS-10 48-term rational-function equation of state (McDougall et al.
# 2003 / IOC 2010), transcribed from pyOM2 for_src/density/density.f90
# (module gsw_eq_of_state).  Density is returned as the anomaly to
# rho0 = 1024 kg/m^3, as in pyOM2.  Pressure p in dbar (~ depth in m);
# p = None falls back to 0 dbar (surface-referenced), which is only used
# by the convective-adjustment check when that option is active.
# ---------------------------------------------------------------------------
_gsw_rho0 = 1024.0
v01 = 999.842089750606; v02 = 2.83994083316191; v03 = -0.0314775926558851
v04 = 0.00118180554507431; v05 = -6.6980010711238; v06 = -0.0298649894720322
v07 = 0.000232785940747916; v08 = -0.0398882237896849; v09 = 0.00050954225738805
v10 = -1.42698467163362e-05; v11 = 1.64503937368292e-07; v12 = -0.0223326962735253
v13 = -0.000343609007985188; v14 = 3.72605072034573e-06; v15 = -0.000180678976374533
v16 = 6.87683721953623e-07; v17 = -3.08703250037421e-07; v18 = -1.98836658792559e-08
v19 = -1.06151907029646e-11; v20 = 1.55093272922008e-10; v21 = 1
v22 = 0.00277592774778565; v23 = -2.34960744413592e-05; v24 = 1.11951335748674e-06
v25 = 6.74368932504277e-10; v26 = -0.00752144809361545; v27 = -2.76430697989441e-05
v28 = 1.26293731509855e-07; v29 = 9.52787508169644e-10; v30 = -1.81114720194989e-11
v31 = -3.30330887138642e-05; v32 = 3.8015645888763e-07; v33 = -7.67287686925904e-09
v34 = -4.63418234111614e-11; v35 = 2.68109723556914e-12; v36 = 5.41932655114874e-06
v37 = -2.7421853949061e-05; v38 = -3.21274647797419e-07; v39 = 3.19141391056163e-09
v40 = -1.93101293154178e-12; v41 = -1.10509757714958e-07; v42 = 6.21142672836386e-10
v43 = -1.11901159287511e-10; v44 = -1.94166021314872e-11; v45 = -1.8648264253656e-14
v46 = 1.11952234487948e-14; v47 = -1.2005077485516e-15; v48 = 6.05790248754687e-17
a01 = 2.83994083316191; a02 = -0.0629551853117702; a03 = 0.00354541663522292
a04 = -0.0298649894720322; a05 = 0.000465571881495832; a06 = 0.00050954225738805
a07 = -2.85396934326724e-05; a08 = 4.93511812104877e-07; a09 = -0.000343609007985188
a10 = 7.45210144069147e-06; a11 = 6.87683721953623e-07; a12 = -1.98836658792559e-08
a13 = -2.12303814059292e-11; a14 = 0.00277592774778565; a15 = -4.69921488827185e-05
a16 = 3.35854007246023e-06; a17 = 2.69747573001711e-09; a18 = -2.76430697989441e-05
a19 = 2.52587463019709e-07; a20 = 2.85836252450893e-09; a21 = -7.24458880779956e-11
a22 = 3.8015645888763e-07; a23 = -1.53457537385181e-08; a24 = -1.39025470233484e-10
a25 = 1.07243889422766e-11; a26 = -3.21274647797419e-07; a27 = 6.38282782112325e-09
a28 = -5.79303879462533e-12; a29 = 6.21142672836386e-10; a30 = -1.94166021314872e-11
a31 = -3.7296528507312e-14; a32 = 1.11952234487948e-14; a33 = 6.05790248754687e-17
b01 = -6.6980010711238; b02 = -0.0298649894720322; b03 = 0.000232785940747916
b04 = -0.0598323356845274; b05 = 0.000764313386082075; b06 = -2.14047700745043e-05
b07 = 2.46755906052438e-07; b08 = -0.000180678976374533; b09 = 6.87683721953623e-07
b10 = 1.55093272922008e-10; b11 = -0.00752144809361545; b12 = -2.76430697989441e-05
b13 = 1.26293731509855e-07; b14 = 9.52787508169644e-10; b15 = -1.81114720194989e-11
b16 = -4.95496330707963e-05; b17 = 5.70234688331445e-07; b18 = -1.15093153038886e-08
b19 = -6.95127351167422e-11; b20 = 4.02164585335372e-12; b21 = 1.08386531022975e-05
b22 = -1.10509757714958e-07; b23 = 6.21142672836386e-10; b24 = 1.11952234487948e-14


def _gsw_num_den(sa, ct, p):
    sqrtsa = OM.np.sqrt(OM.np.maximum(sa, 0.0))
    den = (v01 + ct*(v02 + ct*(v03 + v04*ct))
           + sa*(v05 + ct*(v06 + v07*ct)
           + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct))))
           + p*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct)
           + p*(v17 + ct*(v18 + v19*ct) + v20*sa)))
    num = (v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct)))
           + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa
           + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct)))))
           + p*(v37 + ct*(v38 + ct*(v39 + v40*ct))
           + sa*(v41 + v42*ct) + p*(v43 + ct*(v44 + v45*ct + v46*sa)
           + p*(v47 + v48*ct))))
    return num, den, sqrtsa


@OM.jaxjit
def gsw_rho(sa, ct, p):
    """density anomaly to 1024 as function of SA [g/kg], CT [degC], p [dbar]"""
    p = 0.0 if p is None else p
    num, den, _ = _gsw_num_den(sa, ct, p)
    return den/num - _gsw_rho0


@OM.jaxjit
def gsw_drhodT(sa, ct, p):
    p = 0.0 if p is None else p
    num, den, sqrtsa = _gsw_num_den(sa, ct, p)
    dden = (a01 + ct*(a02 + a03*ct)
            + sa*(a04 + a05*ct + sqrtsa*(a06 + ct*(a07 + a08*ct)))
            + p*(a09 + a10*ct + a11*sa + p*(a12 + a13*ct)))
    dnum = (a14 + ct*(a15 + ct*(a16 + a17*ct))
            + sa*(a18 + ct*(a19 + ct*(a20 + a21*ct))
            + sqrtsa*(a22 + ct*(a23 + ct*(a24 + a25*ct))))
            + p*(a26 + ct*(a27 + a28*ct) + a29*sa
            + p*(a30 + a31*ct + a32*sa + a33*p)))
    rec = 1.0/num
    return (dden - dnum*(rec*den))*rec


@OM.jaxjit
def gsw_drhodS(sa, ct, p):
    p = 0.0 if p is None else p
    num, den, sqrtsa = _gsw_num_den(sa, ct, p)
    dden = (b01 + ct*(b02 + b03*ct)
            + sqrtsa*(b04 + ct*(b05 + ct*(b06 + b07*ct)))
            + p*(b08 + b09*ct + b10*p))
    dnum = (b11 + ct*(b12 + ct*(b13 + ct*(b14 + b15*ct)))
            + sqrtsa*(b16 + ct*(b17 + ct*(b18 + ct*(b19 + b20*ct))))
            + b21*sa + p*(b22 + ct*(b23 + b24*p)))
    rec = 1.0/num
    return (dden - dnum*(rec*den))*rec


@OM.jaxjit
def gsw_drhodTS(sa, ct, p):
    """drho/dCT and drho/dSA together from one evaluation of the rational
    function (used by the isoneutral module; half the cost of calling
    gsw_drhodT and gsw_drhodS separately)"""
    p = 0.0 if p is None else p
    num, den, sqrtsa = _gsw_num_den(sa, ct, p)
    rec = 1.0/num
    fac = rec*den
    ddenT = (a01 + ct*(a02 + a03*ct)
             + sa*(a04 + a05*ct + sqrtsa*(a06 + ct*(a07 + a08*ct)))
             + p*(a09 + a10*ct + a11*sa + p*(a12 + a13*ct)))
    dnumT = (a14 + ct*(a15 + ct*(a16 + a17*ct))
             + sa*(a18 + ct*(a19 + ct*(a20 + a21*ct))
             + sqrtsa*(a22 + ct*(a23 + ct*(a24 + a25*ct))))
             + p*(a26 + ct*(a27 + a28*ct) + a29*sa
             + p*(a30 + a31*ct + a32*sa + a33*p)))
    ddenS = (b01 + ct*(b02 + b03*ct)
             + sqrtsa*(b04 + ct*(b05 + ct*(b06 + b07*ct)))
             + p*(b08 + b09*ct + b10*p))
    dnumS = (b11 + ct*(b12 + ct*(b13 + ct*(b14 + b15*ct)))
             + sqrtsa*(b16 + ct*(b17 + ct*(b18 + ct*(b19 + b20*ct))))
             + b21*sa + p*(b22 + ct*(b23 + b24*p)))
    return (ddenT - dnumT*fac)*rec, (ddenS - dnumS*fac)*rec
