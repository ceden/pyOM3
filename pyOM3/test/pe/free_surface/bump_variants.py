"""variants of the bump harness: implicit control, theta sweep, M sweep"""
import os, sys, numpy as np
sys.path.insert(0, os.environ.get('PYOM3','/home/claude/tree'))
import pyOM3 as OM
from pyOM3.pe_realistic.model import model as mr
from pyOM3.density import get_rho
nx=ny=30; nz=4; dz=1250.; H=nz*dz; dt=216.; grav=9.81
deg=2*np.pi*6.371e6/360.; lat0=70.
dx_m=0.1*deg*np.cos(np.deg2rad(lat0+1.5)); dy_m=0.1*deg
c=np.sqrt(grav*H); mcfl=dt*c*np.sqrt(1/dx_m**2+1/dy_m**2)
def make(explicit,M=None,friction=None):
    par=dict(nx=nx,ny=ny,nz=nz,dt=dt,periodic_in_x=False,periodic_in_y=False,eq_of_state=1,
             explicit_free_surface=explicit,implicit_free_surface=not explicit,superbee_advection=True,
             vmix_scheme='implicit',enable_conv_adjust=False,kappa_T=1e-4,A_v=1e-4,epsilon=1e-12,
             max_iterations=5000,dxt=np.full(nx,dx_m),dyt=np.full(ny,dy_m),dzt=np.full(nz,dz),
             coord_degree=False,x_origin=0.,y_origin=0.)
    if friction: par['Ahbi']=friction
    m=mr(par)
    if M: m.sub_cycling=M
    m.coriolis_t=OM.np.asarray(np.zeros((m.Ny,m.Nx)))
    kb=np.ones((m.Ny,m.Nx)); kb[[0,-1],:]=0; kb[:,[0,-1]]=0; m.kbot=OM.np.asarray(kb)
    m.prepare_for_run()
    m.temp=m.apply_bc(OM.np.asarray(np.full((m.Nz,m.Ny,m.Nx),10.)))*m.maskT
    m.salt=35.*m.maskT; m.rho=get_rho(m.salt,m.temp,m.p0,m.eq_of_state)
    h=m.halo; j=np.arange(m.Ny)[:,None]; i=np.arange(m.Nx)[None,:]
    eta=0.5*np.exp(-(((i-m.Nx/2)/4.)**2+((j-m.Ny/2)/4.)**2))
    ps=grav*eta*np.asarray(m.maskT)[-h-1]
    m.p_s=OM.np.asarray(ps if np.asarray(m.p_s).ndim==2 else np.repeat(ps[None],3,axis=0))
    m.set_forcing=(lambda self,tau: None).__get__(m)
    return m,float(np.sum((ps)[h:-h,h:-h]**2))
def go(explicit,M=None,nst=50,friction=None):
    m,e0=make(explicit,M,friction); h=m.halo; tau,taum1,taum2=2,1,0
    for n in range(nst):
        if explicit: m.time_step_explicit_pressure(1.6,-0.6,0.,tau,taum1,taum2)
        else:        m.time_step_implicit_pressure(1.6,-0.6,0.,tau,taum1,taum2)
        tau,taum1,taum2=(tau+1)%3,(taum1+1)%3,(taum2+1)%3
        u=np.asarray(m.u)
        if not np.isfinite(u).all() or np.abs(u).max()>50: return 'blew up at %d'%(n+1)
    ps=np.asarray(m.p_s); ps=ps[taum1] if ps.ndim==3 else ps
    return 'ratio %10.2f  max|u| %.4f'%(float(np.sum(ps[h:-h,h:-h]**2))/e0,np.abs(np.asarray(m.u)).max())
if __name__=='__main__':
    print('box: dx %.0f dy %.0f H %.0f c %.0f dt %.0f  M_cfl %.1f  c dt/dx %.1f'%(dx_m,dy_m,H,c,dt,mcfl,c*dt/dx_m))
    print('  implicit free surface (control):        %s'%go(False))
    print('  implicit, with biharmonic 1e11:         %s'%go(False,friction=1e11))
    for M in (31,62):
        print('  explicit M=%3d:                         %s'%(M,go(True,M)))
        print('  explicit M=%3d, biharmonic 1e11:        %s'%(M,go(True,M,friction=1e11)))
