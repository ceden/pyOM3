"""
 Simple JAX compatible conjugate gradient (CG) iterative solver with simple preconditioner.
 Also a BiCGStab solver, see https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method.
 
 Both with and with preconditioner (which does not help much).
 
"""
import pyOM3          as OM
from   functools      import partial


class solver():

    
      @partial(OM.jaxjit, static_argnums=(0,))
      def cg(self,y,x,cf):  
          """
           Solves A*x = y for x. cf contains coefficients in A.
           Allows for JAX optimisation.
           epsilon and error is here largest absolute change in x from i to i+1
          """
          def cond_fun(state):  
              p,x,res, n,error,rsold = state 
              return (n < self.max_iterations) & (error > self.epsilon) & (rsold !=0)
        
          def body_fun(state):
              p,x,res,  n,error,rsold = state             
              Ap = self._op(p,cf)
              fxa0 =  self._dot(p,Ap)
              fxa = OM.np.where(fxa0 == 0, 1e-32, fxa0)
              alpha = rsold/fxa
              x   += alpha*p 
              res -= alpha*Ap 
              rsnew = self._dot(res,res)
              p     = self.apply_bc( res + rsnew/rsold*p)            
              rsold = rsnew            
              error = abs(alpha) * self._max(p)  
              n += 1               
              return (p,x,res,n,error,rsold)
                 
          res = x*0
          res = y - self._op(x,cf)
          p   = self.apply_bc( res.copy())       
          init_state = (p,x,res, 0, self.epsilon+1, self._dot(res,res)  ) 
          state = OM.while_loop(cond_fun, body_fun, init_state)    
          p,x,res, n,error,rsold = state  
          return (x,error,n)
       
    
      @partial(OM.jaxjit, static_argnums=(0,))
      def bicgstab(self,b,x,cf):  
          """
          Unpreconditioned BiCGStab from wikipedia for JAX.
          epsilon and error is here largest absolute change in x from i to i+1
          """
          def cond_fun(state):
              x, r, rhat, p, rho, k,error = state
              #rs = self._dot(r, r)              
              return (error > self.epsilon) & (k < self.max_iterations) 

          def body_fun(state):
              x, r, rhat, p, rho, k,error = state
              p = self.apply_bc(p)
              v = self._op(p,cf)
              alpha = rho/self._dot(rhat,v)
              h = x + alpha*p
              s = r - alpha*v
              error = abs(alpha) * self._max(p)  
              #If h is accurate enough, i.e., if s is small enough, then set x_i = h and quit
              #exit_early = self._dot(s, s) < self.epsilon 
              s = self.apply_bc(s)    
              t = self._op(s,cf)
              omega = self._dot(t,s)/self._dot(t,t)
              x = h + omega*s
              r = s - omega*t
              #def positive_branch(y):
              #    (h,omega,s,t) = y
              #    return h, s                  
              #def negative_branch(y):
              #    (h,omega,s,t) = y
              #    return h + omega*s, s - omega*t
              #x,r = OM.cond(exit_early,positive_branch,negative_branch,(h,omega,s,t))                                
              rho_old = rho
              rho = self._dot(rhat,r)
              beta = rho/rho_old  *alpha/omega              
              p = r + beta*(p - omega*v)          
              return (x, r, rhat, p, rho, k+1,error)

          x = self.apply_bc(x)
          r = b - self._op(x,cf)
          rhat = OM.jax.random.normal(OM.jax.random.key(0),shape= r.shape)
          rho = self._dot(r,rhat)
          p = r*1
          x, r, rhat, p, rho, k,error =  OM.while_loop(cond_fun, body_fun, (x, r, rhat, p, rho, 0,self.epsilon+1)  )
          return (x,error,k)
 

    
      @partial(OM.jaxjit, static_argnums=(0,))
      def bicgstab_pre(self,b,x0,cf):  
          """
          Preconditioned BiCGStab adapted from jax.scipy. Does not help much.
          epsilon and error is here summed squared residuals (Ax-b)^2
          """
          def cond_fun(state):
              x, r, *_, k = state
              rs = self._dot(r, r)   
              return (rs > self.epsilon) & (k < self.max_iterations) & (k >= 0)

          def body_fun(state):
              x, r, rhat, alpha, omega, rho, p, q, k = state
              rho_ = self._dot(rhat, r)
              beta = rho_ / rho * alpha / omega
              p_ = r +  beta*(p-  omega*q) 
              phat = self._pre(p_,cf)
              phat = self.apply_bc(phat)
              q_ = self._op(phat,cf)
              alpha_ = rho_ / self._dot(rhat, q_)
              s = r  -  alpha_*q_
              exit_early = self._dot(s, s) < self.epsilon
              shat = self._pre(s,cf)
              shat = self.apply_bc(shat)
              t = self._op(shat,cf)
              omega_ = self._dot(t, s) / self._dot(t, t)  # make cases?
              
              def positive_branch(y):
                  x,phat,s,shat,t,alpha_,omega_ = y
                  return x +  alpha_* phat , s                  
              def negative_branch(y):
                  x,phat,s,shat,t,alpha_,omega_ = y
                  return x + alpha_* phat +  omega_*shat  , s -   omega_*t                  
              x_,r_ = OM.cond(exit_early,positive_branch,negative_branch,(x,phat,s,shat,t,alpha_,omega_))                                
              
              k_ = OM.np.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
              k_ = OM.np.where((rho_ == 0), -10, k_)
              return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

          x0 = self.apply_bc(x0)
          r0 = b - self._op(x0,cf)
          rho0 = alpha0 = omega0 = 1.
          initial_state = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)
          x, *_, k = OM.while_loop(cond_fun, body_fun, initial_state)
          res =    b - self._op(self.apply_bc(x),cf) 
          error = self._dot(res,res)
          return (x,error,k)
            
    
      @partial(OM.jaxjit, static_argnums=(0,))
      def cg_pre(self,y,x,cf):  
          """
           Solves A*x = y for x. cf contains coefficients in A with simple preconditioner.
           Allows for JAX optimisation.
           epsilon and error is here largest absolute change in x from i to i+1
          """
          def cond_fun(state):  
              p,x,res, n,error,rsold = state 
              return (n < self.max_iterations) & (error > self.epsilon) & (rsold !=0)
        
          def body_fun(state):
              p,x,res,  n,error,rsold = state             
              Ap = self._op(p,cf)
              fxa0 =  self._dot(p,Ap)
              fxa = OM.np.where(fxa0 == 0, 1e-32, fxa0)
              alpha = rsold/fxa
              x   += alpha*p 
              res -= alpha*Ap 
              z = self._pre(res,cf)     ###### new
              rsnew = self._dot(z,res)  ###### chg
              p     = self.apply_bc( z + rsnew/rsold*p)      ###### chg        
              rsold = rsnew            
              error = abs(alpha) * self._max(p)  
              n += 1               
              return (p,x,res,n,error,rsold)
                 
          res = x*0
          res = y - self._op(x,cf)
          z = self._pre(res,cf)  #### new
          p   = self.apply_bc( z.copy())  ###### chg  
          rsold = self._dot(z,res) ###### chg 
          init_state = (p,x,res, 0, self.epsilon+1, rsold  ) 
          state = OM.while_loop(cond_fun, body_fun, init_state)    
          p,x,res, n,error,rsold = state    
          return (x,error,n)
 
    
      def cg_nojax(self,y,x,cf): 
          """
           Solves A*x = y for x. cf contains coefficients in A.
           Just for reference and to show how ugly it gets with jax
          """ 
          res = y - self._op(x,cf)
          p   = self.apply_bc(res*1)
          rsold = self._dot(res,res)  
          n, error = 0 , self.epsilon+1 
          while n < self.max_iterations and rsold !=0 and error > self.epsilon:
                Ap = self._op(p,cf)
                fxa0 =  self._dot(p,Ap)
                fxa = OM.np.where(fxa0 == 0, 1e-32, fxa0)  
                alpha = rsold/fxa
                x   +=  alpha * p 
                res -=  alpha * Ap 
                rsnew = self._dot(res,res)
                p   = self.apply_bc(res + rsnew/rsold*p)
                rsold = rsnew            
                error = abs(alpha) * self._max(p)  
                n += 1       
          return (x,error,n)

           
      def cg_pre_nojax(self,y,x,cf):  
         """
           Solves A*x = y for x. cf contains coefficients in A with simple preconditioner.
           Just for reference and to show how ugly it gets with jax
          """ 
         res = y - self._op(x,cf)
         z = self._pre(res,cf)            ###### new
         p = self.apply_bc( z.copy() )    ###### chg
         rsold = self._dot(z,res)         ###### chg
         n, error = 0 , self.epsilon+1 
         while n < self.max_iterations and rsold !=0 and error > self.epsilon:
                Ap = self._op(p,cf)
                fxa0 = self._dot(p,Ap)
                fxa = OM.np.where(fxa0 == 0, 1e-32, fxa0)               
                alpha = rsold / fxa
                x   += alpha * p
                res -= alpha * Ap
                z = self._pre(res,cf)     ###### new
                rsnew = self._dot(z,res)  ###### chg
                p = self.apply_bc( z + (rsnew / rsold) * p )     ###### chg    
                rsold = rsnew            
                error = abs(alpha) * self._max(p)  
                n += 1       
         return (x,error,n)
        

    
    
                
      @partial(OM.jaxjit, static_argnums=(0,))
      def _max(self,p):
          a = OM.np.abs( self.unpad(p) )
          return OM.global_max( OM.np.amax(a) ) if OM.mpi else OM.np.max(a)           

    
      @partial(OM.jaxjit, static_argnums=(0,))
      def _dot(self,p1,p2): 
          a = self.unpad(p1*p2)
          return OM.global_sum( OM.np.sum(a) ) if OM.mpi else OM.np.sum(a)           

    
      @partial(OM.jaxjit, static_argnums=(0,))     
      def _op(self,p1,cf): 
          # there are no off-diagonal elements for surface pressure, in general there might be
          r  = cf[ 0+1, 0+1, :,:]*p1
          r += cf[ 1+1, 0+1, :,:]*self.rolly(p1,-1)
          r += cf[-1+1, 0+1, :,:]*self.rolly(p1, 1) 
          r += cf[ 0+1, 1+1, :,:]*self.rollx(p1,-1)
          r += cf[ 0+1,-1+1, :,:]*self.rollx(p1, 1) 
          return r

    
      @partial(OM.jaxjit, static_argnums=(0,)) 
      def _pre(self,p1,cf):      
          return p1*cf[ 0+1, 0+1, :,:]
          

          