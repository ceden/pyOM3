"""
 Preconditioned conjugate gradient for the non-hydrostatic pressure.

 Solves  A p = b  with  A = -V div grad  (symmetric, positive
 semi-definite, the null space is the constant on each connected wet
 region). cf holds the face coefficients of V div grad,
     cf[0] = a_e   east face of cell (i,j,k):   area_e / (cost dxu)
     cf[1] = a_n   north face:                  area_n / dyu
     cf[2] = a_t   top face:                    area_t / dzw
     cf[3] = diag  -(a_e + a_e[i-1] + a_n + a_n[j-1] + a_t + a_t[k-1]),
                   -1 on dry cells (A is the identity there, b = 0)
 so that  (V div grad p) = diag p + a_e p[i+1] + a_e[i-1] p[i-1] + ...
 The coefficient of the face shared by two cells is the same seen from
 both, which is what makes A symmetric on the variable grid.

 Preconditioner (Marshall et al. 1997; Oceananigans
 ColumnwiseTridiagonalPreconditioner): the vertical part of A solved
 exactly in every column with the Thomas algorithm, horizontal couplings
 dropped. The regularisation (1 + eps) on the diagonal lifts the
 Neumann null space of a column; cells without a vertical neighbour are
 treated as the identity.

 Gauge: the wet-cell mean is removed from the solution and the residual
 after every iteration (Oceananigans ZeroMeanGaugeCondition).

 Convergence criterion: the residual r = b - A p is, up to the factor
 -V/dt, the divergence the corrected velocity would still have, so the
 iteration stops when  dt max|r|/V < epsilon_nh,  with epsilon_nh the
 largest remaining divergence in 1/s (default 1e-12). The error returned
 is that number. (pe_simple's surface solve stops on the change of the
 solution instead; for the pressure that fixes the divergence, the
 divergence itself is the natural measure.)

 JAX compatible: OM.while_loop, OM.for_loop, OM.modify_array only.
"""
import pyOM3          as OM
from   functools      import partial


class solver_nh:

      @partial(OM.jaxjit, static_argnums=(0,))
      def _op_nh(self, p, cf):
          """A p = -(V div grad p), 7-point stencil with the face coefficients"""
          ae, an, at, dg = cf[0], cf[1], cf[2], cf[3]
          r = ( dg*p + ae*self.rollx(p,-1) + self.rollx(ae,1)*self.rollx(p,1)
                     + an*self.rolly(p,-1) + self.rolly(an,1)*self.rolly(p,1)
                     + at*self.rollz(p,-1) + self.rollz(at,1)*self.rollz(p,1) )
          return -r

      @partial(OM.jaxjit, static_argnums=(0,))
      def _pre_nh(self, r, cf, maskT):
          """
          z = M r with M the inverse of the vertical part of A, column by
          column (Thomas algorithm along axis 0). Vertical part of A at
          cell k:  -( a_t[k] (p[k+1]-p[k]) - a_t[k-1] (p[k]-p[k-1]) ),
          lower a[k] = -a_t[k-1], upper c[k] = -a_t[k], diagonal
          b[k] = (a_t[k] + a_t[k-1]) (1 + eps); identity on dry cells and
          on cells with no wet vertical neighbour.
          """
          eps = 0.01
          h, Nz = self.halo, self.Nz
          at   = cf[2]
          atm  = self.rollz(at,1)
          wet  = maskT > 0.0
          iso  = (at + atm) <= 0.0
          ident = OM.np.logical_or(OM.np.logical_not(wet), iso)
          a = OM.np.where(ident, 0.0, -atm)
          c = OM.np.where(ident, 0.0, -at)
          b = OM.np.where(ident, 1.0, (at + atm)*(1.0 + eps))
          d = OM.np.where(wet, r, 0.0)

          cp = OM.np.zeros_like(r)
          dp = OM.np.zeros_like(r)
          cp = OM.modify_array(cp, (h, slice(None), slice(None)), c[h]/b[h])
          dp = OM.modify_array(dp, (h, slice(None), slice(None)), d[h]/b[h])

          def _fwd(k, state):
              cp, dp = state
              m  = b[k] - a[k]*cp[k-1]
              cp = OM.modify_array(cp, (k, slice(None), slice(None)), c[k]/m)
              dp = OM.modify_array(dp, (k, slice(None), slice(None)), (d[k] - a[k]*dp[k-1])/m)
              return (cp, dp)
          cp, dp = OM.for_loop(h+1, Nz-h, _fwd, (cp, dp))

          z = OM.np.zeros_like(r)
          z = OM.modify_array(z, (Nz-h-1, slice(None), slice(None)), dp[Nz-h-1])
          def _bwd(i, z):
              k = Nz-h-2-i
              return OM.modify_array(z, (k, slice(None), slice(None)), dp[k] - cp[k]*z[k+1])
          z = OM.for_loop(0, (Nz-2*h)-1, _bwd, z)
          return z*maskT

      @partial(OM.jaxjit, static_argnums=(0,))
      def _wet_mean(self, x, maskT, n_wet):
          return self._dot(x, maskT)/n_wet

      @partial(OM.jaxjit, static_argnums=(0,))
      def cg_nh(self, b, x, cf, maskT, n_wet):
          """
          Preconditioned conjugate gradient, A x = b, start from x.
          Returns (x, error, iterations).
          """
          def cond_fun(state):
              p, x, res, z, n, error, rz = state
              return (n < self.max_iterations_nh) & (error > self.epsilon_nh) & (rz != 0)

          def body_fun(state):
              p, x, res, z, n, error, rz = state
              Ap    = self._op_nh(p, cf)
              pAp0  = self._dot(p, Ap)
              pAp   = OM.np.where(pAp0 == 0, 1e-32, pAp0)
              alpha = rz/pAp
              x    += alpha*p
              res  -= alpha*Ap
              # gauge: wet-cell mean of solution and residual to zero
              x    = (x   - self._wet_mean(x,  maskT, n_wet))*maskT
              res  = (res - self._wet_mean(res,maskT, n_wet))*maskT
              z     = self._pre_nh(res, cf, maskT)
              rznew = self._dot(res, z)
              p     = self.apply_bc( z + rznew/rz*p )
              error = self.dt*self._max(res/(self.area_t*self.dzt_z))
              return (p, x, res, z, n+1, error, rznew)

          x   = (x - self._wet_mean(x, maskT, n_wet))*maskT
          x   = self.apply_bc(x)
          res = (b - self._op_nh(x, cf))*maskT
          z   = self._pre_nh(res, cf, maskT)
          p   = self.apply_bc(z*1.0)
          rz  = self._dot(res, z)
          state = OM.while_loop(cond_fun, body_fun, (p, x, res, z, 0, self.epsilon_nh + 1.0, rz))
          p, x, res, z, n, error, rz = state
          return (self.apply_bc(x), error, n)
