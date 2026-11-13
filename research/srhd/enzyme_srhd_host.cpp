// Enzyme (LLVM-level reverse AD) of the same SRHD source, host build: the conservative-to-primitive
// kernels and the whole time integration. Unmodified examples/srhd/srhd.cu; Enzyme differentiates the
// iterations of the C2P loop as they run.
#include <cmath>
#include <cstdlib>
#include <cstring>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "srhd.cu"

extern int enzyme_dup, enzyme_const, enzyme_dupnoneed;
template <class... A> void __enzyme_autodiff(void *, A...);

// sum_k P_k Pb_k as a function of (U, gam)
static double c2p_dot(int solver, int n, const double *U, const double *gam, const double *Pb) {
  double *P = (double *)std::malloc(sizeof(double) * 3 * n);
  if (solver == 0) c2p_newton(U, P, gam, n); else c2p_bisect(U, P, gam, n);
  double s = 0;
  for (int k = 0; k < 3 * n; ++k) s += P[k] * Pb[k];
  std::free(P);
  return s;
}
extern "C" void enzyme_c2p_adjoint(int solver, int n, const double *U, const double *gam, const double *Pb,
                                   double *Ub, double *gamb) {
  __enzyme_autodiff((void *)c2p_dot, enzyme_const, solver, enzyme_const, n, enzyme_dup, U, Ub, enzyme_dup, gam,
                    gamb, enzyme_const, Pb);
}

// J = 0.5 dx sum (rho - obs)^2 at t = steps * cfl * dx; q = (rhoL, uL, pL, rhoR, uR, pR, gam)
static double misfit(const double *q, int n, int steps, double cfl, int solver, const double *obs) {
  const double dx = 1.0 / n;
  double g[1] = {q[6]};
  double *W = (double *)std::malloc(sizeof(double) * 3 * n), *U = (double *)std::malloc(sizeof(double) * 3 * n);
  double *U1 = (double *)std::malloc(sizeof(double) * 3 * n), *U2 = (double *)std::malloc(sizeof(double) * 3 * n);
  double *F = (double *)std::malloc(sizeof(double) * 3 * (n + 1));
  for (int i = 0; i < n; ++i) {
    bool left = (i + 0.5) / n < 0.5;
    W[i] = left ? q[0] : q[3];
    W[n + i] = left ? q[1] : q[4];
    W[2 * n + i] = left ? q[2] : q[5];
  }
  p2c(W, U, g, n);
  for (int s = 0; s < steps; ++s) {
    if (solver == 0) c2p_newton(U, W, g, n); else c2p_bisect(U, W, g, n);
    hll_flux(W, F, g, n);
    rk_stage(U, U, F, U1, 0.0, 1.0, cfl, n);
    if (solver == 0) c2p_newton(U1, W, g, n); else c2p_bisect(U1, W, g, n);
    hll_flux(W, F, g, n);
    rk_stage(U, U1, F, U2, 0.5, 0.5, cfl, n);
    std::memcpy(U, U2, sizeof(double) * 3 * n);
  }
  if (solver == 0) c2p_newton(U, W, g, n); else c2p_bisect(U, W, g, n);
  double J = 0;
  for (int i = 0; i < n; ++i) J += 0.5 * dx * (W[i] - obs[i]) * (W[i] - obs[i]);
  std::free(W); std::free(U); std::free(U1); std::free(U2); std::free(F);
  return J;
}
// The inverse problem's misfit (inverse_density.py): unknown initial density field rho0, fixed pressure p0 and
// zero velocity, observations of the primitives at step indices kst[j] with weights w[3n].
static double misfit_obs(const double *rho0, const double *p0, int n, int steps, double cfl, int solver, double gam,
                         int nobs, const int *kst, const double *obs, const double *w) {
  const double dx = 1.0 / n;
  double g[1] = {gam};
  double *W = (double *)std::malloc(sizeof(double) * 3 * n), *U = (double *)std::malloc(sizeof(double) * 3 * n);
  double *U1 = (double *)std::malloc(sizeof(double) * 3 * n), *U2 = (double *)std::malloc(sizeof(double) * 3 * n);
  double *F = (double *)std::malloc(sizeof(double) * 3 * (n + 1));
  for (int i = 0; i < n; ++i) {
    W[i] = rho0[i];
    W[n + i] = 0.0;
    W[2 * n + i] = p0[i];
  }
  p2c(W, U, g, n);
  double J = 0;
  for (int s = 0; s <= steps; ++s) {
    for (int j = 0; j < nobs; ++j)
      if (kst[j] == s) {
        if (solver == 0) c2p_newton(U, W, g, n); else c2p_bisect(U, W, g, n);
        for (int k = 0; k < 3 * n; ++k) J += 0.5 * dx * w[k] * (W[k] - obs[j * 3 * n + k]) * (W[k] - obs[j * 3 * n + k]);
      }
    if (s == steps) break;
    if (solver == 0) c2p_newton(U, W, g, n); else c2p_bisect(U, W, g, n);
    hll_flux(W, F, g, n);
    rk_stage(U, U, F, U1, 0.0, 1.0, cfl, n);
    if (solver == 0) c2p_newton(U1, W, g, n); else c2p_bisect(U1, W, g, n);
    hll_flux(W, F, g, n);
    rk_stage(U, U1, F, U2, 0.5, 0.5, cfl, n);
    std::memcpy(U, U2, sizeof(double) * 3 * n);
  }
  std::free(W); std::free(U); std::free(U1); std::free(U2); std::free(F);
  return J;
}
extern "C" double enzyme_obs_grad(const double *rho0, const double *p0, int n, int steps, double cfl, int solver,
                                  double gam, int nobs, const int *kst, const double *obs, const double *w,
                                  double *grad) {
  for (int i = 0; i < n; ++i) grad[i] = 0;
  __enzyme_autodiff((void *)misfit_obs, enzyme_dup, rho0, grad, enzyme_const, p0, enzyme_const, n, enzyme_const,
                    steps, enzyme_const, cfl, enzyme_const, solver, enzyme_const, gam, enzyme_const, nobs,
                    enzyme_const, kst, enzyme_const, obs, enzyme_const, w);
  return misfit_obs(rho0, p0, n, steps, cfl, solver, gam, nobs, kst, obs, w);
}

extern "C" double enzyme_misfit(const double *q, int n, int steps, double cfl, int solver, const double *obs) {
  return misfit(q, n, steps, cfl, solver, obs);
}
extern "C" void enzyme_misfit_grad(const double *q, int n, int steps, double cfl, int solver, const double *obs,
                                   double *grad) {
  for (int k = 0; k < 7; ++k) grad[k] = 0;
  __enzyme_autodiff((void *)misfit, enzyme_dup, q, grad, enzyme_const, n, enzyme_const, steps, enzyme_const, cfl,
                    enzyme_const, solver, enzyme_const, obs);
}
