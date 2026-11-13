// The Newton C2P of examples/srhd/srhd.cu rewritten the way a user avoids a data-dependent exit for an AD
// tool: a fixed number of iterations and no `break` (after convergence the update is ~0). Used only to
// give Enzyme (and CuAD) a loop with a known trip count in the comparison.
#ifndef C2P_ITERS
#define C2P_ITERS 40
#endif
__global__ void c2p_newton_fixed(const double *U, double *P, const double *gam, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double D = U[i], S = U[n + i], tau = U[2 * n + i], g = gam[0];
    double pmin = fmax(fabs(S) - tau - D, 0.0) + 1e-14;
    double p = fmax((g - 1.0) * tau, pmin);
    for (int it = 0; it < C2P_ITERS; ++it) {
      double v = S / (tau + D + p);
      double W = 1.0 / sqrt(1.0 - v * v);
      double rho = D / W;
      double eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W);
      double f = (g - 1.0) * rho * eps - p;
      double cs2 = g * (g - 1.0) * eps / (1.0 + g * eps);
      double dp = f / (v * v * cs2 - 1.0);
      p = fmax(p - dp, pmin);
    }
    double v = S / (tau + D + p);
    double W = 1.0 / sqrt(1.0 - v * v);
    P[i] = D / W;
    P[n + i] = W * v;
    P[2 * n + i] = p;
  }
}
