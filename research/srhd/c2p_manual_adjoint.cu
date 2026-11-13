// Hand-written implicit adjoint of the conservative-to-primitive step (baseline for the generated one).
// The primal is recovered by the same Newton iteration as c2p_newton in examples/srhd/srhd.cu. The adjoint uses the
// analytic primitive-to-conservative map U(P; gam), P = (rho, u, p), u = W v:
//   dP = A^{-1} dU with A = dU/dP,  so  U_b += A^{-T} P_b  and  gam_b -= (A^{-T} P_b) . dU/dgam,
// one 3x3 solve (Cramer's rule) per cell. Nothing is stored per iteration.
__global__ void c2p_manual_b(const double *U, double *U_b, double *P, double *P_b, const double *gam, double *gam_b,
                             int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double D = U[i], S = U[n + i], tau = U[2 * n + i], g = gam[0];
    double pmin = fmax(fabs(S) - tau - D, 0.0) + 1e-14;
    double p = fmax((g - 1.0) * tau, pmin);
    for (int it = 0; it < 60; ++it) {
      double v = S / (tau + D + p);
      double W = 1.0 / sqrt(1.0 - v * v);
      double rho = D / W;
      double eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W);
      double f = (g - 1.0) * rho * eps - p;
      double cs2 = g * (g - 1.0) * eps / (1.0 + g * eps);
      double dp = f / (v * v * cs2 - 1.0);
      p = fmax(p - dp, pmin);
      if (fabs(dp) <= 1e-14 * p) break;
    }
    double v = S / (tau + D + p);
    double W = 1.0 / sqrt(1.0 - v * v);
    double rho = D / W, u = W * v;
    P[i] = rho; P[n + i] = u; P[2 * n + i] = p;
    // A = d(D, S, tau)/d(rho, u, p), with k = g/(g-1), rho h = rho + k p
    double k = g / (g - 1.0), rh = rho + k * p;
    double a11 = W,             a12 = rho * u / W,                   a13 = 0.0;
    double a21 = W * u,         a22 = rh * (1.0 + 2.0 * u * u) / W, a23 = k * W * u;
    double a31 = W * W - W,     a32 = 2.0 * rh * u - rho * u / W,    a33 = k * W * W - 1.0;
    // solve A^T lam = P_b by Cramer's rule (columns of A^T are the rows of A)
    double b1 = P_b[i], b2 = P_b[n + i], b3 = P_b[2 * n + i];
    P_b[i] = 0; P_b[n + i] = 0; P_b[2 * n + i] = 0;
    double det = a11 * (a22 * a33 - a32 * a23) - a21 * (a12 * a33 - a32 * a13) + a31 * (a12 * a23 - a22 * a13);
    // A^T = [[a11 a21 a31],[a12 a22 a32],[a13 a23 a33]]
    double l1 = (b1 * (a22 * a33 - a32 * a23) - a21 * (b2 * a33 - a32 * b3) + a31 * (b2 * a23 - a22 * b3)) / det;
    double l2 = (a11 * (b2 * a33 - a32 * b3) - b1 * (a12 * a33 - a32 * a13) + a31 * (a12 * b3 - b2 * a13)) / det;
    double l3 = (a11 * (a22 * b3 - b2 * a23) - a21 * (a12 * b3 - b2 * a13) + b1 * (a12 * a23 - a22 * a13)) / det;
    atomicAdd(&U_b[i], l1);
    atomicAdd(&U_b[n + i], l2);
    atomicAdd(&U_b[2 * n + i], l3);
    // dU/dgam at fixed P: dk/dgam = -1/(g-1)^2, D does not depend on gam
    double dk = -1.0 / ((g - 1.0) * (g - 1.0));
    atomicAdd(&gam_b[0], -(l2 * dk * p * W * u + l3 * dk * p * W * W));
  }
}
