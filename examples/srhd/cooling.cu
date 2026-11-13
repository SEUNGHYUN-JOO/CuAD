// Optically thin cooling as an operator-split source step for the SRHD solver (examples/srhd/srhd.cu):
// dp/dt = -L0 rho^2 (p/rho)^alpha in the fluid frame, integrated by backward Euler over dt, cell by cell:
//   r(x) = x - p + dt L0 rho^2 (x/rho)^alpha = 0  for the new pressure x (unique root in (0, p], r' > 0),
// solved by Newton's method with a convergence exit. lam = (L0, alpha) is an array so that its adjoint exists.
// Primitives P = (rho, u, p) in three planes of n cells; rho and u are unchanged.
__global__ void cool_implicit(const double *P, double *Pn, const double *lam, double dt, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double rho = P[i], u = P[n + i], p = P[2 * n + i];
    double L0 = lam[0], al = lam[1];
    double c = dt * L0 * rho * rho;
    double floor = 1e-6 * p;
    double x = p;
    for (int it = 0; it < 60; ++it) {
      double q = pow(x / rho, al);
      double r = x - p + c * q;
      double dr = 1.0 + c * al * q / x;
      double dx = r / dr;
      x = fmax(x - dx, floor);
      if (fabs(dx) <= 1e-14 * x) break;
    }
    Pn[i] = rho;
    Pn[n + i] = u;
    Pn[2 * n + i] = x;
  }
}
