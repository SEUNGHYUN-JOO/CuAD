// Bracketing root solves (reverse mode only: forward mode rejects them, because the tangent of the
// iterations is the tangent of the initial bracket). Same equation as test/cases/convergence.cu.

// Bisection on t^3 + x0 t - x1 = 0 over [0, 4], then y = t^2 + x2 t.
__device__ void bisect_root(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double lo = 0.0, hi = 4.0;
  for (int it = 0; it < 200; ++it) {
    double m = 0.5 * (lo + hi);
    double r = m * m * m + a * m - b;
    if (r > 0) hi = m;
    else lo = m;
    if (hi - lo <= 1e-15 * hi) break;
  }
  double t = 0.5 * (lo + hi);
  y[0] = t * t + c * t;
}

// The bracket itself depends on the inputs (as in a conservative-to-primitive solve): differentiating
// the iterations would carry the bracket's derivative, which is not the root's.
__device__ void bisect_moving_bracket(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double lo = 0.1 * b, hi = b + 1.0;
  for (int it = 0; it < 200; ++it) {
    double m = 0.5 * (lo + hi);
    double r = m * m * m + a * m - b;
    if (r < 0) lo = m;
    else hi = m;
    if (hi - lo <= 1e-15 * hi) break;
  }
  y[0] = lo * lo + c * hi;
}
