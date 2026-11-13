// Iteration-capped root solves of r(t; a, b) = t^3 + a t - b = 0, x = (a, b), y[0] = t after at most K iterations
// from t0 = x[2] (passive). Used to measure how the derivative error of each AD strategy decreases with K.
__device__ void newton_K(const double *x, double *y, int K, double t0) {
  double a = x[0], b = x[1];
  double t = t0;
  for (int it = 0; it < K; ++it) {
    double d = (t * t * t + a * t - b) / (3.0 * t * t + a);
    t = t - d;
    if (fabs(d) <= 1e-300 * fabs(t)) break;
  }
  y[0] = t;
}
__device__ void fixed_point_K(const double *x, double *y, int K, double t0) {
  double a = x[0], b = x[1];
  double t = t0;
  for (int it = 0; it < K; ++it) {
    double tn = cbrt(b - a * t);
    double d = tn - t;
    t = tn;
    if (fabs(d) <= 1e-300 * fabs(t)) break;
  }
  y[0] = t;
}
__device__ void bisect_K(const double *x, double *y, int K, double t0) {
  double a = x[0], b = x[1];
  double lo = 0.0, hi = 4.0;
  for (int it = 0; it < K; ++it) {
    double m = 0.5 * (lo + hi);
    double r = m * m * m + a * m - b;
    if (r > 0) hi = m;
    else lo = m;
    if (hi - lo <= 1e-300 * hi) break;
  }
  y[0] = 0.5 * (lo + hi);
}
