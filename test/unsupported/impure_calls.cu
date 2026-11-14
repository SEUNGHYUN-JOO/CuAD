// Convergence loops that call functions which are not pure (run_tests.sh): reverse mode must reject each loop with
// the reason instead of re-evaluating a function whose value depends on, or changes, memory outside its arguments.

__device__ double g_shift = 0.5;

__device__ double reads_global(double t, double a) { return t * t - a - g_shift; }

__device__ double inner_global(double t) { return t + g_shift; }
__device__ double calls_impure(double t, double a) { return inner_global(t) * t - a; }

__device__ double through_pointer(double t, double a) {
  double v[2] = {t, a};
  double *q = v;
  return q[0] * q[0] - q[1];
}

__global__ void newton_reads_global(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], t = 2.0;
  for (int it = 0; it < 50; ++it) {
    double dt = reads_global(t, a) / (2.0 * t);
    t = t - dt;
    if (fabs(dt) <= 1e-14 * fabs(t)) break;
  }
  y[0] = t;
}

__global__ void newton_calls_impure(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], t = 2.0;
  for (int it = 0; it < 50; ++it) {
    double dt = calls_impure(t, a) / (2.0 * t);
    t = t - dt;
    if (fabs(dt) <= 1e-14 * fabs(t)) break;
  }
  y[0] = t;
}

__global__ void newton_through_pointer(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], t = 2.0;
  for (int it = 0; it < 50; ++it) {
    double dt = through_pointer(t, a) / (2.0 * t);
    t = t - dt;
    if (fabs(dt) <= 1e-14 * fabs(t)) break;
  }
  y[0] = t;
}
