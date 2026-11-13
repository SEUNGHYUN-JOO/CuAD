// Loops with `break` that reverse mode must reject at generation time, each with a located error naming the reason
// (test/run_tests.sh expects 4 errors).

// Reads an active array inside the loop (a running search over the input).
__device__ void search_sum(const double *x, double *y, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s = s + x[i];
    if (s > 1.0) break;
  }
  y[0] = s;
}

// Writes memory inside the loop.
__device__ void writes_memory(const double *x, double *y, int n) {
  double t = x[0], a = x[1];
  for (int it = 0; it < 50; ++it) {
    t = 0.5 * t + a;
    y[1] = t;
    if (fabs(t - 2.0 * a) <= 1e-14) break;
  }
  y[0] = t;
}

// A nested loop in the body.
__device__ void nested_loop(const double *x, double *y, int n) {
  double t = 1.0, a = x[0];
  for (int it = 0; it < 50; ++it) {
    double s = 0.0;
    for (int k = 0; k < 3; ++k) s = s + t * a;
    t = 0.5 * t + 0.1 * s;
    if (fabs(s) < 1e-14) break;
  }
  y[0] = t;
}

// `break` not at the top level of the body.
__device__ void inner_break(const double *x, double *y, int n) {
  double t = 1.0, a = x[0];
  for (int it = 0; it < 50; ++it) {
    if (t > 0) {
      t = 0.5 * t + a;
      if (fabs(t - 2.0 * a) < 1e-14) break;
    }
  }
  y[0] = t;
}
