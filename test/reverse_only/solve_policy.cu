// Loops of the convergence-loop form (`for` + `if (c) break;`, floating locals iterated) that do NOT stop at a fixed
// point. Reverse mode recognises their form statically, so the implicit adjoint must refuse them at run time: one more
// iteration from the exit state moves it (test/verify_solve_policy.cpp expects a trap from each `_b`).
// x = (a, ...), y[0] = result.

// A threshold exit: t -> 0.5 t + a is a contraction (fixed point 2a), but the loop leaves at t > 1.5 a.
__device__ void threshold_exit(const double *x, double *y, int n) {
  double a = x[0];
  double t = 0.0;
  for (int it = 0; it < 100; ++it) {
    t = 0.5 * t + a;
    if (t > 1.5 * a) break;
  }
  y[0] = t;
}

// Time marching of y' = -a y that stops at a threshold: never a fixed point.
__device__ void march_to_threshold(const double *x, double *y, int n) {
  double a = x[0];
  double v = 1.0;
  for (int it = 0; it < 1000; ++it) {
    v = v - 0.01 * a * v;
    if (v < 0.5) break;
  }
  y[0] = v;
}

// Newton on t^3 - 2t + b = 0 from t = 0 cycles between 0 and 1 for b = 2: it never converges and the loop ends
// at its iteration cap.
__device__ void newton_cycle(const double *x, double *y, int n) {
  double b = x[0];
  double t = 0.0;
  for (int it = 0; it < 41; ++it) {
    double d = (t * t * t - 2.0 * t + b) / (3.0 * t * t - 2.0);
    t = t - d;
    if (fabs(d) <= 1e-14 * fabs(t)) break;
  }
  y[0] = t;
}

// A bisection stopped by an iteration cap long before its bracket collapses.
__device__ void bisect_capped(const double *x, double *y, int n) {
  double a = x[0];
  double lo = 0.0, hi = 4.0;
  for (int it = 0; it < 5; ++it) {
    double m = 0.5 * (lo + hi);
    if (m * m - a > 0) hi = m;
    else lo = m;
    if (hi - lo <= 1e-15 * hi) break;
  }
  y[0] = 0.5 * (lo + hi);
}

// Control: the same contraction run to convergence is accepted.
__device__ void contraction_converged(const double *x, double *y, int n) {
  double a = x[0];
  double t = 0.0;
  for (int it = 0; it < 200; ++it) {
    double tn = 0.5 * t + a;
    double d = tn - t;
    t = tn;
    if (fabs(d) <= 1e-15 * fabs(t)) break;
  }
  y[0] = t;
}
