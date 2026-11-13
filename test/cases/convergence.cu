// Exercises: loops that iterate to convergence and leave with `if (c) break;`. Reverse mode
// differentiates the converged point (implicit adjoint), not the iterations. Each kernel maps x[0..n)
// to y[0]; the bracketing (bisection) kernels are in test/reverse_only/bracketing.cu.

// Newton on t^3 + x0 t - x1 = 0, then y = t^2 + x2 t.
__device__ void newton_root(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double t = 1.0;
  for (int it = 0; it < 100; ++it) {
    double r = t * t * t + a * t - b;
    double d = r / (3.0 * t * t + a);
    t = t - d;
    if (fabs(d) <= 1e-15 * fabs(t)) break;
  }
  y[0] = t * t + c * t;
}

// The same root by the contraction t <- cbrt(x1 - x0 t) (linear convergence).
__device__ void fixed_point(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double t = 1.0;
  for (int it = 0; it < 1000; ++it) {
    double tn = cbrt(b - a * t);
    double d = tn - t;
    t = tn;
    if (fabs(d) <= 1e-15 * fabs(t)) break;
  }
  y[0] = t * t + c * t;
}

// Two unknowns by Newton: u + v^3 = x0, u^2 - v = x1 (Jacobian inverted in closed form).
__device__ void newton_system(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double u = 1.0, v = 0.5;
  for (int it = 0; it < 100; ++it) {
    double f = u + v * v * v - a, g = u * u - v - b;
    double j11 = 1.0, j12 = 3.0 * v * v, j21 = 2.0 * u, j22 = -1.0;
    double det = j11 * j22 - j12 * j21;
    double du = (j22 * f - j12 * g) / det, dv = (j11 * g - j21 * f) / det;
    u = u - du;
    v = v - dv;
    if (fabs(du) + fabs(dv) <= 1e-15 * (fabs(u) + fabs(v))) break;
  }
  y[0] = u * v + c * u;
}

// A root solve inside a loop that carries its result: each step's equation depends on the last root.
__device__ void solve_in_loop(const double *x, double *y, int n) {
  double s = 0.2;
  for (int k = 0; k < n; ++k) {
    double a = 1.0 + x[k] * x[k], b = s + x[k];
    double t = 1.0;
    for (int it = 0; it < 100; ++it) {
      double r = t * t * t + a * t - b;
      double d = r / (3.0 * t * t + a);
      t = t - d;
      if (fabs(d) <= 1e-15 * fabs(t)) break;
    }
    s = 0.5 * s + t;
  }
  y[0] = s;
}
