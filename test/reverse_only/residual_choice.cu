// Which residual a multi-state root finder solves, and when that cannot be decided. x = (a, b).
//
// Ambiguity: the loop below drives f = x - a to zero and also tests g = x - b in its exit condition, so both are
// candidate residuals. For a != b only f vanishes at the exit state. For a == b both vanish there with different
// derivatives of the root, (1, 0) and (0, 1); nothing in the vanishing values tells them apart, so the adjoint traps.
__device__ void steer_tested(const double *x, double *y, int n) {
  double a = x[0], b = x[1];
  double x0 = 0.0, x1 = 1.0;
  double f0 = x0 - a, f1 = x1 - a, g = x1 - b;
  for (int it = 0; it < 50; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = x1 - a;
    g = x1 - b;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1) || g == 0.0) break;
  }
  y[0] = x1;
}

// The same loop where g is computed beside the iteration and never used by it: g does not steer the loop, so it is
// no candidate, and a == b gives the derivative of the root of f.
__device__ void side_value(const double *x, double *y, int n) {
  double a = x[0], b = x[1];
  double x0 = 0.0, x1 = 1.0;
  double f0 = x0 - a, f1 = x1 - a, g = x1 - b;
  for (int it = 0; it < 50; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = x1 - a;
    g = x1 - b;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1)) break;
  }
  y[0] = x1;
}

// A seeded iterated variable that is zero at the exit state only by accident: h = x1 - 1/2 vanishes when the root
// is 1/2, but its derivative along the solution is that of the root. Dropping its seed (as for a residual) would
// give dy/da = 1 instead of 2; it is neither at the root nor a residual, so the adjoint traps.
__device__ void zero_aux(const double *x, double *y, int n) {
  double a = x[0];
  double x0 = 0.0, x1 = 1.0;
  double f0 = x0 - a, f1 = x1 - a, h = x1 - 0.5;
  for (int it = 0; it < 50; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = x1 - a;
    h = x1 - 0.5;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1)) break;
  }
  y[0] = x1 + h;
}

// Algebraic variants of the secant iteration for t^3 + a t - b = 0: the residual in Horner form, scaled, partly in
// a body local, and computed twice (two candidates that vanish together and agree). All give dt/d(a, b).
__device__ void cubic_horner(const double *x, double *y, int n) {
  double a = x[0], b = x[1];
  double x0 = 1.0, x1 = 2.0;
  double f0 = x0 * (x0 * x0 + a) - b, f1 = x1 * (x1 * x1 + a) - b;
  for (int it = 0; it < 100; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = x1 * (x1 * x1 + a) - b;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1)) break;
  }
  y[0] = x1;
}

__device__ void cubic_scaled(const double *x, double *y, int n) {
  double a = x[0], b = x[1];
  double x0 = 1.0, x1 = 2.0;
  double f0 = 2.0 * (x0 * x0 * x0 + a * x0) - 2.0 * b, f1 = 2.0 * (x1 * x1 * x1 + a * x1) - 2.0 * b;
  for (int it = 0; it < 100; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = 2.0 * (x1 * x1 * x1 + a * x1) - 2.0 * b;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1)) break;
  }
  y[0] = x1;
}

__device__ void cubic_local(const double *x, double *y, int n) {
  double a = x[0], b = x[1];
  double x0 = 1.0, x1 = 2.0;
  double f0 = x0 * x0 * x0 + a * x0 - b, f1 = x1 * x1 * x1 + a * x1 - b;
  for (int it = 0; it < 100; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    double u = x1 * x1;
    f1 = u * x1 + a * x1 - b;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1)) break;
  }
  y[0] = x1;
}

__device__ void cubic_twice(const double *x, double *y, int n) {
  double a = x[0], b = x[1];
  double x0 = 1.0, x1 = 2.0;
  double f0 = x0 * x0 * x0 + a * x0 - b, f1 = x1 * x1 * x1 + a * x1 - b, h1 = 3.0 * f1;
  for (int it = 0; it < 100; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = x1 * x1 * x1 + a * x1 - b;
    h1 = 3.0 * (x1 * x1 * x1 + a * x1 - b);
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1) || h1 == 0.0) break;
  }
  y[0] = x1;
}
