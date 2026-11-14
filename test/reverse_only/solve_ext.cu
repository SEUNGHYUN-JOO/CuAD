// Convergence loops beyond the basic forms (verify_solve_ext.cpp): slow contractions, double roots, `while` and
// `do` loops, a `for` loop that stops on its condition, a pure function called in the body, and Newton's method for a
// 2 x 2 system with scalar or array state. The inputs are x = (a, b); y[0] is the result.

__device__ double cubic(double t, double a, double b) { return t * t * t + a * t - b; }

// t^3 + a t - b = 0 by a relaxation that contracts at rate 0.999, run to convergence
__global__ void slow_tight(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = 1.0;
  for (int it = 0; it < 400000; ++it) {
    double dt = 1.372e-4 * (t * t * t + a * t - b);
    t = t - dt;
    if (fabs(dt) <= 1e-16 * fabs(t)) break;
  }
  y[0] = t;
}

// the same relaxation stopped when its step falls below 1e-8: the distance to the root is still about 1e-5 relative
__global__ void slow_loose(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = 1.0;
  for (int it = 0; it < 400000; ++it) {
    double dt = 1.372e-4 * (t * t * t + a * t - b);
    t = t - dt;
    if (fabs(dt) <= 1e-8 * fabs(t)) break;
  }
  y[0] = t;
}

// (t - a)^2 = b by Newton's method: a double root at b = 0
__global__ void double_newton(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = a + 1.0;
  for (int it = 0; it < 200; ++it) {
    double f = (t - a) * (t - a) - b;
    double dt = f / (2.0 * (t - a));
    t = t - dt;
    if (fabs(dt) <= 1e-13 * fabs(t)) break;
  }
  y[0] = t;
}

// the same equation by bisection on [a, a + 2]
__global__ void double_bisect(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double lo = a, hi = a + 2.0;
  for (int it = 0; it < 200; ++it) {
    double m = 0.5 * (lo + hi);
    double f = (m - a) * (m - a) - b;
    if (f > 0) hi = m;
    else lo = m;
    if (hi - lo <= 1e-15 * hi) break;
  }
  y[0] = 0.5 * (lo + hi);
}

// the same equation by the secant method
__global__ void double_secant(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double x0 = a + 2.0, x1 = a + 1.5;
  double f0 = (x0 - a) * (x0 - a) - b;
  for (int it = 0; it < 200; ++it) {
    double f1 = (x1 - a) * (x1 - a) - b;
    if (f1 == 0 || f1 == f0) break;
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    if (fabs(x1 - x0) <= 1e-14 * fabs(x1)) break;
  }
  y[0] = x1;
}

// Newton's method on the cubic written as a `while` loop with an iteration counter
__global__ void while_newton(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = 2.0, dt = 1.0;
  int it = 0;
  while (fabs(dt) > 1e-15 * fabs(t) && it < 100) {
    dt = (t * t * t + a * t - b) / (3.0 * t * t + a);
    t = t - dt;
    ++it;
  }
  y[0] = t;
}

// ... as a `do` loop
__global__ void do_newton(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = 2.0, dt;
  int it = 0;
  do {
    dt = (t * t * t + a * t - b) / (3.0 * t * t + a);
    t = t - dt;
    it++;
  } while (fabs(dt) > 1e-15 * fabs(t) && it < 100);
  y[0] = t;
}

// ... as a `for` loop that stops on its condition
__global__ void for_cond_newton(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = 2.0, dt = 1.0;
  for (int it = 0; it < 100 && fabs(dt) > 1e-15 * fabs(t); ++it) {
    dt = (t * t * t + a * t - b) / (3.0 * t * t + a);
    t = t - dt;
  }
  y[0] = t;
}

// ... with the residual computed by a pure device function
__global__ void call_newton(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double t = 2.0;
  for (int it = 0; it < 100; ++it) {
    double dt = cubic(t, a, b) / (3.0 * t * t + a);
    t = t - dt;
    if (fabs(dt) <= 1e-15 * fabs(t)) break;
  }
  y[0] = t;
}

// Newton's method for u + v = a, u v = b (u is the larger root of z^2 - a z + b) with two scalar unknowns
__global__ void system_scalar(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double u = 2.5, v = 0.5;
  for (int it = 0; it < 100; ++it) {
    double f0 = u + v - a, f1 = u * v - b;
    double det = u - v;
    double du = (u * f0 - f1) / det, dv = (f1 - v * f0) / det;
    u = u - du;
    v = v - dv;
    if (fabs(du) + fabs(dv) <= 1e-15 * (fabs(u) + fabs(v))) break;
  }
  y[0] = u;
}

// ... with the unknowns in a local array
__global__ void system_array(const double *x, double *y, int n) {
  if (n < 1) return;
  double a = x[0], b = x[1];
  double z[2] = {2.5, 0.5};
  for (int it = 0; it < 100; ++it) {
    double f0 = z[0] + z[1] - a, f1 = z[0] * z[1] - b;
    double det = z[0] - z[1];
    double d0 = (z[0] * f0 - f1) / det, d1 = (f1 - z[1] * f0) / det;
    z[0] = z[0] - d0;
    z[1] = z[1] - d1;
    if (fabs(d0) + fabs(d1) <= 1e-15 * (fabs(z[0]) + fabs(z[1]))) break;
  }
  y[0] = z[0];
}
