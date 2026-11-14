// Convergence loops written as in a physics code, used to measure what the run-time checks of the implicit adjoint
// accept and what the accepted derivatives are worth (check_reliability.cpp). Active inputs x = (a, b, c); the
// result is y[0] = t. Kernel scalars (relaxation weight, start, loop tolerances) are passive.

// Fixed-point relaxation for t^3 + a t - b = 0 with contraction rate rho = 1 - w (3 t*^2 + a) at the root.
__global__ void fp_relax(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double a = x[0], b = x[1];
  double t = t0;
  for (int it = 0; it < 2000000; ++it) {
    double f = t * t * t + a * t - b;
    double dt = w * f;
    t = t - dt;
    if (fabs(dt) <= tolp * fabs(t) + atolp) break;
  }
  y[0] = t;
}

// Newton's method for t^3 + a t - b = 0.
__global__ void newton_cubic(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double a = x[0], b = x[1];
  double t = t0;
  for (int it = 0; it < 200; ++it) {
    double f = t * t * t + a * t - b;
    double fp = 3.0 * t * t + a;
    double dt = f / fp;
    t = t - dt;
    if (fabs(dt) <= tolp * fabs(t) + atolp) break;
  }
  y[0] = t;
}

// Newton's method for t^3 + a t + b = 0 (three real roots for a = -7, b = 6): the start selects the root.
__global__ void newton_three(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double a = x[0], b = x[1];
  double t = t0;
  for (int it = 0; it < 200; ++it) {
    double f = t * t * t + a * t + b;
    double fp = 3.0 * t * t + a;
    double dt = f / fp;
    t = t - dt;
    if (fabs(dt) <= tolp * fabs(t) + atolp) break;
  }
  y[0] = t;
}

// Nearly double root: (t - 1)^2 - d = 0 with d = x[0], root 1 + sqrt(d), by Newton from t0 and by bisection on [1, 2].
__global__ void newton_double(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double d = x[0];
  double t = t0;
  for (int it = 0; it < 200; ++it) {
    double f = (t - 1.0) * (t - 1.0) - d;
    double fp = 2.0 * (t - 1.0);
    double dt = f / fp;
    t = t - dt;
    if (fabs(dt) <= tolp * fabs(t) + atolp) break;
  }
  y[0] = t;
}

__global__ void bisect_double(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double d = x[0];
  double lo = 1.0, hi = 2.0;
  for (int it = 0; it < 200; ++it) {
    double m = 0.5 * (lo + hi);
    double f = (m - 1.0) * (m - 1.0) - d;
    if (f > 0) hi = m;
    else lo = m;
    if (hi - lo <= tolp * hi) break;
  }
  y[0] = 0.5 * (lo + hi);
}

// Active lower bound c = x[2] above the root of t^3 + a t - b: Newton clamped to the bound, and bisection on [c, 3].
__global__ void newton_floor(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double a = x[0], b = x[1], tmin = x[2];
  double t = t0;
  for (int it = 0; it < 60; ++it) {
    double f = t * t * t + a * t - b;
    double fp = 3.0 * t * t + a;
    double dt = f / fp;
    t = fmax(t - dt, tmin);
    if (fabs(dt) <= tolp * fabs(t) + atolp) break;
  }
  y[0] = t;
}

__global__ void bisect_floor(const double *x, double *y, double w, double t0, double tolp, double atolp, int n) {
  double a = x[0], b = x[1];
  double lo = x[2], hi = 3.0;
  for (int it = 0; it < 200; ++it) {
    double m = 0.5 * (lo + hi);
    double f = m * m * m + a * m - b;
    if (f > 0) hi = m;
    else lo = m;
    if (hi - lo <= tolp * hi) break;
  }
  y[0] = 0.5 * (lo + hi);
}
