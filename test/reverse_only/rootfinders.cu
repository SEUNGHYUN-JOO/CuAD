// Multi-state root finders on the same equation as test/cases/convergence.cu: t^3 + a t - b = 0, x = (a, b, c),
// y[0] = t^2 + c t. Each keeps several floating variables across iterations (points, residual values, step sizes),
// so one iteration is not a smooth contraction at the root; reverse mode differentiates the root through the
// residual it finds in the loop (the function value computed from one trial point and the inputs).

// Secant method.
__device__ void secant_root(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double x0 = 1.0, x1 = 2.0;
  double f0 = x0 * x0 * x0 + a * x0 - b;
  double f1 = x1 * x1 * x1 + a * x1 - b;
  for (int it = 0; it < 100; ++it) {
    double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = x1 * x1 * x1 + a * x1 - b;
    if (fabs(x1 - x0) <= 1e-15 * fabs(x1)) break;
  }
  y[0] = x1 * x1 + c * x1;
}

// Regula falsi with the Illinois modification, bracket [0, 4].
__device__ void illinois_root(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double lo = 0.0, hi = 4.0;
  double flo = lo * lo * lo + a * lo - b;
  double fhi = hi * hi * hi + a * hi - b;
  double t = hi;
  for (int it = 0; it < 200; ++it) {
    t = (lo * fhi - hi * flo) / (fhi - flo);
    double ft = t * t * t + a * t - b;
    if (fabs(ft) <= 1e-14) break;
    if (ft * fhi < 0) {
      lo = hi;
      flo = fhi;
    } else {
      flo = 0.5 * flo;
    }
    hi = t;
    fhi = ft;
  }
  y[0] = t * t + c * t;
}

// Brent's method (the Numerical Recipes zbrent structure, with `break` for its return), bracket [0, 4].
__device__ void brent_root(const double *x, double *y, int n) {
  double a = x[0], b = x[1], c = x[2];
  double pa = 0.0, pb = 4.0, pc = 4.0, d = 0.0, e = 0.0;
  double fa = pa * pa * pa + a * pa - b;
  double fb = pb * pb * pb + a * pb - b;
  double fc = fb;
  for (int it = 0; it < 100; ++it) {
    if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
      pc = pa;
      fc = fa;
      e = pb - pa;
      d = e;
    }
    if (fabs(fc) < fabs(fb)) {
      pa = pb;
      pb = pc;
      pc = pa;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    double tol1 = 2.0e-16 * fabs(pb) + 0.5e-15;
    double xm = 0.5 * (pc - pb);
    if (fabs(xm) <= tol1 || fb == 0.0) break;
    if (fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
      double s = fb / fa;
      double p = 0.0, q = 0.0;
      if (pa == pc) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        double qq = fa / fc, r = fb / fc;
        p = s * (2.0 * xm * qq * (qq - r) - (pb - pa) * (r - 1.0));
        q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (p > 0) q = -q;
      p = fabs(p);
      double m1 = 3.0 * xm * q - fabs(tol1 * q), m2 = fabs(e * q);
      if (2.0 * p < (m1 < m2 ? m1 : m2)) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    pa = pb;
    fa = fb;
    if (fabs(d) > tol1) pb = pb + d;
    else pb = pb + (xm >= 0 ? tol1 : -tol1);
    fb = pb * pb * pb + a * pb - b;
  }
  y[0] = pb * pb + c * pb;
}
