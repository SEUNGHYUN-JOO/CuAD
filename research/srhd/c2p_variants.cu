// The conservative-to-primitive step of examples/srhd/srhd.cu solved by two more root finders that GPU relativistic
// codes use: regula falsi with the Illinois modification and Brent's method (Numerical Recipes structure). Same
// residual f(p) = (gam-1) rho eps - p, same bracket [pmin, pmax] as c2p_bisect. Written as ordinary CUDA.

__global__ void c2p_illinois(const double *U, double *P, const double *gam, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double D = U[i], S = U[n + i], tau = U[2 * n + i], g = gam[0];
    double lo = fmax(fabs(S) - tau - D, 0.0) + 1e-14;
    double hi = fmax(2.0 * (g - 1.0) * tau, 2.0 * lo);
    double vl = S / (tau + D + lo), Wl = 1.0 / sqrt(1.0 - vl * vl);
    double flo = (g - 1.0) * (D / Wl) * (tau + D * (1.0 - Wl) + lo * (1.0 - Wl * Wl)) / (D * Wl) - lo;
    double vh = S / (tau + D + hi), Wh = 1.0 / sqrt(1.0 - vh * vh);
    double fhi = (g - 1.0) * (D / Wh) * (tau + D * (1.0 - Wh) + hi * (1.0 - Wh * Wh)) / (D * Wh) - hi;
    double p = hi;
    for (int it = 0; it < 200; ++it) {
      p = (lo * fhi - hi * flo) / (fhi - flo);
      double v = S / (tau + D + p);
      double W = 1.0 / sqrt(1.0 - v * v);
      double eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W);
      double fp = (g - 1.0) * (D / W) * eps - p;
      if (fabs(fp) <= 1e-15 * p) break;
      if (fp * fhi < 0) {
        lo = hi;
        flo = fhi;
      } else {
        flo = 0.5 * flo;
      }
      hi = p;
      fhi = fp;
    }
    double v = S / (tau + D + p);
    double W = 1.0 / sqrt(1.0 - v * v);
    P[i] = D / W;
    P[n + i] = W * v;
    P[2 * n + i] = p;
  }
}

__global__ void c2p_brent(const double *U, double *P, const double *gam, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double D = U[i], S = U[n + i], tau = U[2 * n + i], g = gam[0];
    double pa = fmax(fabs(S) - tau - D, 0.0) + 1e-14;
    double pb = fmax(2.0 * (g - 1.0) * tau, 2.0 * pa);
    double va = S / (tau + D + pa), Wa = 1.0 / sqrt(1.0 - va * va);
    double fa = (g - 1.0) * (D / Wa) * (tau + D * (1.0 - Wa) + pa * (1.0 - Wa * Wa)) / (D * Wa) - pa;
    double vb = S / (tau + D + pb), Wb = 1.0 / sqrt(1.0 - vb * vb);
    double fb = (g - 1.0) * (D / Wb) * (tau + D * (1.0 - Wb) + pb * (1.0 - Wb * Wb)) / (D * Wb) - pb;
    double pc = pb, fc = fb, d = 0.0, e = 0.0;
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
      double tol1 = 2.0e-16 * fabs(pb) + 1e-300;
      double xm = 0.5 * (pc - pb);
      if (fabs(xm) <= tol1 || fb == 0.0) break;
      if (fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
        double s = fb / fa;
        double pp = 0.0, q = 0.0;
        if (pa == pc) {
          pp = 2.0 * xm * s;
          q = 1.0 - s;
        } else {
          double qq = fa / fc, r = fb / fc;
          pp = s * (2.0 * xm * qq * (qq - r) - (pb - pa) * (r - 1.0));
          q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
        }
        if (pp > 0) q = -q;
        pp = fabs(pp);
        double m1 = 3.0 * xm * q - fabs(tol1 * q), m2 = fabs(e * q);
        if (2.0 * pp < (m1 < m2 ? m1 : m2)) {
          e = d;
          d = pp / q;
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
      double v = S / (tau + D + pb);
      double W = 1.0 / sqrt(1.0 - v * v);
      double eps = (tau + D * (1.0 - W) + pb * (1.0 - W * W)) / (D * W);
      fb = (g - 1.0) * (D / W) * eps - pb;
    }
    double v = S / (tau + D + pb);
    double W = 1.0 / sqrt(1.0 - v * v);
    P[i] = D / W;
    P[n + i] = W * v;
    P[2 * n + i] = pb;
  }
}
