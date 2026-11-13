// One-dimensional special-relativistic hydrodynamics (c = 1), ideal gas p = (gam - 1) rho eps.
// Conserved U = (D, S, tau) = (rho W, rho h W^2 v, rho h W^2 - p - D), h = 1 + gam/(gam-1) p/rho.
// Primitives P = (rho, u, p) with the four-velocity u = W v, so that |v| < 1 after reconstruction.
// Arrays hold three planes of n cells (plane k at offset k*n). Outflow boundaries: neighbour indices
// are clamped to [0, n). Fluxes live on the n+1 faces, face f between cells f-1 and f.
// Scheme: minmod-limited linear reconstruction of (rho, u, p), HLL flux, SSP-RK2 in the driver.
// The adiabatic index is an array of length one so that its adjoint can be requested.
// Written as ordinary CUDA: nothing in it is specific to differentiation.

// Conservative to primitive: the pressure solves f(p) = (gam-1) rho eps - p = 0, Newton iteration.
__global__ void c2p_newton(const double *U, double *P, const double *gam, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double D = U[i], S = U[n + i], tau = U[2 * n + i], g = gam[0];
    double pmin = fmax(fabs(S) - tau - D, 0.0) + 1e-14;
    double p = fmax((g - 1.0) * tau, pmin);
    for (int it = 0; it < 60; ++it) {
      double v = S / (tau + D + p);
      double W = 1.0 / sqrt(1.0 - v * v);
      double rho = D / W;
      double eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W);
      double f = (g - 1.0) * rho * eps - p;
      double cs2 = g * (g - 1.0) * eps / (1.0 + g * eps);
      double dp = f / (v * v * cs2 - 1.0);
      p = fmax(p - dp, pmin);
      if (fabs(dp) <= 1e-14 * p) break;
    }
    double v = S / (tau + D + p);
    double W = 1.0 / sqrt(1.0 - v * v);
    P[i] = D / W;
    P[n + i] = W * v;
    P[2 * n + i] = p;
  }
}

// The same inversion by bisection on the bracket [pmin, pmax], as robust GPU codes often do.
__global__ void c2p_bisect(const double *U, double *P, const double *gam, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double D = U[i], S = U[n + i], tau = U[2 * n + i], g = gam[0];
    double lo = fmax(fabs(S) - tau - D, 0.0) + 1e-14;
    double hi = fmax(2.0 * (g - 1.0) * tau, 2.0 * lo);
    for (int it = 0; it < 200; ++it) {
      double m = 0.5 * (lo + hi);
      double v = S / (tau + D + m);
      double W = 1.0 / sqrt(1.0 - v * v);
      double eps = (tau + D * (1.0 - W) + m * (1.0 - W * W)) / (D * W);
      double f = (g - 1.0) * (D / W) * eps - m;
      if (f > 0) lo = m;
      else hi = m;
      if (hi - lo <= 1e-15 * hi) break;
    }
    double p = 0.5 * (lo + hi);
    double v = S / (tau + D + p);
    double W = 1.0 / sqrt(1.0 - v * v);
    P[i] = D / W;
    P[n + i] = W * v;
    P[2 * n + i] = p;
  }
}

__device__ double minmod(double a, double b) {
  return a * b <= 0 ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}

// HLL flux on every face from the reconstructed left and right states.
__global__ void hll_flux(const double *P, double *F, const double *gam, int n) {
  for (int f = blockIdx.x * blockDim.x + threadIdx.x; f <= n; f += gridDim.x * blockDim.x) {
    double g = gam[0];
    int i0 = f - 2 < 0 ? 0 : (f - 2 > n - 1 ? n - 1 : f - 2);
    int i1 = f - 1 < 0 ? 0 : (f - 1 > n - 1 ? n - 1 : f - 1);
    int i2 = f < n ? f : n - 1;
    int i3 = f + 1 < n ? f + 1 : n - 1;
    double qL[3], qR[3];
    for (int k = 0; k < 3; ++k) {
      double a = P[k * n + i0], b = P[k * n + i1], c = P[k * n + i2], d = P[k * n + i3];
      qL[k] = b + 0.5 * minmod(b - a, c - b);
      qR[k] = c - 0.5 * minmod(c - b, d - c);
    }
    // left state
    double rl = qL[0], ul = qL[1], pl = qL[2];
    double Wl = sqrt(1.0 + ul * ul), vl = ul / Wl;
    double hl = 1.0 + g / (g - 1.0) * pl / rl;
    double Dl = rl * Wl, Sl = rl * hl * Wl * Wl * vl, El = rl * hl * Wl * Wl - pl - Dl;
    double c2l = g * pl / (rl * hl);
    // characteristic speeds (v -/+ cs) / (1 -/+ v cs), written as (v (1 - cs^2) -/+ cs (1 - v^2)) / (1 - v^2 cs^2)
    double al = (vl * (1.0 - c2l) - sqrt(c2l) * (1.0 - vl * vl)) / (1.0 - vl * vl * c2l);
    double bl = (vl * (1.0 - c2l) + sqrt(c2l) * (1.0 - vl * vl)) / (1.0 - vl * vl * c2l);
    // right state
    double rr = qR[0], ur = qR[1], pr = qR[2];
    double Wr = sqrt(1.0 + ur * ur), vr = ur / Wr;
    double hr = 1.0 + g / (g - 1.0) * pr / rr;
    double Dr = rr * Wr, Sr = rr * hr * Wr * Wr * vr, Er = rr * hr * Wr * Wr - pr - Dr;
    double c2r = g * pr / (rr * hr);
    double ar = (vr * (1.0 - c2r) - sqrt(c2r) * (1.0 - vr * vr)) / (1.0 - vr * vr * c2r);
    double br = (vr * (1.0 - c2r) + sqrt(c2r) * (1.0 - vr * vr)) / (1.0 - vr * vr * c2r);
    double sl = fmin(0.0, fmin(al, ar)), sr = fmax(0.0, fmax(bl, br));
    double inv = 1.0 / (sr - sl);
    F[f] = (sr * Dl * vl - sl * Dr * vr + sl * sr * (Dr - Dl)) * inv;
    F[(n + 1) + f] = (sr * (Sl * vl + pl) - sl * (Sr * vr + pr) + sl * sr * (Sr - Sl)) * inv;
    F[2 * (n + 1) + f] = (sr * (Sl - Dl * vl) - sl * (Sr - Dr * vr) + sl * sr * (Er - El)) * inv;
  }
}

// One SSP-RK stage: Un = a U0 + b (U - dtdx (F_{i+1/2} - F_{i-1/2})).
__global__ void rk_stage(const double *U0, const double *U, const double *F, double *Un, double a, double b,
                         double dtdx, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    for (int k = 0; k < 3; ++k)
      Un[k * n + i] = a * U0[k * n + i] + b * (U[k * n + i] - dtdx * (F[k * (n + 1) + i + 1] - F[k * (n + 1) + i]));
}

// Primitive to conservative, for the initial data.
__global__ void p2c(const double *P, double *U, const double *gam, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double g = gam[0], rho = P[i], u = P[n + i], p = P[2 * n + i];
    double W = sqrt(1.0 + u * u), v = u / W, h = 1.0 + g / (g - 1.0) * p / rho;
    U[i] = rho * W;
    U[n + i] = rho * h * W * W * v;
    U[2 * n + i] = rho * h * W * W - p - rho * W;
  }
}
