// Two-dimensional special-relativistic hydrodynamics (c = 1), ideal gas p = (gam - 1) rho eps.
// nx x ny cells, cell c = j*nx + i. Conserved U = (D, Sx, Sy, tau), primitives P = (rho, ux, uy, p) with the
// spatial four-velocity u = W v; four planes of N = nx*ny values each (plane k at offset k*N).
// Outflow boundaries by clamping neighbor indices. Fluxes: x faces (nx+1)*ny, face (i, j) between cells i-1 and i
// of row j; y faces nx*(ny+1), face (i, j) between rows j-1 and j. Minmod-limited linear reconstruction of
// (rho, ux, uy, p), HLL flux, unsplit SSP-RK2 in the driver. Written as ordinary CUDA.

// Conservative to primitive, Newton on f(p) = (gam-1) rho eps - p. E = tau + D > |S| for physical states, so
// v < 1 for every p > 0.
__global__ void c2p2_newton(const double *U, double *P, const double *gam, int N) {
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N; c += gridDim.x * blockDim.x) {
    double D = U[c], Sx = U[N + c], Sy = U[2 * N + c], tau = U[3 * N + c], g = gam[0];
    double S2 = Sx * Sx + Sy * Sy;
    double pmin = 1e-14 * (tau + D);
    double p = fmax((g - 1.0) * tau, pmin);
    for (int it = 0; it < 60; ++it) {
      double E = tau + D + p;
      double v2 = S2 / (E * E);
      double W = 1.0 / sqrt(1.0 - v2);
      double rho = D / W;
      double eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W);
      double f = (g - 1.0) * rho * eps - p;
      double cs2 = g * (g - 1.0) * eps / (1.0 + g * eps);
      double dp = f / (v2 * cs2 - 1.0);
      p = fmax(p - dp, pmin);
      if (fabs(dp) <= 1e-14 * p) break;
    }
    double E = tau + D + p;
    double W = 1.0 / sqrt(1.0 - S2 / (E * E));
    P[c] = D / W;
    P[N + c] = W * Sx / E;
    P[2 * N + c] = W * Sy / E;
    P[3 * N + c] = p;
  }
}

// The same inversion by bisection on [pmin, pmax].
__global__ void c2p2_bisect(const double *U, double *P, const double *gam, int N) {
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N; c += gridDim.x * blockDim.x) {
    double D = U[c], Sx = U[N + c], Sy = U[2 * N + c], tau = U[3 * N + c], g = gam[0];
    double S2 = Sx * Sx + Sy * Sy;
    double lo = 1e-14 * (tau + D);
    double hi = fmax(2.0 * (g - 1.0) * tau, 2.0 * lo);
    for (int it = 0; it < 200; ++it) {
      double m = 0.5 * (lo + hi);
      double E = tau + D + m;
      double W = 1.0 / sqrt(1.0 - S2 / (E * E));
      double eps = (tau + D * (1.0 - W) + m * (1.0 - W * W)) / (D * W);
      double f = (g - 1.0) * (D / W) * eps - m;
      if (f > 0) lo = m;
      else hi = m;
      if (hi - lo <= 1e-15 * hi) break;
    }
    double p = 0.5 * (lo + hi);
    double E = tau + D + p;
    double W = 1.0 / sqrt(1.0 - S2 / (E * E));
    P[c] = D / W;
    P[N + c] = W * Sx / E;
    P[2 * N + c] = W * Sy / E;
    P[3 * N + c] = p;
  }
}

__device__ double minmod2(double a, double b) {
  return a * b <= 0 ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}

// HLL flux on the faces normal to direction dir (0: x, 1: y).
__global__ void hll2(const double *P, double *F, const double *gam, int nx, int ny, int dir) {
  const int N = nx * ny;
  const int fx = dir == 0 ? nx + 1 : nx, fy = dir == 0 ? ny : ny + 1, NF = fx * fy;
  for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < NF; f += gridDim.x * blockDim.x) {
    double g = gam[0];
    int fi = f % fx, fj = f / fx;
    int n = dir == 0 ? nx : ny;       // cells along the direction
    int a = dir == 0 ? fi : fj;       // face position along it: between cells a-1 and a
    int i0 = a - 2 < 0 ? 0 : (a - 2 > n - 1 ? n - 1 : a - 2);
    int i1 = a - 1 < 0 ? 0 : (a - 1 > n - 1 ? n - 1 : a - 1);
    int i2 = a < n ? a : n - 1;
    int i3 = a + 1 < n ? a + 1 : n - 1;
    int c0 = dir == 0 ? fj * nx + i0 : i0 * nx + fi;
    int c1 = dir == 0 ? fj * nx + i1 : i1 * nx + fi;
    int c2 = dir == 0 ? fj * nx + i2 : i2 * nx + fi;
    int c3 = dir == 0 ? fj * nx + i3 : i3 * nx + fi;
    double qL[4], qR[4];
    for (int k = 0; k < 4; ++k) {
      double q0 = P[k * N + c0], q1 = P[k * N + c1], q2 = P[k * N + c2], q3 = P[k * N + c3];
      qL[k] = q1 + 0.5 * minmod2(q1 - q0, q2 - q1);
      qR[k] = q2 - 0.5 * minmod2(q2 - q1, q3 - q2);
    }
    double hg = g / (g - 1.0);
    // left state
    double rl = qL[0], uxl = qL[1], uyl = qL[2], pl = qL[3];
    double Wl = sqrt(1.0 + uxl * uxl + uyl * uyl);
    double vxl = uxl / Wl;
    double vyl = uyl / Wl;
    double hl = 1.0 + hg * pl / rl;
    double Dl = rl * Wl;
    double Sxl = rl * hl * Wl * uxl;
    double Syl = rl * hl * Wl * uyl;
    double El = rl * hl * Wl * Wl - pl - Dl;
    double vnl = dir == 0 ? vxl : vyl;
    double v2l = vxl * vxl + vyl * vyl;
    double c2l = g * pl / (rl * hl);
    double ql = sqrt(c2l * (1.0 - v2l) * (1.0 - v2l * c2l - vnl * vnl * (1.0 - c2l)));
    double aml = (vnl * (1.0 - c2l) - ql) / (1.0 - v2l * c2l);
    double apl = (vnl * (1.0 - c2l) + ql) / (1.0 - v2l * c2l);
    // right state
    double rr = qR[0], uxr = qR[1], uyr = qR[2], pr = qR[3];
    double Wr = sqrt(1.0 + uxr * uxr + uyr * uyr);
    double vxr = uxr / Wr;
    double vyr = uyr / Wr;
    double hr = 1.0 + hg * pr / rr;
    double Dr = rr * Wr;
    double Sxr = rr * hr * Wr * uxr;
    double Syr = rr * hr * Wr * uyr;
    double Er = rr * hr * Wr * Wr - pr - Dr;
    double vnr = dir == 0 ? vxr : vyr;
    double v2r = vxr * vxr + vyr * vyr;
    double c2r = g * pr / (rr * hr);
    double qr = sqrt(c2r * (1.0 - v2r) * (1.0 - v2r * c2r - vnr * vnr * (1.0 - c2r)));
    double amr = (vnr * (1.0 - c2r) - qr) / (1.0 - v2r * c2r);
    double apr = (vnr * (1.0 - c2r) + qr) / (1.0 - v2r * c2r);
    double sl = fmin(0.0, fmin(aml, amr)), sr = fmax(0.0, fmax(apl, apr));
    double inv = 1.0 / (sr - sl);
    double px = dir == 0 ? 1.0 : 0.0, py = 1.0 - px;
    double Snl = dir == 0 ? Sxl : Syl, Snr = dir == 0 ? Sxr : Syr;
    F[f] = (sr * Dl * vnl - sl * Dr * vnr + sl * sr * (Dr - Dl)) * inv;
    F[NF + f] = (sr * (Sxl * vnl + px * pl) - sl * (Sxr * vnr + px * pr) + sl * sr * (Sxr - Sxl)) * inv;
    F[2 * NF + f] = (sr * (Syl * vnl + py * pl) - sl * (Syr * vnr + py * pr) + sl * sr * (Syr - Syl)) * inv;
    F[3 * NF + f] = (sr * (Snl - Dl * vnl) - sl * (Snr - Dr * vnr) + sl * sr * (Er - El)) * inv;
  }
}

// One unsplit SSP-RK stage: Un = a U0 + b (U - dtdx (Fx_E - Fx_W) - dtdy (Fy_N - Fy_S)).
__global__ void rk2d(const double *U0, const double *U, const double *Fx, const double *Fy, double *Un, double a,
                     double b, double dtdx, double dtdy, int nx, int ny) {
  const int N = nx * ny, NFx = (nx + 1) * ny, NFy = nx * (ny + 1);
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N; c += gridDim.x * blockDim.x) {
    int i = c % nx, j = c / nx;
    int w = j * (nx + 1) + i, s = j * nx + i;
    for (int k = 0; k < 4; ++k)
      Un[k * N + c] = a * U0[k * N + c] +
                      b * (U[k * N + c] - dtdx * (Fx[k * NFx + w + 1] - Fx[k * NFx + w]) -
                           dtdy * (Fy[k * NFy + s + nx] - Fy[k * NFy + s]));
  }
}

// Primitive to conservative, for the initial data.
__global__ void p2c2(const double *P, double *U, const double *gam, int N) {
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N; c += gridDim.x * blockDim.x) {
    double g = gam[0], rho = P[c], ux = P[N + c], uy = P[2 * N + c], p = P[3 * N + c];
    double W = sqrt(1.0 + ux * ux + uy * uy), h = 1.0 + g / (g - 1.0) * p / rho;
    U[c] = rho * W;
    U[N + c] = rho * h * W * ux;
    U[2 * N + c] = rho * h * W * uy;
    U[3 * N + c] = rho * h * W * W - p - rho * W;
  }
}
