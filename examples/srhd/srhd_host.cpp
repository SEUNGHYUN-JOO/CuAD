// Host build of the SRHD kernels (test/cuad_rt.h style emulation: one block of one thread runs every
// grid-stride loop serially). Exposes the primal solve to Python through ctypes.
#include <cmath>
#include <vector>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "srhd.cu"

// Riemann problem on [0,1], interface at 0.5, fixed dt = cfl*dx (signal speeds are below c = 1).
// solver: 0 = Newton C2P, 1 = bisection C2P. Returns the primitives at t = steps*dt in P[3n].
extern "C" void srhd_run(int n, int steps, double cfl, double gam, double rhoL, double uL, double pL,
                         double rhoR, double uR, double pR, int solver, double *P) {
  const double dx = 1.0 / n, dtdx = cfl;
  std::vector<double> U(3 * n), U1(3 * n), U2(3 * n), F(3 * (n + 1)), W(3 * n);
  const double g[1] = {gam};
  for (int i = 0; i < n; ++i) {
    bool left = (i + 0.5) * dx < 0.5;
    W[i] = left ? rhoL : rhoR;
    W[n + i] = left ? uL : uR;
    W[2 * n + i] = left ? pL : pR;
  }
  p2c(W.data(), U.data(), g, n);
  auto c2p = [&](const double *u, double *w) {
    if (solver == 0) c2p_newton(u, w, g, n); else c2p_bisect(u, w, g, n);
  };
  for (int s = 0; s < steps; ++s) {
    c2p(U.data(), W.data());
    hll_flux(W.data(), F.data(), g, n);
    rk_stage(U.data(), U.data(), F.data(), U1.data(), 0.0, 1.0, dtdx, n);
    c2p(U1.data(), W.data());
    hll_flux(W.data(), F.data(), g, n);
    rk_stage(U.data(), U1.data(), F.data(), U2.data(), 0.5, 0.5, dtdx, n);
    U.swap(U2);
  }
  c2p(U.data(), P);
}
