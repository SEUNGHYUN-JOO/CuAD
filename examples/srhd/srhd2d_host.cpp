// Host build of the 2-D SRHD kernels and their CuAD adjoint, for verification on small grids.
// srhd2_obs: J = sum_j 0.5 dx dy sum_k w_k (P_k(t_j) - obs_jk)^2 over observations at step indices kst[j]; with
// W0b != 0 also dJ/dW0 (4N) and dJ/dgam, by storing the stage states and reversing the generated kernels.
#include <cmath>
#include <vector>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#ifndef CUAD_TAPE_SIZE
#define CUAD_TAPE_SIZE 256
#endif
#include "cuad_rt.h"
#include "srhd2d.cu"
#include "srhd2d_b.cuh"
#include "srhd2d_d.cuh"

typedef std::vector<double> Vec;
static void c2p(int s, const double *U, double *P, const double *g, int N) {
  if (s == 0) c2p2_newton(U, P, g, N); else c2p2_bisect(U, P, g, N);
}
static void c2p_b(int s, const double *U, double *Ub, double *P, double *Pb, const double *g, double *gb, int N) {
  if (s == 0) c2p2_newton_b(U, Ub, P, Pb, g, gb, N); else c2p2_bisect_b(U, Ub, P, Pb, g, gb, N);
}

extern "C" double srhd2_obs(int nx, int ny, int steps, double cfl, int solver, const double *W0, double gam, int nobs,
                            const int *kst, const double *obs, const double *w, double *W0b, double *gamb,
                            double *Pobs) {
  const int N = nx * ny, m = 4 * N, NFx = 4 * (nx + 1) * ny, NFy = 4 * nx * (ny + 1);
  // dt = cfl min(dx, dy)
  const double dx = 1.0 / nx, dy = 1.0 / ny, dt = cfl * (dx < dy ? dx : dy), dtdx = dt / dx, dty = dt / dy;
  const double g[1] = {gam};
  std::vector<Vec> Us(steps + 1, Vec(m)), U1s(steps, Vec(m));
  Vec W(m), Fx(NFx), Fy(NFy), Pb(m * (size_t)nobs, 0.0);
  std::vector<int> at(steps + 1, -1);
  for (int j = 0; j < nobs; ++j) at[kst[j]] = j;
  p2c2(W0, Us[0].data(), g, N);
  for (int s = 0; s < steps; ++s) {
    c2p(solver, Us[s].data(), W.data(), g, N);
    hll2(W.data(), Fx.data(), g, nx, ny, 0);
    hll2(W.data(), Fy.data(), g, nx, ny, 1);
    rk2d(Us[s].data(), Us[s].data(), Fx.data(), Fy.data(), U1s[s].data(), 0.0, 1.0, dtdx, dty, nx, ny);
    c2p(solver, U1s[s].data(), W.data(), g, N);
    hll2(W.data(), Fx.data(), g, nx, ny, 0);
    hll2(W.data(), Fy.data(), g, nx, ny, 1);
    rk2d(Us[s].data(), U1s[s].data(), Fx.data(), Fy.data(), Us[s + 1].data(), 0.5, 0.5, dtdx, dty, nx, ny);
  }
  double J = 0;
  for (int j = 0; j < nobs; ++j) {
    c2p(solver, Us[kst[j]].data(), W.data(), g, N);
    for (int k = 0; k < m; ++k) {
      double r = W[k] - obs[(size_t)j * m + k];
      J += 0.5 * dx * dy * w[k] * r * r;
      Pb[(size_t)j * m + k] = dx * dy * w[k] * r;
      if (Pobs) Pobs[(size_t)j * m + k] = W[k];
    }
  }
  if (!W0b) return J;
  double gb[1] = {0};
  Vec Ub(m, 0), U1b(m), U0b(m), Wb(m), Fxb(NFx), Fyb(NFy), Un(m);
  if (at[steps] >= 0) c2p_b(solver, Us[steps].data(), Ub.data(), W.data(), &Pb[(size_t)at[steps] * m], g, gb, N);
  for (int s = steps - 1; s >= 0; --s) {
    std::fill(U1b.begin(), U1b.end(), 0.0); std::fill(U0b.begin(), U0b.end(), 0.0);
    std::fill(Fxb.begin(), Fxb.end(), 0.0); std::fill(Fyb.begin(), Fyb.end(), 0.0); std::fill(Wb.begin(), Wb.end(), 0.0);
    c2p(solver, U1s[s].data(), W.data(), g, N);
    rk2d_b(Us[s].data(), U0b.data(), U1s[s].data(), U1b.data(), Fx.data(), Fxb.data(), Fy.data(), Fyb.data(), Un.data(),
           Ub.data(), 0.5, 0.5, dtdx, dty, nx, ny);
    hll2_b(W.data(), Wb.data(), Fx.data(), Fxb.data(), g, gb, nx, ny, 0);
    hll2_b(W.data(), Wb.data(), Fy.data(), Fyb.data(), g, gb, nx, ny, 1);
    c2p_b(solver, U1s[s].data(), U1b.data(), W.data(), Wb.data(), g, gb, N);
    std::fill(Ub.begin(), Ub.end(), 0.0);
    std::fill(Fxb.begin(), Fxb.end(), 0.0); std::fill(Fyb.begin(), Fyb.end(), 0.0); std::fill(Wb.begin(), Wb.end(), 0.0);
    c2p(solver, Us[s].data(), W.data(), g, N);
    rk2d_b(Us[s].data(), Ub.data(), Us[s].data(), Ub.data(), Fx.data(), Fxb.data(), Fy.data(), Fyb.data(), Un.data(),
           U1b.data(), 0.0, 1.0, dtdx, dty, nx, ny);
    hll2_b(W.data(), Wb.data(), Fx.data(), Fxb.data(), g, gb, nx, ny, 0);
    hll2_b(W.data(), Wb.data(), Fy.data(), Fyb.data(), g, gb, nx, ny, 1);
    c2p_b(solver, Us[s].data(), Ub.data(), W.data(), Wb.data(), g, gb, N);
    for (int k = 0; k < m; ++k) Ub[k] += U0b[k];
    if (at[s] >= 0) c2p_b(solver, Us[s].data(), Ub.data(), W.data(), &Pb[(size_t)at[s] * m], g, gb, N);
  }
  Vec U0(m);
  for (int k = 0; k < m; ++k) W0b[k] = 0;
  p2c2_b(W0, W0b, U0.data(), Ub.data(), g, gb, N);
  *gamb = gb[0];
  return J;
}

// Directional derivative of the same J by the CuAD tangent (Newton C2P only: forward mode rejects bisection):
// J_d = dJ/dW0 . W0d + dJ/dgam gamd, for the adjoint dot-product test.
extern "C" double srhd2_obs_tangent(int nx, int ny, int steps, double cfl, const double *W0, const double *W0d,
                                    double gam, double gamd, int nobs, const int *kst, const double *obs,
                                    const double *w) {
  const int N = nx * ny, m = 4 * N, NFx = 4 * (nx + 1) * ny, NFy = 4 * nx * (ny + 1);
  const double dx = 1.0 / nx, dy = 1.0 / ny, dt = cfl * (dx < dy ? dx : dy), dtdx = dt / dx, dty = dt / dy;
  const double g[1] = {gam}, gd[1] = {gamd};
  Vec U(m), Ud(m), U1(m), U1d(m), U2(m), U2d(m), W(m), Wd(m), Fx(NFx), Fxd(NFx), Fy(NFy), Fyd(NFy);
  p2c2_d(W0, W0d, U.data(), Ud.data(), g, gd, N);
  double Jd = 0;
  for (int s = 0; s <= steps; ++s) {
    for (int j = 0; j < nobs; ++j)
      if (kst[j] == s) {
        c2p2_newton_d(U.data(), Ud.data(), W.data(), Wd.data(), g, gd, N);
        for (int k = 0; k < m; ++k) Jd += dx * dy * w[k] * (W[k] - obs[(size_t)j * m + k]) * Wd[k];
      }
    if (s == steps) break;
    c2p2_newton_d(U.data(), Ud.data(), W.data(), Wd.data(), g, gd, N);
    hll2_d(W.data(), Wd.data(), Fx.data(), Fxd.data(), g, gd, nx, ny, 0);
    hll2_d(W.data(), Wd.data(), Fy.data(), Fyd.data(), g, gd, nx, ny, 1);
    rk2d_d(U.data(), Ud.data(), U.data(), Ud.data(), Fx.data(), Fxd.data(), Fy.data(), Fyd.data(), U1.data(), U1d.data(),
           0.0, 0.0, 1.0, 0.0, dtdx, 0.0, dty, 0.0, nx, ny);
    c2p2_newton_d(U1.data(), U1d.data(), W.data(), Wd.data(), g, gd, N);
    hll2_d(W.data(), Wd.data(), Fx.data(), Fxd.data(), g, gd, nx, ny, 0);
    hll2_d(W.data(), Wd.data(), Fy.data(), Fyd.data(), g, gd, nx, ny, 1);
    rk2d_d(U.data(), Ud.data(), U1.data(), U1d.data(), Fx.data(), Fxd.data(), Fy.data(), Fyd.data(), U2.data(),
           U2d.data(), 0.5, 0.0, 0.5, 0.0, dtdx, 0.0, dty, 0.0, nx, ny);
    U.swap(U2); Ud.swap(U2d);
  }
  return Jd;
}
