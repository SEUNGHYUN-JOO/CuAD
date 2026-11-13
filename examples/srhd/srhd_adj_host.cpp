// Host build of the SRHD kernels with their CuAD tangent (srhd_d.cuh) and adjoint (srhd_b.cuh).
// Kernel-level entry points for finite-difference / dot-product checks, and the gradient of a
// final-time density misfit with respect to the Riemann data and the adiabatic index, by the discrete
// adjoint of the whole time integration (states stored per stage, kernels reversed in order).
#include <cmath>
#include <vector>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#ifndef CUAD_TAPE_SIZE
#define CUAD_TAPE_SIZE 64
#endif
#include "cuad_rt.h"
#include "srhd.cu"
#include "srhd_d.cuh"
#include "srhd_b.cuh"

typedef std::vector<double> Vec;

extern "C" void c2p_apply(int solver, int n, const double *U, const double *gam, double *P) {
  if (solver == 0) c2p_newton(U, P, gam, n); else c2p_bisect(U, P, gam, n);
}
extern "C" void c2p_adjoint(int solver, int n, const double *U, const double *gam, const double *Pb_in,
                            double *Ub, double *gamb) {
  Vec P(3 * n), Pb(Pb_in, Pb_in + 3 * n);
  if (solver == 0) c2p_newton_b(U, Ub, P.data(), Pb.data(), gam, gamb, n);
  else c2p_bisect_b(U, Ub, P.data(), Pb.data(), gam, gamb, n);
}
extern "C" void c2p_tangent(int solver, int n, const double *U, const double *Ud, const double *gam,
                            const double *gamd, double *Pd) {
  Vec P(3 * n);
  if (solver == 0) c2p_newton_d(U, Ud, P.data(), Pd, gam, gamd, n);
  else c2p_bisect_d(U, Ud, P.data(), Pd, gam, gamd, n);
}

// q = (rhoL, uL, pL, rhoR, uR, pR, gam); interface at x = 0.5 of [0, 1].
static void initial(const double *q, int n, Vec &W) {
  W.assign(3 * n, 0);
  for (int i = 0; i < n; ++i) {
    bool left = (i + 0.5) / n < 0.5;
    W[i] = left ? q[0] : q[3];
    W[n + i] = left ? q[1] : q[4];
    W[2 * n + i] = left ? q[2] : q[5];
  }
}
static void c2p(int solver, const double *U, double *P, const double *g, int n) {
  if (solver == 0) c2p_newton(U, P, g, n); else c2p_bisect(U, P, g, n);
}
static void c2p_b(int solver, const double *U, double *Ub, double *P, double *Pb, const double *g, double *gb, int n) {
  if (solver == 0) c2p_newton_b(U, Ub, P, Pb, g, gb, n); else c2p_bisect_b(U, Ub, P, Pb, g, gb, n);
}

// Final-time primitives (used to make synthetic observations).
extern "C" void srhd_final(const double *q, int n, int steps, double cfl, int solver, double *P) {
  const double g[1] = {q[6]};
  Vec W0, U(3 * n), U1(3 * n), U2(3 * n), W(3 * n), F(3 * (n + 1));
  initial(q, n, W0);
  p2c(W0.data(), U.data(), g, n);
  for (int s = 0; s < steps; ++s) {
    c2p(solver, U.data(), W.data(), g, n);
    hll_flux(W.data(), F.data(), g, n);
    rk_stage(U.data(), U.data(), F.data(), U1.data(), 0.0, 1.0, cfl, n);
    c2p(solver, U1.data(), W.data(), g, n);
    hll_flux(W.data(), F.data(), g, n);
    rk_stage(U.data(), U1.data(), F.data(), U2.data(), 0.5, 0.5, cfl, n);
    U.swap(U2);
  }
  c2p(solver, U.data(), P, g, n);
}

// J = 0.5 dx sum (rho - obs)^2 at t = steps * cfl * dx. With grad != 0 also dJ/dq by the adjoint.
extern "C" double srhd_misfit(const double *q, int n, int steps, double cfl, int solver, const double *obs,
                              double *grad) {
  const double dx = 1.0 / n;
  const double g[1] = {q[6]};
  Vec W0, U(3 * n), W(3 * n), F(3 * (n + 1)), P(3 * n);
  initial(q, n, W0);
  p2c(W0.data(), U.data(), g, n);
  std::vector<Vec> Us(steps + 1, Vec(3 * n)), U1s(steps, Vec(3 * n));
  Us[0] = U;
  for (int s = 0; s < steps; ++s) {
    c2p(solver, Us[s].data(), W.data(), g, n);
    hll_flux(W.data(), F.data(), g, n);
    rk_stage(Us[s].data(), Us[s].data(), F.data(), U1s[s].data(), 0.0, 1.0, cfl, n);
    c2p(solver, U1s[s].data(), W.data(), g, n);
    hll_flux(W.data(), F.data(), g, n);
    rk_stage(Us[s].data(), U1s[s].data(), F.data(), Us[s + 1].data(), 0.5, 0.5, cfl, n);
  }
  c2p(solver, Us[steps].data(), P.data(), g, n);
  double J = 0;
  for (int i = 0; i < n; ++i) J += 0.5 * dx * (P[i] - obs[i]) * (P[i] - obs[i]);
  if (!grad) return J;

  double gb[1] = {0};
  Vec Ub(3 * n, 0), Pb(3 * n, 0);
  for (int i = 0; i < n; ++i) Pb[i] = dx * (P[i] - obs[i]);
  c2p_b(solver, Us[steps].data(), Ub.data(), P.data(), Pb.data(), g, gb, n);
  for (int s = steps - 1; s >= 0; --s) {
    Vec U1b(3 * n, 0), U0b(3 * n, 0), Fb(3 * (n + 1), 0), Wb(3 * n, 0), Un(3 * n);
    // U_{s+1} = 0.5 U_s + 0.5 (U1 - dtdx dF(W(U1)))
    c2p(solver, U1s[s].data(), W.data(), g, n);
    rk_stage_b(Us[s].data(), U0b.data(), U1s[s].data(), U1b.data(), F.data(), Fb.data(), Un.data(), Ub.data(),
               0.5, 0.5, cfl, n);
    hll_flux_b(W.data(), Wb.data(), F.data(), Fb.data(), g, gb, n);
    c2p_b(solver, U1s[s].data(), U1b.data(), W.data(), Wb.data(), g, gb, n);
    // U1 = U_s - dtdx dF(W(U_s))
    Vec Wb2(3 * n, 0), Fb2(3 * (n + 1), 0), Usb(3 * n, 0);
    c2p(solver, Us[s].data(), W.data(), g, n);
    rk_stage_b(Us[s].data(), Usb.data(), Us[s].data(), Usb.data(), F.data(), Fb2.data(), Un.data(), U1b.data(),
               0.0, 1.0, cfl, n);
    hll_flux_b(W.data(), Wb2.data(), F.data(), Fb2.data(), g, gb, n);
    c2p_b(solver, Us[s].data(), Usb.data(), W.data(), Wb2.data(), g, gb, n);
    for (int k = 0; k < 3 * n; ++k) Ub[k] = Usb[k] + U0b[k];
  }
  Vec W0b(3 * n, 0);
  p2c_b(W0.data(), W0b.data(), U.data(), Ub.data(), g, gb, n);
  for (int k = 0; k < 7; ++k) grad[k] = 0;
  for (int i = 0; i < n; ++i) {
    bool left = (i + 0.5) / n < 0.5;
    for (int c = 0; c < 3; ++c) grad[(left ? 0 : 3) + c] += W0b[c * n + i];
  }
  grad[6] = gb[0];
  return J;
}
