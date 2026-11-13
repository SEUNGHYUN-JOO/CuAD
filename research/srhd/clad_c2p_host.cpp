// Clad (source transformation in Clang, reverse mode) on the unchanged C2P kernels of examples/srhd/srhd.cu,
// host build: gradient of sum_k P_k Pb_k with respect to U and gam, compared in check_clad_host.py.
#include <cmath>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "srhd.cu"
#include "clad/Differentiator/Differentiator.h"

#define NC 64
double dot_newton(const double *U, const double *gam, const double *Pb) {
  double P[3 * NC];
  c2p_newton(U, P, gam, NC);
  double s = 0;
  for (int k = 0; k < 3 * NC; ++k) s += P[k] * Pb[k];
  return s;
}
double dot_bisect(const double *U, const double *gam, const double *Pb) {
  double P[3 * NC];
  c2p_bisect(U, P, gam, NC);
  double s = 0;
  for (int k = 0; k < 3 * NC; ++k) s += P[k] * Pb[k];
  return s;
}

extern "C" void clad_c2p_adjoint(int solver, const double *U, const double *gam, const double *Pb, double *Ub,
                                 double *gamb) {
  if (solver == 0) {
    auto g = clad::gradient(dot_newton, "U, gam");
    g.execute(U, gam, Pb, Ub, gamb);
  } else {
    auto g = clad::gradient(dot_bisect, "U, gam");
    g.execute(U, gam, Pb, Ub, gamb);
  }
}
