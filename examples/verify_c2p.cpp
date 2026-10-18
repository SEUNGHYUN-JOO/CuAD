// End-to-end verification of cuad on the real CuDAFLOW c2pKernel.
//   forward: tangent vs central finite difference (all 6 outputs)
//   reverse: dot-product identity  <w_b, J q_d> == <q_b, J^T w_b> == <q_b, q_d>
#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "cuad_rt.h"

#include "c2p.cu"       // primal (Blk, Thermo, th_p, c2pKernel)
#include "c2p_d.cuh"    // tangent
#include "c2p_b.cuh"    // adjoint

int main() {
  int fails = 0;
  const float eps = 3e-4f; // FP32: modest eps
  Blk B{}; B.ncT = 1; B.coff = 0;
  Thermo th{}; th.mode = 0;

  float q[6]  = {1.2f, 0.36f, -0.12f, 0.06f, 2.5f, 0.1f};
  float qd[6] = {0.7f, -0.3f, 0.5f, 0.2f, 1.1f, -0.4f}; // arbitrary direction

  // ---- forward: tangent vs finite difference ----
  float w[6] = {0}, wd[6] = {0};
  c2pKernel(B, nullptr, q, w, th);
  c2pKernel_d(B, nullptr, q, qd, w, wd, th);
  float qp[6], qm[6], wp[6] = {0}, wm[6] = {0};
  for (int k = 0; k < 6; k++) { qp[k] = q[k] + eps*qd[k]; qm[k] = q[k] - eps*qd[k]; }
  c2pKernel(B, nullptr, qp, wp, th);
  c2pKernel(B, nullptr, qm, wm, th);
  const char *nm[6] = {"rho","u","v","w","p","nut"};
  for (int k = 0; k < 6; k++) {
    float fd = (wp[k] - wm[k]) / (2*eps);
    float err = std::fabs(wd[k] - fd) / (1.f + std::fabs(fd));
    bool ok = err < 2e-3f;
    std::printf("  c2p tangent w[%s]   d=% .6e  fd=% .6e  relerr=%.1e  [%s]\n",
                nm[k], wd[k], fd, err, ok ? "PASS" : "FAIL");
    if (!ok) fails++;
  }

  // ---- reverse: dot-product identity ----
  float wb[6] = {0.9f, 0.2f, -0.5f, 0.7f, 1.3f, -0.6f}; // arbitrary seed
  double lhs = 0; for (int k = 0; k < 6; k++) lhs += (double)wb[k]*wd[k]; // <wb, J qd>
  float qb[6] = {0};
  c2pKernel_b(B, nullptr, q, qb, w, wb, th);
  double rhs = 0; for (int k = 0; k < 6; k++) rhs += (double)qb[k]*qd[k];  // <J^T wb, qd>
  double err = std::fabs(lhs - rhs) / (1.0 + std::fabs(lhs));
  bool ok = err < 1e-5;
  std::printf("  c2p adjoint dot-test   <wb,Jqd>=% .8e  <J^Twb,qd>=% .8e  relerr=%.1e  [%s]\n",
              lhs, rhs, err, ok ? "PASS" : "FAIL");
  if (!ok) fails++;

  std::printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fails);
  return fails ? 1 : 0;
}
