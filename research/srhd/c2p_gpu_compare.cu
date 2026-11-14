// GPU comparison on the conservative-to-primitive kernels of examples/srhd/srhd.cu (unchanged source):
// Enzyme reverse AD (differentiates the iterations, caching what the loop with `break` needs) versus the
// CuAD implicit adjoint (srhd_b.cuh). Same Clang 19, sm_90, one cell per thread for both.
#include <cuda_runtime.h>
#include "cuad_rt.h"
#include "srhd.cu"
#include "srhd_b.cuh"
#include "c2p_fixed.cu"
#include "c2p_fixed_b.cuh"
#include "c2p_manual_adjoint.cu"
#include <cstdlib>
#ifndef ENZYME_HEAP_MIB
#define ENZYME_HEAP_MIB 1024
#endif

namespace dev { // the same kernels as device functions, for Enzyme
#pragma push_macro("__global__")
#undef __global__
#define __global__ __device__
#include "srhd.cu"
#include "c2p_fixed.cu"
#pragma pop_macro("__global__")
}
#ifdef WITH_ABLATION
namespace devr { // the C2P kernels with every pointer parameter declared __restrict__ (srhd_restrict.cu)
#pragma push_macro("__global__")
#undef __global__
#define __global__ __device__
#include "srhd_restrict.cu"
#pragma pop_macro("__global__")
}
namespace nochk { // the CuAD adjoint compiled without its run-time checks, to measure their cost
#pragma push_macro("CUAD_SOLVE_CHECK")
#undef CUAD_SOLVE_CHECK
#define CUAD_SOLVE_CHECK 0
#include "srhd_b.cuh"
#pragma pop_macro("CUAD_SOLVE_CHECK")
}
namespace allchk { // the CuAD adjoint with every check, including the second derivative evaluation of Class I
#pragma push_macro("CUAD_SOLVE_CHECK")
#undef CUAD_SOLVE_CHECK
#define CUAD_SOLVE_CHECK 2
#include "srhd_b.cuh"
#pragma pop_macro("CUAD_SOLVE_CHECK")
}
#endif
extern __device__ int enzyme_dup, enzyme_const;
template <class... A> __device__ void __enzyme_autodiff(void *, A...);

__global__ void enzyme_c2p_b(int solver, const double *U, double *Ub, double *P, double *Pb, const double *g,
                             double *gb, int n) {
  if (solver == 0)
    __enzyme_autodiff((void *)dev::c2p_newton, enzyme_dup, U, Ub, enzyme_dup, P, Pb, enzyme_dup, g, gb,
                      enzyme_const, n);
  else if (solver == 1)
    __enzyme_autodiff((void *)dev::c2p_bisect, enzyme_dup, U, Ub, enzyme_dup, P, Pb, enzyme_dup, g, gb,
                      enzyme_const, n);
  else
    __enzyme_autodiff((void *)dev::c2p_newton_fixed, enzyme_dup, U, Ub, enzyme_dup, P, Pb, enzyme_dup, g, gb,
                      enzyme_const, n);
}
#ifdef WITH_ABLATION
__global__ void enzyme_c2p_b_restrict(int solver, const double *U, double *Ub, double *P, double *Pb, const double *g,
                                      double *gb, int n) {
  if (solver == 0)
    __enzyme_autodiff((void *)devr::c2p_newton, enzyme_dup, U, Ub, enzyme_dup, P, Pb, enzyme_dup, g, gb,
                      enzyme_const, n);
  else
    __enzyme_autodiff((void *)devr::c2p_bisect, enzyme_dup, U, Ub, enzyme_dup, P, Pb, enzyme_dup, g, gb,
                      enzyme_const, n);
}
#endif
// solver: 0 Newton with break, 1 bisection, 2 Newton with a fixed iteration count
static void primal(int s, int gs, const double *U, double *P, const double *g, int n) {
  if (s == 0) c2p_newton<<<gs, 256>>>(U, P, g, n);
  else if (s == 1) c2p_bisect<<<gs, 256>>>(U, P, g, n);
  else c2p_newton_fixed<<<gs, 256>>>(U, P, g, n);
}
static void cuad_b(int s, int gs, const double *U, double *Ub, double *P, double *Pb, const double *g, double *gb, int n) {
  if (s == 0) c2p_newton_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
  else if (s == 1) c2p_bisect_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
  else c2p_newton_fixed_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
}

// tool: 0 primal, 1 CuAD adjoint, 2 Enzyme adjoint, 3 hand-written implicit adjoint (Newton primal; solver ignored),
// with WITH_ABLATION also 4 Enzyme on the __restrict__ kernels and 5 the CuAD adjoint without run-time checks
// and 6 the CuAD adjoint with every check (CUAD_SOLVE_CHECK=2) (solvers 0 and 1 only).
// Pb is copied in for every launch (the adjoints consume it). The device heap (Enzyme's cache) is ENZYME_HEAP_MIB,
// or the environment variable C2P_HEAP_MIB if set, fixed for the process at the first call.
extern "C" int c2p_gpu(int tool, int solver, int n, const double *Uh, double gam, const double *Pbh, double *Ubh,
                       double *gbh, int reps, double *times, int *regs, int *local) {
  static bool heap = false;
  cudaError_t err = cudaSuccess;
  if (!heap) {
    size_t mib = ENZYME_HEAP_MIB;
    if (const char *e = std::getenv("C2P_HEAP_MIB")) mib = (size_t)std::atol(e);
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, mib << 20);
    if (err) return int(err);
    heap = true;
  }
  const size_t sz = sizeof(double) * 3 * n;
  const int gs = (n + 255) / 256;
  double *U = 0, *Ub = 0, *P = 0, *Pb = 0, *Pb0 = 0, *g = 0, *gb = 0;
  cudaEvent_t a = 0, b = 0;
  cudaFuncAttributes at;
#define CK(x) do { err = (x); if (err != cudaSuccess) goto done; } while (0)
  if (tool == 0) CK(cudaFuncGetAttributes(&at, solver == 0 ? c2p_newton : solver == 1 ? c2p_bisect : c2p_newton_fixed));
  else if (tool == 1)
    CK(cudaFuncGetAttributes(&at, solver == 0 ? c2p_newton_b : solver == 1 ? c2p_bisect_b : c2p_newton_fixed_b));
  else if (tool == 2) CK(cudaFuncGetAttributes(&at, enzyme_c2p_b));
#ifdef WITH_ABLATION
  else if (tool == 4) CK(cudaFuncGetAttributes(&at, enzyme_c2p_b_restrict));
  else if (tool == 5) CK(cudaFuncGetAttributes(&at, solver == 0 ? nochk::c2p_newton_b : nochk::c2p_bisect_b));
  else if (tool == 6) CK(cudaFuncGetAttributes(&at, solver == 0 ? allchk::c2p_newton_b : allchk::c2p_bisect_b));
#endif
  else CK(cudaFuncGetAttributes(&at, c2p_manual_b));
  *regs = at.numRegs; *local = (int)at.localSizeBytes;
  CK(cudaMalloc(&U, sz)); CK(cudaMalloc(&Ub, sz)); CK(cudaMalloc(&P, sz)); CK(cudaMalloc(&Pb, sz));
  CK(cudaMalloc(&Pb0, sz)); CK(cudaMalloc(&g, 8)); CK(cudaMalloc(&gb, 8));
  CK(cudaMemcpy(U, Uh, sz, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Pb0, Pbh, sz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g, &gam, 8, cudaMemcpyHostToDevice));
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int r = -2; r < reps; ++r) {
    CK(cudaMemset(Ub, 0, sz)); CK(cudaMemset(gb, 0, 8));
    CK(cudaMemcpy(Pb, Pb0, sz, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(a));
    if (tool == 0) primal(solver, gs, U, P, g, n);
    else if (tool == 1) cuad_b(solver, gs, U, Ub, P, Pb, g, gb, n);
    else if (tool == 2) enzyme_c2p_b<<<gs, 256>>>(solver, U, Ub, P, Pb, g, gb, n);
#ifdef WITH_ABLATION
    else if (tool == 4) enzyme_c2p_b_restrict<<<gs, 256>>>(solver, U, Ub, P, Pb, g, gb, n);
    else if (tool == 5) {
      if (solver == 0) nochk::c2p_newton_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
      else nochk::c2p_bisect_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
    } else if (tool == 6) {
      if (solver == 0) allchk::c2p_newton_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
      else allchk::c2p_bisect_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
    }
#endif
    else c2p_manual_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
    CK(cudaGetLastError()); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    if (r >= 0) { float ms; CK(cudaEventElapsedTime(&ms, a, b)); times[r] = ms; }
  }
  CK(cudaMemcpy(Ubh, Ub, sz, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gbh, gb, 8, cudaMemcpyDeviceToHost));
done:
  for (double *p : {U, Ub, P, Pb, Pb0, g, gb}) if (p) cudaFree(p);
  if (a) cudaEventDestroy(a);
  if (b) cudaEventDestroy(b);
  return int(err);
}
