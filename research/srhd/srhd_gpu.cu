// GPU driver for the SRHD solver and its CuAD adjoint (examples/srhd/srhd.cu, srhd_b.cuh).
// Gradient of J = 0.5 dx sum_k w_k (P_k(T) - obs_k)^2 with respect to the whole initial primitive field
// W0 (3n values) and the adiabatic index, by storing the stage states and running the generated adjoint
// kernels in reverse order. Times: forward sweep and reverse sweep, CUDA events.
#include <cuda_runtime.h>
#include <vector>
#include "cuad_rt.h"
#include "srhd.cu"
#include "srhd_b.cuh"

__global__ void seed_misfit(const double *P, const double *obs, const double *w, double *Pb, double *Jc, double dx,
                            int m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    double r = P[k] - obs[k];
    Pb[k] = dx * w[k] * r;
    Jc[k] = 0.5 * dx * w[k] * r * r;
  }
}

__global__ void add_into(double *a, const double *b, int m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) a[k] += b[k];
}

static void c2p(int solver, const double *U, double *P, const double *g, int n, int gs) {
  if (solver == 0) c2p_newton<<<gs, 256>>>(U, P, g, n); else c2p_bisect<<<gs, 256>>>(U, P, g, n);
}
static void c2p_b(int solver, const double *U, double *Ub, double *P, double *Pb, const double *g, double *gb, int n,
                  int gs) {
  if (solver == 0) c2p_newton_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
  else c2p_bisect_b<<<gs, 256>>>(U, Ub, P, Pb, g, gb, n);
}

// Observations of the primitives at nobs step indices kst[j] (state U_k, 0 <= k <= steps), obs[j*3n...], one
// weight vector w[3n]: J = sum_j 0.5 dx sum_k w_k (P_k(t_j) - obs_jk)^2. W0bh != 0: also the gradient (W0b[3n],
// gamb). Ph != 0: the observed primitives (nobs x 3n). Returns cudaError_t.
extern "C" int srhd_gpu_obs(int n, int steps, double cfl, int solver, const double *W0h, double gam, int nobs,
                            const int *kst, const double *obsh, const double *wh, double *J, double *W0bh,
                            double *gamb, double *times, double *Ph) {
  const int gs = (n + 1 + 255) / 256, m = 3 * n;
  const size_t sz = sizeof(double) * m, fz = sizeof(double) * 3 * (n + 1);
  cudaError_t err = cudaSuccess;
  double *g = 0, *gb = 0, *W0 = 0, *W = 0, *F = 0, *Us = 0, *U1s = 0, *obs = 0, *w = 0, *Pb = 0, *Jc = 0;
  double *Ub = 0, *U1b = 0, *U0b = 0, *Wb = 0, *Fb = 0, *Un = 0, *W0b = 0;
  cudaEvent_t e0 = 0, e1 = 0, e2 = 0, e3 = 0;
  std::vector<double> Jh(m);
  std::vector<int> at(steps + 1, -1);
  for (int j = 0; j < nobs; ++j)
    if (kst[j] >= 0 && kst[j] <= steps) at[kst[j]] = j;
#define CK(x) do { err = (x); if (err != cudaSuccess) goto done; } while (0)
  CK(cudaMalloc(&g, 8)); CK(cudaMalloc(&gb, 8)); CK(cudaMalloc(&W0, sz)); CK(cudaMalloc(&W, sz));
  CK(cudaMalloc(&F, fz)); CK(cudaMalloc(&Us, sz * (steps + 1))); CK(cudaMalloc(&U1s, sz * steps));
  CK(cudaMalloc(&obs, sz * nobs)); CK(cudaMalloc(&w, sz)); CK(cudaMalloc(&Pb, sz * nobs)); CK(cudaMalloc(&Jc, sz));
  CK(cudaMemcpy(g, &gam, 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(W0, W0h, sz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(obs, obsh, sz * nobs, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w, wh, sz, cudaMemcpyHostToDevice));
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));
  CK(cudaEventRecord(e0));
  p2c<<<gs, 256>>>(W0, Us, g, n);
  for (int s = 0; s < steps; ++s) {
    double *U = Us + (size_t)s * m, *U1 = U1s + (size_t)s * m, *U2 = Us + (size_t)(s + 1) * m;
    c2p(solver, U, W, g, n, gs);
    hll_flux<<<gs, 256>>>(W, F, g, n);
    rk_stage<<<gs, 256>>>(U, U, F, U1, 0.0, 1.0, cfl, n);
    c2p(solver, U1, W, g, n, gs);
    hll_flux<<<gs, 256>>>(W, F, g, n);
    rk_stage<<<gs, 256>>>(U, U1, F, U2, 0.5, 0.5, cfl, n);
  }
  *J = 0;
  for (int j = 0; j < nobs; ++j) {
    c2p(solver, Us + (size_t)kst[j] * m, W, g, n, gs);
    seed_misfit<<<(m + 255) / 256, 256>>>(W, obs + (size_t)j * m, w, Pb + (size_t)j * m, Jc, 1.0 / n, m);
    CK(cudaGetLastError());
    CK(cudaMemcpy(Jh.data(), Jc, sz, cudaMemcpyDeviceToHost));
    if (Ph) CK(cudaMemcpy(Ph + (size_t)j * m, W, sz, cudaMemcpyDeviceToHost));
    for (double v : Jh) *J += v;
  }
  CK(cudaEventRecord(e1));
  if (W0bh) {
    CK(cudaMalloc(&Ub, sz)); CK(cudaMalloc(&U1b, sz)); CK(cudaMalloc(&U0b, sz)); CK(cudaMalloc(&Wb, sz));
    CK(cudaMalloc(&Fb, fz)); CK(cudaMalloc(&Un, sz)); CK(cudaMalloc(&W0b, sz));
    CK(cudaMemset(gb, 0, 8)); CK(cudaMemset(Ub, 0, sz)); CK(cudaMemset(W0b, 0, sz));
    CK(cudaEventRecord(e3));
    if (at[steps] >= 0) c2p_b(solver, Us + (size_t)steps * m, Ub, W, Pb + (size_t)at[steps] * m, g, gb, n, gs);
    for (int s = steps - 1; s >= 0; --s) {
      double *U = Us + (size_t)s * m, *U1 = U1s + (size_t)s * m;
      cudaMemsetAsync(U1b, 0, sz); cudaMemsetAsync(U0b, 0, sz); cudaMemsetAsync(Fb, 0, fz); cudaMemsetAsync(Wb, 0, sz);
      c2p(solver, U1, W, g, n, gs);
      rk_stage_b<<<gs, 256>>>(U, U0b, U1, U1b, F, Fb, Un, Ub, 0.5, 0.5, cfl, n);
      hll_flux_b<<<gs, 256>>>(W, Wb, F, Fb, g, gb, n);
      c2p_b(solver, U1, U1b, W, Wb, g, gb, n, gs);
      // Ub now free: it becomes the adjoint of U_s, accumulated from both stages
      cudaMemsetAsync(Ub, 0, sz); cudaMemsetAsync(Fb, 0, fz); cudaMemsetAsync(Wb, 0, sz);
      c2p(solver, U, W, g, n, gs);
      rk_stage_b<<<gs, 256>>>(U, Ub, U, Ub, F, Fb, Un, U1b, 0.0, 1.0, cfl, n);
      hll_flux_b<<<gs, 256>>>(W, Wb, F, Fb, g, gb, n);
      c2p_b(solver, U, Ub, W, Wb, g, gb, n, gs);
      add_into<<<(m + 255) / 256, 256>>>(Ub, U0b, m);
      if (at[s] >= 0) c2p_b(solver, U, Ub, W, Pb + (size_t)at[s] * m, g, gb, n, gs);
    }
    CK(cudaGetLastError());
    p2c_b<<<gs, 256>>>(W0, W0b, Us, Ub, g, gb, n);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e2));
    CK(cudaEventSynchronize(e2));
    CK(cudaMemcpy(W0bh, W0b, sz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gamb, gb, 8, cudaMemcpyDeviceToHost));
  }
  CK(cudaEventSynchronize(e1));
done:
  if (e0 && e1) { float a = 0; cudaEventElapsedTime(&a, e0, e1); times[0] = a; }
  if (W0bh && e3 && e2) { float b = 0; cudaEventElapsedTime(&b, e3, e2); times[1] = b; }
  for (double *p : {g, gb, W0, W, F, Us, U1s, obs, w, Pb, Jc, Ub, U1b, U0b, Wb, Fb, Un, W0b})
    if (p) cudaFree(p);
  if (e0) cudaEventDestroy(e0);
  if (e1) cudaEventDestroy(e1);
  if (e2) cudaEventDestroy(e2);
  if (e3) cudaEventDestroy(e3);
  return int(err);
}

// a single observation at the final time
extern "C" int srhd_gpu(int n, int steps, double cfl, int solver, const double *W0h, double gam, const double *obsh,
                        const double *wh, double *J, double *W0bh, double *gamb, double *times, double *Ph) {
  return srhd_gpu_obs(n, steps, cfl, solver, W0h, gam, 1, &steps, obsh, wh, J, W0bh, gamb, times, Ph);
}
