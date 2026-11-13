// GPU driver of the 2-D SRHD solver (examples/srhd/srhd2d.cu) and its CuAD adjoint (srhd2d_b.cuh), same algorithm as
// examples/srhd/srhd2d_host.cpp: J = sum_j 0.5 dx dy sum_k w_k (P_k(t_j) - obs_jk)^2 over observations at step indices
// kst[j]; gradient with respect to the whole initial primitive field (4N) and gam by storing the stage states.
#include <cuda_runtime.h>
#include <vector>
#include "cuad_rt.h"
#include "srhd2d.cu"
#include "srhd2d_b.cuh"

__global__ void seed2(const double *P, const double *obs, const double *w, double *Pb, double *Jc, double dA, int m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) {
    double r = P[k] - obs[k];
    Pb[k] = dA * w[k] * r;
    Jc[k] = 0.5 * dA * w[k] * r * r;
  }
}
__global__ void add2(double *a, const double *b, int m) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) a[k] += b[k];
}

extern "C" int srhd2_gpu(int nx, int ny, int steps, double cfl, int solver, const double *W0h, double gam, int nobs,
                         const int *kst, const double *obsh, const double *wh, double *J, double *W0bh, double *gamb,
                         double *times, double *Ph) {
  const int N = nx * ny, m = 4 * N, NFx = 4 * (nx + 1) * ny, NFy = 4 * nx * (ny + 1);
  const double dx = 1.0 / nx, dy = 1.0 / ny, dt = cfl * (dx < dy ? dx : dy), dtdx = dt / dx, dty = dt / dy;
  const int B = 256, gN = (N + B - 1) / B, gm = (m + B - 1) / B, gFx = ((nx + 1) * ny + B - 1) / B, gFy = (nx * (ny + 1) + B - 1) / B;
  const size_t sz = sizeof(double) * m;
  cudaError_t err = cudaSuccess;
  double *g = 0, *gb = 0, *W0 = 0, *W = 0, *Fx = 0, *Fy = 0, *Us = 0, *U1s = 0, *obs = 0, *w = 0, *Pb = 0, *Jc = 0;
  double *Ub = 0, *U1b = 0, *U0b = 0, *Wb = 0, *Fxb = 0, *Fyb = 0, *Un = 0, *W0b = 0;
  cudaEvent_t e0 = 0, e1 = 0, e2 = 0, e3 = 0;
  std::vector<double> Jh(m);
  std::vector<int> at(steps + 1, -1);
  for (int j = 0; j < nobs; ++j)
    if (kst[j] >= 0 && kst[j] <= steps) at[kst[j]] = j;
  auto c2p = [&](const double *U, double *P) {
    if (solver == 0) c2p2_newton<<<gN, B>>>(U, P, g, N); else c2p2_bisect<<<gN, B>>>(U, P, g, N);
  };
  auto c2pb = [&](const double *U, double *Ub_, double *P, double *Pb_) {
    if (solver == 0) c2p2_newton_b<<<gN, B>>>(U, Ub_, P, Pb_, g, gb, N);
    else c2p2_bisect_b<<<gN, B>>>(U, Ub_, P, Pb_, g, gb, N);
  };
#define CK(x) do { err = (x); if (err != cudaSuccess) goto done; } while (0)
  CK(cudaMalloc(&g, 8)); CK(cudaMalloc(&gb, 8)); CK(cudaMalloc(&W0, sz)); CK(cudaMalloc(&W, sz));
  CK(cudaMalloc(&Fx, sizeof(double) * NFx)); CK(cudaMalloc(&Fy, sizeof(double) * NFy));
  CK(cudaMalloc(&Us, sz * (steps + 1))); CK(cudaMalloc(&U1s, sz * steps));
  CK(cudaMalloc(&obs, sz * nobs)); CK(cudaMalloc(&w, sz)); CK(cudaMalloc(&Pb, sz * nobs)); CK(cudaMalloc(&Jc, sz));
  CK(cudaMemcpy(g, &gam, 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(W0, W0h, sz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(obs, obsh, sz * nobs, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w, wh, sz, cudaMemcpyHostToDevice));
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));
  CK(cudaEventRecord(e0));
  p2c2<<<gN, B>>>(W0, Us, g, N);
  for (int s = 0; s < steps; ++s) {
    double *U = Us + (size_t)s * m, *U1 = U1s + (size_t)s * m, *U2 = Us + (size_t)(s + 1) * m;
    c2p(U, W);
    hll2<<<gFx, B>>>(W, Fx, g, nx, ny, 0);
    hll2<<<gFy, B>>>(W, Fy, g, nx, ny, 1);
    rk2d<<<gN, B>>>(U, U, Fx, Fy, U1, 0.0, 1.0, dtdx, dty, nx, ny);
    c2p(U1, W);
    hll2<<<gFx, B>>>(W, Fx, g, nx, ny, 0);
    hll2<<<gFy, B>>>(W, Fy, g, nx, ny, 1);
    rk2d<<<gN, B>>>(U, U1, Fx, Fy, U2, 0.5, 0.5, dtdx, dty, nx, ny);
  }
  *J = 0;
  for (int j = 0; j < nobs; ++j) {
    c2p(Us + (size_t)kst[j] * m, W);
    seed2<<<gm, B>>>(W, obs + (size_t)j * m, w, Pb + (size_t)j * m, Jc, dx * dy, m);
    CK(cudaGetLastError());
    CK(cudaMemcpy(Jh.data(), Jc, sz, cudaMemcpyDeviceToHost));
    if (Ph) CK(cudaMemcpy(Ph + (size_t)j * m, W, sz, cudaMemcpyDeviceToHost));
    for (double v : Jh) *J += v;
  }
  CK(cudaEventRecord(e1));
  if (W0bh) {
    CK(cudaMalloc(&Ub, sz)); CK(cudaMalloc(&U1b, sz)); CK(cudaMalloc(&U0b, sz)); CK(cudaMalloc(&Wb, sz));
    CK(cudaMalloc(&Fxb, sizeof(double) * NFx)); CK(cudaMalloc(&Fyb, sizeof(double) * NFy)); CK(cudaMalloc(&Un, sz));
    CK(cudaMalloc(&W0b, sz));
    CK(cudaMemset(gb, 0, 8)); CK(cudaMemset(Ub, 0, sz)); CK(cudaMemset(W0b, 0, sz));
    CK(cudaEventRecord(e3));
    if (at[steps] >= 0) c2pb(Us + (size_t)steps * m, Ub, W, Pb + (size_t)at[steps] * m);
    for (int s = steps - 1; s >= 0; --s) {
      double *U = Us + (size_t)s * m, *U1 = U1s + (size_t)s * m;
      cudaMemsetAsync(U1b, 0, sz); cudaMemsetAsync(U0b, 0, sz); cudaMemsetAsync(Wb, 0, sz);
      cudaMemsetAsync(Fxb, 0, sizeof(double) * NFx); cudaMemsetAsync(Fyb, 0, sizeof(double) * NFy);
      c2p(U1, W);
      rk2d_b<<<gN, B>>>(U, U0b, U1, U1b, Fx, Fxb, Fy, Fyb, Un, Ub, 0.5, 0.5, dtdx, dty, nx, ny);
      hll2_b<<<gFx, B>>>(W, Wb, Fx, Fxb, g, gb, nx, ny, 0);
      hll2_b<<<gFy, B>>>(W, Wb, Fy, Fyb, g, gb, nx, ny, 1);
      c2pb(U1, U1b, W, Wb);
      cudaMemsetAsync(Ub, 0, sz); cudaMemsetAsync(Wb, 0, sz);
      cudaMemsetAsync(Fxb, 0, sizeof(double) * NFx); cudaMemsetAsync(Fyb, 0, sizeof(double) * NFy);
      c2p(U, W);
      rk2d_b<<<gN, B>>>(U, Ub, U, Ub, Fx, Fxb, Fy, Fyb, Un, U1b, 0.0, 1.0, dtdx, dty, nx, ny);
      hll2_b<<<gFx, B>>>(W, Wb, Fx, Fxb, g, gb, nx, ny, 0);
      hll2_b<<<gFy, B>>>(W, Wb, Fy, Fyb, g, gb, nx, ny, 1);
      c2pb(U, Ub, W, Wb);
      add2<<<gm, B>>>(Ub, U0b, m);
      if (at[s] >= 0) c2pb(U, Ub, W, Pb + (size_t)at[s] * m);
    }
    CK(cudaGetLastError());
    p2c2_b<<<gN, B>>>(W0, W0b, Us, Ub, g, gb, N);
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
  for (double *p : {g, gb, W0, W, Fx, Fy, Us, U1s, obs, w, Pb, Jc, Ub, U1b, U0b, Wb, Fxb, Fyb, Un, W0b})
    if (p) cudaFree(p);
  for (cudaEvent_t e : {e0, e1, e2, e3})
    if (e) cudaEventDestroy(e);
  return int(err);
}
