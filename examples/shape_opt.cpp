// Shape optimization driven by a cuad-generated adjoint.
//
// Mesh: a column of N hexahedral cells (1 x 1 x N) between z-layers ζ_0..ζ_N.
// Interior layers ζ_1..ζ_{N-1} are the DESIGN VARIABLES (ends fixed at 0 and L).
// Objective:  J(ζ) = ½ Σ_c ( V_c(ζ) - V*_c )²   — match a target volume dist.
//
// Cell volumes V_c and their sensitivities come from the REAL CuDAFLOW volume
// kernel differentiated by cuad:
//   V_c        = volKernel(x,y,z)
//   dJ/dz_node = volKernel_b(x, x_b, y, y_b, z, z_b, vol, vol_b),  seeded with
//                vol_b_c = ∂J/∂V_c = (V_c - V*_c)
//   dJ/dζ_k    = Σ_{4 nodes on layer k} z_b[node]
// Then plain gradient descent on the interior layers. The adjoint gradient is
// first checked against a finite-difference gradient, then the loop runs.
#include <cmath>
#include <cstdio>
#include <vector>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "cuad_rt.h"
#include "vol.cu"
#include "vol_b.cuh"

static const int N = 6;      // cells
static const float L = 1.0f; // column height

static Blk makeB() {
  Blk B{};
  B.ni = B.nj = 1; B.nk = N;
  B.niN = B.njN = 2; B.nkN = N + 1;
  B.nit = B.ni + 2*NG; B.njt = B.nj + 2*NG; B.nkt = B.nk + 2*NG;
  B.coff = 0; B.noff = 0;
  return B;
}
static Blk B = makeB();
static const int NNODES = 2 * 2 * (N + 1);

// build node coordinate arrays from the layer positions ζ (size N+1)
static void buildNodes(const std::vector<float>& zeta, std::vector<float>& X,
                       std::vector<float>& Y, std::vector<float>& Z) {
  X.assign(NNODES, 0); Y.assign(NNODES, 0); Z.assign(NNODES, 0);
  for (int k = 0; k <= N; k++)
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        long n = nidx(B, i, j, k);
        X[n] = (float)i; Y[n] = (float)j; Z[n] = zeta[k];
      }
}
static std::vector<float> cellVolumes(const std::vector<float>& X,
                                      const std::vector<float>& Y,
                                      const std::vector<float>& Z) {
  std::vector<float> vol((size_t)B.nit * B.njt * B.nkt, 0.f);
  volKernel(B, X.data(), Y.data(), Z.data(), vol.data());
  std::vector<float> V(N);
  for (int k = 0; k < N; k++) V[k] = vol[B.coff + cidx(B, 0, 0, k)];
  return V;
}

// objective and its adjoint gradient dJ/dζ_k (all layers; ends masked later)
static float objAndGrad(const std::vector<float>& zeta,
                        const std::vector<float>& Vstar,
                        std::vector<float>& gZeta) {
  std::vector<float> X, Y, Z; buildNodes(zeta, X, Y, Z);
  std::vector<float> V = cellVolumes(X, Y, Z);
  float J = 0.f;
  std::vector<float> vol((size_t)B.nit * B.njt * B.nkt, 0.f);
  std::vector<float> volb(vol.size(), 0.f);
  for (int k = 0; k < N; k++) {
    float r = V[k] - Vstar[k];
    J += 0.5f * r * r;
    volb[B.coff + cidx(B, 0, 0, k)] = r;   // ∂J/∂V_k
  }
  // reverse: node-coordinate adjoints
  std::vector<float> Xb(NNODES, 0), Yb(NNODES, 0), Zb(NNODES, 0);
  volKernel_b(B, X.data(), Xb.data(), Y.data(), Yb.data(), Z.data(), Zb.data(),
              vol.data(), volb.data());
  // reduce node adjoints onto layer design variables
  gZeta.assign(N + 1, 0.f);
  for (int k = 0; k <= N; k++)
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++)
        gZeta[k] += Zb[nidx(B, i, j, k)];
  return J;
}

int main() {
  // target: geometric stretching of the column (non-uniform cell heights)
  std::vector<float> Vstar(N);
  { float s = 0, r = 1.35f, h = 1;
    std::vector<float> hh(N);
    for (int k = 0; k < N; k++) { hh[k] = h; s += h; h *= r; }
    for (int k = 0; k < N; k++) Vstar[k] = L * hh[k] / s; }

  // initial guess: uniform layers
  std::vector<float> zeta(N + 1);
  for (int k = 0; k <= N; k++) zeta[k] = L * k / N;

  // --- gradient check: adjoint vs finite difference ---
  {
    std::vector<float> g; float J0 = objAndGrad(zeta, Vstar, g);
    const float eps = 1e-4f; int worst = 0; float maxerr = 0;
    for (int k = 1; k < N; k++) { // interior only
      std::vector<float> zp = zeta, zm = zeta; zp[k] += eps; zm[k] -= eps;
      std::vector<float> gd;
      float Jp = objAndGrad(zp, Vstar, gd), Jm = objAndGrad(zm, Vstar, gd);
      float fd = (Jp - Jm) / (2 * eps);
      float err = std::fabs(g[k] - fd) / (1.f + std::fabs(fd));
      if (err > maxerr) { maxerr = err; worst = k; }
    }
    std::printf("gradient check (adjoint vs FD): max relerr = %.2e at layer %d  [%s]\n\n",
                maxerr, worst, maxerr < 1e-3f ? "PASS" : "FAIL");
    (void)J0;
  }

  // --- gradient descent on interior layers ---
  std::printf("optimizing column mesh to match a stretched target volume dist.\n");
  float lr = 0.3f; // < 2/lambda_max of the tridiagonal Hessian (~4) for stability
  std::vector<float> g;
  float J = objAndGrad(zeta, Vstar, g);
  std::printf("  iter %4d   J = %.3e\n", 0, J);
  for (int it = 1; it <= 600; it++) {
    for (int k = 1; k < N; k++) zeta[k] -= lr * g[k]; // ends fixed
    J = objAndGrad(zeta, Vstar, g);
    if (it % 100 == 0 || it == 1)
      std::printf("  iter %4d   J = %.3e\n", it, J);
  }

  std::vector<float> X, Y, Z; buildNodes(zeta, X, Y, Z);
  std::vector<float> V = cellVolumes(X, Y, Z);
  std::printf("\n  cell:      ");
  for (int k = 0; k < N; k++) std::printf(" %6.4f", V[k]);
  std::printf("\n  target:    ");
  for (int k = 0; k < N; k++) std::printf(" %6.4f", Vstar[k]);
  double maxdev = 0;
  for (int k = 0; k < N; k++) maxdev = std::fmax(maxdev, std::fabs(V[k]-Vstar[k]));
  std::printf("\n  final J = %.3e,  max|V-V*| = %.2e  [%s]\n",
              J, maxdev, (J < 1e-8 && maxdev < 1e-4) ? "CONVERGED" : "check");
  return 0;
}
