// Verify cuad on the real CuDAFLOW volume kernel (volKernel -> faceI/J/K -> faceVec).
// One hexahedral cell (8 nodes); node coords are the differentiation variables.
//   forward: tangent vs finite difference
//   reverse: dot-product identity  <vol_b, J x_d> == <x_b, x_d>   (over all coords)
#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "cuad_rt.h"

#include "vol_reuse.cu"
#include "vol_reuse_d.cuh"
#include "vol_reuse_b.cuh"

static Blk makeB() {
  Blk B{};
  B.ni = B.nj = B.nk = 1;
  B.niN = B.njN = B.nkN = 2;         // 2 nodes per direction -> 8 nodes
  B.nit = B.ni + 2 * NG; B.njt = B.nj + 2 * NG; B.nkt = B.nk + 2 * NG;
  B.coff = 0; B.noff = 0;
  return B;
}

int main() {
  Blk B = makeB();
  const int NN = 8; // nodes
  // unit cube, mildly perturbed so all derivatives are exercised
  float X[NN] = {0.02f, 1.01f, 0.03f, 0.98f, 0.01f, 1.03f, -0.02f, 1.00f};
  float Y[NN] = {0.01f, -0.02f, 1.02f, 0.99f, 0.03f, 0.00f, 0.97f, 1.01f};
  float Z[NN] = {0.00f, 0.01f, -0.01f, 0.02f, 1.02f, 0.98f, 1.01f, 0.99f};
  // node ordering must match nidx(i,j,k) = (k*njN+j)*niN+i with niN=njN=2.

  long vidx = B.coff + cidx(B, 0, 0, 0);
  int volLen = B.nit * B.njt * B.nkt;

  auto runVol = [&](float *x, float *y, float *z) {
    static float vol[512];
    for (int t = 0; t < volLen; t++) vol[t] = 0;
    volKernel(B, x, y, z, vol);
    return vol[vidx];
  };

  int fails = 0;
  const float eps = 2e-4f;

  // arbitrary tangent direction over all 24 coordinates
  float Xd[NN]={.7f,-.3f,.5f,.2f,-.6f,.4f,.1f,-.8f};
  float Yd[NN]={.2f,.9f,-.4f,.6f,-.1f,.3f,-.7f,.5f};
  float Zd[NN]={-.5f,.4f,.8f,-.2f,.6f,-.3f,.1f,.9f};

  // ---- forward: tangent vs FD ----
  static float vol[512], vold[512];
  for (int t = 0; t < volLen; t++) { vol[t]=0; vold[t]=0; }
  volKernel_d(B, X, Xd, Y, Yd, Z, Zd, vol, vold);
  float voldT = vold[vidx];
  float Xp[NN],Xm[NN],Yp[NN],Ym[NN],Zp[NN],Zm[NN];
  for (int i=0;i<NN;i++){Xp[i]=X[i]+eps*Xd[i];Xm[i]=X[i]-eps*Xd[i];
                         Yp[i]=Y[i]+eps*Yd[i];Ym[i]=Y[i]-eps*Yd[i];
                         Zp[i]=Z[i]+eps*Zd[i];Zm[i]=Z[i]-eps*Zd[i];}
  float fd = (runVol(Xp,Yp,Zp) - runVol(Xm,Ym,Zm)) / (2*eps);
  float ferr = std::fabs(voldT - fd) / (1.f + std::fabs(fd));
  bool fok = ferr < 2e-3f;
  std::printf("  vol tangent   d=% .6e  fd=% .6e  relerr=%.1e  [%s]\n",
              voldT, fd, ferr, fok?"PASS":"FAIL");
  if (!fok) fails++;

  // ---- reverse: dot-product identity  <vol_b, J x_d> == <x_b, x_d> ----
  static float volb[512];
  for (int t=0;t<volLen;t++) volb[t]=0;
  volb[vidx] = 1.3f; // arbitrary output-adjoint seed
  double lhs = (double)volb[vidx] * voldT / 1.0; // <vol_b, J x_d> (single output)
  lhs = (double)volb[vidx] * voldT;
  float Xb[NN]={0},Yb[NN]={0},Zb[NN]={0};
  volKernel_b(B, X, Xb, Y, Yb, Z, Zb, vol, volb);
  double rhs = 0;
  for (int i=0;i<NN;i++) rhs += (double)Xb[i]*Xd[i] + (double)Yb[i]*Yd[i] + (double)Zb[i]*Zd[i];
  double derr = std::fabs(lhs - rhs) / (1.0 + std::fabs(lhs));
  bool rok = derr < 1e-4;
  std::printf("  vol adjoint dot-test   <vb,Jxd>=% .8e  <xb,xd>=% .8e  relerr=%.1e  [%s]\n",
              lhs, rhs, derr, rok?"PASS":"FAIL");
  if (!rok) fails++;

  // sanity: near-unit-cube volume
  std::printf("  (cell volume = %.4f)\n", vol[vidx]);

  std::printf(fails?"\nRESULT: %d FAILED\n":"\nRESULT: all passed\n", fails);
  return fails?1:0;
}
