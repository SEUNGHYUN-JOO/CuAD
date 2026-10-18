// Verify cuad on the FIML-augmented SA source, then run a field inversion:
// recover the FIML field beta(x) from target source data using the adjoint
// dsrc/dbeta that cuad generated (sa_source_b).
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
#include "sa_fiml.cu"
#include "sa_fiml_d.cuh"
#include "sa_fiml_b.cuh"

static int check(const char *n, float a, float fd) {
  float e = std::fabs(a - fd) / (1.f + std::fabs(fd));
  bool ok = e < 3e-3f;
  std::printf("  %-18s d=% .6e  fd=% .6e  relerr=%.1e  [%s]\n", n, a, fd, e,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int fails = 0;
  const float nu = 1e-3f, eps = 1e-4f;
  float nut = 0.05f, d = 0.10f, Om = 100.f, beta = 1.2f;

  // ---- forward tangent vs FD, per input ----
  auto S = [&](float a, float b, float c, float e, float f) {
    return sa_source(a, b, c, e, f);
  };
  {
    float dum;
    float dnut; sa_source_d(nut,1,d,0,Om,0,nu,0,beta,0,dnut);
    fails += check("dsrc/dnut", dnut, (S(nut+eps,d,Om,nu,beta)-S(nut-eps,d,Om,nu,beta))/(2*eps));
    float dbeta; sa_source_d(nut,0,d,0,Om,0,nu,0,beta,1,dbeta);
    fails += check("dsrc/dbeta", dbeta, (S(nut,d,Om,nu,beta+eps)-S(nut,d,Om,nu,beta-eps))/(2*eps));
    float dd; sa_source_d(nut,0,d,1,Om,0,nu,0,beta,0,dd);
    fails += check("dsrc/dd", dd, (S(nut,d+eps,Om,nu,beta)-S(nut,d-eps,Om,nu,beta))/(2*eps));
    (void)dum;
  }
  // ---- reverse dot-test:  <s_b, J v> == <v, J^T s_b> ----
  {
    float vnut=0.7f,vd=-0.3f,vOm=0.5f,vbeta=0.9f, sb=1.3f;
    float jd; sa_source_d(nut,vnut,d,vd,Om,vOm,nu,0,beta,vbeta,jd);
    double lhs = (double)sb*jd;
    float nb=0,db=0,ob=0,nub=0,bb=0;
    sa_source_b(nut,nb,d,db,Om,ob,nu,nub,beta,bb,sb);
    double rhs = (double)nb*vnut + (double)db*vd + (double)ob*vOm + (double)bb*vbeta;
    double e = std::fabs(lhs-rhs)/(1.0+std::fabs(lhs));
    std::printf("  SA dot-test        <sb,Jv>=% .8e  <J^Tsb,v>=% .8e  relerr=%.1e  [%s]\n",
                lhs, rhs, e, e<1e-4?"PASS":"FAIL");
    if (e>=1e-4) fails++;
  }

  std::printf("\n=== FIML field inversion (recover Beta(x) from source data) ===\n");
  const int N = 24;
  std::vector<float> NUT(N), D(N), OM(N), betaStar(N), srcStar(N);
  for (int c = 0; c < N; c++) {
    float s = (float)c / (N - 1);
    NUT[c] = 0.02f + 0.06f * s;           // varying eddy viscosity
    D[c]   = 0.05f + 0.20f * s;           // varying wall distance
    OM[c]  = 50.f + 150.f * (1 - s);      // varying vorticity
    betaStar[c] = 1.0f + 0.6f * std::sin(6.0f * s); // "true" FIML field
    srcStar[c]  = sa_source(NUT[c], D[c], OM[c], nu, betaStar[c]); // observations
  }
  std::vector<float> Beta(N, 1.0f); // initial guess: no augmentation
  auto objgrad = [&](std::vector<float>& g) {
    float J = 0; g.assign(N, 0.f);
    for (int c = 0; c < N; c++) {
      float sc = sa_source(NUT[c], D[c], OM[c], nu, Beta[c]);
      float rC = sc - srcStar[c];
      J += 0.5f * rC * rC;
      float nb=0,db=0,ob=0,nub=0,bb=0;
      sa_source_b(NUT[c], nb, D[c], db, OM[c], ob, nu, nub, Beta[c], bb, rC); // seed=dJ/dsrc
      g[c] = bb; // dJ/dbeta_c
    }
    return J;
  };
  std::vector<float> g;
  float J = objgrad(g);
  std::printf("  iter %3d   J = %.4e\n", 0, J);
  float lr = 1.2f;
  for (int it = 1; it <= 120; it++) {
    for (int c = 0; c < N; c++) Beta[c] -= lr * g[c];
    J = objgrad(g);
    if (it % 30 == 0) std::printf("  iter %3d   J = %.4e\n", it, J);
  }
  float maxerr = 0;
  for (int c = 0; c < N; c++) maxerr = std::fmax(maxerr, std::fabs(Beta[c]-betaStar[c]));
  std::printf("  recovered Beta vs true: max|Beta-Beta*| = %.2e  [%s]\n",
              maxerr, maxerr < 1e-3f ? "RECOVERED" : "check");

  std::printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fails);
  return fails ? 1 : 0;
}
