// NACA 0012 inverse shape design driven by a cuad-generated adjoint.
//
// A chordwise stack of thin hex cells wraps the (symmetric) airfoil: cell c
// spans stations x_c..x_{c+1}, from the lower surface (-θ) to the upper (+θ),
// unit depth in z, so its volume is the local cross-sectional area slab.
// DESIGN VARIABLES: the half-thickness θ_i at each station.
// Objective:  J(θ) = ½ Σ_c (V_c(θ) - V*_c)²   — match a target area dist.
// Cell volumes and the shape gradient dJ/dθ come from the real CuDAFLOW volume
// kernel differentiated by cuad (volKernel / volKernel_b).
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

static const int M = 16;         // chordwise cells (M+1 stations)
static Blk B;
static std::vector<float> Xst;   // station x (fixed), size M+1

static void initB() {
  B = Blk{};
  B.ni = M; B.nj = 1; B.nk = 1;
  B.niN = M + 1; B.njN = 2; B.nkN = 2;
  B.nit = B.ni + 2*NG; B.njt = B.nj + 2*NG; B.nkt = B.nk + 2*NG;
  B.coff = 0; B.noff = 0;
}
static int NNODES() { return (M + 1) * 2 * 2; }

// NACA 4-digit half-thickness at chord fraction x for max-thickness t
static float yt(float x, float t) {
  return 5*t*(0.2969f*std::sqrt(x) - 0.1260f*x - 0.3516f*x*x
              + 0.2843f*x*x*x - 0.1015f*x*x*x*x);
}

static void buildNodes(const std::vector<float>& th, std::vector<float>& X,
                       std::vector<float>& Y, std::vector<float>& Z) {
  X.assign(NNODES(),0); Y.assign(NNODES(),0); Z.assign(NNODES(),0);
  for (int i = 0; i <= M; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        long n = nidx(B, i, j, k);
        X[n] = Xst[i];
        Y[n] = (j ? th[i] : -th[i]); // upper / lower surface
        Z[n] = (float)k;
      }
}
static std::vector<float> volumes(const std::vector<float>& X,
                                  const std::vector<float>& Y,
                                  const std::vector<float>& Z) {
  std::vector<float> vol((size_t)B.nit*B.njt*B.nkt, 0.f);
  volKernel(B, X.data(), Y.data(), Z.data(), vol.data());
  std::vector<float> V(M);
  for (int c = 0; c < M; c++) V[c] = vol[B.coff + cidx(B, c, 0, 0)];
  return V;
}
// objective + adjoint gradient dJ/dθ_i (whole shape gradient from cuad)
static float objGrad(const std::vector<float>& th, const std::vector<float>& Vst,
                     std::vector<float>& g) {
  std::vector<float> X,Y,Z; buildNodes(th,X,Y,Z);
  std::vector<float> V = volumes(X,Y,Z);
  std::vector<float> vol((size_t)B.nit*B.njt*B.nkt,0.f), volb(vol.size(),0.f);
  float J = 0;
  for (int c = 0; c < M; c++) {
    float r = V[c] - Vst[c]; J += 0.5f*r*r;
    volb[B.coff + cidx(B,c,0,0)] = r;           // ∂J/∂V_c
  }
  std::vector<float> Xb(NNODES(),0),Yb(NNODES(),0),Zb(NNODES(),0);
  volKernel_b(B, X.data(),Xb.data(), Y.data(),Yb.data(), Z.data(),Zb.data(),
              vol.data(), volb.data());
  // reduce node adjoints onto θ: y_upper = +θ (+1), y_lower = -θ (-1)
  g.assign(M+1, 0.f);
  for (int i = 0; i <= M; i++)
    for (int k = 0; k < 2; k++) {
      g[i] += Yb[nidx(B,i,1,k)];   // upper
      g[i] -= Yb[nidx(B,i,0,k)];   // lower
    }
  return J;
}

int main() {
  initB();
  Xst.resize(M+1);
  for (int i = 0; i <= M; i++) { // cosine spacing, LE=0 TE=1
    float a = (float)i/M; Xst[i] = 0.5f*(1 - std::cos(3.14159265f*a));
  }
  // start: NACA 0012 ; target: a thicker/loaded shape (t=0.15 * chordwise load)
  std::vector<float> th(M+1), thStar(M+1);
  for (int i = 0; i <= M; i++) {
    th[i]     = yt(Xst[i], 0.12f);
    thStar[i] = yt(Xst[i], 0.15f) * (1.0f + 0.15f*std::sin(3.14159f*Xst[i]));
  }
  { std::vector<float> X,Y,Z; buildNodes(thStar,X,Y,Z);
    static std::vector<float> Vstar; Vstar = volumes(X,Y,Z);

    // --- gradient check: adjoint vs FD ---
    std::vector<float> g; float J0 = objGrad(th, Vstar, g);
    const float eps = 1e-4f; float maxerr=0; int worst=0;
    for (int i = 1; i < M; i++) {
      std::vector<float> tp=th, tm=th, gd; tp[i]+=eps; tm[i]-=eps;
      float Jp=objGrad(tp,Vstar,gd), Jm=objGrad(tm,Vstar,gd);
      float fd=(Jp-Jm)/(2*eps), e=std::fabs(g[i]-fd)/(1e-6f+std::fabs(fd));
      if (e>maxerr){maxerr=e;worst=i;}
    }
    std::printf("gradient check (adjoint vs FD): max relerr = %.2e at station %d  [%s]\n\n",
                maxerr, worst, maxerr<2e-3f?"PASS":"FAIL");
    (void)J0;

    // --- optimize the airfoil thickness distribution ---
    std::printf("NACA 0012 -> target: inverse shape design (adjoint gradient)\n");
    float lr = 20.0f; std::vector<float> g2;
    float J = objGrad(th, Vstar, g2);
    std::printf("  iter %4d   J = %.4e\n", 0, J);
    for (int it = 1; it <= 4000; it++) {
      for (int i = 1; i <= M; i++) th[i] -= lr*g2[i]; // LE (i=0) fixed at 0
      J = objGrad(th, Vstar, g2);
      if (it%800==0) std::printf("  iter %4d   J = %.4e\n", it, J);
    }
    float amax=0; for (int i=0;i<=M;i++) amax=std::fmax(amax,std::fabs(th[i]-thStar[i]));
    std::printf("\n  station x:  "); for(int i=0;i<=M;i+=2) std::printf(" %5.3f",Xst[i]);
    std::printf("\n  final th:   "); for(int i=0;i<=M;i+=2) std::printf(" %5.3f",th[i]);
    std::printf("\n  target th:  "); for(int i=0;i<=M;i+=2) std::printf(" %5.3f",thStar[i]);
    std::printf("\n  final J = %.3e,  max|th-th*| = %.2e  [%s]\n",
                J, amax, (J<1e-7 && amax<2e-3f)?"CONVERGED":"check");
  }
  return 0;
}
