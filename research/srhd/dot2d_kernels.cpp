// Kernel-by-kernel dot tests of the 2-D SRHD adjoint (srhd2d_b.cuh) against central differences:
// <Fb, F(x + h d) - F(x - h d)> / 2h  versus  <x_b, d>  with x_b = adjoint of Fb.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#define CUAD_TAPE_SIZE 256
#include "cuad_rt.h"
#include "srhd2d.cu"
#include "srhd2d_b.cuh"
#include "srhd2d_d.cuh"

typedef std::vector<double> V;
static std::mt19937_64 rng(1);
static double un(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
static double dot(const V &a, const V &b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }

int main() {
  const int nx = 9, ny = 7, N = nx * ny;
  V P(4 * N), dP(4 * N);
  // smooth fields (minmod mostly returns a slope) unless SRHD_RANDOM_STATE is set
  for (int c = 0; c < N; ++c) {
#ifdef SRHD_RANDOM_STATE
    P[c] = un(0.5, 2); P[N + c] = un(-1, 1); P[2 * N + c] = un(-1, 1); P[3 * N + c] = un(0.1, 3);
#else
    double x = (c % nx + 0.5) / nx, y = (c / nx + 0.5) / ny;
    P[c] = 1 + 0.3 * std::sin(4 * x + 1) * std::cos(3 * y); P[N + c] = 0.4 * std::sin(3 * y + 2.1 * x + 0.5);
    P[2 * N + c] = -0.3 * std::cos(2 * x - 1.3 * y + 0.2); P[3 * N + c] = 1 + 0.5 * std::exp(-((x - .4) * (x - .4) + (y - .5) * (y - .5)) / 0.05);
#endif
  }
  for (int k = 0; k < 4 * N; ++k) dP[k] = un(-1, 1) * 1e-2 * std::fabs(P[k]);
  double g[1] = {5.0 / 3}, dg = 0.013;
  const double h = 1e-6;
  // hll2, both directions
  for (int dir = 0; dir < 2; ++dir) {
    int NF = 4 * (dir == 0 ? (nx + 1) * ny : nx * (ny + 1));
    V F(NF), Fb(NF), Fp(NF), Fm(NF);
    for (auto &v : Fb) v = un(-1, 1);
    V Pp(P), Pm(P);
    for (int k = 0; k < 4 * N; ++k) { Pp[k] += h * dP[k]; Pm[k] -= h * dP[k]; }
    double gp[1] = {g[0] + h * dg}, gm[1] = {g[0] - h * dg};
    hll2(Pp.data(), Fp.data(), gp, nx, ny, dir);
    hll2(Pm.data(), Fm.data(), gm, nx, ny, dir);
    double fd = 0;
    for (int k = 0; k < NF; ++k) fd += Fb[k] * (Fp[k] - Fm[k]) / (2 * h);
    V Pb(4 * N, 0.0), Fb2(Fb);
    double gb[1] = {0};
    hll2_b(P.data(), Pb.data(), F.data(), Fb2.data(), g, gb, nx, ny, dir);
    double ad = dot(Pb, dP) + gb[0] * dg;
    V Fd(NF), F3(NF);
    double gd[1] = {dg};
    hll2_d(P.data(), dP.data(), F3.data(), Fd.data(), g, gd, nx, ny, dir);
    double tan = dot(Fb, Fd);
    // per-plane breakdown of the tangent error
    double worst = 0; int wk = -1;
    for (int k = 0; k < NF; ++k) {
      double e = std::fabs(Fd[k] - (Fp[k] - Fm[k]) / (2 * h));
      if (e > worst) { worst = e; wk = k; }
    }
    int nf = NF / 4;
    std::printf("hll2 dir %d: adjoint %.12e  tangent %.12e  fd %.12e  rel(adj) %.2e  worst face %d plane %d err %.3e\n",
                dir, ad, tan, fd, std::fabs(ad - fd) / std::fabs(fd), wk % nf, wk / nf, worst);
  }
  // c2p2 (both solvers) and p2c2
  V U(4 * N);
  p2c2(P.data(), U.data(), g, N);
  V dU(4 * N);
  for (int k = 0; k < 4 * N; ++k) dU[k] = un(-1, 1) * 1e-2 * std::fabs(U[k]);
  for (int s = 0; s < 2; ++s) {
    V Q(4 * N), Qp(4 * N), Qm(4 * N), Qb(4 * N), Up(U), Um(U);
    for (auto &v : Qb) v = un(-1, 1);
    for (int k = 0; k < 4 * N; ++k) { Up[k] += h * dU[k]; Um[k] -= h * dU[k]; }
    double gp[1] = {g[0] + h * dg}, gm[1] = {g[0] - h * dg};
    if (s == 0) { c2p2_newton(Up.data(), Qp.data(), gp, N); c2p2_newton(Um.data(), Qm.data(), gm, N); }
    else { c2p2_bisect(Up.data(), Qp.data(), gp, N); c2p2_bisect(Um.data(), Qm.data(), gm, N); }
    double fd = 0;
    for (int k = 0; k < 4 * N; ++k) fd += Qb[k] * (Qp[k] - Qm[k]) / (2 * h);
    V Ub(4 * N, 0.0), Qb2(Qb);
    double gb[1] = {0};
    if (s == 0) c2p2_newton_b(U.data(), Ub.data(), Q.data(), Qb2.data(), g, gb, N);
    else c2p2_bisect_b(U.data(), Ub.data(), Q.data(), Qb2.data(), g, gb, N);
    double ad = dot(Ub, dU) + gb[0] * dg;
    std::printf("c2p2 %s: rel %.2e\n", s ? "bisect" : "newton", std::fabs(ad - fd) / std::fabs(fd));
  }
  {
    V Up(4 * N), Um(4 * N), Ub(4 * N), Pb(4 * N, 0.0), Pp(P), Pm(P);
    for (auto &v : Ub) v = un(-1, 1);
    for (int k = 0; k < 4 * N; ++k) { Pp[k] += h * dP[k]; Pm[k] -= h * dP[k]; }
    double gp[1] = {g[0] + h * dg}, gm[1] = {g[0] - h * dg};
    p2c2(Pp.data(), Up.data(), gp, N); p2c2(Pm.data(), Um.data(), gm, N);
    double fd = 0;
    for (int k = 0; k < 4 * N; ++k) fd += Ub[k] * (Up[k] - Um[k]) / (2 * h);
    V Ub2(Ub), U0(4 * N);
    double gb[1] = {0};
    p2c2_b(P.data(), Pb.data(), U0.data(), Ub2.data(), g, gb, N);
    std::printf("p2c2: rel %.2e\n", std::fabs(dot(Pb, dP) + gb[0] * dg - fd) / std::fabs(fd));
  }
  {
    int NFx = 4 * (nx + 1) * ny, NFy = 4 * nx * (ny + 1);
    V Fx(NFx), Fy(NFy), dFx(NFx), dFy(NFy), U0(4 * N), dU0(4 * N);
    for (auto &v : Fx) v = un(-1, 1); for (auto &v : Fy) v = un(-1, 1); for (auto &v : U0) v = un(-1, 1);
    for (auto &v : dFx) v = un(-1, 1); for (auto &v : dFy) v = un(-1, 1); for (auto &v : dU0) v = un(-1, 1);
    V Un(4 * N), Unb(4 * N), U0b(4 * N, 0.0), Ub(4 * N, 0.0), Fxb(NFx, 0.0), Fyb(NFy, 0.0);
    for (auto &v : Unb) v = un(-1, 1);
    // linear: J(dir) = rk2d(dU0, dU, dFx, dFy) with a = 0.5, b = 0.5
    V Ud(4 * N); for (auto &v : Ud) v = un(-1, 1);
    rk2d(dU0.data(), Ud.data(), dFx.data(), dFy.data(), Un.data(), 0.5, 0.5, 0.3, 0.4, nx, ny);
    double lhs = dot(Unb, Un);
    V Unb2(Unb);
    rk2d_b(U0.data(), U0b.data(), U0.data(), Ub.data(), Fx.data(), Fxb.data(), Fy.data(), Fyb.data(), Un.data(),
           Unb2.data(), 0.5, 0.5, 0.3, 0.4, nx, ny);
    double rhs = dot(U0b, dU0) + dot(Ub, Ud) + dot(Fxb, dFx) + dot(Fyb, dFy);
    std::printf("rk2d (linear): rel %.2e\n", std::fabs(lhs - rhs) / std::fabs(lhs));
  }
  return 0;
}
