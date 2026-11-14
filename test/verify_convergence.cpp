// Host check of convergence loops (test/cases/convergence.cu, test/reverse_only/bracketing.cu):
// the implicit adjoint and the implicit tangent (forward mode) against central finite differences, for the
// smooth and the bracketing loops. Each kernel maps x[0..n) to y[0].
#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
#ifndef CUAD_TAPE_SIZE
#define CUAD_TAPE_SIZE 256
#endif
#include "cuad_rt.h"
#include "cases/convergence.cu"
#include "out/convergence_b.cuh"
#include "out/convergence_d.cuh"
#include "reverse_only/bracketing.cu"
#include "out/bracketing_b.cuh"
#include "out/bracketing_d.cuh"

typedef void (*Fn)(const double *, double *, int);
typedef void (*Adj)(const double *, double *, double *, double *, int);
typedef void (*Tan)(const double *, const double *, double *, double *, int);

static int check(const char *name, Fn f, Adj fb, Tan fd_) {
  const int n = 3;
  double x[n] = {2.0, 5.0, 0.7}, xb[n] = {0}, y = 0, yb = 1.0, worst = 0, worstT = 0;
  fb(x, xb, &y, &yb, n);
  for (int i = 0; i < n; i++) {
    double h = 1e-6, xp[n], xm[n], yp = 0, ym = 0;
    for (int k = 0; k < n; k++) { xp[k] = x[k]; xm[k] = x[k]; }
    xp[i] += h; xm[i] -= h; f(xp, &yp, n); f(xm, &ym, n);
    double fd = (yp - ym) / (2 * h);
    worst = fmax(worst, fabs(xb[i] - fd) / (1 + fabs(fd)));
    if (fd_) {
      double xd[n] = {0}, yt = 0, ytd = 0;
      xd[i] = 1;
      fd_(x, xd, &yt, &ytd, n);
      worstT = fmax(worstT, fabs(ytd - fd) / (1 + fabs(fd)));
    }
  }
  bool ok = worst < 1e-7 && worstT < 1e-7;
  if (fd_) std::printf("  %-22s adjoint-vs-FD %.2e  tangent-vs-FD %.2e  [%s]\n", name, worst, worstT, ok ? "PASS" : "FAIL");
  else std::printf("  %-22s adjoint-vs-FD %.2e  (reverse only)  [%s]\n", name, worst, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int fails = 0;
  fails += check("newton_root", newton_root, newton_root_b, newton_root_d);
  fails += check("fixed_point", fixed_point, fixed_point_b, fixed_point_d);
  fails += check("newton_system", newton_system, newton_system_b, newton_system_d);
  fails += check("solve_in_loop", solve_in_loop, solve_in_loop_b, solve_in_loop_d);
  fails += check("bisect_root", bisect_root, bisect_root_b, bisect_root_d);
  fails += check("bisect_moving_bracket", bisect_moving_bracket, bisect_moving_bracket_b, bisect_moving_bracket_d);
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
