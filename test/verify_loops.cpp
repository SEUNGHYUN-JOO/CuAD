// Host-side check of test/cases/loops.cu: adjoint against central finite differences.
#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
#define CUAD_TAPE_SIZE 4096
#include "cuad_rt.h"
#include "cases/loops.cu"
#include "out/loops_b.cuh"

typedef void (*Fn)(const double *, double *, int);
typedef void (*Adj)(const double *, double *, double *, double *, int);

static int check(const char *name, Fn f, Adj fb) {
  const int n = 7;
  double x[n], xb[n] = {0}, y = 0, yb = 1.0, worst = 0;
  for (int i = 0; i < n; i++) x[i] = 0.2 + 0.1 * i;
  fb(x, xb, &y, &yb, n);
  for (int i = 0; i < n; i++) {
    double h = 1e-6, xp[n], xm[n], yp = 0, ym = 0;
    for (int k = 0; k < n; k++) { xp[k] = x[k]; xm[k] = x[k]; }
    xp[i] += h; xm[i] -= h; f(xp, &yp, n); f(xm, &ym, n);
    double fd = (yp - ym) / (2 * h);
    worst = fmax(worst, fabs(xb[i] - fd) / (1 + fabs(fd)));
  }
  bool ok = worst < 1e-7;
  std::printf("  %-8s adjoint-vs-FD %.2e  [%s]\n", name, worst, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int fails = check("prodloop", prodloop, prodloop_b) + check("sinloop", sinloop, sinloop_b) +
              check("mixloop", mixloop, mixloop_b);
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
