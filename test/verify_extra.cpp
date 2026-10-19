// Host-side check of test/cases/extra.cu: tangent against central finite differences and
// the adjoint against the tangent (dot-product test), at several points.
#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;

#include "cuad_rt.h"
#include "cases/extra.cu"
#include "out/extra_d.cuh"
#include "out/extra_b.cuh"

typedef void (*Fn)(const double *, double *, int);
typedef void (*Tan)(const double *, const double *, double *, double *, int);
typedef void (*Adj)(const double *, double *, double *, double *, int);

static int check(const char *name, Fn f, Tan fd_, Adj fb, double a, double b) {
  double x[2] = {a, b}, xd[2] = {1.0, 0.5}, y = 0, yd = 0, h = 1e-6;
  fd_(x, xd, &y, &yd, 1);
  double xp[2] = {a + h, b + 0.5 * h}, xm[2] = {a - h, b - 0.5 * h}, yp = 0, ym = 0;
  f(xp, &yp, 1); f(xm, &ym, 1);
  double fd = (yp - ym) / (2 * h);
  double xb[2] = {0, 0}, yb = 1.0;
  fb(x, xb, &y, &yb, 1);
  double adj = xb[0] * xd[0] + xb[1] * xd[1];
  double e1 = fabs(yd - fd) / (1 + fabs(fd)), e2 = fabs(adj - yd) / (1 + fabs(yd));
  bool ok = e1 < 1e-6 && e2 < 1e-12;
  std::printf("  %-8s x=(%+.2f,%+.2f)  tangent-vs-FD %.2e  adjoint-vs-tangent %.2e  [%s]\n", name, a, b, e1, e2,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int fails = 0;
  double pts[][2] = {{0.37, 0.9}, {0.61, 0.1}, {1.13, 0.4}};
  for (auto &p : pts) fails += check("nested", nested, nested_d, nested_b, p[0], p[1]);
  double pm[][2] = {{0.37, 0.9}, {-0.61, 0.1}, {1.13, -0.4}};
  for (auto &p : pm) fails += check("mathfn", mathfn, mathfn_d, mathfn_b, p[0], p[1]);
  std::printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fails);
  return fails ? 1 : 0;
}
