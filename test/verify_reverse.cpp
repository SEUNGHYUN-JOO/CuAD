// Host-side finite-difference verification of cuad-generated ADJOINTS.
// Seeds the output adjoint (y_b = 1) and checks that the accumulated input
// adjoints equal the gradient of that output w.r.t. each input (central FD).
// A __global__ kernel's scalar parameters are uniform across the launch and are
// held passive (no adjoint argument), so only the array inputs are checked here.
// A __global__ kernel's scalar parameters are uniform across the launch and are
// held passive (no adjoint argument), so only the array inputs are checked here.
#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;

#include "cuad_rt.h"
#include "cases/scalar.cu"
#include "cases/saxpy.cu"
#include "cases/stencil.cu"
#include "cases/overwrite.cu"
#include "cases/device_fn.cu"
#include "cases/atomic.cu"
#include "out/scalar_b.cuh"
#include "out/saxpy_b.cuh"
#include "out/stencil_b.cuh"
#include "out/overwrite_b.cuh"
#include "out/device_fn_b.cuh"
#include "out/atomic_b.cuh"

static int check(const char *name, double adj, double fd) {
  double err = std::fabs(adj - fd) / (1.0 + std::fabs(fd));
  const char *ok = (err < 1e-6) ? "PASS" : "FAIL";
  std::printf("  %-22s adjoint=% .10e  fd=% .10e  relerr=%.2e  [%s]\n", name,
              adj, fd, err, ok);
  return err < 1e-6 ? 0 : 1;
}

int main() {
  int fails = 0;
  const double eps = 1e-6;

  // ---- scalar: y = sin(a*x)+x*x ----
  {
    double x = 0.37, a = 1.9, y = 0;
    double x_b = 0, y_b = 1;
    apply_f_b(&x, &x_b, &y, &y_b, a, 1);
    double xp = x + eps, xm = x - eps, yp = 0, ym = 0;
    apply_f(&xp, &yp, a, 1); apply_f(&xm, &ym, a, 1);
    fails += check("apply_f x_b", x_b, (yp - ym) / (2 * eps));
  }

  // ---- saxpy: poly ----
  {
    double x = 0.8, c0 = 0.5, c1 = -1.3, c2 = 2.1, y = 0;
    double x_b = 0, y_b = 1;
    poly_b(&x, &x_b, &y, &y_b, c0, c1, c2, 1);
    double xp = x + eps, xm = x - eps, yp = 0, ym = 0;
    poly(&xp, &yp, c0, c1, c2, 1); poly(&xm, &ym, c0, c1, c2, 1);
    fails += check("poly x_b", x_b, (yp - ym) / (2 * eps));
  }

  // ---- stencil: weighted (loop + array) ----
  {
    double x = 1.3, y = 0;
    double x_b = 0, y_b = 1;
    weighted_b(&x, &x_b, &y, &y_b, 1);
    double xp = x + eps, xm = x - eps, yp = 0, ym = 0;
    weighted(&xp, &yp, 1); weighted(&xm, &ym, 1);
    fails += check("weighted x_b", x_b, (yp - ym) / (2 * eps));
  }

  // ---- overwrite: tape (true overwrites + *=) ----
  {
    double x = 0.7, a = 1.6, y = 0;
    double x_b = 0, y_b = 1;
    ow_b(&x, &x_b, &y, &y_b, a, 1);
    double xp = x + eps, xm = x - eps, yp = 0, ym = 0;
    ow(&xp, &yp, a, 1); ow(&xm, &ym, a, 1);
    fails += check("ow x_b", x_b, (yp - ym) / (2 * eps));
  }

  // ---- device_fn: cross-call adjoint (layer -> activation_b) ----
  {
    double x = 0.6, w = 1.4, b = -0.2, y = 0;
    double x_b = 0, y_b = 1;
    layer_b(&x, &x_b, &y, &y_b, w, b, 1);
    double xp = x + eps, xm = x - eps, yp = 0, ym = 0;
    layer(&xp, &yp, w, b, 1); layer(&xm, &ym, w, b, 1);
    fails += check("layer x_b", x_b, (yp - ym) / (2 * eps));
  }

  // ---- atomic: reverse of an atomicAdd scatter (gather) ----
  {
    Idx m{0, 1};
    double src = 0.9, dst = 0;
    double src_b = 0, dst_b = 1;
    scatter_b(&src, &src_b, &dst, &dst_b, m, 1);
    double p = 0, q = 0, sp = src + eps, sm = src - eps;
    scatter(&sp, &p, m, 1); scatter(&sm, &q, m, 1);
    fails += check("scatter src_b", src_b, (p - q) / (2 * eps));
  }

  std::printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fails);
  return fails ? 1 : 0;
}
