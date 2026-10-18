// Host-side finite-difference verification of cuad-generated tangents.
// Compiles the CUDA kernels as plain host functions (qualifiers defined away,
// thread indices stubbed so the single element i==0 is processed) and checks
// the analytic tangent against a central finite difference.
#include <cmath>
#include <cstdio>
#include <cstdlib>

// --- neutralize CUDA-isms so kernels compile as host code, single element ---
#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#include "cuad_rt.h" // host atomicAdd shim + tape macros

// primal kernels
#include "cases/scalar.cu"
#include "cases/saxpy.cu"
#include "cases/stencil.cu"
#include "cases/device_fn.cu"
#include "cases/atomic.cu"
// generated tangents
#include "out/scalar_d.cuh"
#include "out/saxpy_d.cuh"
#include "out/stencil_d.cuh"
#include "out/device_fn_d.cuh"
#include "out/atomic_d.cuh"

static int check(const char *name, double tangent, double fd) {
  double err = std::fabs(tangent - fd) /
               (1.0 + std::fabs(fd)); // relative-ish
  const char *ok = (err < 1e-6) ? "PASS" : "FAIL";
  std::printf("  %-22s tangent=% .10e  fd=% .10e  relerr=%.2e  [%s]\n", name,
              tangent, fd, err, ok);
  return err < 1e-6 ? 0 : 1;
}

int main() {
  int fails = 0;
  const double eps = 1e-6;

  // ---- scalar: apply_f(x,y,a,n),  y=sin(a*x)+x*x ----
  {
    double x = 0.37, a = 1.9;
    double xd = 1.0, ad = 0.0; // direction: d/dx
    double y = 0, yd = 0;
    apply_f(&x, &y, a, 1);
    apply_f_d(&x, &xd, &y, &yd, a, ad, 1);
    double yp = 0, ym = 0, xp = x + eps, xm = x - eps;
    apply_f(&xp, &yp, a, 1);
    apply_f(&xm, &ym, a, 1);
    fails += check("apply_f d/dx", yd, (yp - ym) / (2 * eps));

    // direction d/da
    xd = 0.0; ad = 1.0; yd = 0;
    apply_f_d(&x, &xd, &y, &yd, a, ad, 1);
    double ya = 0, yb = 0;
    apply_f(&x, &ya, a + eps, 1);
    apply_f(&x, &yb, a - eps, 1);
    fails += check("apply_f d/da", yd, (ya - yb) / (2 * eps));
  }

  // ---- saxpy: poly(x,y,c0,c1,c2,n) ----
  {
    double x = 0.8, c0 = 0.5, c1 = -1.3, c2 = 2.1;
    double y = 0;
    // d/dx
    {
      double xd = 1, yd = 0;
      poly(&x, &y, c0, c1, c2, 1);
      poly_d(&x, &xd, &y, &yd, c0, 0, c1, 0, c2, 0, 1);
      double yp = 0, ym = 0, xp = x + eps, xm = x - eps;
      poly(&xp, &yp, c0, c1, c2, 1);
      poly(&xm, &ym, c0, c1, c2, 1);
      fails += check("poly d/dx", yd, (yp - ym) / (2 * eps));
    }
    // d/dc2
    {
      double xd = 0, yd = 0;
      poly_d(&x, &xd, &y, &yd, c0, 0, c1, 0, c2, 1, 1);
      double yp = 0, ym = 0;
      poly(&x, &yp, c0, c1, c2 + eps, 1);
      poly(&x, &ym, c0, c1, c2 - eps, 1);
      fails += check("poly d/dc2", yd, (yp - ym) / (2 * eps));
    }
  }

  // ---- stencil: weighted(x,y,n), loop + local array + constant local ----
  {
    double x = 1.3, y = 0;
    double xd = 1, yd = 0;
    weighted(&x, &y, 1);
    weighted_d(&x, &xd, &y, &yd, 1);
    double yp = 0, ym = 0, xp = x + eps, xm = x - eps;
    weighted(&xp, &yp, 1);
    weighted(&xm, &ym, 1);
    fails += check("weighted d/dx", yd, (yp - ym) / (2 * eps));
  }

  // ---- device_fn: layer calls value-returning __device__ activation() ----
  {
    double x = 0.6, w = 1.4, b = -0.2, y = 0;
    // d/dx
    {
      double xd = 1, yd = 0;
      layer(&x, &y, w, b, 1);
      layer_d(&x, &xd, &y, &yd, w, 0, b, 0, 1);
      double yp = 0, ym = 0, xp = x + eps, xm = x - eps;
      layer(&xp, &yp, w, b, 1);
      layer(&xm, &ym, w, b, 1);
      fails += check("layer d/dx", yd, (yp - ym) / (2 * eps));
    }
    // d/db
    {
      double xd = 0, yd = 0;
      layer_d(&x, &xd, &y, &yd, w, 0, b, 1, 1);
      double yp = 0, ym = 0;
      layer(&x, &yp, w, b + eps, 1);
      layer(&x, &ym, w, b - eps, 1);
      fails += check("layer d/db", yd, (yp - ym) / (2 * eps));
    }
  }

  // ---- atomic: struct index fields + atomicAdd scatter ----
  {
    Idx m{0, 1};
    double src = 0.9, dst = 0, dst_d = 0, src_d = 1;
    scatter_d(&src, &src_d, &dst, &dst_d, m, 1);
    double p = 0, q = 0, sp = src + eps, sm = src - eps;
    scatter(&sp, &p, m, 1); scatter(&sm, &q, m, 1);
    fails += check("scatter d/dsrc", dst_d, (p - q) / (2 * eps));
  }

  std::printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fails);
  return fails ? 1 : 0;
}
