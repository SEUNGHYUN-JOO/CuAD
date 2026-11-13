// Derivative error versus iteration count K for three strategies on t^3 + a t - b = 0 (a = 2, b = 5):
// CuAD implicit adjoint (reverse mode), CuAD tangent of the iterations (forward mode; not for bisection, which
// forward mode rejects) and Enzyme v0.0.307 reverse mode of the iterations. Reference: the exact root (Cardano)
// and dt/da = -t/(3t^2 + a), dt/db = 1/(3t^2 + a). Prints JSON lines.
#include <cmath>
#include <cstdio>
#include <initializer_list>

#define __global__
#define __device__
#define __host__
#define CUAD_TAPE_SIZE 256
#include "cuad_rt.h"
#include "sweep_kernels.cu"
#include "sweep_b.cuh"
#include "sweep_d.cuh"

extern int enzyme_dup, enzyme_const;
template <class... A> void __enzyme_autodiff(void *, A...);

typedef void (*Fn)(const double *, double *, int, double);
#define ENZ(f, x, gx, K, t0) do { double y_ = 0, yb_ = 1; \
  __enzyme_autodiff((void *)f, enzyme_dup, x, gx, enzyme_dup, &y_, &yb_, enzyme_const, K, enzyme_const, t0); } while (0)

int main() {
  const double a = 2.0, b = 5.0;
  // Cardano for the depressed cubic t^3 + a t - b = 0 (a > 0: one real root)
  const double q = -b, p = a, D = q * q / 4 + p * p * p / 27;
  const double ts = std::cbrt(-q / 2 + std::sqrt(D)) + std::cbrt(-q / 2 - std::sqrt(D));
  const double ga = -ts / (3 * ts * ts + a), gb = 1 / (3 * ts * ts + a);
  for (double t0 : {1.0, 3.0}) {
    for (int K = 1; K <= 60; ++K) {
      const char *names[3] = {"newton", "fixed_point", "bisection"};
      for (int s = 0; s < 3; ++s) {
        double x[2] = {a, b}, y = 0, yb = 1, gx[2] = {0, 0};
        Fn f = s == 0 ? newton_K : s == 1 ? fixed_point_K : bisect_K;
        f(x, &y, K, t0);
        double val = std::fabs(y - ts) / ts;
        // CuAD implicit adjoint
        double yy = 0, t0b = 0; // the starting value's adjoint (zero at a converged point) is not used
        if (s == 0) newton_K_b(x, gx, &yy, &yb, K, t0, t0b);
        else if (s == 1) fixed_point_K_b(x, gx, &yy, &yb, K, t0, t0b);
        else bisect_K_b(x, gx, &yy, &yb, K, t0, t0b);
        double e_imp = std::fmax(std::fabs(gx[0] - ga) / std::fabs(ga), std::fabs(gx[1] - gb) / std::fabs(gb));
        // CuAD tangent (smooth loops only)
        double e_tan = -1;
        if (s < 2) {
          double g2[2];
          for (int k = 0; k < 2; ++k) {
            double xd[2] = {0, 0}, yd = 0, y2 = 0;
            xd[k] = 1;
            if (s == 0) newton_K_d(x, xd, &y2, &yd, K, t0, 0.0); else fixed_point_K_d(x, xd, &y2, &yd, K, t0, 0.0);
            g2[k] = yd;
          }
          e_tan = std::fmax(std::fabs(g2[0] - ga) / std::fabs(ga), std::fabs(g2[1] - gb) / std::fabs(gb));
        }
        // Enzyme
        double ge[2] = {0, 0};
        if (s == 0) ENZ(newton_K, x, ge, K, t0); else if (s == 1) ENZ(fixed_point_K, x, ge, K, t0); else ENZ(bisect_K, x, ge, K, t0);
        double e_enz = std::fmax(std::fabs(ge[0] - ga) / std::fabs(ga), std::fabs(ge[1] - gb) / std::fabs(gb));
        std::printf("{\"t0\": %g, \"K\": %d, \"solver\": \"%s\", \"value_error\": %.3e, \"cuad_implicit\": %.3e, "
                    "\"cuad_tangent_unrolled\": %.3e, \"enzyme_unrolled\": %.3e}\n",
                    t0, K, names[s], val, e_imp, e_tan, e_enz);
      }
    }
  }
  return 0;
}
