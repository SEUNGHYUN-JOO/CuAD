// Reliability of the run-time checks of the implicit adjoint (kernels in check_reliability.cu). Every adjoint runs in
// a forked child so that a trap of the generated code is recorded as such. The check tolerances CUAD_SOLVE_TOL,
// CUAD_SOLVE_ATOL and CUAD_SOLVE_MAXIT are run-time variables here (compile with -DCUAD_SOLVE_TOL=g_tol
// -DCUAD_SOLVE_ATOL=g_atol -DCUAD_SOLVE_MAXIT=g_maxit). Output: one JSON object per run on stdout.
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <sys/wait.h>
#include <unistd.h>

static double g_tol = 1e-6, g_atol = 0.0;
static long g_maxit = 200;

#define __global__
#define __device__
#define __host__
#define CUAD_TAPE_SIZE 512
#include "cuad_rt.h"
#include "check_reliability.cu"
#include "out/check_reliability_b.cuh"

typedef void (*Prim)(const double *, double *, double, double, double, double, int);
typedef void (*Adj)(const double *, double *, double *, double *, double, double, double, double, int);
struct Par { double w, t0, tolp, atolp; };

// gradient of y[0] with respect to x in a child; false if it trapped
static bool adjoint(Adj fb, const double x[3], Par p, double g[3]) {
  int fd[2];
  if (pipe(fd)) return false;
  pid_t pid = fork();
  if (pid == 0) {
    close(fd[0]);
    double xb[3] = {0, 0, 0}, y = 0, yb = 1;
    fb(x, xb, &y, &yb, p.w, p.t0, p.tolp, p.atolp, 1);
    if (write(fd[1], xb, sizeof xb) != (ssize_t)sizeof xb) _exit(3);
    _exit(0);
  }
  close(fd[1]);
  ssize_t got = read(fd[0], g, 3 * sizeof(double));
  close(fd[0]);
  int st = 0;
  waitpid(pid, &st, 0);
  return !WIFSIGNALED(st) && got == (ssize_t)(3 * sizeof(double));
}

// root of t^3 + a t + s b = 0 nearest to t, polished in long double
static long double polish(long double t, long double a, long double b, int s) {
  for (int k = 0; k < 200; ++k) {
    long double f = t * t * t + a * t + s * b, fp = 3 * t * t + a;
    if (fp == 0) break;
    long double dt = f / fp;
    t -= dt;
    if (fabsl(dt) <= 1e-30L * fabsl(t)) break;
  }
  return t;
}

// iterations of the primal (replays the loop to count them)
static int count_fp(double a, double b, Par p) {
  double t = p.t0; int it = 0;
  for (; it < 2000000; ++it) { double dt = p.w * (t * t * t + a * t - b); t -= dt; if (fabs(dt) <= p.tolp * fabs(t) + p.atolp) break; }
  return it + 1;
}

struct Num { char s[32]; };
static Num num(double v) {  // JSON number, null if not finite
  Num n;
  if (std::isfinite(v)) std::snprintf(n.s, sizeof n.s, "%.17g", v);
  else std::snprintf(n.s, sizeof n.s, "null");
  return n;
}

static void emit(const char *cs, const char *label, double param, Par p, bool ok, const double g[3], const double ex[3],
                 int ncomp, double t, double texact, int iters) {
  double ge = NAN, gm = 0;
  if (ok) {
    double e = 0;
    for (int k = 0; k < ncomp; ++k) { e = fmax(e, fabs(g[k] - ex[k])); gm = fmax(gm, fabs(ex[k])); }
    ge = e / gm;
  }
  double ve = texact != 0 ? fabs(t - texact) / fabs(texact) : fabs(t - texact);
  char gs[32] = "null";
  if (ok && std::isfinite(ge)) std::snprintf(gs, sizeof gs, "%.3e", ge);
  std::printf("{\"case\": \"%s\", \"label\": \"%s\", \"param\": %.6g, \"tolp\": %.0e, \"atolp\": %.0e, \"eps_s\": %.0e, "
              "\"atol\": %.0e, \"maxit\": %ld, \"trapped\": %s, \"grad_rel_error\": %s, \"value_rel_error\": %.3e, "
              "\"iterations\": %d, \"value\": %.17g, \"grad\": [%s, %s, %s], \"exact\": [%s, %s, %s]}\n",
              cs, label, param, p.tolp, p.atolp, g_tol, g_atol, g_maxit, ok ? "false" : "true", gs, ve, iters, t,
              num(ok ? g[0] : NAN).s, num(ok ? g[1] : NAN).s, num(ok ? g[2] : NAN).s, num(ex[0]).s, num(ex[1]).s,
              num(ex[2]).s);
}

int main() {
  const double tolps[] = {1e-4, 1e-6, 1e-8, 1e-10, 1e-12}, epss[] = {1e-4, 1e-6, 1e-8};
  const double A = 2.0, B = 5.0;
  const double ts = (double)polish(1.3L, A, B, -1);
  const double fps = 3 * ts * ts + A;
  const double exc[3] = {-ts / fps, 1 / fps, 0};

  // 1. fixed-point relaxation: contraction rate rho, primal tolerance, check tolerance, adjoint iteration cap
  for (double rho : {0.5, 0.9, 0.99, 0.999})
    for (long mx : {200L, 1000000L})
      for (double tp : tolps)
        for (double es : epss) {
          g_tol = es; g_atol = 0; g_maxit = mx;
          Par p{(1 - rho) / fps, 2.0, tp, 0.0};
          double x[3] = {A, B, 0}, y = 0, g[3] = {NAN, NAN, NAN};
          fp_relax(x, &y, p.w, p.t0, p.tolp, p.atolp, 1);
          bool ok = adjoint(fp_relax_b, x, p, g);
          emit("fixed_point", "rho", rho, p, ok, g, exc, 2, y, ts, count_fp(A, B, p));
        }

  // 2. Newton on the same cubic
  for (double tp : tolps)
    for (double es : epss) {
      g_tol = es; g_atol = 0; g_maxit = 200;
      Par p{0, 2.0, tp, 0.0};
      double x[3] = {A, B, 0}, y = 0, g[3] = {NAN, NAN, NAN};
      newton_cubic(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
      int it = 0; { double t = p.t0; for (; it < 200; ++it) { double dt = (t*t*t + A*t - B) / (3*t*t + A); t -= dt; if (fabs(dt) <= tp * fabs(t)) break; } ++it; }
      bool ok = adjoint(newton_cubic_b, x, p, g);
      emit("newton", "-", 0, p, ok, g, exc, 2, y, ts, it);
    }

  // 3. root at or near zero: t^3 + 2 t - b, b -> 0, by the relaxation (rho = 0.5) that also stops at an absolute
  //    step of 1e-12 (Newton lands on 0 exactly in floating point and is recorded for comparison)
  for (int solver = 0; solver < 2; ++solver)
    for (double b : {1.0, 1e-4, 1e-8, 0.0})
      for (double at : {0.0, 1e-12, 1e-9}) {
        g_tol = 1e-6; g_atol = at; g_maxit = 200;
        double tz = b == 0 ? 0.0 : (double)polish((long double)b / A, A, b, -1);
        double f0 = 3 * tz * tz + A, ex[3] = {-tz / f0, 1 / f0, 0};
        Par p{0.5 / f0, 1.0, 1e-10, 1e-12};
        double x[3] = {A, b, 0}, y = 0, g[3] = {NAN, NAN, NAN};
        if (solver == 0) fp_relax(x, &y, p.w, p.t0, p.tolp, p.atolp, 1);
        else newton_cubic(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
        bool ok = adjoint(solver == 0 ? fp_relax_b : newton_cubic_b, x, p, g);
        emit(solver == 0 ? "zero_root_fixed_point" : "zero_root_newton", "b", b, p, ok, g, ex, 2, y, tz,
             solver == 0 ? count_fp(A, b, p) : 0);
      }

  // 4. nearly double root (t - 1)^2 = d by Newton and bisection
  for (double d : {1e-2, 1e-4, 1e-8, 1e-12, 0.0})
    for (int solver = 0; solver < 2; ++solver)
      for (double tp : {1e-4, 1e-8, 1e-12})
        for (double es : epss) {
          g_tol = es; g_atol = 0; g_maxit = 200;
          Par p{0, 2.0, tp, 0.0};
          double x[3] = {d, 0, 0}, y = 0, g[3] = {NAN, NAN, NAN};
          if (solver == 0) newton_double(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
          else bisect_double(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
          double tr = 1 + std::sqrt(d), ex[3] = {d > 0 ? 0.5 / std::sqrt(d) : INFINITY, 0, 0};
          bool ok = adjoint(solver == 0 ? newton_double_b : bisect_double_b, x, p, g);
          emit(solver == 0 ? "double_root_newton" : "double_root_bisection", "d", d, p, ok, g, ex, 1, y, tr, 0);
        }

  // 5. three roots of t^3 - 7 t + 6 (1, 2, -3): the start selects the root
  for (double t0 : {0.0, 1.5, 1.6, 2.5, -5.0}) {
    g_tol = 1e-6; g_atol = 0; g_maxit = 200;
    Par p{0, t0, 1e-12, 0.0};
    double x[3] = {-7.0, 6.0, 0}, y = 0, g[3] = {NAN, NAN, NAN};
    newton_three(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
    double tr = (double)polish(y, -7.0, 6.0, 1), f0 = 3 * tr * tr - 7.0, ex[3] = {-tr / f0, -1 / f0, 0};
    bool ok = adjoint(newton_three_b, x, p, g);
    emit("three_roots", "t0", t0, p, ok, g, ex, 2, y, tr, 0);
  }

  // 6. active lower bound c = 1.5 above the root 1.328 of t^3 + 2 t - 5: the derivative is that of the bound
  {
    g_tol = 1e-6; g_atol = 0; g_maxit = 200;
    const double ex[3] = {0, 0, 1};
    for (int solver = 0; solver < 2; ++solver) {
      Par p{0, 2.0, 1e-12, 0.0};
      double x[3] = {A, B, 1.5}, y = 0, g[3] = {NAN, NAN, NAN};
      if (solver == 0) newton_floor(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
      else bisect_floor(x, &y, 0, p.t0, p.tolp, p.atolp, 1);
      bool ok = adjoint(solver == 0 ? newton_floor_b : bisect_floor_b, x, p, g);
      emit(solver == 0 ? "bound_newton" : "bound_bisection", "c", 1.5, p, ok, g, ex, 3, y, 1.5, 0);
    }
  }
  return 0;
}
