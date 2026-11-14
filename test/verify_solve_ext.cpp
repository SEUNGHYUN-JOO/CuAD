// Convergence loops beyond the basic forms (test/reverse_only/solve_ext.cu), reverse and forward mode: the implicit
// derivative of a slow contraction needs no adjoint iteration; a contraction stopped far from its fixed point traps;
// a double root traps in every class; `while`, `do` and condition-terminated `for` loops, a pure call in the body and
// a 2 x 2 Newton system with scalar or array unknowns give the exact derivative. Each derivative runs in a child
// process so that a trap is seen as such.
#include <cmath>
#include <csignal>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>

#define __global__
#define __device__
#define __host__
#define CUAD_TAPE_SIZE 4096
#include "cuad_rt.h"
#include "reverse_only/solve_ext.cu"
#include "out/solve_ext_b.cuh"
#ifdef WITH_TANGENT
#include "out/solve_ext_d.cuh"
#endif

typedef void (*Adj)(const double *, double *, double *, double *, int);
typedef void (*Tan)(const double *, const double *, double *, double *, int);

// the gradient of y[0] with respect to (a, b) in a child; returns false if it trapped
template <class F>
static bool child(F f, double g[2]) {
  int fd[2];
  if (pipe(fd)) return false;
  pid_t pid = fork();
  if (pid == 0) {
    close(fd[0]);
    double r[2] = {NAN, NAN};
    f(r);
    if (write(fd[1], r, sizeof r) != (ssize_t)sizeof r) _exit(3);
    _exit(0);
  }
  close(fd[1]);
  ssize_t got = read(fd[0], g, 2 * sizeof(double));
  close(fd[0]);
  int st = 0;
  waitpid(pid, &st, 0);
  return !WIFSIGNALED(st) && got == (ssize_t)(2 * sizeof(double));
}

int main() {
  const double A = 2.0, B = 5.0, D = B * B / 4 + A * A * A / 27;
  const double t = std::cbrt(B / 2 + std::sqrt(D)) + std::cbrt(B / 2 - std::sqrt(D));
  const double ta = -t / (3 * t * t + A), tb = 1 / (3 * t * t + A);
  struct Case { const char *name; Adj fb; Tan fd; double a, b; bool trap; double ga, gb; };
#ifdef WITH_TANGENT
#define T(f) f##_d
#else
#define T(f) nullptr
#endif
  Case cases[] = {
      {"slow contraction, converged", slow_tight_b, T(slow_tight), A, B, false, ta, tb},
      {"slow contraction, stopped early", slow_loose_b, T(slow_loose), A, B, true, 0, 0},
      {"Newton, simple root", double_newton_b, T(double_newton), 1, 1e-4, false, 1, 50},
      {"Newton, double root", double_newton_b, T(double_newton), 1, 0, true, 0, 0},
      {"bisection, simple root", double_bisect_b, T(double_bisect), 1, 1e-4, false, 1, 50},
      {"bisection, double root", double_bisect_b, T(double_bisect), 1, 0, true, 0, 0},
      {"secant, simple root", double_secant_b, T(double_secant), 1, 1e-4, false, 1, 50},
      {"secant, double root", double_secant_b, T(double_secant), 1, 0, true, 0, 0},
      {"while loop", while_newton_b, T(while_newton), A, B, false, ta, tb},
      {"do loop", do_newton_b, T(do_newton), A, B, false, ta, tb},
      {"for loop ended by its condition", for_cond_newton_b, T(for_cond_newton), A, B, false, ta, tb},
      {"pure call in the body", call_newton_b, T(call_newton), A, B, false, ta, tb},
      {"2x2 system, scalar unknowns", system_scalar_b, T(system_scalar), 3, 2, false, 2, -1},
      {"2x2 system, array unknowns", system_array_b, T(system_array), 3, 2, false, 2, -1},
  };
  int fails = 0;
  for (int mode = 0; mode < 2; ++mode) {
#ifndef WITH_TANGENT
    if (mode == 1) break;
#endif
    std::printf("  %s mode\n", mode ? "forward" : "reverse");
    for (auto &k : cases) {
      double g[2] = {NAN, NAN};
      bool ok = child([&](double r[2]) {
        double x[2] = {k.a, k.b}, y = 0;
        if (mode == 0) {
          double xb[2] = {0, 0}, yb = 1;
          k.fb(x, xb, &y, &yb, 2);
          r[0] = xb[0];
          r[1] = xb[1];
        } else {
          for (int j = 0; j < 2; ++j) {
            double xd[2] = {0, 0}, yd = 0;
            xd[j] = 1;
            k.fd(x, xd, &y, &yd, 2);
            r[j] = yd;
          }
        }
      }, g);
      bool pass;
      if (k.trap) {
        pass = !ok;
        std::printf("    %-34s %s (expected trap)  [%s]\n", k.name, ok ? "returned" : "trapped", pass ? "PASS" : "FAIL");
      } else if (!ok) {
        pass = false;
        std::printf("    %-34s trapped  [FAIL]\n", k.name);
      } else {
        double e = std::fmax(std::fabs(g[0] - k.ga), std::fabs(g[1] - k.gb)) / std::fmax(std::fabs(k.ga), std::fabs(k.gb));
        pass = e < 1e-9;
        std::printf("    %-34s derivative vs exact %.2e  [%s]\n", k.name, e, pass ? "PASS" : "FAIL");
      }
      fails += !pass;
    }
  }
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails != 0;
}
