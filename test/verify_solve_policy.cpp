// Run-time policy of the implicit adjoint (test/reverse_only/solve_policy.cu): a loop of the convergence-loop form that
// stopped away from a fixed point must trap (abort on the host) instead of returning the derivative of a fixed point it
// did not reach. Death tests by fork, as in verify_tape_bounds.cpp; the converged control must run and be correct.
#include <cmath>
#include <csignal>
#include <cstdio>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define __global__
#define __device__
#define __host__
#define CUAD_TAPE_SIZE 256
#include "cuad_rt.h"
#include "reverse_only/solve_policy.cu"
#include "out/solve_policy_b.cuh"

typedef void (*Adj)(const double *, double *, double *, double *, int);
static Adj g_fb;
static double g_x;
static void run() {
  double x[1] = {g_x}, xb[1] = {0}, y = 0, yb = 1;
  g_fb(x, xb, &y, &yb, 1);
}
static bool aborts(Adj fb, double x) {
  g_fb = fb; g_x = x;
  pid_t c = fork();
  if (c < 0) return false;
  if (c == 0) {
    rlimit l = {0, 0};
    setrlimit(RLIMIT_CORE, &l);
    run();
    _exit(0);
  }
  int st = 0;
  if (waitpid(c, &st, 0) != c) return false;
  return WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
}

int main() {
  int fails = 0;
  struct { const char *name; Adj fb; double x; } traps[] = {
      {"threshold_exit", threshold_exit_b, 1.0},
      {"march_to_threshold", march_to_threshold_b, 1.0},
      {"newton_cycle", newton_cycle_b, 2.0},
      {"bisect_capped", bisect_capped_b, 2.0},
  };
  for (auto &t : traps) {
    bool ok = aborts(t.fb, t.x);
    std::printf("  %-22s stopped away from a fixed point: trap  [%s]\n", t.name, ok ? "PASS" : "FAIL");
    fails += !ok;
  }
  double x[1] = {0.8}, xb[1] = {0}, y = 0, yb = 1;
  contraction_converged_b(x, xb, &y, &yb, 1);
  bool ok = std::fabs(xb[0] - 2.0) < 1e-12; // t* = 2a
  std::printf("  %-22s converged: dt*/da = %.15f (exact 2)  [%s]\n", "contraction_converged", xb[0], ok ? "PASS" : "FAIL");
  fails += !ok;
  {
    double xo[1] = {2.0}, xob[1] = {0}, yo = 0, yob = 1;
    bisect_outside_b(xo, xob, &yo, &yob, 1);
    bool ok2 = std::fabs(xob[0] - 0.1) < 1e-12; // y = hi0 = 0.5 + 0.1 a
    std::printf("  %-22s root outside the bracket: dy/da = %.15f (the bound's, exact 0.1)  [%s]\n", "bisect_outside",
                xob[0], ok2 ? "PASS" : "FAIL");
    fails += !ok2;
  }
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
