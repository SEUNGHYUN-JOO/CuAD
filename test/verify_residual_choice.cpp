// Residual choice in multi-state root finders (test/reverse_only/residual_choice.cu): ambiguous candidates and
// accidentally vanishing seeded variables must trap; unused side values are no candidates; algebraic variants of one
// equation give its exact implicit derivative. Each adjoint runs in a child process so that a trap is seen as such.
#include <cmath>
#include <csignal>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>

#define __global__
#define __device__
#define __host__
#define CUAD_TAPE_SIZE 512
#include "cuad_rt.h"
#include "reverse_only/residual_choice.cu"
#include "out/residual_choice_b.cuh"

typedef void (*Adj)(const double *, double *, double *, double *, int);

// adjoint of y[0] in a child; returns false if it trapped
static bool run(Adj fb, double a, double b, double g[2]) {
  int fd[2];
  if (pipe(fd)) return false;
  pid_t pid = fork();
  if (pid == 0) {
    close(fd[0]);
    double x[2] = {a, b}, xb[2] = {0, 0}, y = 0, yb = 1;
    fb(x, xb, &y, &yb, 2);
    if (write(fd[1], xb, sizeof xb) != (ssize_t)sizeof xb) _exit(3);
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
  // root of t^3 + a t - b at a = 2, b = 5 and its exact derivative
  const double A = 2.0, B = 5.0, D = B * B / 4 + A * A * A / 27;
  const double t = std::cbrt(B / 2 + std::sqrt(D)) + std::cbrt(B / 2 - std::sqrt(D));
  const double ta = -t / (3 * t * t + A), tb = 1 / (3 * t * t + A);
  struct Case { const char *name; Adj fb; double a, b; bool trap; double ga, gb; } cases[] = {
      {"steer_tested a!=b", steer_tested_b, 0.3, 0.7, false, 1.0, 0.0},
      {"steer_tested a==b", steer_tested_b, 0.5, 0.5, true, 0, 0},
      {"side_value a==b", side_value_b, 0.5, 0.5, false, 1.0, 0.0},
      {"zero_aux root 1/2", zero_aux_b, 0.5, 0.0, true, 0, 0},
      {"cubic_horner", cubic_horner_b, A, B, false, ta, tb},
      {"cubic_scaled", cubic_scaled_b, A, B, false, ta, tb},
      {"cubic_local", cubic_local_b, A, B, false, ta, tb},
      {"cubic_twice", cubic_twice_b, A, B, false, ta, tb},
  };
  int fails = 0;
  for (auto &k : cases) {
    double g[2] = {NAN, NAN};
    bool ok = run(k.fb, k.a, k.b, g);
    bool pass;
    if (k.trap) {
      pass = !ok;
      std::printf("  %-20s %s (expected trap)  [%s]\n", k.name, ok ? "returned" : "trapped", pass ? "PASS" : "FAIL");
    } else if (!ok) {
      pass = false;
      std::printf("  %-20s trapped  [FAIL]\n", k.name);
    } else {
      double e = std::fmax(std::fabs(g[0] - k.ga), std::fabs(g[1] - k.gb)) / std::fmax(std::fabs(k.ga), std::fabs(k.gb));
      pass = e < 1e-10;
      std::printf("  %-20s adjoint vs exact %.2e  [%s]\n", k.name, e, pass ? "PASS" : "FAIL");
    }
    fails += !pass;
  }
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
