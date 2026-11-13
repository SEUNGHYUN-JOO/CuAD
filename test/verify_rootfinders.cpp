// Multi-state root finders (test/reverse_only/rootfinders.cu): the adjoint of y = t^2 + c t at the root of
// t^3 + a t - b = 0 against the exact implicit derivative, run in a child process so that a trap is reported as such.
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
#include "reverse_only/rootfinders.cu"
#include "out/rootfinders_b.cuh"

typedef void (*Adj)(const double *, double *, double *, double *, int);

int main() {
  const double a = 2.0, b = 5.0, c = 0.7;
  const double q = -b, p = a, D = q * q / 4 + p * p * p / 27;
  const double t = std::cbrt(-q / 2 + std::sqrt(D)) + std::cbrt(-q / 2 - std::sqrt(D));
  const double dy = 2 * t + c, ga = dy * (-t / (3 * t * t + a)), gb = dy * (1 / (3 * t * t + a)), gc = t;
  struct { const char *name; Adj fb; } cases[] = {
      {"secant_root", secant_root_b}, {"illinois_root", illinois_root_b}, {"brent_root", brent_root_b}};
  int fails = 0;
  for (auto &k : cases) {
    int fd[2];
    if (pipe(fd)) return 2;
    pid_t pid = fork();
    if (pid == 0) {
      close(fd[0]);
      double x[3] = {a, b, c}, xb[3] = {0, 0, 0}, y = 0, yb = 1;
      k.fb(x, xb, &y, &yb, 3);
      if (write(fd[1], xb, sizeof xb) != (ssize_t)sizeof xb) _exit(3);
      _exit(0);
    }
    close(fd[1]);
    double xb[3] = {NAN, NAN, NAN};
    ssize_t got = read(fd[0], xb, sizeof xb);
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st) || got != (ssize_t)sizeof xb) {
      std::printf("  %-14s trapped  [FAIL]\n", k.name);
      ++fails;
      continue;
    }
    double e = std::fmax(std::fmax(std::fabs(xb[0] - ga) / std::fabs(ga), std::fabs(xb[1] - gb) / std::fabs(gb)),
                         std::fabs(xb[2] - gc) / std::fabs(gc));
    bool ok = e < 1e-10;
    std::printf("  %-14s adjoint vs exact %.2e  [%s]\n", k.name, e, ok ? "PASS" : "FAIL");
    fails += !ok;
  }
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
