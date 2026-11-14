// Multi-state root finders (test/reverse_only/rootfinders.cu): the adjoint and the tangent of y = t^2 + c t at the
// root of t^3 + a t - b = 0 against the exact implicit derivative, run in a child process so that a trap is reported
// as such.
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
#include "out/rootfinders_d.cuh"

typedef void (*Adj)(const double *, double *, double *, double *, int);
typedef void (*Tan)(const double *, const double *, double *, double *, int);

int main() {
  const double a = 2.0, b = 5.0, c = 0.7;
  const double q = -b, p = a, D = q * q / 4 + p * p * p / 27;
  const double t = std::cbrt(-q / 2 + std::sqrt(D)) + std::cbrt(-q / 2 - std::sqrt(D));
  const double dy = 2 * t + c, ga = dy * (-t / (3 * t * t + a)), gb = dy * (1 / (3 * t * t + a)), gc = t;
  struct { const char *name; Adj fb; Tan fd; } cases[] = {{"secant_root", secant_root_b, secant_root_d},
                                                           {"illinois_root", illinois_root_b, illinois_root_d},
                                                           {"brent_root", brent_root_b, brent_root_d}};
  int fails = 0;
  for (int mode = 0; mode < 2; ++mode)
    for (auto &k : cases) {
      int fd[2];
      if (pipe(fd)) return 2;
      pid_t pid = fork();
      if (pid == 0) {
        close(fd[0]);
        double x[3] = {a, b, c}, g[3] = {0, 0, 0}, y = 0;
        if (mode == 0) {
          double yb = 1;
          k.fb(x, g, &y, &yb, 3);
        } else {
          for (int j = 0; j < 3; ++j) {
            double xd[3] = {0, 0, 0}, yd = 0;
            xd[j] = 1;
            k.fd(x, xd, &y, &yd, 3);
            g[j] = yd;
          }
        }
        if (write(fd[1], g, sizeof g) != (ssize_t)sizeof g) _exit(3);
        _exit(0);
      }
      close(fd[1]);
      double g[3] = {NAN, NAN, NAN};
      ssize_t got = read(fd[0], g, sizeof g);
      close(fd[0]);
      int st = 0;
      waitpid(pid, &st, 0);
      const char *what = mode ? "tangent" : "adjoint";
      if (WIFSIGNALED(st) || got != (ssize_t)sizeof g) {
        std::printf("  %-14s %s trapped  [FAIL]\n", k.name, what);
        ++fails;
        continue;
      }
      double e = std::fmax(std::fmax(std::fabs(g[0] - ga) / std::fabs(ga), std::fabs(g[1] - gb) / std::fabs(gb)),
                           std::fabs(g[2] - gc) / std::fabs(gc));
      bool ok = e < 1e-10;
      std::printf("  %-14s %s vs exact %.2e  [%s]\n", k.name, what, e, ok ? "PASS" : "FAIL");
      fails += !ok;
    }
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
