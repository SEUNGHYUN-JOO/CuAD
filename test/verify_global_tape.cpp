// Host check of the global tape (-DCUAD_GLOBAL_TAPE, test/cuad_rt.h): every thread keeps its tape
// in one column of a shared allocation, and nested *_b calls stack their frames in that column.
// Emulated threads (std::thread, each with its own cuad_host_tid) run the adjoints of
// test/cases/carried_arrays.cu and test/cases/global_tape.cu at the same time; each result must
// match finite differences and, bit for bit, the same thread's result computed alone. A thread
// index outside the allocation, a missing allocation and an exhausted capacity must abort.
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define __global__
#define __device__
#define __host__
#define CUAD_GLOBAL_TAPE
#define CUAD_TAPE_PROFILE
long long cuad_tape_peak = 0;
#include "cuad_rt.h"
#include "cases/carried_arrays.cu"
#include "out/carried_arrays_b.cuh"
#include "cases/global_tape.cu"
#include "out/global_tape_b.cuh"

typedef void (*Fn)(const double *, double *, int);
typedef void (*Adj)(const double *, double *, double *, double *, int);
struct Kernel { const char *name; Fn f; Adj fb; long long peak; };

static const int N = 7, T = 8, REPEAT = 40, CAP = 4096;

static void input(int t, double *x) {
  for (int i = 0; i < N; i++) x[i] = 0.3 + 0.13 * i - 0.05 * (i % 3) + 0.011 * t;
}
static void adjoint(Adj fb, const double *x, double *xb) {
  double y = 0, yb = 1.0;
  for (int i = 0; i < N; i++) xb[i] = 0;
  fb(x, xb, &y, &yb, N);
}
static double fdError(Fn f, const double *x, const double *xb) {
  double worst = 0;
  for (int i = 0; i < N; i++) {
    double h = 1e-6, xp[N], xm[N], yp = 0, ym = 0;
    for (int k = 0; k < N; k++) { xp[k] = x[k]; xm[k] = x[k]; }
    xp[i] += h; xm[i] -= h; f(xp, &yp, N); f(xm, &ym, N);
    double fd = (yp - ym) / (2 * h);
    worst = fmax(worst, fabs(xb[i] - fd) / (1 + fabs(fd)));
  }
  return worst;
}
static bool topsZero() {
  for (int t = 0; t < T; t++)
    if (cuad_gtop[t] != 0) return false;
  return true;
}

// child processes for the fatal cases
static Adj g_fb;
static int g_tid, g_cap;
static bool g_alloc;
static void runOne() {
  cuad_tape_free();
  if (g_alloc && cuad_tape_alloc(T, g_cap)) _exit(2);
  cuad_host_tid = g_tid;
  double x[N], xb[N];
  input(0, x);
  adjoint(g_fb, x, xb);
}
static bool aborts(Adj fb, int tid, int cap, bool alloc) {
  g_fb = fb; g_tid = tid; g_cap = cap; g_alloc = alloc;
  pid_t child = fork();
  if (child < 0) return false;
  if (child == 0) {
    rlimit limit = {0, 0};
    setrlimit(RLIMIT_CORE, &limit);
    runOne();
    _exit(0);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) return false;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
static bool exitsCleanly(Adj fb, int cap) {
  g_fb = fb; g_tid = T - 1; g_cap = cap; g_alloc = true;
  pid_t child = fork();
  if (child < 0) return false;
  if (child == 0) { runOne(); _exit(0); }
  int status = 0;
  if (waitpid(child, &status, 0) != child) return false;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main() {
  std::vector<Kernel> ks;
#define K(f) ks.push_back({#f, f, f##_b, 0})
  K(recur_fixed); K(recur_prefix); K(read_then_overwrite); K(recompute_state); K(lstm_like);
  K(dense_rnn); K(recurrent_state); K(modulo_index); K(quotient_index); K(square_index);
  K(private_update); K(offset_private); K(multidim_state); K(tri_region); K(neg_offset);
  K(sign_branch); K(nested_taped); K(through_plain);
  const int nk = (int)ks.size();
  int fails = 0;
  if (cuad_tape_alloc(T, CAP)) { std::printf("allocation failed\n"); return 1; }

  // each thread alone: reference results, finite differences, stacked tape depth
  std::vector<double> ref((size_t)nk * T * N);
  for (int k = 0; k < nk; k++) {
    double worst = 0;
    cuad_tape_peak = 0;
    for (int t = 0; t < T; t++) {
      cuad_host_tid = t;
      double x[N], *xb = &ref[((size_t)k * T + t) * N];
      input(t, x);
      adjoint(ks[k].fb, x, xb);
      worst = fmax(worst, fdError(ks[k].f, x, xb));
    }
    ks[k].peak = cuad_tape_peak;
    bool ok = worst < 1e-7 && topsZero();
    std::printf("  %-20s alone: adjoint-vs-FD %.2e, tape depth %lld  [%s]\n", ks[k].name, worst,
                ks[k].peak, ok ? "PASS" : "FAIL");
    fails += !ok;
  }
  long long most = 0;
  for (auto &k : ks) most = k.peak > most ? k.peak : most;
  cuad_tape_peak = most; // the concurrent phase reaches no new depth, so it does not write the counter

  // all threads at once on the shared tape
  std::atomic<long long> mismatches(0);
  std::vector<std::thread> pool;
  for (int t = 0; t < T; t++)
    pool.emplace_back([&, t] {
      cuad_host_tid = t;
      double x[N], xb[N];
      input(t, x);
      for (int r = 0; r < REPEAT; r++)
        for (int k = 0; k < nk; k++) {
          adjoint(ks[k].fb, x, xb);
          for (int i = 0; i < N; i++)
            if (xb[i] != ref[((size_t)k * T + t) * N + i]) mismatches++;
        }
    });
  for (auto &th : pool) th.join();
  bool together = mismatches == 0 && topsZero();
  std::printf("  %d threads x %d kernels x %d repeats on one tape: %lld mismatches  [%s]\n", T, nk, REPEAT,
              (long long)mismatches, together ? "PASS" : "FAIL");
  fails += !together;

  // nested frames need exactly their stacked depth; less, an index outside the launch or no tape aborts
  bool fatal = true;
  for (const char *name : {"recur_prefix", "nested_taped", "through_plain"})
    for (auto &k : ks)
      if (std::string(k.name) == name) {
        bool fits = exitsCleanly(k.fb, (int)k.peak), over = aborts(k.fb, T - 1, (int)k.peak - 1, true);
        std::printf("  %-20s capacity %lld runs, %lld aborts  [%s]\n", name, k.peak, k.peak - 1,
                    fits && over ? "PASS" : "FAIL");
        fatal = fatal && fits && over;
      }
  bool range = aborts(ks[0].fb, T, CAP, true) && aborts(ks[0].fb, -1, CAP, true) &&
               aborts(ks[0].fb, 0, CAP, false);
  std::printf("  thread index outside the allocation, no allocation: abort  [%s]\n", range ? "PASS" : "FAIL");
  fails += !fatal + !range;
  cuad_tape_free();
  std::printf("RESULT: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
