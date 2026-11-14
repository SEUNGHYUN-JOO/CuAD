// cuad runtime support for generated derivative code (host + device).
//
//  * Reverse-mode tape: a per-thread LIFO stack. By default it is a plain local
//    array declared (CUAD_TAPE_DECL) at the top of each generated *_b function, so every CUDA thread
//    gets its own private tape with zero extra plumbing. Bump CUAD_TAPE_SIZE if
//    a kernel pushes more than the default within one thread. Exhaustion traps
//    instead of silently corrupting the thread's memory.
//  * atomicAdd: generated reverse code accumulates shared (function-parameter)
//    adjoints with atomicAdd to be race-free across threads; forward code emits
//    atomicAdd tangents. On the host (verification builds) we provide a scalar
//    shim so the same code compiles and runs single-threaded.
#pragma once

#if !defined(__CUDA_ARCH__)
#include <cstdlib>
#endif

//  * -DCUAD_GLOBAL_TAPE: the tape lives in one device allocation instead of a
//    local array. CUDA reserves local memory for every thread that can be
//    resident, so a long per-thread tape (a recurrence over many time steps)
//    does not fit in local memory; the global tape is sized at run time for the
//    threads actually launched. Entry j of thread t is at tape[j*stride + t], so
//    a warp that pushes in lockstep writes contiguous memory. Before a kernel
//    launch the host calls cuad_tape_alloc(threads, capacity) once; each thread
//    may hold `capacity` entries across all nested *_b calls. A thread index
//    outside [0, threads) or an exhausted capacity traps.
#ifndef CUAD_TAPE_SIZE
#define CUAD_TAPE_SIZE 256
#endif
// Operate on the function-local _cuad_tape / _cuad_tp declared by each *_b fn.
// Generated calls use these macros as statements. In CUDA device code, trap
// reports a launch error at synchronization; on the host, abort is immediate.
#if defined(__CUDA_ARCH__)
#define cuad_tape_fail() asm volatile("trap;")
#else
#define cuad_tape_fail() std::abort()
#endif
// -DCUAD_TAPE_PROFILE (host builds): record the deepest tape position reached in
// cuad_tape_peak, which the including program defines (`long long cuad_tape_peak;`).
#if defined(CUAD_TAPE_PROFILE) && !defined(__CUDA_ARCH__)
extern long long cuad_tape_peak;
#define cuad_tape_note() do { if (_cuad_tp > cuad_tape_peak) cuad_tape_peak = _cuad_tp; } while (0)
#else
#define cuad_tape_note() do {} while (0)
#endif
#if defined(__CUDACC__)
#define CUAD_HD __host__ __device__
#define CUAD_DEVVAR __device__
#else
#define CUAD_HD
#define CUAD_DEVVAR
#endif

#if !defined(CUAD_GLOBAL_TAPE)
// Per-thread local tape. CUAD_TAPE_DECL opens it at the top of a *_b function;
// a nested *_b call has its own array, so publishing the top is not needed.
#define CUAD_TAPE_DECL double _cuad_tape[CUAD_TAPE_SIZE]; int _cuad_tp = 0
#define CUAD_TAPE_PUBLISH() do {} while (0)
#define CUAD_TAPE_RESTORE() do {} while (0)
#define cuad_push(v) do { \
  if (_cuad_tp < 0 || _cuad_tp >= CUAD_TAPE_SIZE) cuad_tape_fail(); \
  _cuad_tape[_cuad_tp++] = (v); \
  cuad_tape_note(); \
} while (0)
#define cuad_pop(v) do { \
  if (_cuad_tp <= 0 || _cuad_tp > CUAD_TAPE_SIZE) cuad_tape_fail(); \
  (v) = _cuad_tape[--_cuad_tp]; \
} while (0)
#else
// Global tape. cuad_gtop[t] is the first free entry of thread t: a *_b function
// starts its frame there (_cuad_tp0), publishes its own top before calling another
// *_b function and restores its start after the call, so every function leaves
// cuad_gtop[t] as it found it (0 between launches).
static CUAD_DEVVAR double *cuad_gtape = 0;
static CUAD_DEVVAR int *cuad_gtop = 0;
static CUAD_DEVVAR long long cuad_gstride = 0;
static CUAD_DEVVAR int cuad_gcap = 0;
#if defined(__CUDACC__)
// One host/device function: CUDA front ends also check device code in the host pass. The global
// tape is device memory, so a CUDA build that runs a *_b function on the host traps (index -1).
static __host__ __device__ __forceinline__ long long cuad_tid() {
#if defined(__CUDA_ARCH__)
  const long long block = ((long long)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  return block * ((long long)blockDim.x * blockDim.y * blockDim.z) +
         ((long long)threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
#else
  return -1;
#endif
}
#else
// Host builds emulate threads: each host thread sets its own index.
static thread_local long long cuad_host_tid = 0;
static inline long long cuad_tid() { return cuad_host_tid; }
#endif
#define CUAD_TAPE_DECL \
  const long long _cuad_t = cuad_tid(); \
  if (!cuad_gtape || _cuad_t < 0 || _cuad_t >= cuad_gstride) cuad_tape_fail(); \
  const long long _cuad_ts = cuad_gstride; \
  const int _cuad_tcap = cuad_gcap; \
  double *const _cuad_tape = cuad_gtape + _cuad_t; \
  const int _cuad_tp0 = cuad_gtop[_cuad_t]; \
  int _cuad_tp = _cuad_tp0
#define CUAD_TAPE_PUBLISH() do { cuad_gtop[_cuad_t] = _cuad_tp; } while (0)
#define CUAD_TAPE_RESTORE() do { cuad_gtop[_cuad_t] = _cuad_tp0; } while (0)
#define cuad_push(v) do { \
  if (_cuad_tp < _cuad_tp0 || _cuad_tp >= _cuad_tcap) cuad_tape_fail(); \
  _cuad_tape[(long long)_cuad_tp++ * _cuad_ts] = (v); \
  cuad_tape_note(); \
} while (0)
#define cuad_pop(v) do { \
  if (_cuad_tp <= _cuad_tp0 || _cuad_tp > _cuad_tcap) cuad_tape_fail(); \
  (v) = _cuad_tape[(long long)--_cuad_tp * _cuad_ts]; \
} while (0)

// Host side: allocate the tape for `threads` threads with `capacity` entries each
// (returns 0 on success). Device memory under nvcc, host memory otherwise.
#if defined(__CUDACC__)
static inline int cuad_tape_alloc(long long threads, int capacity) {
  double *tape = 0;
  int *top = 0;
  if (threads <= 0 || capacity <= 0) return 1;
  if (cudaMalloc(&tape, sizeof(double) * (size_t)threads * (size_t)capacity) != cudaSuccess) return 1;
  if (cudaMalloc(&top, sizeof(int) * (size_t)threads) != cudaSuccess) { cudaFree(tape); return 1; }
  if (cudaMemset(top, 0, sizeof(int) * (size_t)threads) != cudaSuccess ||
      cudaMemcpyToSymbol(cuad_gtape, &tape, sizeof tape) != cudaSuccess ||
      cudaMemcpyToSymbol(cuad_gtop, &top, sizeof top) != cudaSuccess ||
      cudaMemcpyToSymbol(cuad_gstride, &threads, sizeof threads) != cudaSuccess ||
      cudaMemcpyToSymbol(cuad_gcap, &capacity, sizeof capacity) != cudaSuccess) {
    cudaFree(tape); cudaFree(top); return 1;
  }
  return 0;
}
static inline void cuad_tape_free() {
  double *tape = 0;
  int *top = 0;
  const long long zero = 0;
  cudaMemcpyFromSymbol(&tape, cuad_gtape, sizeof tape);
  cudaMemcpyFromSymbol(&top, cuad_gtop, sizeof top);
  cudaFree(tape);
  cudaFree(top);
  tape = 0; top = 0;
  cudaMemcpyToSymbol(cuad_gtape, &tape, sizeof tape);
  cudaMemcpyToSymbol(cuad_gtop, &top, sizeof top);
  cudaMemcpyToSymbol(cuad_gstride, &zero, sizeof zero);
}
#else
static inline int cuad_tape_alloc(long long threads, int capacity) {
  if (threads <= 0 || capacity <= 0) return 1;
  cuad_gtape = (double *)std::calloc((size_t)threads * (size_t)capacity, sizeof(double));
  cuad_gtop = (int *)std::calloc((size_t)threads, sizeof(int));
  if (!cuad_gtape || !cuad_gtop) { std::free(cuad_gtape); std::free(cuad_gtop); cuad_gtape = 0; cuad_gtop = 0; return 1; }
  cuad_gstride = threads;
  cuad_gcap = capacity;
  return 0;
}
static inline void cuad_tape_free() {
  std::free(cuad_gtape);
  std::free(cuad_gtop);
  cuad_gtape = 0; cuad_gtop = 0; cuad_gstride = 0; cuad_gcap = 0;
}
#endif
#endif

// Implicit derivatives of convergence loops (ReverseMode.h, ForwardMode.h). For a smooth update x <- G(x, p) the
// generated code forms G_x and G_p from one reverse (or tangent) sweep per state component and solves
// (I - G_x)^T l = s (or (I - G_x) x' = G_p p') directly, so a slowly contracting loop needs no more work than a fast
// one. CUAD_SOLVE_MAXIT bounded the adjoint fixed-point iteration of earlier versions; it is kept for compatibility
// and no longer used.
#ifndef CUAD_SOLVE_MAXIT
#define CUAD_SOLVE_MAXIT 200
#endif
// The loop's exit state must lie within this relative distance of a solution: for a smooth update the distance
// |(I - G_x)^{-1} (G(x) - x)| estimated from one more iteration (not just the size of that step, which is smaller
// by 1 - |G_x| for a slow contraction), for a bracketing loop the width of the collapsed bracket, for a root finder
// |R / R_t|. A loop that left for another reason (a threshold, an iteration cap) traps instead of receiving the
// derivative of a solution it did not reach.
#ifndef CUAD_SOLVE_TOL
#define CUAD_SOLVE_TOL 1e-6
#endif
// Absolute part of the same tests, in the units of the iterated variables (default 0: purely relative). With 0, a
// state or root at exactly zero passes only if the test quantity vanishes exactly; otherwise the adjoint traps, so
// a code whose solution can sit at the origin should set the scale of its variables here.
#ifndef CUAD_SOLVE_ATOL
#define CUAD_SOLVE_ATOL 0.0
#endif
// A loop with several candidate residuals (secant, regula falsi, Brent): when more than one vanishes at the exit
// state, they must locate the same root and imply the same derivative of it to this relative tolerance; otherwise
// the relation the loop solved is ambiguous and the adjoint traps.
// The same tolerance bounds the conditioning of the solution: the derivative is evaluated again at the edge of the
// interval in which the solution is known (the estimated distance, the bracket, |R / R_t|) and must agree with the
// derivative at the exit state. A double root, where the derivative does not exist, or a root so ill-conditioned
// that the loop's tolerance leaves its derivative undetermined, traps.
#ifndef CUAD_SOLVE_AGREE
#define CUAD_SOLVE_AGREE 1e-4
#endif
#define cuad_solve_fail() cuad_tape_fail()
// -DCUAD_SOLVE_CHECK=0 compiles out every test above together with the extra evaluations it needs (one more
// iteration, the second derivative evaluation). It exists to measure what the tests cost; the derivative of a loop
// that did not reach a solution is then returned without warning. The default 1 skips the second derivative
// evaluation of a smooth update whose every entry of G_x lies below 1/4 (an optimization aimed at Newton's method,
// which converges quadratically there; it is a heuristic, not a guarantee). -DCUAD_SOLVE_CHECK=2 never skips it.
#ifndef CUAD_SOLVE_CHECK
#define CUAD_SOLVE_CHECK 1
#endif

// Solve the small dense system A x = b in place (b becomes x) by Gaussian elimination with partial pivoting. Every
// index is a compile-time constant after unrolling, so A and b stay in registers. Returns 0 if A is singular to
// working precision (a pivot below 1e-13 times the largest entry), in which case the solution is not isolated.
template <int K>
static inline CUAD_HD int cuad_lsolve(double (&A)[K][K], double (&b)[K]) {
  double big = 0;
#pragma unroll
  for (int i = 0; i < K; ++i)
#pragma unroll
    for (int j = 0; j < K; ++j)
      big = (A[i][j] < 0 ? -A[i][j] : A[i][j]) > big ? (A[i][j] < 0 ? -A[i][j] : A[i][j]) : big;
#pragma unroll
  for (int c = 0; c < K; ++c) {
#pragma unroll
    for (int r = c + 1; r < K; ++r) { // bring the largest remaining entry of column c to row c
      double ar = A[r][c] < 0 ? -A[r][c] : A[r][c], ac = A[c][c] < 0 ? -A[c][c] : A[c][c];
      if (ar > ac) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
          double t = A[r][j]; A[r][j] = A[c][j]; A[c][j] = t;
        }
        double t = b[r]; b[r] = b[c]; b[c] = t;
      }
    }
    double piv = A[c][c] < 0 ? -A[c][c] : A[c][c];
    if (!(piv > 1e-13 * big))
      return 0;
#pragma unroll
    for (int r = c + 1; r < K; ++r) {
      double f = A[r][c] / A[c][c];
#pragma unroll
      for (int j = c; j < K; ++j)
        A[r][j] -= f * A[c][j];
      b[r] -= f * b[c];
    }
  }
#pragma unroll
  for (int r = K - 1; r >= 0; --r) {
    double s = b[r];
#pragma unroll
    for (int j = r + 1; j < K; ++j)
      s -= A[r][j] * b[j];
    b[r] = s / A[r][r];
  }
  return 1;
}

// Bounds of the element ranges that a loop's per-iteration checkpoint saves (see ReverseMode.h).
static inline CUAD_HD long long cuad_imin(long long a, long long b) { return a < b ? a : b; }
static inline CUAD_HD long long cuad_imax(long long a, long long b) { return a > b ? a : b; }

#if !defined(__CUDACC__)
static inline double atomicAdd(double *a, double v) {
  double o = *a;
  *a += v;
  return o;
}
static inline float atomicAdd(float *a, float v) {
  float o = *a;
  *a += v;
  return o;
}
#endif
