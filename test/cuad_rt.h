// cuad runtime support for generated derivative code (host + device).
//
//  * Reverse-mode tape: a per-thread LIFO stack. It is a plain local array
//    declared at the top of each generated *_b function, so every CUDA thread
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
#define cuad_push(v) do { \
  if (_cuad_tp < 0 || _cuad_tp >= CUAD_TAPE_SIZE) cuad_tape_fail(); \
  _cuad_tape[_cuad_tp++] = (v); \
  cuad_tape_note(); \
} while (0)
#define cuad_pop(v) do { \
  if (_cuad_tp <= 0 || _cuad_tp > CUAD_TAPE_SIZE) cuad_tape_fail(); \
  (v) = _cuad_tape[--_cuad_tp]; \
} while (0)

// Bounds of the element ranges that a loop's per-iteration checkpoint saves (see ReverseMode.h).
#if defined(__CUDACC__)
#define CUAD_HD __host__ __device__
#else
#define CUAD_HD
#endif
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
