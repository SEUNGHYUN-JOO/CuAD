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
#define cuad_push(v) do { \
  if (_cuad_tp < 0 || _cuad_tp >= CUAD_TAPE_SIZE) cuad_tape_fail(); \
  _cuad_tape[_cuad_tp++] = (v); \
} while (0)
#define cuad_pop(v) do { \
  if (_cuad_tp <= 0 || _cuad_tp > CUAD_TAPE_SIZE) cuad_tape_fail(); \
  (v) = _cuad_tape[--_cuad_tp]; \
} while (0)

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
