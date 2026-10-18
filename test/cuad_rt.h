// cuad runtime support for generated derivative code (host + device).
//
//  * Reverse-mode tape: a per-thread LIFO stack. It is a plain local array
//    declared at the top of each generated *_b function, so every CUDA thread
//    gets its own private tape with zero extra plumbing. Bump CUAD_TAPE_SIZE if
//    a kernel pushes more than the default within one thread.
//  * atomicAdd: generated reverse code accumulates shared (function-parameter)
//    adjoints with atomicAdd to be race-free across threads; forward code emits
//    atomicAdd tangents. On the host (verification builds) we provide a scalar
//    shim so the same code compiles and runs single-threaded.
#pragma once

#ifndef CUAD_TAPE_SIZE
#define CUAD_TAPE_SIZE 256
#endif
// Operate on the function-local _cuad_tape / _cuad_tp declared by each *_b fn.
#define cuad_push(v) (_cuad_tape[_cuad_tp++] = (v))
#define cuad_pop(v) ((v) = _cuad_tape[--_cuad_tp])

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
