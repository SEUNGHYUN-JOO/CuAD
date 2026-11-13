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
#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ long long cuad_tid() {
  const long long block = ((long long)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  return block * ((long long)blockDim.x * blockDim.y * blockDim.z) +
         ((long long)threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
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
