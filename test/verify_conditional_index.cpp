#include <cmath>
#include <cstdio>

#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;

#include "cuad_rt.h"
#include "cases/conditional_index.cu"
#include "out/conditional_index_b.cuh"

int main() {
  double x[2] = {0.3, 0.7}, y[2] = {0, 0};
  double xb[2] = {0, 0}, yb[2] = {1, 1};
  for (unsigned i = 0; i < 2; ++i) {
    threadIdx.x = i;
    conditional_index_b(x, xb, y, yb, 2);
  }
  double want0 = 2.0 + x[1], want1 = x[0];
  bool ok = std::fabs(xb[0] - want0) < 1e-12 &&
            std::fabs(xb[1] - want1) < 1e-12;
  std::printf("conditional index: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
