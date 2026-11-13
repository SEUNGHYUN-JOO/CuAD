// The inactive branch contains an index outside the input allocation at i=0.
__global__ void conditional_index(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = i == 0 ? 2.0 * x[i] : x[i - 1] * x[i];
}
