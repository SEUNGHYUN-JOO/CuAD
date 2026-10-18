// SAXPY-ish with a reduction-free body and a loop, to exercise for/if/compound.
__global__ void poly(const double *x, double *y, double c0, double c1, double c2,
                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double xi = x[i];
        double acc = c0;
        acc += c1 * xi;
        acc += c2 * xi * xi;
        y[i] = sqrt(acc) / (1.0 + xi);
    }
}
