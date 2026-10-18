// Exercises: a constant local (should get NO _d), a local array companion,
// and a small loop accumulating into a scalar.
__global__ void weighted(const double *x, double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double w[3];
        w[0] = 0.25;
        w[1] = 0.5;
        w[2] = 0.25;
        double h = 2.0;          // constant local: no derivative
        double acc = 0.0;
        for (int k = 0; k < 3; k++) {
            acc += w[k] * x[i] * x[i];
        }
        y[i] = acc / h;
    }
}
