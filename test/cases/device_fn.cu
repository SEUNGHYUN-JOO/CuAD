// Exercises a value-returning __device__ helper called from a kernel.
__device__ double activation(double z, double bias)
{
    double s = 1.0 / (1.0 + exp(-(z + bias)));
    return s;
}

__global__ void layer(const double *x, double *y, double w, double b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double z = w * x[i];
        double a = activation(z, b);
        y[i] = a * a;
    }
}
