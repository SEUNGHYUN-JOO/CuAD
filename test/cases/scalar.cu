// A simple elementwise kernel: y = f(x) with f = sin(a*x) + x*x
__global__ void apply_f(const double *x, double *y, double a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double xi = x[i];
        double t = sin(a * xi) + xi * xi;
        y[i] = t;
    }
}
