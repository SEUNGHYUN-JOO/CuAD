// Constructs that CuAD must reject with a located error (each one touches active data).
__global__ void k_while(const double *x, double *y, int n)
{
    int i = 0;
    double v = x[i];
    while (v < 10.0) {
        v = v * 2.0;
    }
    y[i] = v;
}

__global__ void k_break(const double *x, double *y, int n)
{
    double s = 0.0;
    for (int j = 0; j < n; j++) {
        if (x[j] < 0.0)
            break;
        s = s + x[j] * x[j];
    }
    y[0] = s;
}

__global__ void k_unknown(const double *x, double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = erf(x[i]);
}
