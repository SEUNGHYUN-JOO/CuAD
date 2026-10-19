// Exercises: value-returning device functions nested in an expression (both modes),
// and the math rules for copysign, hypot, cbrt and floor.
__device__ double speed(double r, double m)
{
    double u = m / r;
    return fabs(u) + sqrt(1.4 / r);
}

__global__ void nested(const double *x, double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = 0.5 * (speed(x[i], x[i + n]) + speed(x[i + n], x[i])) * x[i];
        y[i] = s;
    }
}

__global__ void mathfn(const double *x, double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double a = x[i];
        double b = x[i + n];
        double c = copysign(a * a, b - 0.3) + hypot(a, b) + cbrt(a + 2.0) + floor(4.0 * a) * a;
        y[i] = c;
    }
}
