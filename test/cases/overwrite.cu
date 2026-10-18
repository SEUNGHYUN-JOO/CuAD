// Exercises the tape: a local reassigned several times (true overwrites) and a
// *= compound assignment — none of this is single-assignment.
__global__ void ow(const double *x, double *y, double a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double t = x[i];
        t = t * t + a;      // overwrite t (old t needed by its own adjoint)
        t *= a;             // *= : adjoint needs the pre-multiply value
        double s = t + x[i];
        t = s * s;          // overwrite again, using s
        y[i] = t + a * x[i];
    }
}
