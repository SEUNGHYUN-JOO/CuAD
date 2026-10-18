// Exercises: non-active struct fields (int index members, like CuDAFLOW's B.ncT),
// and atomicAdd scatter (reverse becomes a gather; parameter adjoints accumulate
// atomically).
struct Idx { int base, stride; };

__global__ void scatter(const double *src, double *dst, Idx m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = m.base + i % m.stride;   // struct fields are int -> inactive
        double s = src[i];
        double v = s * s + 2.0 * s;
        atomicAdd(&dst[j], v);
    }
}
