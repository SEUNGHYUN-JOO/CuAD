// Real CuDAFLOW kernel: conservative -> primitive (c2pKernel from src/flux.cu),
// made self-contained for cuad. Struct field indices, a grid-stride loop, a
// value-returning __device__ helper (th_p), fmaxf, and float math.
#define NG 3
#define NV 6
#define GM1 0.4f

struct Blk {
    int ni, nj, nk, nit, njt, nkt;
    long ncT, coff;
    long noff;
    int dev, moving;
};
struct Thermo {
    int mode;
    float R, cvtr, thetaV;
};

// pressure from internal-energy density (calorically-perfect gas: p = (g-1)*ei)
__device__ float th_p(float rho, float ei, const Thermo &th) {
    return GM1 * ei;
}

__global__ void c2pKernel(Blk B, const Blk *__restrict__ bb,
                          const float *__restrict__ q, float *__restrict__ w,
                          Thermo th) {
    if (bb) B = bb[blockIdx.y];
    long o = (long)NV * B.coff;
    for (long c = blockIdx.x * (long)blockDim.x + threadIdx.x; c < B.ncT;
         c += (long)gridDim.x * blockDim.x) {
        float rho = fmaxf(q[o + c], 1e-8f);
        float ri = 1.f / rho;
        float u = q[o + B.ncT + c] * ri, v = q[o + 2 * B.ncT + c] * ri,
              ww = q[o + 3 * B.ncT + c] * ri;
        float p = th_p(rho, q[o + 4 * B.ncT + c] - 0.5f * rho * (u * u + v * v + ww * ww), th);
        w[o + c] = rho;
        w[o + B.ncT + c] = u;
        w[o + 2 * B.ncT + c] = v;
        w[o + 3 * B.ncT + c] = ww;
        w[o + 4 * B.ncT + c] = fmaxf(p, 1e-8f);
        w[o + 5 * B.ncT + c] = q[o + 5 * B.ncT + c] * ri;
    }
}
