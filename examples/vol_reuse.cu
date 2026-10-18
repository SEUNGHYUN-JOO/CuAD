// Real CuDAFLOW geometry: cell volume from node coordinates (volKernel + faceI/J/K
// + faceVec from src/geom.cu & metrics.cuh). Heavy: a grid-stride loop, three
// levels of __device__ calls, and helpers that WRITE output arrays S[3]/C[3].
// (volKernel reuses S/C across 6 faces in CuDAFLOW; here each face uses its own
// array so no reused-temporary tape is needed — the math is identical.)
#define NG 3

struct Blk {
    int ni, nj, nk, nit, njt, nkt;
    int niN, njN, nkN;
    long ncT, coff, noff;
};
__host__ __device__ inline long nidx(const Blk &B, int i, int j, int k) {
    return ((long)k * B.njN + j) * B.niN + i;
}
__host__ __device__ inline long cidx(const Blk &B, int i, int j, int k) {
    return ((long)(k + NG) * B.njt + (j + NG)) * B.nit + (i + NG);
}

// face area-vector S (½ diagonal cross product) and centroid C, from 4 corners
__host__ __device__ inline void
faceVec(float ax, float ay, float az, float bx, float by, float bz, float cx,
        float cy, float cz, float dx, float dy, float dz, float S[3],
        float C[3]) {
    float d1x = cx - ax, d1y = cy - ay, d1z = cz - az;
    float d2x = dx - bx, d2y = dy - by, d2z = dz - bz;
    S[0] = 0.5f * (d1y * d2z - d1z * d2y);
    S[1] = 0.5f * (d1z * d2x - d1x * d2z);
    S[2] = 0.5f * (d1x * d2y - d1y * d2x);
    C[0] = 0.25f * (ax + bx + cx + dx);
    C[1] = 0.25f * (ay + by + cy + dy);
    C[2] = 0.25f * (az + bz + cz + dz);
}
__host__ __device__ inline void faceI(const Blk &B, const float *x,
                                      const float *y, const float *z, int i,
                                      int j, int k, float S[3], float C[3]) {
    long a = B.noff + nidx(B, i, j, k), b = B.noff + nidx(B, i, j + 1, k);
    long c = B.noff + nidx(B, i, j + 1, k + 1), d = B.noff + nidx(B, i, j, k + 1);
    faceVec(x[a], y[a], z[a], x[b], y[b], z[b], x[c], y[c], z[c], x[d], y[d],
            z[d], S, C);
}
__host__ __device__ inline void faceJ(const Blk &B, const float *x,
                                      const float *y, const float *z, int i,
                                      int j, int k, float S[3], float C[3]) {
    long a = B.noff + nidx(B, i, j, k), b = B.noff + nidx(B, i, j, k + 1);
    long c = B.noff + nidx(B, i + 1, j, k + 1), d = B.noff + nidx(B, i + 1, j, k);
    faceVec(x[a], y[a], z[a], x[b], y[b], z[b], x[c], y[c], z[c], x[d], y[d],
            z[d], S, C);
}
__host__ __device__ inline void faceK(const Blk &B, const float *x,
                                      const float *y, const float *z, int i,
                                      int j, int k, float S[3], float C[3]) {
    long a = B.noff + nidx(B, i, j, k), b = B.noff + nidx(B, i + 1, j, k);
    long c = B.noff + nidx(B, i + 1, j + 1, k), d = B.noff + nidx(B, i, j + 1, k);
    faceVec(x[a], y[a], z[a], x[b], y[b], z[b], x[c], y[c], z[c], x[d], y[d],
            z[d], S, C);
}

__global__ void volKernel(Blk B, const float *x, const float *y, const float *z,
                          float *vol) {
    long n = (long)B.ni * B.nj * B.nk;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < n;
         t += (long)gridDim.x * blockDim.x) {
        int i = t % B.ni, j = (t / B.ni) % B.nj, k = t / ((long)B.ni * B.nj);
        float S[3], C[3], v = 0.f;
        faceI(B, x, y, z, i, j, k, S, C);
        v -= S[0] * C[0] + S[1] * C[1] + S[2] * C[2];
        faceI(B, x, y, z, i + 1, j, k, S, C);
        v += S[0] * C[0] + S[1] * C[1] + S[2] * C[2];
        faceJ(B, x, y, z, i, j, k, S, C);
        v -= S[0] * C[0] + S[1] * C[1] + S[2] * C[2];
        faceJ(B, x, y, z, i, j + 1, k, S, C);
        v += S[0] * C[0] + S[1] * C[1] + S[2] * C[2];
        faceK(B, x, y, z, i, j, k, S, C);
        v -= S[0] * C[0] + S[1] * C[1] + S[2] * C[2];
        faceK(B, x, y, z, i, j, k + 1, S, C);
        v += S[0] * C[0] + S[1] * C[1] + S[2] * C[2];
        vol[B.coff + cidx(B, i, j, k)] = v / 3.f;
    }
}
