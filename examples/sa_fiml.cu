// FIML-augmented Spalart-Allmaras turbulence source term (self-contained).
// The FIML field beta multiplies the SA production; field inversion recovers
// beta from data via the adjoint dsrc/dbeta, which cuad generates here.
//   forward:  sa_source_d  (tangent)
//   reverse:  sa_source_b  (adjoint) -> dJ/dbeta for field inversion
#define SA_CB1 0.1355f
#define SA_CB2 0.622f
#define SA_SIG 0.66666667f
#define SA_KAP 0.41f
#define SA_CW2 0.3f
#define SA_CW3 2.0f
#define SA_CV1 7.1f

// SA source per cell: beta * Production - Destruction.
__device__ float sa_source(float nut, float d, float Om, float nu, float beta) {
    float chi = nut / nu;
    float chi3 = chi * chi * chi;
    float fv1 = chi3 / (chi3 + SA_CV1 * SA_CV1 * SA_CV1);
    float fv2 = 1.f - chi / (1.f + chi * fv1);
    float d2 = d * d;
    float k2d2 = SA_KAP * SA_KAP * d2;
    float Stil = Om + nut / k2d2 * fv2;
    Stil = fmaxf(Stil, 1e-8f);
    float r = fminf(nut / (Stil * k2d2), 10.f);
    float g = r + SA_CW2 * (powf(r, 6.f) - r);
    float cw3_6 = powf(SA_CW3, 6.f);
    float fw = g * powf((1.f + cw3_6) / (powf(g, 6.f) + cw3_6), 1.f / 6.f);
    float cw1 = SA_CB1 / (SA_KAP * SA_KAP) + (1.f + SA_CB2) / SA_SIG;
    float P = SA_CB1 * Stil * nut;              // production
    float D = cw1 * fw * (nut / d) * (nut / d); // destruction
    return beta * P - D;
}

__global__ void saFimlKernel(const float *nut, const float *d, const float *Om,
                             const float *beta, float nu, float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        src[i] = sa_source(nut[i], d[i], Om[i], nu, beta[i]);
    }
}
