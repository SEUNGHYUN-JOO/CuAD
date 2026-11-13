// Host entry points for the four C2P root finders (Newton and bisection from examples/srhd/srhd.cu, Illinois and
// Brent from c2p_variants.cu): primal, CuAD adjoint, and (built with -DWITH_ENZYME and the Enzyme plugin) Enzyme's
// reverse mode of the same source. solver: 0 Newton, 1 bisection, 2 Illinois, 3 Brent.
#include <cmath>
#include <cstdlib>
#include <vector>
#define __global__
#define __device__
#define __host__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx;
static struct { unsigned x = 1, y = 1, z = 1; } blockDim, gridDim;
#ifndef WITH_ENZYME
#define CUAD_TAPE_SIZE 256
#include "cuad_rt.h"
#endif
#include "srhd.cu"
#include "c2p_variants.cu"
#ifndef WITH_ENZYME
#include "srhd_b.cuh"
#include "c2p_variants_b.cuh"
#endif

static void prim(int s, const double *U, double *P, const double *g, int n) {
  if (s == 0) c2p_newton(U, P, g, n);
  else if (s == 1) c2p_bisect(U, P, g, n);
  else if (s == 2) c2p_illinois(U, P, g, n);
  else c2p_brent(U, P, g, n);
}
extern "C" void c2pv_apply(int s, int n, const double *U, const double *g, double *P) { prim(s, U, P, g, n); }

#ifndef WITH_ENZYME
extern "C" void c2pv_adjoint(int s, int n, const double *U, const double *g, const double *Pb_in, double *Ub, double *gb) {
  std::vector<double> P(3 * n), Pb(Pb_in, Pb_in + 3 * n);
  if (s == 0) c2p_newton_b(U, Ub, P.data(), Pb.data(), g, gb, n);
  else if (s == 1) c2p_bisect_b(U, Ub, P.data(), Pb.data(), g, gb, n);
  else if (s == 2) c2p_illinois_b(U, Ub, P.data(), Pb.data(), g, gb, n);
  else c2p_brent_b(U, Ub, P.data(), Pb.data(), g, gb, n);
}
#else
extern int enzyme_dup, enzyme_const;
template <class... A> void __enzyme_autodiff(void *, A...);
static double dotp(int s, int n, const double *U, const double *g, const double *Pb) {
  double *P = (double *)std::malloc(sizeof(double) * 3 * n);
  prim(s, U, P, g, n);
  double r = 0;
  for (int k = 0; k < 3 * n; ++k) r += P[k] * Pb[k];
  std::free(P);
  return r;
}
extern "C" void c2pv_adjoint(int s, int n, const double *U, const double *g, const double *Pb, double *Ub, double *gb) {
  __enzyme_autodiff((void *)dotp, enzyme_const, s, enzyme_const, n, enzyme_dup, U, Ub, enzyme_dup, g, gb, enzyme_const, Pb);
}
#endif
