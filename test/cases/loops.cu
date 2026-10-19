// Exercises: scalars carried across the iterations of a loop that declares locals (a recompute loop).
// The backward loop recomputes each iteration from the values these hold when it starts, so they must be
// checkpointed per iteration - unless they are pure accumulators, which need nothing.
__device__ void prodloop(const double *x, double *y, int n) {
  double p = 1.0;
  for (int j = 0; j < n; j++) {
    double t = x[j] + 0.5;
    p = p * t;
  }
  y[0] = p;
}
__device__ void sinloop(const double *x, double *y, int n) {
  double s = x[0];
  for (int j = 0; j < n; j++) {
    double t = 0.3 * x[j];
    s = sin(s) + t * s;
  }
  y[0] = s;
}
__device__ void mixloop(const double *x, double *y, int n) {
  double s = 0.0;
  double q = 1.0;
  for (int j = 0; j < n; j++) {
    double t = x[j] * x[j];
    s = s + t * q;
    q = q * 0.9 + x[0] * t;
  }
  y[0] = s + q;
}
