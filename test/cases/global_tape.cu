// Exercises: nested calls between functions that use the tape. Under -DCUAD_GLOBAL_TAPE every
// thread has one tape column, so a callee's frame must start above its caller's entries and each
// function must leave the published top as it found it. Each kernel maps x[0..n) to y[0].

// A value-returning helper with its own tape (a scalar carried through a recompute loop).
__device__ double carry(const double *x, int n) {
  double h = x[0];
  for (int t = 1; t < n; ++t) {
    double u = sin(h) * x[t];
    h = u + 0.5 * h;
  }
  return h;
}

// A taped function that calls a taped function (three frames deep from a kernel).
__device__ double twice(const double *x, int n) {
  double s = x[0];
  for (int k = 0; k < 2; ++k) {
    double v = carry(x, n);
    s = s * v + 0.1;
  }
  return s;
}

// No tape of its own: two taped calls, so a callee that did not restore the top would let the
// second call start higher than the first.
__device__ double plain(const double *x, int n) {
  return twice(x, n) * carry(x + 1, n - 1) + x[1];
}

// The caller holds checkpoints of `s` on the tape while the callee pushes and pops above them.
__device__ void nested_taped(const double *x, double *y, int n) {
  double s = x[0];
  for (int k = 0; k < 3; ++k) {
    double v = carry(x, n);
    s = s * v + cos(s);
  }
  y[0] = s;
}

// A carried array in the caller, taped calls reached through a function without a tape.
__device__ void through_plain(const double *x, double *y, int n) {
  double a[4];
  a[0] = x[0];
  for (int k = 1; k < 4; ++k) {
    double m = plain(x, n);
    a[k] = a[k - 1] * m + 0.25 * x[k];
  }
  y[0] = a[3];
}
