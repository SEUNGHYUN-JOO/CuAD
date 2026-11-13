// Exercises: local arrays whose values are carried from one loop iteration to the next.
// Each kernel maps x[0..n) to the scalar y[0]; the adjoint is checked against central differences.

// A fixed element updated in place every iteration (a scalar recurrence held in an array).
__device__ void recur_fixed(const double *x, double *y, int n) {
  double a[2];
  a[0] = x[0];
  for (int t = 1; t < n; ++t)
    a[0] = a[0] * x[t] + 0.1;
  y[0] = a[0];
}

// Each iteration reads the element the previous iteration wrote.
__device__ void recur_prefix(const double *x, double *y, int n) {
  double a[16];
  a[0] = x[0];
  for (int i = 1; i < n; ++i)
    a[i] = a[i - 1] * x[i] + sin(a[i - 1]);
  y[0] = a[n - 1];
}

// An element is read and then overwritten (a plain overwrite, no self reference).
__device__ void read_then_overwrite(const double *x, double *y, int n) {
  double a[2];
  a[0] = x[0];
  double s = 0;
  for (int i = 0; i < n; ++i) {
    s = s + a[0] * x[i];
    a[0] = x[i] * x[i];
  }
  y[0] = s;
}

// A recompute loop (it declares locals) carrying a two-element state.
__device__ void recompute_state(const double *x, double *y, int n) {
  double h[2];
  h[0] = x[0];
  h[1] = x[1];
  for (int t = 0; t < n; ++t) {
    double u = x[t];
    double h0 = tanh(h[0] * u + h[1]);
    h[1] = h[1] * 0.5 + h[0] * u;
    h[0] = h0;
  }
  y[0] = h[0] + h[1];
}

// Elementwise multi-layer LSTM in the layout of ADBench's lstm (2 layers, width 2).
__device__ void lstm_like(const double *x, double *y, int n) {
  double state[8], output[2];
  for (int i = 0; i < 8; ++i)
    state[i] = 0.1 * x[i % n];
  double total = 0;
  for (int t = 0; t < n - 1; ++t) {
    for (int i = 0; i < 2; ++i)
      output[i] = x[t] * (0.5 + 0.1 * i);
    for (int k = 0; k < 2; ++k) {
      for (int i = 0; i < 2; ++i) {
        double in = k == 0 ? output[i] : state[2 * (k - 1) * 2 + i];
        double hidden = state[2 * k * 2 + i];
        double forget = 1 / (1 + exp(-(in * x[(k + i) % n] + 0.1)));
        double ingate = 1 / (1 + exp(-(hidden * x[(k + 2 * i + 1) % n] - 0.2)));
        double outgate = 1 / (1 + exp(-(in * 0.7 + hidden * 0.3)));
        double change = tanh(hidden * x[(t + k + i) % n] + 0.05);
        state[(2 * k + 1) * 2 + i] = state[(2 * k + 1) * 2 + i] * forget + ingate * change;
        state[2 * k * 2 + i] = outgate * tanh(state[(2 * k + 1) * 2 + i]);
      }
    }
    for (int i = 0; i < 2; ++i)
      output[i] = state[2 * 2 + i] * 1.3 + 0.1;
    double sum = 0;
    for (int i = 0; i < 2; ++i)
      sum += exp(output[i]);
    double lse = log(sum + 2);
    for (int i = 0; i < 2; ++i)
      total += x[(t + 1) % n] * (output[i] - lse);
  }
  y[0] = total;
}

// A dense recurrence through a temporary array: every new element reads every old one.
__device__ void dense_rnn(const double *x, double *y, int n) {
  double h[4], nh[4];
  for (int j = 0; j < 4; ++j)
    h[j] = 0.2 * x[j % n];
  for (int t = 0; t < n; ++t) {
    for (int j = 0; j < 4; ++j) {
      double s = x[t];
      for (int k = 0; k < 4; ++k)
        s += 0.3 * x[(j * 4 + k) % n] * h[k];
      nh[j] = tanh(s);
    }
    for (int j = 0; j < 4; ++j)
      h[j] = nh[j];
  }
  y[0] = h[0] + 2 * h[1] + 3 * h[2] + 4 * h[3];
}

// Formerly rejected: a recurrence through an element written at a non-injective index.
__device__ void recurrent_state(const double *x, double *y, int n) {
  double state[2];
  state[0] = x[0];
  for (int t = 0; t < n - 1; ++t)
    state[0] = state[0] * x[t + 1];
  y[0] = state[0];
}
__device__ void modulo_index(const double *x, double *y, int n) {
  double state[2] = {1, 1};
  for (int i = 0; i < n; ++i)
    state[i % 2] *= x[i];
  y[0] = state[0] + state[1];
}
__device__ void quotient_index(const double *x, double *y, int n) {
  double state[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = 0; i < n; ++i)
    state[i / 2] *= x[i];
  y[0] = state[0] + 2 * state[1] + 3 * state[3];
}
__device__ void square_index(const double *x, double *y, int n) {
  double state[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = -2; i < 3; ++i)
    state[i * i] *= x[i + 2];
  y[0] = state[1] + state[4];
}

// Iteration-private updates (one index, injective in i): a single entry snapshot suffices.
__device__ void private_update(const double *x, double *y, int n) {
  double a[8];
  for (int i = 0; i < 8; ++i)
    a[i] = x[i % n];
  for (int i = 0; i < n; ++i) {
    double t = x[i] * 0.5;
    a[i] = a[i] * t + sin(a[i]);
    a[i] = a[i] * a[i];
  }
  y[0] = a[0] + a[3] + a[n - 1];
}

// The same with a loop-invariant offset in the index.
__device__ void offset_private(const double *x, double *y, int n) {
  double s[8];
  for (int i = 0; i < 8; ++i)
    s[i] = 0.1 * x[i % n];
  for (int k = 0; k < 2; ++k) {
    int base = (2 * k + 1) * 2;
    for (int i = 0; i < 2; ++i) {
      double g = tanh(s[base + i] * x[k + i]);
      s[base + i] = s[base + i] * g + x[i];
    }
  }
  y[0] = s[2] + s[3] + s[6] * s[7];
}

// A carried two-dimensional array.
__device__ void multidim_state(const double *x, double *y, int n) {
  double g[2][2];
  g[0][0] = x[0]; g[0][1] = x[1]; g[1][0] = x[2]; g[1][1] = 0.5;
  for (int t = 0; t < n; ++t) {
    g[0][1] = g[0][1] * x[t] + g[1][0];
    g[1][0] = sin(g[0][1]) * g[1][1];
    g[1][1] = g[1][1] + 0.1 * g[0][0] * x[t];
  }
  y[0] = g[0][1] + g[1][0] + g[1][1];
}

// Write-region checkpoints: index ranges the analysis must bound from above.
// A triangular inner range starting at the checkpointed loop's own counter.
__device__ void tri_region(const double *x, double *y, int n) {
  double a[8];
  for (int i = 0; i < 8; ++i)
    a[i] = 0.1 * (i + 1) * x[i % n];
  for (int t = 0; t < 4; ++t) {
    for (int j = t; j < 4; ++j)
      a[j] = a[j] * x[t] + 0.3 * sin(a[7 - j]);
    a[7 - t] = a[7 - t] + a[t] * x[t + 1];
  }
  y[0] = a[0] + a[2] * a[5] + a[7];
}
// A negative coefficient and a runtime offset held in a never-modified local.
__device__ void neg_offset(const double *x, double *y, int n) {
  double a[8];
  for (int i = 0; i < 8; ++i)
    a[i] = x[i % n];
  int off = n - 4;
  for (int t = 0; t < n; ++t) {
    for (int j = 0; j < 4; ++j) {
      double g = tanh(a[off + j] * x[t]);
      a[7 - 2 * j] = a[7 - 2 * j] * 0.9 + g * x[j];
    }
  }
  y[0] = a[1] + a[3] * a[5] + a[7] + a[6];
}
// A coefficient of run-time sign, and a write inside a branch.
__device__ void sign_branch(const double *x, double *y, int n) {
  double a[16];
  for (int i = 0; i < 16; ++i)
    a[i] = 0.2 * x[i % n];
  int s = n > 5 ? 2 : -2;
  int base = n > 5 ? 0 : 15;
  for (int t = 0; t < n; ++t) {
    for (int j = 0; j < 4; ++j) {
      if (x[t] > 0.5)
        a[base + s * j] = a[base + s * j] * x[t] + a[base + s * j + 1];
      else
        a[base + s * j + 1] = sin(a[base + s * j]) + x[t];
    }
  }
  y[0] = a[0] + a[1] + a[2] * a[4] + a[6] + a[7];
}
