// Minimal Clad check: Newton root with `break`, scalars only, and variants that isolate the SRHD kernel's features
// (a name declared both inside the loop and after it; fmax clamp).
#include <cmath>
#include <cstdio>
#include "clad/Differentiator/Differentiator.h"

double newton_plain(double a, double b) { // t^3 + a t - b = 0
  double t = 1.0;
  for (int it = 0; it < 100; ++it) {
    double d = (t * t * t + a * t - b) / (3 * t * t + a);
    t = t - d;
    if (fabs(d) <= 1e-15 * fabs(t)) break;
  }
  return t;
}
double newton_clamp(double a, double b) {
  double t = 1.0, lo = 1e-3;
  for (int it = 0; it < 100; ++it) {
    double d = (t * t * t + a * t - b) / (3 * t * t + a);
    t = fmax(t - d, lo);
    if (fabs(d) <= 1e-15 * fabs(t)) break;
  }
  return t;
}
double newton_shadow(double a, double b) {
  double t = 1.0;
  for (int it = 0; it < 100; ++it) {
    double v = t * t;
    double d = (v * t + a * t - b) / (3 * v + a);
    t = t - d;
    if (fabs(d) <= 1e-15 * fabs(t)) break;
  }
  double v = t * t;
  return v;
}
double newton_nobreak(double a, double b) {
  double t = 1.0;
  for (int it = 0; it < 40; ++it) {
    double d = (t * t * t + a * t - b) / (3 * t * t + a);
    t = t - d;
  }
  return t;
}

double bisect_min(double a, double b) {
  double lo = 0.0, hi = 4.0;
  for (int it = 0; it < 200; ++it) {
    double m = 0.5 * (lo + hi);
    if (m * m * m + a * m - b > 0) hi = m;
    else lo = m;
    if (hi - lo <= 1e-15 * hi) break;
  }
  return 0.5 * (lo + hi);
}

template <class F> void check(const char *name, F f, double (*p)(double, double), double scale_out) {
  const double a = 2, b = 5;
  double ga = 0, gb = 0;
  f.execute(a, b, &ga, &gb);
  double h = 1e-6;
  double fa = (p(a + h, b) - p(a - h, b)) / (2 * h), fb = (p(a, b + h) - p(a, b - h)) / (2 * h);
  std::printf("%-15s clad (%.10g, %.10g)  fd (%.10g, %.10g)\n", name, ga, gb, fa, fb);
}

int main() {
  check("plain", clad::gradient(newton_plain), newton_plain, 1);
  check("clamp", clad::gradient(newton_clamp), newton_clamp, 1);
  check("shadow", clad::gradient(newton_shadow), newton_shadow, 1);
  check("no-break 40", clad::gradient(newton_nobreak), newton_nobreak, 1);
  check("bisection", clad::gradient(bisect_min), bisect_min, 1);
  return 0;
}
