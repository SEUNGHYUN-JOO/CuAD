"""Exact solution of the 1-D special-relativistic Riemann problem (no tangential velocity), ideal gas.

Independent of the CUDA solver: rarefactions through the Riemann invariant
  J(+/-) = artanh(v) +/- (2/sqrt(G-1)) artanh(cs/sqrt(G-1))   (isentropic, p/rho^G const),
shocks through the Taub adiabat [h^2] = (h_b/rho_b + h_a/rho_a)[p] and the mass flux j
(Marti & Mueller, J. Fluid Mech. 258, 317, 1994). Every shock is checked against the
Rankine-Hugoniot conditions of (D, S, tau).
"""
import numpy as np
from scipy.optimize import brentq


class Gas:
    def __init__(self, G):
        self.G = G

    def h(self, rho, p):
        return 1 + self.G / (self.G - 1) * p / rho

    def cs(self, rho, p):
        return np.sqrt(self.G * p / (rho * self.h(rho, p)))

    def cons(self, rho, v, p):
        W = 1 / np.sqrt(1 - v * v); h = self.h(rho, p)
        return np.array([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W])

    def flux(self, rho, v, p):
        D, S, tau = self.cons(rho, v, p)
        return np.array([D * v, S * v + p, S - D * v])


def rarefaction(gas, a, p, side):
    """State behind a rarefaction of the given side (-1 left, +1 right) at pressure p < p_a."""
    G = gas.G; k = np.sqrt(G - 1)
    rho_a, v_a, p_a = a
    rho = rho_a * (p / p_a) ** (1 / G)
    J = np.arctanh(v_a) - side * 2 / k * np.arctanh(gas.cs(rho_a, p_a) / k)
    v = np.tanh(J + side * 2 / k * np.arctanh(gas.cs(rho, p) / k))
    return rho, v


def shock(gas, a, p, side):
    """State behind a shock of the given side at pressure p > p_a, and the shock speed."""
    G = gas.G
    rho_a, v_a, p_a = a
    h_a = gas.h(rho_a, p_a); W_a = 1 / np.sqrt(1 - v_a ** 2)
    def taub(h):
        rho = G * p / ((G - 1) * (h - 1))
        return h * h - h_a * h_a - (h / rho + h_a / rho_a) * (p - p_a)
    hi = h_a * 2
    while taub(hi) < 0:
        hi *= 2
    h = brentq(taub, 1 + 1e-15, hi, xtol=1e-15, rtol=1e-15)
    rho = G * p / ((G - 1) * (h - 1))
    j = side * np.sqrt(-(p - p_a) / (h / rho - h_a / rho_a))
    V = (rho_a ** 2 * W_a ** 2 * v_a + side * abs(j) * np.sqrt(j * j + rho_a ** 2)) / (rho_a ** 2 * W_a ** 2 + j * j)
    W_s = 1 / np.sqrt(1 - V * V)
    v = (h_a * W_a * v_a + W_s * (p - p_a) / j) / (h_a * W_a + (p - p_a) * (W_s * v_a / j + 1 / (rho_a * W_a)))
    # Rankine-Hugoniot check: [U] V = [F]
    res = (gas.cons(rho, v, p) - gas.cons(rho_a, v_a, p_a)) * V - (gas.flux(rho, v, p) - gas.flux(rho_a, v_a, p_a))
    scale = np.abs(gas.flux(rho_a, v_a, p_a)).max() + np.abs(gas.cons(rho, v, p)).max()
    assert np.abs(res).max() < 1e-10 * scale, ('Rankine-Hugoniot violated', res)
    return rho, v, V


def behind(gas, a, p, side):
    return rarefaction(gas, a, p, side) if p < a[2] else shock(gas, a, p, side)[:2]


def solve(G, L, R):
    """Return the star pressure and velocity and a sampler x/t -> (rho, v, p)."""
    gas = Gas(G)
    f = lambda p: behind(gas, L, p, -1)[1] - behind(gas, R, p, +1)[1]
    lo, hi = 1e-14 * max(L[2], R[2], 1e-300) + 1e-300, 10 * max(L[2], R[2])
    while f(hi) > 0:
        hi *= 10
    ps = brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)
    rhoL_s, vs = behind(gas, L, ps, -1)
    rhoR_s, _ = behind(gas, R, ps, +1)

    def wave(a, rho_s, side):
        """Head/tail speeds and a sampler of one wave."""
        if ps < a[2]:  # rarefaction
            lam = lambda rho, v, p: (v + side * gas.cs(rho, p)) / (1 + side * v * gas.cs(rho, p))
            head = lam(*a); tail = lam(rho_s, vs, ps)
            def fan(xi):
                g = lambda p: lam(*(rarefaction(gas, a, p, side)), p) - xi
                p = brentq(g, ps * (1 - 1e-12), a[2] * (1 + 1e-12), xtol=1e-15, rtol=1e-15)
                rho, v = rarefaction(gas, a, p, side)
                return rho, v, p
            return head, tail, fan
        V = shock(gas, a, ps, side)[2]
        return V, V, None

    hL, tL, fanL = wave(L, rhoL_s, -1)
    hR, tR, fanR = wave(R, rhoR_s, +1)

    def sample(xi):
        if xi < hL:
            return tuple(L)
        if xi < tL:
            return fanL(xi)
        if xi < vs:
            return rhoL_s, vs, ps
        if xi < tR:
            return rhoR_s, vs, ps
        if xi < hR:
            return fanR(xi)
        return tuple(R)
    return ps, vs, sample


if __name__ == '__main__':
    # Marti & Mueller (2003) problem 1: rho 10 | 1, p 13.33 | 1e-8 (0 in the review), G = 5/3
    ps, vs, s = solve(5 / 3, (10.0, 0.0, 13.33), (1.0, 0.0, 1e-8))
    print('problem 1: p* = %.6f, v* = %.6f, shell density = %.6f' % (ps, vs, s(vs + 1e-9)[0]))
    # problem 2: rho 1 | 1, p 1000 | 0.01, G = 5/3
    ps, vs, s = solve(5 / 3, (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01))
    print('problem 2: p* = %.6f, v* = %.6f, shell density = %.6f' % (ps, vs, s(vs + 1e-9)[0]))
