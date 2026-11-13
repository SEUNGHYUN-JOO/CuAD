"""Four C2P root finders (Newton, bisection, Illinois, Brent) on the state grid of c2p_robustness.py (W in [1, 1e3],
p/rho in [1e-8, 1e2]): the full 3x3 Jacobian dP/dU from CuAD's adjoint and from Enzyme's, against the exact Jacobian
A^{-1} computed in 50-digit arithmetic. Same solution, four implementations: the implicit derivative must not depend
on the solver; differentiating the iterations may."""
import ctypes as ct
import json
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 50
out = Path(__file__).parent / 'out'
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
libs = {}
for tool, so in (('cuad', 'c2p_variants_cuad.so'), ('enzyme', 'c2p_variants_enzyme.so')):
    l = ct.CDLL(str(out / so))
    l.c2pv_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
    l.c2pv_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
    libs[tool] = l
G = mp.mpf(5) / 3

def p2c(rho, u, p):
    W = mp.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    return [rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W]

Ws = np.logspace(0, 3, 13); Ws[0] = 1.0
thetas = np.logspace(-8, 2, 11)
S, R = [], []
for W in Ws:
    for th in thetas:
        x = [mp.mpf(1), mp.sqrt(mp.mpf(W) ** 2 - 1), mp.mpf(th)]
        U = p2c(*x); A = mp.matrix(3, 3)
        for j in range(3):
            h = mp.mpf('1e-20') * (1 + abs(x[j])); xp = list(x); xm = list(x); xp[j] += h; xm[j] -= h
            Up, Um = p2c(*xp), p2c(*xm)
            for i in range(3):
                A[i, j] = (Up[i] - Um[i]) / (2 * h)
        Ai = A ** -1
        S.append([float(U[0]), float(U[1]), float(U[2]), float(x[0]), float(x[1]), float(x[2])])
        R.append([[float(Ai[i, j]) for j in range(3)] for i in range(3)])
S = np.array(S); R = np.array(R); n = len(S)
U = np.concatenate([S[:, 0], S[:, 1], S[:, 2]]); g = np.array([5 / 3])
rows = []
for s, name in enumerate(['newton', 'bisection', 'illinois', 'brent']):
    for tool, l in libs.items():
        P = np.zeros(3 * n); l.c2pv_apply(s, n, U, g, P)
        # relative error per primitive as in c2p_robustness.py (a velocity that is exactly zero is not a 0/0)
        rec = np.max(np.abs(P.reshape(3, n).T - S[:, 3:]) / np.maximum(np.abs(S[:, 3:]), 1e-300), axis=1)
        J = np.zeros((n, 3, 3))
        for i in range(3):
            Pb = np.zeros(3 * n); Pb[i * n:(i + 1) * n] = 1.0
            Ub = np.zeros(3 * n); gb = np.zeros(1)
            l.c2pv_adjoint(s, n, U, g, Pb, Ub, gb)
            J[:, i, :] = Ub.reshape(3, n).T
        err = np.linalg.norm(J - R, axis=(1, 2)) / np.linalg.norm(R, axis=(1, 2))
        ok = rec < 1e-8  # states where the primal itself is accurate
        row = dict(solver=name, tool=tool, states=n, primal_accurate_states=int(ok.sum()),
                   jacobian_error_median=float(np.median(err[ok])), jacobian_error_max=float(np.max(err[ok])),
                   states_with_error_above_1e_6=int(np.sum(err[ok] > 1e-6)))
        rows.append(row); print(json.dumps(row), flush=True)
(out / 'c2p-variants.json').write_text(json.dumps(rows, indent=2) + '\n')
