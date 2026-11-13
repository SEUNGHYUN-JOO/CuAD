"""Robustness of the C2P derivatives over the state space, against an independent high-precision reference.

States: Lorentz factor W in [1, 1e3] (v along x) and specific pressure p/rho in [1e-8, 1e2], rho = 1, Gamma = 5/3.
Reference: the primitive-to-conservative map U(P) and its Jacobian A = dU/dP evaluated in 50-digit arithmetic
(mpmath); the exact C2P derivative is dP/dU = A^{-1} (implicit function theorem at the exact solution).
Tested: CuAD reverse mode of examples/srhd/srhd.cu (implicit adjoint of the Newton and of the bisection C2P),
three adjoint sweeps (seeds on rho, u, p) give the full 3x3 dP/dU. Also reported: primitive recovery error and
cond(A). Output: out/c2p-robustness.json (+ .npz for the figure).
"""
import ctypes as ct
import json
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 50
out = Path(__file__).parent / 'out'
lib = ct.CDLL(str(out / 'srhd_adj_host.so'))
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
lib.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
lib.c2p_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
G = mp.mpf(5) / 3

def p2c(rho, u, p):
    W = mp.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    return [rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W]

Ws = np.logspace(0, 3, 13)
Ws[0] = 1.0
thetas = np.logspace(-8, 2, 11)
states, ref, conds = [], [], []
for W in Ws:
    for th in thetas:
        rho = mp.mpf(1); u = mp.sqrt(mp.mpf(W) ** 2 - 1); p = rho * mp.mpf(th)
        x = [rho, u, p]
        U = p2c(*x)
        A = mp.matrix(3, 3)
        for j in range(3):  # exact Jacobian columns by high-precision complex-free central differences
            hstep = mp.mpf('1e-20') * (1 + abs(x[j]))
            xp = list(x); xm = list(x); xp[j] += hstep; xm[j] -= hstep
            Up, Um = p2c(*xp), p2c(*xm)
            for i in range(3):
                A[i, j] = (Up[i] - Um[i]) / (2 * hstep)
        Ainv = A ** -1
        states.append((float(U[0]), float(U[1]), float(U[2]), float(rho), float(u), float(p)))
        ref.append([[float(Ainv[i, j]) for j in range(3)] for i in range(3)])
        conds.append(float(mp.norm(A, 1) * mp.norm(Ainv, 1)))
S = np.array(states); R = np.array(ref); n = len(S)
U = np.concatenate([S[:, 0], S[:, 1], S[:, 2]])
P_exact = np.concatenate([S[:, 3], S[:, 4], S[:, 5]])
g = np.array([5 / 3])
res = {}
for solver, name in ((0, 'newton'), (1, 'bisection')):
    P = np.zeros(3 * n); lib.c2p_apply(solver, n, U, g, P)
    rec = np.max(np.abs(P.reshape(3, n) - P_exact.reshape(3, n)) / np.maximum(np.abs(P_exact.reshape(3, n)), 1e-300), axis=0)
    J = np.zeros((n, 3, 3))
    for i in range(3):
        Pb = np.zeros(3 * n); Pb[i * n:(i + 1) * n] = 1.0
        Ub = np.zeros(3 * n); gb = np.zeros(1)
        lib.c2p_adjoint(solver, n, U, g, Pb, Ub, gb)
        J[:, i, :] = Ub.reshape(3, n).T  # row i: d P_i / d U
    err = np.linalg.norm(J - R, axis=(1, 2)) / np.linalg.norm(R, axis=(1, 2))
    res[name] = dict(recovery_error=rec.reshape(len(Ws), len(thetas)).tolist(),
                     jacobian_error=err.reshape(len(Ws), len(thetas)).tolist(),
                     finite=bool(np.isfinite(J).all()))
    print(name, 'max recovery err %.2e' % np.nanmax(rec), 'max jacobian err %.2e' % np.nanmax(err),
          'median %.2e' % np.nanmedian(err), 'nonfinite', int((~np.isfinite(err)).sum()))
cond = np.array(conds).reshape(len(Ws), len(thetas))
print('cond(A) range %.1e .. %.1e' % (cond.min(), cond.max()))
for name in res:
    e = np.array(res[name]['jacobian_error']).ravel(); c = cond.ravel()
    ok = np.isfinite(e) & (e > 0)
    print(name, 'error / (cond * eps): median %.2e max %.2e' % (np.median(e[ok] / (c[ok] * 2.2e-16)), np.max(e[ok] / (c[ok] * 2.2e-16))))
(out / 'c2p-robustness.json').write_text(json.dumps(dict(W=Ws.tolist(), p_over_rho=thetas.tolist(), cond=cond.tolist(),
                                                        **res), indent=2) + '\n')
np.savez(out / 'c2p-robustness.npz', W=Ws, theta=thetas, cond=cond,
         err_newton=np.array(res['newton']['jacobian_error']), err_bisect=np.array(res['bisection']['jacobian_error']),
         rec_newton=np.array(res['newton']['recovery_error']), rec_bisect=np.array(res['bisection']['recovery_error']))
