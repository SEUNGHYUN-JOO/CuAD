"""Host verification of CuAD's derivatives of the SRHD solver (examples/srhd).

1. Conservative-to-primitive kernel, both solvers: <P_b, dP> by central differences in a random
   direction (dU, dgam) against the CuAD adjoint <U_b, dU> + gam_b dgam (implicit adjoint of the root
   solve) and against the CuAD tangent <P_b, P_d> (forward mode differentiates the iterations).
2. Whole time integration: gradient of the final-time density misfit with respect to the seven Riemann
   parameters (rhoL, uL, pL, rhoR, uR, pR, gam) by the discrete adjoint, against central differences.
"""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
lib = ct.CDLL(str(out / 'srhd_adj_host.so'))
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
lib.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
lib.c2p_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
lib.c2p_tangent.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
lib.srhd_misfit.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_void_p]
lib.srhd_misfit.restype = ct.c_double
names = ['newton', 'bisection']
report = {'c2p': [], 'trajectory': []}

def cons(rho, u, p, G):
    W = np.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    return np.concatenate([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W])

rng = np.random.default_rng(7)
n = 64
for G in (4 / 3, 5 / 3):
    rho = rng.uniform(0.5, 2, n); u = rng.uniform(-3, 3, n); p = rng.uniform(0.01, 10, n)
    U = cons(rho, u, p, G); g = np.array([G])
    for solver in (0, 1):
        P = np.zeros(3 * n); lib.c2p_apply(solver, n, U, g, P)
        prim_err = float(np.max(np.abs(P - np.concatenate([rho, u, p])) / (1 + np.abs(np.concatenate([rho, u, p])))))
        worst_adj = worst_tan = 0.0
        for trial in range(5):
            dU = rng.normal(size=3 * n) * np.abs(U) * 1e-2; dg = np.array([rng.normal() * 1e-2])
            Pb = rng.normal(size=3 * n)
            def phi(t):
                Q = np.zeros(3 * n); lib.c2p_apply(solver, n, U + t * dU, g + t * dg, Q); return Q @ Pb
            h = 1e-5
            fd = (phi(h) - phi(-h)) / (2 * h)
            Ub = np.zeros(3 * n); gb = np.zeros(1)
            lib.c2p_adjoint(solver, n, U, g, Pb.copy(), Ub, gb)
            adj = Ub @ dU + gb[0] * dg[0]
            Pd = np.zeros(3 * n); lib.c2p_tangent(solver, n, U, dU, g, dg, Pd)
            tan = Pd @ Pb
            worst_adj = max(worst_adj, abs(adj - fd) / max(abs(fd), 1e-300))
            worst_tan = max(worst_tan, abs(tan - fd) / max(abs(fd), 1e-300))
        row = dict(gam=G, c2p=names[solver], primitive_recovery_error=prim_err,
                   adjoint_vs_fd=worst_adj, tangent_vs_fd=worst_tan)
        report['c2p'].append(row); print(json.dumps(row), flush=True)

# whole trajectory: Marti-Mueller problem 1 data, observations from perturbed parameters
q_true = np.array([10.0, 0.0, 13.33, 1.0, 0.0, 1e-2, 5 / 3])
q = np.array([9.0, 0.05, 12.0, 1.1, -0.05, 2e-2, 1.6])
lib.srhd_final.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr]
for n, solver in ((100, 0), (100, 1), (200, 0)):
    steps = n  # t = steps * 0.4 / n = 0.4
    Ptrue = np.zeros(3 * n)
    lib.srhd_final(q_true, n, steps, 0.4, solver, Ptrue)
    obs = np.ascontiguousarray(Ptrue[:n])
    grad = np.zeros(7)
    J = lib.srhd_misfit(q, n, steps, 0.4, solver, obs, grad.ctypes.data)
    fd = np.zeros(7)
    for k in range(7):
        h = 1e-6 * max(abs(q[k]), 1e-2)
        qp = q.copy(); qm = q.copy(); qp[k] += h; qm[k] -= h
        fd[k] = (lib.srhd_misfit(qp, n, steps, 0.4, solver, obs, None) -
                 lib.srhd_misfit(qm, n, steps, 0.4, solver, obs, None)) / (2 * h)
    rel = np.abs(grad - fd) / np.maximum(np.abs(fd), 1e-12 * np.abs(fd).max())
    row = dict(n=n, steps=steps, c2p=names[solver], J=J, adjoint=grad.tolist(), fd=fd.tolist(),
               max_rel_error=float(rel.max()))
    report['trajectory'].append(row); print(json.dumps(row), flush=True)
(out / 'adjoint-check.json').write_text(json.dumps(report, indent=2) + '\n')
