"""Clad v2.5 versus CuAD and Enzyme on the unchanged SRHD C2P kernels (host, 64 cells): dot test of the
gradient of sum P.Pb with respect to (U, gam) against central differences, same states as check_enzyme_host.py."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
clad = ct.CDLL(str(out / 'clad_c2p_host.so'))
clad.clad_c2p_adjoint.argtypes = [ct.c_int, ptr, ptr, ptr, ptr, ptr]
enz = ct.CDLL(str(out / 'enzyme_srhd_host.so'))
enz.enzyme_c2p_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
cuad = ct.CDLL(str(out / 'srhd_adj_host.so'))
cuad.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
cuad.c2p_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]

def cons(rho, u, p, G):
    W = np.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    return np.concatenate([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W])

rng = np.random.default_rng(7)
n = 64
rows = []
for G in (4 / 3, 5 / 3):
    rho = rng.uniform(0.5, 2, n); u = rng.uniform(-3, 3, n); p = rng.uniform(0.01, 10, n)
    U = cons(rho, u, p, G); g = np.array([G])
    for solver in (0, 1):
        err = {'clad': 0.0, 'enzyme': 0.0, 'cuad': 0.0}
        for trial in range(5):
            dU = rng.normal(size=3 * n) * np.abs(U) * 1e-2; dg = rng.normal() * 1e-2
            Pb = rng.normal(size=3 * n)
            def phi(t):
                Q = np.zeros(3 * n); cuad.c2p_apply(solver, n, U + t * dU, g + t * dg, Q); return Q @ Pb
            fd = (phi(1e-5) - phi(-1e-5)) / 2e-5
            for tool in err:
                Ub = np.zeros(3 * n); gb = np.zeros(1)
                if tool == 'clad': clad.clad_c2p_adjoint(solver, U, g, Pb, Ub, gb)
                elif tool == 'enzyme': enz.enzyme_c2p_adjoint(solver, n, U, g, Pb, Ub, gb)
                else: cuad.c2p_adjoint(solver, n, U, g, Pb.copy(), Ub, gb)
                err[tool] = max(err[tool], abs(Ub @ dU + gb[0] * dg - fd) / abs(fd))
        row = dict(gam=G, c2p=['newton', 'bisection'][solver], **{t + '_vs_fd': e for t, e in err.items()})
        rows.append(row); print(json.dumps(row), flush=True)
(out / 'clad-host-check.json').write_text(json.dumps(rows, indent=2) + '\n')
