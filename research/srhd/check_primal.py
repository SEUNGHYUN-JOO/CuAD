"""Primal check of examples/srhd/srhd.cu (host build) against the exact Riemann solution.
Marti & Mueller problems 1 and 2, t = 0.4, both C2P solvers, L1 density error versus resolution."""
import ctypes as ct
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exact_riemann import solve

out = Path(__file__).parent / 'out'
lib = ct.CDLL(str(out / 'srhd_host.so'))
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
lib.srhd_run.argtypes = [ct.c_int, ct.c_int, ct.c_double, ct.c_double] + [ct.c_double] * 6 + [ct.c_int, ptr]
lib.srhd_run.restype = None

problems = {'MM1': (5 / 3, (10.0, 0.0, 13.33), (1.0, 0.0, 1e-8)),
            'MM2': (5 / 3, (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01))}
cfl, T = 0.4, 0.4
rows = []
for name, (G, L, R) in problems.items():
    _, _, sample = solve(G, L, R)
    for n in (200, 400, 800, 1600):
        steps = int(round(T / (cfl / n)))
        x = (np.arange(n) + 0.5) / n
        exact = np.array([sample((xi - 0.5) / T) for xi in x])
        for solver in (0, 1):
            P = np.zeros(3 * n)
            lib.srhd_run(n, steps, cfl, G, L[0], 0.0, L[2], R[0], 0.0, R[2], solver, P)
            rho, u, p = P[:n], P[n:2 * n], P[2 * n:]
            v = u / np.sqrt(1 + u * u)
            row = dict(problem=name, n=n, c2p=['newton', 'bisection'][solver], finite=bool(np.isfinite(P).all()),
                       l1_rho=float(np.mean(np.abs(rho - exact[:, 0]))),
                       l1_v=float(np.mean(np.abs(v - exact[:, 1]))),
                       l1_p=float(np.mean(np.abs(p - exact[:, 2]))))
            rows.append(row); print(json.dumps(row), flush=True)
        if n == 1600:
            np.savez(out / f'primal-{name}-1600.npz', x=x, exact=exact, P=P)
for name in problems:
    for c in ('newton', 'bisection'):
        r = [row['l1_rho'] for row in rows if row['problem'] == name and row['c2p'] == c]
        print(name, c, 'L1(rho) orders:', np.round(np.log2(np.array(r[:-1]) / np.array(r[1:])), 2))
(out / 'primal.json').write_text(json.dumps(rows, indent=2) + '\n')
