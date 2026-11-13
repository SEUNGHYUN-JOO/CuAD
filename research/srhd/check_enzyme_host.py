"""Enzyme versus CuAD on the SRHD source, host builds: C2P kernel (dot test against central
differences) and the gradient through the whole time integration. Same inputs as check_adjoint.py."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
enz = ct.CDLL(str(out / 'enzyme_srhd_host.so'))
cuad = ct.CDLL(str(out / 'srhd_adj_host.so'))
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
enz.enzyme_c2p_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
enz.enzyme_misfit_grad.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ptr]
cuad.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
cuad.c2p_adjoint.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr, ptr, ptr]
cuad.srhd_misfit.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_void_p]
cuad.srhd_misfit.restype = ct.c_double
cuad.srhd_final.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr]
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
        we = wc = 0.0
        for trial in range(5):
            dU = rng.normal(size=3 * n) * np.abs(U) * 1e-2; dg = np.array([rng.normal() * 1e-2])
            Pb = rng.normal(size=3 * n)
            def phi(t):
                Q = np.zeros(3 * n); cuad.c2p_apply(solver, n, U + t * dU, g + t * dg, Q); return Q @ Pb
            fd = (phi(1e-5) - phi(-1e-5)) / 2e-5
            Ub = np.zeros(3 * n); gb = np.zeros(1)
            enz.enzyme_c2p_adjoint(solver, n, U, g, Pb, Ub, gb)
            we = max(we, abs(Ub @ dU + gb[0] * dg[0] - fd) / abs(fd))
            Ub = np.zeros(3 * n); gb = np.zeros(1)
            cuad.c2p_adjoint(solver, n, U, g, Pb.copy(), Ub, gb)
            wc = max(wc, abs(Ub @ dU + gb[0] * dg[0] - fd) / abs(fd))
        row = dict(gam=G, c2p=names[solver], enzyme_vs_fd=we, cuad_vs_fd=wc)
        report['c2p'].append(row); print(json.dumps(row), flush=True)

q_true = np.array([10.0, 0.0, 13.33, 1.0, 0.0, 1e-2, 5 / 3])
q = np.array([9.0, 0.05, 12.0, 1.1, -0.05, 2e-2, 1.6])
for n, solver in ((100, 0), (100, 1)):
    steps = n
    Pt = np.zeros(3 * n); cuad.srhd_final(q_true, n, steps, 0.4, solver, Pt); obs = np.ascontiguousarray(Pt[:n])
    gc = np.zeros(7); cuad.srhd_misfit(q, n, steps, 0.4, solver, obs, gc.ctypes.data)
    ge = np.zeros(7); enz.enzyme_misfit_grad(q, n, steps, 0.4, solver, obs, ge)
    fd = np.zeros(7)
    for k in range(7):
        h = 1e-6 * max(abs(q[k]), 1e-2)
        qp = q.copy(); qm = q.copy(); qp[k] += h; qm[k] -= h
        fd[k] = (cuad.srhd_misfit(qp, n, steps, 0.4, solver, obs, None) -
                 cuad.srhd_misfit(qm, n, steps, 0.4, solver, obs, None)) / (2 * h)
    rel = lambda a: float(np.max(np.abs(a - fd) / np.abs(fd)))
    row = dict(n=n, c2p=names[solver], enzyme_vs_fd=rel(ge), cuad_vs_fd=rel(gc),
               enzyme=ge.tolist(), cuad=gc.tolist(), fd=fd.tolist())
    report['trajectory'].append(row)
    print(json.dumps({k: row[k] for k in ('n', 'c2p', 'enzyme_vs_fd', 'cuad_vs_fd')}), flush=True)
(out / 'enzyme-host-check.json').write_text(json.dumps(report, indent=2) + '\n')
