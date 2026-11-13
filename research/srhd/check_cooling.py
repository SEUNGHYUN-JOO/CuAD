"""Second implicit kernel: backward-Euler cooling solved by Newton per cell (examples/srhd/cooling.cu).
1. Kernel: CuAD implicit adjoint and CuAD tangent (iterations differentiated) against central differences.
2. Whole integration with operator-split cooling after every RK2 step: gradient of the final density misfit with
   respect to the Riemann data (7), L0 and alpha, against central differences, both C2P solvers."""
import ctypes as ct
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
lib = ct.CDLL(str(out / 'srhd_adj_host.so'))
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
lib.cool_apply.argtypes = [ct.c_int, ptr, ptr, ct.c_double, ptr]
lib.cool_adjoint.argtypes = [ct.c_int, ptr, ptr, ct.c_double, ptr, ptr, ptr]
lib.cool_tangent.argtypes = [ct.c_int, ptr, ptr, ptr, ptr, ct.c_double, ptr]
lib.srhd_cool_misfit.argtypes = [ptr, ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_void_p, ct.c_void_p]
lib.srhd_cool_misfit.restype = ct.c_double
report = {'kernel': [], 'trajectory': []}
rng = np.random.default_rng(9)
n = 64


def newton_exits(P, lam, dt):
    """Per cell: True if the Newton loop of cool_implicit leaves through its convergence test (same arithmetic)."""
    rho, p = P[:n], P[2 * n:]
    c = dt * lam[0] * rho * rho
    floor = np.minimum(1e-6 * rho, p)
    ok = np.zeros(n, bool)
    for i in range(n):
        x = p[i]
        for _ in range(60):
            q = (x / rho[i]) ** lam[1]
            dx = (x - p[i] + c[i] * q) / (1.0 + c[i] * lam[1] * q / x)
            x = max(x - dx, floor[i])
            if abs(dx) <= 1e-14 * x:
                ok[i] = True
                break
    return ok


if len(sys.argv) > 1 and sys.argv[1] == 'adjoint':      # child process: the adjoint of a case that must trap
    P = np.load(out / 'cooling-nonmonotone.npy')
    lib.cool_adjoint(n, P, np.array([0.5, -0.5]), 0.05, np.ones(3 * n), np.zeros(3 * n), np.zeros(2))
    print('adjoint returned'); sys.exit(0)

# alpha = -0.5 makes the residual non-monotone: Newton can fail to converge, and the implicit adjoint must trap
P = np.concatenate([rng.uniform(0.5, 2, n), rng.uniform(-2, 2, n), rng.uniform(0.05, 5, n)])
np.save(out / 'cooling-nonmonotone.npy', P)
ok = newton_exits(P, (0.5, -0.5), 0.05)
child = subprocess.run([sys.executable, __file__, 'adjoint'], capture_output=True, text=True)
report['nonmonotone'] = dict(L0=0.5, alpha=-0.5, dt=0.05, cells=n, cells_without_convergence=int((~ok).sum()),
                             adjoint_exit_code=child.returncode, adjoint_returned='adjoint returned' in child.stdout)
print(json.dumps(report['nonmonotone']), flush=True)
rng = np.random.default_rng(9)
for lam, dt in (((5.0, 0.5), 0.01), ((50.0, 1.5), 0.02), ((0.5, 0.25), 0.05)):
    P = np.concatenate([rng.uniform(0.5, 2, n), rng.uniform(-2, 2, n), rng.uniform(0.05, 5, n)])
    lm = np.array(lam)
    Pn = np.zeros(3 * n); lib.cool_apply(n, P, lm, dt, Pn)
    drop = float(np.median(1 - Pn[2 * n:] / P[2 * n:]))
    wa = wt = 0.0
    for t in range(5):
        dP = rng.normal(size=3 * n) * np.abs(P) * 1e-2; dl = rng.normal(size=2) * 1e-2
        Pb = rng.normal(size=3 * n)
        phi = lambda s: (lambda Q: (lib.cool_apply(n, P + s * dP, lm + s * dl, dt, Q), Q @ Pb)[1])(np.zeros(3 * n))
        fd = (phi(1e-6) - phi(-1e-6)) / 2e-6
        Pb_ = np.zeros(3 * n); lb = np.zeros(2)
        lib.cool_adjoint(n, P, lm, dt, Pb.copy(), Pb_, lb)
        Pd = np.zeros(3 * n); lib.cool_tangent(n, P, dP, lm, dl, dt, Pd)
        wa = max(wa, abs(Pb_ @ dP + lb @ dl - fd) / abs(fd)); wt = max(wt, abs(Pd @ Pb - fd) / abs(fd))
    row = dict(L0=lam[0], alpha=lam[1], dt=dt, median_pressure_drop=drop, adjoint_vs_fd=wa, tangent_vs_fd=wt)
    report['kernel'].append(row); print(json.dumps(row), flush=True)

q = np.array([9.0, 0.05, 12.0, 1.1, -0.05, 2e-2, 1.6]); lam = np.array([2.0, 0.5])
q_true = np.array([10.0, 0.0, 13.33, 1.0, 0.0, 1e-2, 5 / 3]); lam_true = np.array([3.0, 0.7])
for n_, solver in ((100, 0), (100, 1)):
    steps = n_
    obs = np.zeros(n_)
    lib.srhd_cool_misfit(q_true, lam_true, n_, steps, 0.4, solver, np.zeros(n_), None, obs.ctypes.data)
    g = np.zeros(9)
    J = lib.srhd_cool_misfit(q, lam, n_, steps, 0.4, solver, obs, g.ctypes.data, None)
    fd = np.zeros(9)
    x0 = np.concatenate([q, lam])
    for k in range(9):
        h = 1e-6 * max(abs(x0[k]), 1e-2)
        xp = x0.copy(); xm = x0.copy(); xp[k] += h; xm[k] -= h
        fd[k] = (lib.srhd_cool_misfit(xp[:7], xp[7:], n_, steps, 0.4, solver, obs, None, None) -
                 lib.srhd_cool_misfit(xm[:7], xm[7:], n_, steps, 0.4, solver, obs, None, None)) / (2 * h)
    rel = np.abs(g - fd) / np.abs(fd)
    row = dict(n=n_, c2p=['newton', 'bisection'][solver], J=J, dJ_dL0=g[7], dJ_dalpha=g[8],
               max_rel_error=float(rel.max()), rel_errors=rel.tolist())
    report['trajectory'].append(row); print(json.dumps({k: row[k] for k in ('n', 'c2p', 'J', 'dJ_dL0', 'dJ_dalpha', 'max_rel_error')}), flush=True)
(out / 'cooling-check.json').write_text(json.dumps(report, indent=2) + '\n')
