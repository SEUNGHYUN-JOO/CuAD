"""GPU run of the SRHD CuAD adjoint: the gradient field against the host adjoint (itself checked against
finite differences in check_adjoint.py), then forward and reverse times versus resolution."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
gpu = ct.CDLL(str(out / 'srhd_gpu.so')).srhd_gpu
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
gpu.argtypes = [ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_double, ptr, ptr, ct.POINTER(ct.c_double),
                ct.c_void_p, ct.POINTER(ct.c_double), ptr, ct.c_void_p]
gpu.restype = ct.c_int
host = ct.CDLL(str(out / 'srhd_adj_host.so'))
host.srhd_misfit.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_void_p]
host.srhd_misfit.restype = ct.c_double
host.srhd_final.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr]

def riemann(q, n):
    left = (np.arange(n) + 0.5) / n < 0.5
    return np.concatenate([np.where(left, q[0], q[3]), np.where(left, q[1], q[4]), np.where(left, q[2], q[5])])

def run(W0, gam, obs, w, n, steps, solver, grad=True):
    J = ct.c_double(); gb = ct.c_double(); t = np.zeros(2)
    W0b = np.zeros(3 * n)
    st = gpu(n, steps, 0.4, solver, np.ascontiguousarray(W0), gam, np.ascontiguousarray(obs), np.ascontiguousarray(w),
             ct.byref(J), W0b.ctypes.data if grad else None, ct.byref(gb), t, None)
    assert st == 0, st
    return J.value, W0b, gb.value, t

q_true = np.array([10.0, 0.0, 13.33, 1.0, 0.0, 1e-2, 5 / 3])
q = np.array([9.0, 0.05, 12.0, 1.1, -0.05, 2e-2, 1.6])
report = {'agreement': [], 'timing': []}
for solver in (0, 1):
    n = 100; steps = n
    Pt = np.zeros(3 * n); host.srhd_final(q_true, n, steps, 0.4, solver, Pt)
    obs = np.concatenate([Pt[:n], np.zeros(2 * n)]); w = np.concatenate([np.ones(n), np.zeros(2 * n)])
    J, W0b, gb, t = run(riemann(q, n), q[6], obs, w, n, steps, solver)
    left = (np.arange(n) + 0.5) / n < 0.5
    gq = np.array([W0b[c * n:(c + 1) * n][mask].sum() for mask in (left, ~left) for c in range(3)] + [gb])
    gh = np.zeros(7); Jh = host.srhd_misfit(q, n, steps, 0.4, solver, np.ascontiguousarray(Pt[:n]), gh.ctypes.data)
    row = dict(c2p=['newton', 'bisection'][solver], J_gpu=J, J_host=Jh,
               max_rel_grad_diff=float(np.max(np.abs(gq - gh) / np.abs(gh))))
    report['agreement'].append(row); print(json.dumps(row), flush=True)
for n in (256, 1024, 4096):
    steps = n
    W0 = riemann(q_true, n); obs = W0.copy(); w = np.ones(3 * n)
    for solver in (0, 1):
        run(W0, 5 / 3, obs, w, n, steps, solver)  # warm-up
        ts = np.array([run(W0, 5 / 3, obs, w, n, steps, solver)[3] for _ in range(3)])
        row = dict(n=n, steps=steps, c2p=['newton', 'bisection'][solver], forward_ms=float(np.median(ts[:, 0])),
                   reverse_ms=float(np.median(ts[:, 1])), reverse_over_forward=float(np.median(ts[:, 1] / ts[:, 0])))
        report['timing'].append(row); print(json.dumps(row), flush=True)
(out / 'gpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
