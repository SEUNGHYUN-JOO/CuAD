"""C2P adjoint on the GPU: CuAD implicit adjoint, Enzyme v0.0.307 (iterations differentiated), a hand-written
implicit adjoint (3x3 solve per cell), and the primal, for 2^12 .. 2^22 cells.
Solvers: 0 Newton with `break` (srhd.cu), 1 bisection (srhd.cu), 2 Newton with 40 fixed iterations (c2p_fixed.cu,
the rewrite a user makes to avoid the data-dependent exit). Accuracy: dot test against central differences of the
host primal (n = 4096). Cost: kernel time per launch (CUDA events, median of REPS after two warm-ups, Pb copied in
before each timed launch, outside the timed interval), registers and local memory per thread. Enzyme's device heap is
1 GiB (C2P_HEAP_MIB overrides); minimal heap sizes are measured by c2p_enzyme_heap.py."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
f = ct.CDLL(str(out / 'c2p_gpu_compare.so')).c2p_gpu
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
f.argtypes = [ct.c_int, ct.c_int, ct.c_int, ptr, ct.c_double, ptr, ptr, ptr, ct.c_int, ptr,
              ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
f.restype = ct.c_int
host = ct.CDLL(str(out / 'srhd_adj_host.so'))
host.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
REPS = 5
tools = ['primal', 'cuad', 'enzyme', 'manual']
solvers = ['newton', 'bisection', 'newton_fixed40']

def cons(rho, u, p, G):
    W = np.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    return np.concatenate([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W])

rng = np.random.default_rng(11)
G = 5 / 3
rows = []
for n in (1 << 12, 1 << 16, 1 << 20, 1 << 22):
    rho = rng.uniform(0.5, 2, n); u = rng.uniform(-3, 3, n); p = rng.uniform(0.01, 10, n)
    U = cons(rho, u, p, G); Pb = rng.normal(size=3 * n)
    for solver in (0, 1, 2):
        fd = dU = None
        if n <= 4096:
            dU = rng.normal(size=3 * n) * np.abs(U) * 1e-2
            def phi(t):
                Q = np.zeros(3 * n); host.c2p_apply(solver % 2, n, U + t * dU, np.array([G]), Q); return Q @ Pb
            fd = (phi(1e-5) - phi(-1e-5)) / 2e-5
        for tool in (0, 1, 2, 3):
            if tool == 3 and solver != 0:
                continue
            Ub = np.zeros(3 * n); gb = np.zeros(1); times = np.zeros(REPS); regs = ct.c_int(); loc = ct.c_int()
            st = f(tool, solver, n, U, G, Pb, Ub, gb, REPS, times, ct.byref(regs), ct.byref(loc))
            row = dict(n=n, c2p=solvers[solver], tool=tools[tool], status=st,
                       median_ms=float(np.median(times)) if st == 0 else None,
                       min_ms=float(np.min(times)) if st == 0 else None,
                       registers=regs.value, local_bytes=loc.value)
            if tool and st == 0 and fd is not None:
                row['dot_test_rel_error'] = float(abs(Ub @ dU - fd) / abs(fd))
            rows.append(row); print(json.dumps(row), flush=True)
(out / 'c2p-gpu-compare.json').write_text(json.dumps(rows, indent=2) + '\n')
