"""Where the cost of the C2P gradient comes from, at 2^20 cells on the GPU (c2p_gpu_ablation.so, built from
c2p_gpu_compare.cu with -DWITH_ABLATION):
  - Enzyme on the unchanged kernels and on a copy whose pointer parameters are all __restrict__ (aliasing information),
  - the CuAD implicit adjoint with and without its run-time checks (-DCUAD_SOLVE_CHECK=0),
  - the primal and the hand-written implicit adjoint for reference,
with registers and local memory per thread, the spread over three runs of ten launches, a dot test against central
differences on 4096 cells, and the smallest device heap with which Enzyme runs on the __restrict__ kernels."""
import ctypes as ct
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
out = here / 'out'
lib = out / 'c2p_gpu_ablation.so'
f = ct.CDLL(str(lib)).c2p_gpu
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
f.argtypes = [ct.c_int, ct.c_int, ct.c_int, ptr, ct.c_double, ptr, ptr, ptr, ct.c_int, ptr,
              ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
f.restype = ct.c_int
host = ct.CDLL(str(out / 'srhd_adj_host.so'))
host.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]
names = {0: 'primal', 1: 'cuad', 5: 'cuad_without_checks', 3: 'hand_written', 2: 'enzyme', 4: 'enzyme_restrict'}
G = 5 / 3


def states(n, seed=11):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.5, 2, n); u = rng.uniform(-3, 3, n); p = rng.uniform(0.01, 10, n)
    W = np.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    U = np.concatenate([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W])
    return U, rng.normal(size=3 * n), rng


def run(tool, solver, n, U, Pb, reps):
    Ub = np.zeros(3 * n); gb = np.zeros(1); t = np.zeros(reps); r = ct.c_int(); l = ct.c_int()
    st = f(tool, solver, n, U, G, Pb, Ub, gb, reps, t, ct.byref(r), ct.byref(l))
    return st, Ub, t, r.value, l.value


rows = []
for solver, sname in ((0, 'newton'), (1, 'bisection')):
    # correctness on 4096 cells
    n = 4096
    U, Pb, rng = states(n)
    dU = rng.normal(size=3 * n) * np.abs(U) * 1e-2

    def phi(s):
        Q = np.zeros(3 * n); host.c2p_apply(solver, n, U + s * dU, np.array([G]), Q); return Q @ Pb
    fd = (phi(1e-5) - phi(-1e-5)) / 2e-5
    dot = {}
    for tool in names:
        if tool == 3 and solver != 0:
            continue
        st, Ub, _, _, _ = run(tool, solver, n, U, Pb, 1)
        if tool:
            dot[tool] = float(abs(Ub @ dU - fd) / abs(fd)) if st == 0 else None
    # timing on 2^20 cells: three runs of ten launches
    n = 1 << 20
    U, Pb, _ = states(n)
    for tool in names:
        if tool == 3 and solver != 0:
            continue
        times, st, regs, loc = [], 0, 0, 0
        for _ in range(3):
            st, _, t, regs, loc = run(tool, solver, n, U, Pb, 10)
            if st:
                break
            times.append(t)
        row = dict(n=n, c2p=sname, tool=names[tool], status=st, registers=regs, local_bytes=loc,
                   dot_test_rel_error=dot.get(tool))
        if st == 0:
            a = np.concatenate(times)
            row.update(median_ms=float(np.median(a)), min_ms=float(a.min()), max_ms=float(a.max()),
                       run_medians_ms=[float(np.median(x)) for x in times])
        rows.append(row); print(json.dumps(row), flush=True)

# smallest heap for Enzyme on the __restrict__ kernels (fresh process per trial: the heap limit is fixed at first use)
probe = r'''
import ctypes as ct, numpy as np, sys
f = ct.CDLL(sys.argv[1]).c2p_gpu
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
f.argtypes = [ct.c_int, ct.c_int, ct.c_int, ptr, ct.c_double, ptr, ptr, ptr, ct.c_int, ptr, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
n, solver = int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(11); G = 5 / 3
rho = rng.uniform(0.5, 2, n); u = rng.uniform(-3, 3, n); p = rng.uniform(0.01, 10, n)
W = np.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
U = np.concatenate([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W]); Pb = rng.normal(size=3 * n)
Ub = np.zeros(3 * n); gb = np.zeros(1); t = np.zeros(1); r = ct.c_int(); l = ct.c_int()
st = f(4, solver, n, U, G, Pb, Ub, gb, 1, t, ct.byref(r), ct.byref(l))
print(st, int(np.isfinite(Ub).all() and np.abs(Ub).sum() > 0))
'''
heap = []
for solver, sname in ((0, 'newton'), (1, 'bisection')):
    mib, found = 1, None
    while mib <= 8192:
        env = dict(os.environ, C2P_HEAP_MIB=str(mib))
        r = subprocess.run([sys.executable, '-c', probe, str(lib), str(1 << 20), str(solver)],
                           capture_output=True, text=True, env=env, timeout=600)
        if r.stdout.split() == ['0', '1']:
            found = mib; break
        mib *= 2
    heap.append(dict(n=1 << 20, c2p=sname, tool='enzyme_restrict', min_heap_mib_power_of_two=found))
    print(json.dumps(heap[-1]), flush=True)
(out / 'c2p-gpu-ablation.json').write_text(json.dumps(dict(timing=rows, heap=heap), indent=2) + '\n')
