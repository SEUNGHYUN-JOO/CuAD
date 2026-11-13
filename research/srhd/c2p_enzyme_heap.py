"""Smallest device heap with which Enzyme's C2P adjoint runs, per cell count and solver (one fresh process per
trial, because the heap limit cannot change after the first allocation). Doubling search from 1 MiB up to 8 GiB.
CuAD's implicit adjoint and the hand-written one use no heap (local memory per thread: see c2p-gpu-compare.json)."""
import json
import os
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
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
st = f(2, solver, n, U, G, Pb, Ub, gb, 1, t, ct.byref(r), ct.byref(l))
print(st, int(np.isfinite(Ub).all() and np.abs(Ub).sum() > 0))
'''
rows = []
for n in (1 << 12, 1 << 16, 1 << 20):
    for solver, name in ((0, 'newton'), (1, 'bisection'), (2, 'newton_fixed40')):
        mib, found = 1, None
        while mib <= 8192:
            env = dict(os.environ, C2P_HEAP_MIB=str(mib))
            r = subprocess.run([sys.executable, '-c', probe, str(here / 'out' / 'c2p_gpu_compare.so'), str(n), str(solver)],
                               capture_output=True, text=True, env=env, timeout=600)
            ok = r.stdout.split() == ['0', '1']
            if ok:
                found = mib; break
            mib *= 2
        rows.append(dict(n=n, c2p=name, min_heap_mib_power_of_two=found))
        print(json.dumps(rows[-1]), flush=True)
(here / 'out' / 'c2p-enzyme-heap.json').write_text(json.dumps(rows, indent=2) + '\n')
