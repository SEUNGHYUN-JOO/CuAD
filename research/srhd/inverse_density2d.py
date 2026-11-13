"""2-D inverse problem with the GPU SRHD adjoint: a planar relativistic shock, driven by a hot slab (rho = 1, p = 10,
x < 0.15), sweeps a cold medium (p = 0.1) with dense and rarefied clumps; reconstruct the medium's initial density
field (every cell right of the slab) from rho and p observed at t = 0.15, 0.30, 0.45 on the whole grid.
Checks first: GPU gradient against the host adjoint (examples/srhd/srhd2d_host.cpp) on a small grid, and against central
differences in random directions on the full grid. Then L-BFGS-B on log rho0 with a small gradient regularizer.
Usage: python inverse_density2d.py [n (grid n x n)] [c2p: 0 newton, 1 bisection] [noise]"""
import ctypes as ct
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

out = Path(__file__).parent / 'out'
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
iptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
gpu = ct.CDLL(str(out / 'srhd2d_gpu.so')).srhd2_gpu
gpu.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_double, ct.c_int, iptr, ptr, ptr,
                ct.POINTER(ct.c_double), ct.c_void_p, ct.POINTER(ct.c_double), ptr, ct.c_void_p]
gpu.restype = ct.c_int
host = ct.CDLL(str(out / 'srhd2d_host.so')).srhd2_obs
host.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_double, ct.c_int, iptr, ptr, ptr,
                 ct.c_void_p, ct.c_void_p, ct.c_void_p]
host.restype = ct.c_double

n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
solver = int(sys.argv[2]) if len(sys.argv) > 2 else 1
noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
G, cfl, beta = 5 / 3, 0.25, 1e-7 if noise == 0 else 2e-6
report = {}

def setup(nx, ny, times):
    dt = cfl / max(nx, ny)
    kst = np.ascontiguousarray(np.round(np.array(times) / dt).astype(np.int32))
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    return kst, X.ravel(), Y.ravel()

def medium(X, Y):
    clumps = [(0.35, 0.30, 0.06, 0.6), (0.50, 0.65, 0.07, -0.4), (0.65, 0.40, 0.05, 0.8), (0.42, 0.80, 0.05, 0.5)]
    rho = np.ones_like(X)
    for cx, cy, s, a in clumps:
        rho += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    return rho

def run(nx, ny, kst, W0, obs, w, grad, solver_=solver):
    N = nx * ny; J = ct.c_double(); gb = ct.c_double(); t = np.zeros(2)
    W0b = np.zeros(4 * N); P = np.zeros(len(kst) * 4 * N)
    st = gpu(nx, ny, int(kst[-1]), cfl, solver_, np.ascontiguousarray(W0), G, len(kst), kst, np.ascontiguousarray(obs),
             np.ascontiguousarray(w), ct.byref(J), W0b.ctypes.data if grad else None, ct.byref(gb), t, P.ctypes.data)
    assert st == 0, st
    return J.value, W0b, P, t

# 1. GPU against host adjoint, small grid
nx, ny = 20, 16
kst, X, Y = setup(nx, ny, [0.1, 0.2])
N = nx * ny
W0 = np.concatenate([np.where(X < 0.15, 1.0, medium(X, Y)), 0.05 * np.sin(3 * Y + X), 0.05 * np.cos(2 * X - Y),
                     np.where(X < 0.15, 10.0, 0.1)])
obs = np.concatenate([run(nx, ny, kst, W0 * 1.01, np.zeros(2 * 4 * N), np.ones(4 * N), False)[2]])
w = np.ones(4 * N)
Jg, gg, _, _ = run(nx, ny, kst, W0, obs, w, True)
gh = np.zeros(4 * N); gbh = ct.c_double()
Jh = host(nx, ny, int(kst[-1]), cfl, solver, W0, G, len(kst), kst, obs, w, gh.ctypes.data, ct.addressof(gbh), None)
report['gpu_vs_host'] = dict(J_gpu=Jg, J_host=Jh, max_rel_grad_diff=float(np.max(np.abs(gg - gh)) / np.max(np.abs(gh))))
print(json.dumps(report['gpu_vs_host']), flush=True)

# 2. the inverse problem
nx = ny = n; N = n * n
kst, X, Y = setup(nx, ny, [0.15, 0.30, 0.45])
free = X >= 0.15
rho_true = np.where(free, medium(X, Y), 1.0)
p0 = np.where(free, 0.1, 10.0)
w = np.concatenate([np.ones(N), np.zeros(2 * N), np.ones(N)])
W0t = np.concatenate([rho_true, np.zeros(2 * N), p0])
_, _, P_true, ttrue = run(nx, ny, kst, W0t, np.zeros(len(kst) * 4 * N), w, False)
rng = np.random.default_rng(21)
obs = P_true * (1 + noise * rng.normal(size=P_true.size))
dA = 1.0 / N
idx = np.flatnonzero(free)
grid = -np.ones(N, dtype=int); grid[idx] = np.arange(idx.size)
I = idx % n; Jy = idx // n
right = np.array([grid[j * n + i + 1] if i + 1 < n else -1 for i, j in zip(I, Jy)])
up = np.array([grid[(j + 1) * n + i] if j + 1 < n else -1 for i, j in zip(I, Jy)])
stats = dict(evals=0, gpu_ms=0.0)

def full(z):
    rho0 = np.ones(N); rho0[free] = np.exp(z); return rho0

def objective(z):
    rho0 = full(z)
    J, g, _, t = run(nx, ny, kst, np.concatenate([rho0, np.zeros(2 * N), p0]), obs, w, True)
    stats['evals'] += 1; stats['gpu_ms'] += t[0] + t[1]; stats['last_misfit'] = J; stats['last_z'] = z.copy()
    R = 0.0; gR = np.zeros_like(z)
    for nb in (right, up):
        ok = nb >= 0
        d = z[nb[ok]] - z[ok]
        R += 0.5 * beta * np.sum(d * d)
        np.add.at(gR, nb[ok], beta * d); np.add.at(gR, np.flatnonzero(ok), -beta * d)
    return J + R, g[:N][free] * rho0[free] + gR

zc = 0.05 * np.sin(5 * X[free] + 3 * Y[free]) + 0.01 * rng.normal(size=idx.size)
fc, gc = objective(zc)
checks = []
for k in range(3):
    dz = rng.normal(size=idx.size)
    for h in (1e-5, 1e-6):
        fd = (objective(zc + h * dz)[0] - objective(zc - h * dz)[0]) / (2 * h)
        checks.append(dict(h=h, rel_error=float(abs(gc @ dz - fd) / abs(fd))))
print(json.dumps({'n': n, 'unknowns': int(idx.size), 'steps': int(kst[-1]), 'gradient_checks': checks}), flush=True)
amp = np.linalg.norm(rho_true[free] - 1)
hist, misfits = [], []
stats.update(evals=0, gpu_ms=0.0, last_misfit=None, last_z=None)
t0 = time.time()
def cb(z):
    hist.append(float(np.linalg.norm(full(z)[free] - rho_true[free]) / amp))
    # data misfit of the accepted iterate (re-evaluated if the last objective call was another line-search point)
    if stats['last_z'] is not None and np.array_equal(z, stats['last_z']):
        misfits.append(stats['last_misfit'])
    else:
        misfits.append(run(nx, ny, kst, np.concatenate([full(z), np.zeros(2 * N), p0]), obs, w, False)[0])
res = minimize(objective, np.zeros(idx.size), jac=True, method='L-BFGS-B', callback=cb,
               options=dict(maxiter=300, ftol=1e-15, gtol=1e-14, maxcor=20))
wall = time.time() - t0
# discrepancy principle (Morozov) with the threshold from the noise statistics alone (relative noise of standard
# deviation `noise` on each observed value), as it would be set for real data: E|obs - P|^2 = noise^2 P^2 ~ noise^2 obs^2,
# times the usual safety factor tau > 1 for the fluctuation of one noise sample about its expectation
tau = 1.1
noise_misfit = float(tau * 0.5 * dA * np.sum(np.tile(w, len(kst)) * (noise * obs) ** 2))
k_dp = next((i for i, m in enumerate(misfits) if m is not None and noise > 0 and m <= noise_misfit), None)
fwd_ms = ttrue[0]
summary = dict(n=n, unknowns=int(idx.size), steps=int(kst[-1]), c2p=['newton', 'bisection'][solver], noise=noise, beta=beta,
               iterations=int(res.nit), evaluations=stats['evals'], message=str(res.message), objective_final=float(res.fun),
               misfit_of_the_truth=float(0.5 * dA * np.sum((np.tile(w, len(kst)) * (P_true - obs) ** 2))),
               rel_l2_error_rho0_final=float(hist[-1]) if hist else None, error_history=hist[::10] + hist[-1:],
               noise_misfit_threshold=noise_misfit, discrepancy_tau=tau,
               discrepancy_iterate=(k_dp + 1 if k_dp is not None else None),  # 1-based L-BFGS-B iterate
               rel_l2_error_at_discrepancy=(hist[k_dp] if k_dp is not None else None),
               min_rel_l2_error=float(min(hist)) if hist else None,
               wall_s=wall, gpu_ms_total=stats['gpu_ms'], forward_ms=fwd_ms,
               fd_gradient_estimate_s=2 * idx.size * fwd_ms / 1e3, gradient_checks=checks, gpu_vs_host=report['gpu_vs_host'])
print(json.dumps({k: v for k, v in summary.items() if k != 'error_history'}), flush=True)
rho_fit = full(res.x)
np.savez(out / f'inverse2d-{n}-{solver}-noise{noise:g}.npz', X=X, Y=Y, rho_true=rho_true, rho_fit=rho_fit,
         P_true=P_true, kst=kst, n=n)
(out / f'inverse2d-{n}-{solver}-noise{noise:g}.json').write_text(
    json.dumps(dict(summary=summary, history=hist, misfit_history=misfits), indent=2) + '\n')
