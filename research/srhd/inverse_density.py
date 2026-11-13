"""Inverse problem with the GPU SRHD adjoint: recover the density structure of the medium that a relativistic
shock sweeps through, from density and pressure profiles observed at a few times.

Setup on [0, 1], Gamma = 5/3: a hot slab (rho = 1, p = 10) for x < 0.15 drives a relativistic shock into a cold
medium (p = 0.1, at rest) whose density rho0(x) = 1 + 0.4 sin(6 pi x) exp(-((x - 0.5)/0.18)^2) is unknown.
Observations: rho and p on every cell at t = 0.15, 0.3, 0.45, 0.6, with 1 % Gaussian noise. Unknowns: log rho0
on every cell right of the slab. Objective: misfit + 0.5 beta dx sum ((d/dx) log rho0)^2, L-BFGS-B.
Gradient: CuAD adjoint of the whole integration with the chosen C2P (bisection by default; Enzyme's gradient
through that C2P is wrong: check_enzyme_host.py). A finite-difference gradient needs two solves per unknown.
Usage: python inverse_density.py [n] [c2p: 0 newton, 1 bisection] [gradient: cuad (GPU) | enzyme (host)] [noise] [beta] [case: 0 sinusoid, 1 clumps, k>=2 random field seed k]
With `enzyme` the objective and its gradient come from Enzyme v0.0.307 applied to the same source on the host
(enzyme_srhd_host.cpp); observations are the same GPU solution in both cases.
"""
import ctypes as ct
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

out = Path(__file__).parent / 'out'
gpu = ct.CDLL(str(out / 'srhd_gpu.so')).srhd_gpu_obs
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
iptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
gpu.argtypes = [ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_double, ct.c_int, iptr, ptr, ptr,
                ct.POINTER(ct.c_double), ct.c_void_p, ct.POINTER(ct.c_double), ptr, ct.c_void_p]
gpu.restype = ct.c_int

n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
solver = int(sys.argv[2]) if len(sys.argv) > 2 else 1
tool = sys.argv[3] if len(sys.argv) > 3 else 'cuad'
if tool == 'enzyme':
    enz = ct.CDLL(str(out / 'enzyme_srhd_host.so')).enzyme_obs_grad
    enz.argtypes = [ptr, ptr, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ct.c_double, ct.c_int,
                    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), ptr, ptr, ptr]
    enz.restype = ct.c_double
noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
beta = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
G, cfl = 5 / 3, 0.4
times_obs = np.array([0.15, 0.3, 0.45, 0.6])
dt = cfl / n
kst = np.ascontiguousarray(np.round(times_obs / dt).astype(np.int32))
steps = int(kst[-1])
dx = 1.0 / n
x = (np.arange(n) + 0.5) * dx
free = x >= 0.15
case = int(sys.argv[6]) if len(sys.argv) > 6 else 0
if case == 0:    # windowed sinusoid
    medium = 1 + 0.4 * np.sin(6 * np.pi * x) * np.exp(-((x - 0.5) / 0.18) ** 2)
elif case == 1:  # two clumps, one dense and one rarefied
    medium = 1 + 0.6 * np.exp(-((x - 0.35) / 0.04) ** 2) - 0.35 * np.exp(-((x - 0.62) / 0.06) ** 2)
else:            # smooth random field: five Fourier modes with amplitudes drawn from seed `case`
    cr = np.random.default_rng(case)
    medium = 1 + sum(cr.uniform(-0.12, 0.12) * np.sin(k * np.pi * x + cr.uniform(0, 2 * np.pi)) for k in range(2, 12, 2))
rho_true = np.where(free, medium, 1.0)
p0 = np.where(free, 0.1, 10.0)
w = np.concatenate([np.ones(n), np.zeros(n), np.ones(n)])  # density and pressure observed
stats = dict(solves=0, gpu_forward_ms=0.0, gpu_reverse_ms=0.0)

def gpu_run(rho0, obs, grad):
    J = ct.c_double(); gb = ct.c_double(); t = np.zeros(2); W0b = np.zeros(3 * n); P = np.zeros(len(kst) * 3 * n)
    st = gpu(n, steps, cfl, solver, np.concatenate([rho0, np.zeros(n), p0]), G, len(kst), kst, obs, w, ct.byref(J),
             W0b.ctypes.data if grad else None, ct.byref(gb), t, P.ctypes.data)
    assert st == 0, st
    stats['solves'] += 1; stats['gpu_forward_ms'] += t[0]; stats['gpu_reverse_ms'] += t[1] if grad else 0
    return J.value, W0b[:n], P

rng = np.random.default_rng(3)
_, _, P_true = gpu_run(rho_true, np.zeros(len(kst) * 3 * n), False)
obs = np.ascontiguousarray(P_true * (1 + noise * rng.normal(size=P_true.size)))

def full(z):
    rho0 = np.ones(n); rho0[free] = np.exp(z); return rho0

def objective(z):
    rho0 = full(z)
    if tool == 'enzyme':
        g = np.zeros(n)
        J = enz(rho0, p0, n, steps, cfl, solver, G, len(kst), kst, obs, w, g)
        stats['solves'] += 1
    else:
        J, g, _ = gpu_run(rho0, obs, True)
    d = np.diff(z)
    R = 0.5 * beta * np.sum(d * d) / dx
    gR = np.zeros_like(z); gR[:-1] -= beta * d / dx; gR[1:] += beta * d / dx
    return J + R, g[free] * rho0[free] + gR

# gradient check at a generic (non-uniform) state, three random directions, two step sizes
m_free = int(free.sum())
zc = 0.05 * np.sin(np.linspace(0, 9.0, m_free)) + 0.01 * rng.normal(size=m_free)
fc, gc = objective(zc)
checks = []
for k in range(3):
    dz = rng.normal(size=m_free)
    for h in (1e-5, 1e-6):
        fd = (objective(zc + h * dz)[0] - objective(zc - h * dz)[0]) / (2 * h)
        checks.append(dict(h=h, rel_error=float(abs(gc @ dz - fd) / abs(fd))))
print(json.dumps({'n': n, 'unknowns': m_free, 'steps': steps, 'gradient_checks': checks}), flush=True)

hist = []
amp = np.linalg.norm(rho_true[free] - 1)
def cb(z):
    rho0 = full(z)
    hist.append(dict(rel_l2_error_rho0=float(np.linalg.norm(rho0[free] - rho_true[free]) / amp)))
z0 = np.zeros(m_free)
f0 = objective(z0)[0]
stats.update(solves=0, gpu_forward_ms=0.0, gpu_reverse_ms=0.0)
t0 = time.time()
res = minimize(objective, z0, jac=True, method='L-BFGS-B', callback=cb,
               options=dict(maxiter=400, ftol=1e-15, gtol=1e-14, maxcor=20))
wall = time.time() - t0
rho_fit = full(res.x)
misfit_noise = 0.5 * dx * np.sum((w * (P_true - obs).reshape(len(kst), 3 * n)) ** 2 / np.maximum(w, 1e-300))
summary = dict(case=case, gradient=tool, n=n, unknowns=m_free, steps=steps, c2p=['newton', 'bisection'][solver], noise=noise, beta=beta,
               observation_times=times_obs.tolist(), iterations=int(res.nit), objective_evaluations=int(res.nfev),
               message=str(res.message), objective_initial=float(f0), objective_final=float(res.fun),
               misfit_of_the_truth=float(misfit_noise),
               rel_l2_error_rho0_final=float(np.linalg.norm(rho_fit[free] - rho_true[free]) / amp),
               wall_s=wall, solves=stats['solves'], gpu_forward_ms_total=stats['gpu_forward_ms'],
               gpu_reverse_ms_total=stats['gpu_reverse_ms'], gradient_checks=checks,
               fd_gradient_cost='2 x %d forward solves per gradient' % m_free)
print(json.dumps(summary), flush=True)
np.savez(out / f'inverse-density-{n}-{solver}-{tool}-noise{noise:g}-case{case}.npz', x=x, rho_true=rho_true, rho_fit=rho_fit, obs=obs, P_true=P_true,
         kst=kst)
(out / f'inverse-density-{n}-{solver}-{tool}-noise{noise:g}-case{case}.json').write_text(json.dumps(dict(summary=summary, history=hist), indent=2) + '\n')
