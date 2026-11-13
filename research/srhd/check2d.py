"""Host verification of the 2-D SRHD solver and its CuAD adjoint (examples/srhd/srhd2d.cu).
1. A 1-D Riemann problem (Marti-Mueller 1) run along x and along y on an nx x 4 / 4 x ny grid equals the 1-D
   solver (srhd.cu) cell by cell: the transverse fluxes cancel exactly.
2. Cylindrical blast wave: the solution is symmetric under x <-> y.
3. Gradient of a two-time misfit with respect to the whole initial primitive field (4N values) and gam, both C2P
   solvers, against central differences in random directions. The initial state varies in both directions in every
   field: where a field is uniform along a direction, its slopes are exactly zero and minmod(0, 0) is a kink at which
   finite differences give a one-sided derivative while AD gives the derivative of the branch taken.
4. Dot-product test of the adjoint against the CuAD tangent (no finite differences)."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
iptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
l2 = ct.CDLL(str(out / 'srhd2d_host.so'))
l2.srhd2_obs.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_double, ct.c_int, iptr, ptr,
                         ptr, ct.c_void_p, ct.c_void_p, ct.c_void_p]
l2.srhd2_obs.restype = ct.c_double
l1 = ct.CDLL(str(out / 'srhd_host.so'))
l1.srhd_run.argtypes = [ct.c_int, ct.c_int, ct.c_double, ct.c_double] + [ct.c_double] * 6 + [ct.c_int, ptr]
report = {}

def final2(nx, ny, steps, cfl, solver, W0, gam):
    m = 4 * nx * ny; P = np.zeros(m); k = np.array([steps], dtype=np.int32)
    l2.srhd2_obs(nx, ny, steps, cfl, solver, W0, gam, 1, k, np.zeros(m), np.zeros(m), None, None, P.ctypes.data)
    return P

# 1. 1-D consistency
n, cfl, G = 200, 0.4, 5 / 3
steps = int(round(0.4 / (cfl / n)))
P1 = np.zeros(3 * n); l1.srhd_run(n, steps, cfl, G, 10.0, 0.0, 13.33, 1.0, 0.0, 1e-8, 1, P1)
x = (np.arange(n) + 0.5) / n
left = x < 0.5
rows = []
for dir in (0, 1):
    nx, ny = (n, 4) if dir == 0 else (4, n)
    N = nx * ny
    xx = np.tile(x, ny) if dir == 0 else np.repeat(x, nx)
    lm = xx < 0.5
    W0 = np.concatenate([np.where(lm, 10.0, 1.0), np.zeros(N), np.zeros(N), np.where(lm, 13.33, 1e-8)])
    P = final2(nx, ny, steps, cfl, 1, W0, G)
    rho = P[:N].reshape(ny, nx); un = (P[N:2 * N] if dir == 0 else P[2 * N:3 * N]).reshape(ny, nx)
    ut = (P[2 * N:3 * N] if dir == 0 else P[N:2 * N])
    r2 = rho[0, :] if dir == 0 else rho[:, 0]
    u2 = un[0, :] if dir == 0 else un[:, 0]
    rows.append(dict(direction='xy'[dir], max_rho_diff=float(np.max(np.abs(r2 - P1[:n]))),
                     max_u_diff=float(np.max(np.abs(u2 - P1[n:2 * n]))), max_transverse_u=float(np.max(np.abs(ut)))))
    print(json.dumps(rows[-1]), flush=True)
report['consistency_1d'] = rows

# 2. cylindrical blast wave symmetry
nx = ny = 64; N = nx * ny; cfl2 = 0.25
xc = (np.arange(nx) + 0.5) / nx
X, Y = np.meshgrid(xc, xc)
r = np.hypot(X - 0.5, Y - 0.5).ravel()
W0 = np.concatenate([np.ones(N), np.zeros(N), np.zeros(N), np.where(r < 0.1, 10.0, 0.01)])
steps2 = int(round(0.3 / (cfl2 / nx)))
P = final2(nx, ny, steps2, cfl2, 0, W0, G)
rho = P[:N].reshape(ny, nx)
sym = dict(steps=steps2, finite=bool(np.isfinite(P).all()), max_rho=float(rho.max()),
           transpose_asym=float(np.max(np.abs(rho - rho.T))), mirror_asym=float(np.max(np.abs(rho - rho[:, ::-1]))))
print(json.dumps(sym), flush=True)
report['blast_symmetry'] = sym
np.save(out / 'blast2d-rho.npy', rho)

# 3. gradient vs central differences
rng = np.random.default_rng(5)
nx, ny = 12, 10; N = nx * ny; m = 4 * N
X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
r = np.hypot(X - 0.4, Y - 0.5).ravel()
W0 = np.concatenate([1 + 0.2 * np.sin(5 * X.ravel()) * np.cos(4 * Y.ravel()), 0.1 * np.sin(3 * Y.ravel() + 2.1 * X.ravel()),
                     -0.1 * np.cos(2 * X.ravel() - 1.3 * Y.ravel()), (0.2 + 4.8 * np.exp(-(r / 0.15) ** 2)) * (1 + 0.1 * np.sin(7 * X.ravel() + 0.9 * Y.ravel()))])
kst = np.array([6, 12], dtype=np.int32)
obs = np.concatenate([final2(nx, ny, 6, 0.25, 0, W0 * 1.01, G), final2(nx, ny, 12, 0.25, 0, W0 * 1.01, G)])
w = np.ones(m)
grows = []
for solver in (0, 1):
    W0b = np.zeros(m); gb = ct.c_double()
    J = l2.srhd2_obs(nx, ny, 12, 0.25, solver, W0, G, 2, kst, obs, w, W0b.ctypes.data, ct.addressof(gb), None)
    worst = []
    for trial in range(3):
        d = rng.normal(size=m) * np.abs(W0) * 1e-2; dg = 1e-2 * rng.normal()
        res = {}
        for h in (1e-4, 1e-5, 1e-6):
            fp = l2.srhd2_obs(nx, ny, 12, 0.25, solver, W0 + h * d, G + h * dg, 2, kst, obs, w, None, None, None)
            fm = l2.srhd2_obs(nx, ny, 12, 0.25, solver, W0 - h * d, G - h * dg, 2, kst, obs, w, None, None, None)
            fd = (fp - fm) / (2 * h)
            res[h] = abs(W0b @ d + gb.value * dg - fd) / abs(fd)
        worst.append(min(res.values()))
    grows.append(dict(c2p=['newton', 'bisection'][solver], J=J, best_step_rel_errors=worst))
    print(json.dumps(grows[-1]), flush=True)
report['gradient'] = grows

# 4. dot-product test: adjoint against the CuAD tangent (no finite differences, so limiter kinks do not enter)
l2.srhd2_obs_tangent.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ptr, ptr, ct.c_double, ct.c_double,
                                 ct.c_int, iptr, ptr, ptr]
l2.srhd2_obs_tangent.restype = ct.c_double
W0b = np.zeros(m); gb = ct.c_double()
l2.srhd2_obs(nx, ny, 12, 0.25, 0, W0, G, 2, kst, obs, w, W0b.ctypes.data, ct.addressof(gb), None)
drows = []
for trial in range(3):
    d = rng.normal(size=m) * np.abs(W0) * 1e-2; dg = 1e-2 * rng.normal()
    Jd = l2.srhd2_obs_tangent(nx, ny, 12, 0.25, W0, d, G, dg, 2, kst, obs, w)
    adj = W0b @ d + gb.value * dg
    drows.append(dict(tangent=Jd, adjoint=adj, rel=abs(adj - Jd) / abs(Jd)))
    print(json.dumps(drows[-1]), flush=True)
report['dot_test'] = drows
(out / 'check2d.json').write_text(json.dumps(report, indent=2) + '\n')
