"""1. Primal order of accuracy on a smooth problem with an exact solution: a density wave advected at uniform velocity
   v = 0.5 and uniform pressure (a relativistic contact: rho(x, t) = rho0(x - v t), u and p constant), Gaussian bump
   kept away from the outflow boundaries, t = 0.5, n = 64 .. 1024, L1 and Linf errors and observed orders.
2. Taylor test of the 2-D adjoint (srhd2d_host.so, generic smooth fields, two observation times): the remainder
   |J(W0 + h d) - J(W0) - h grad J . d| must fall like h^2 (first-order term exact), for h = 1e-1 .. 1e-6, both C2P
   solvers. Independent of finite-difference step choice."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent / 'out'
ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
iptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
l1 = ct.CDLL(str(out / 'srhd_host.so'))
l1.srhd_run_field.argtypes = [ct.c_int, ct.c_int, ct.c_double, ct.c_double, ptr, ct.c_int, ptr]
report = {'smooth_order': [], 'taylor_2d': []}

G, v0, T, cfl = 5 / 3, 0.5, 0.5, 0.4
u0 = v0 / np.sqrt(1 - v0 * v0)
rho_exact = lambda x: 1 + 0.5 * np.exp(-((x - 0.3) / 0.06) ** 2)
prev = None
for n in (64, 128, 256, 512, 1024):
    x = (np.arange(n) + 0.5) / n
    steps = int(round(T / (cfl / n)))
    t = steps * cfl / n
    W0 = np.concatenate([rho_exact(x), np.full(n, u0), np.full(n, 1.0)])
    for solver in (0, 1):
        P = np.zeros(3 * n); l1.srhd_run_field(n, steps, cfl, G, W0, solver, P)
        e = P[:n] - rho_exact(x - v0 * t)
        row = dict(n=n, c2p=['newton', 'bisection'][solver], L1=float(np.mean(np.abs(e))), Linf=float(np.max(np.abs(e))),
                   u_dev=float(np.max(np.abs(P[n:2 * n] - u0))), p_dev=float(np.max(np.abs(P[2 * n:] - 1.0))))
        report['smooth_order'].append(row)
rows = report['smooth_order']
for solver in ('newton', 'bisection'):
    r = [x for x in rows if x['c2p'] == solver]
    for a, b in zip(r[:-1], r[1:]):
        b['order_L1'] = float(np.log2(a['L1'] / b['L1'])); b['order_Linf'] = float(np.log2(a['Linf'] / b['Linf']))
for x in rows:
    print(json.dumps(x), flush=True)

l2 = ct.CDLL(str(out / 'srhd2d_host.so'))
l2.srhd2_obs.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ptr, ct.c_double, ct.c_int, iptr, ptr,
                         ptr, ct.c_void_p, ct.c_void_p, ct.c_void_p]
l2.srhd2_obs.restype = ct.c_double
nx, ny = 16, 12; N = nx * ny; m = 4 * N
X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
X, Y = X.ravel(), Y.ravel()
r = np.hypot(X - 0.4, Y - 0.5)
W0 = np.concatenate([1 + 0.2 * np.sin(5 * X + 0.3) * np.cos(4 * Y), 0.1 * np.sin(3 * Y + 2.1 * X),
                     -0.1 * np.cos(2 * X - 1.3 * Y), (0.2 + 4.8 * np.exp(-(r / 0.15) ** 2)) * (1 + 0.1 * np.sin(7 * X + 0.9 * Y))])
kst = np.array([8, 16], dtype=np.int32); w = np.ones(m)
obs = np.zeros(2 * m)
P = np.zeros(2 * m)
l2.srhd2_obs(nx, ny, 16, 0.25, 0, W0 * 1.02, G, 2, kst, obs, w, None, None, P.ctypes.data)
obs = P.copy()
rng = np.random.default_rng(13)
d = rng.normal(size=m) * np.abs(W0) * 1e-1
for solver in (0, 1):
    gW = np.zeros(m); gg = ct.c_double()
    J0 = l2.srhd2_obs(nx, ny, 16, 0.25, solver, W0, G, 2, kst, obs, w, gW.ctypes.data, ct.addressof(gg), None)
    lin = gW @ d
    hs = [10.0 ** (-k) for k in range(1, 7)]
    rem = []
    for h in hs:
        Jh = l2.srhd2_obs(nx, ny, 16, 0.25, solver, W0 + h * d, G, 2, kst, obs, w, None, None, None)
        rem.append(abs(Jh - J0 - h * lin))
    orders = [float(np.log10(rem[i] / rem[i + 1])) for i in range(len(rem) - 1)]
    row = dict(c2p=['newton', 'bisection'][solver], J=J0, h=hs, remainder=rem, observed_orders=orders,
               first_order_only_remainder=[abs(l2.srhd2_obs(nx, ny, 16, 0.25, solver, W0 + h * d, G, 2, kst, obs, w,
                                                            None, None, None) - J0) for h in hs])
    report['taylor_2d'].append(row)
    print(json.dumps({k: row[k] for k in ('c2p', 'J', 'remainder', 'observed_orders')}), flush=True)
(out / 'smooth-and-taylor.json').write_text(json.dumps(report, indent=2) + '\n')
