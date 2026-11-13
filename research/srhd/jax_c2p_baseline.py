"""Existing implicit-differentiation tool as a baseline: the same conservative-to-primitive step rewritten in JAX,
gradient of sum P.Pb with respect to (U, gam) on the GPU.
  optx_newton     optimistix.root_find(Newton, ImplicitAdjoint)   - the user writes the residual f(p; U, gam)
  optx_bisection  optimistix.root_find(Bisection, ImplicitAdjoint) with the same bracket as srhd.cu
  jax_unrolled40  40 Newton iterations in lax.fori_loop, differentiated through (reverse mode)
Accuracy: dot test against central differences of the CuAD host primal (n = 4096), same states and seeds as
compare_c2p_gpu.py. Cost: median wall time of the jitted gradient (compilation excluded, block_until_ready).
Run with research/build/venv-jax."""
import ctypes as ct
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

jax.config.update('jax_enable_x64', True)
out = Path(__file__).parent / 'out'
REPS = 5

def residual(p, args):
    D, S, tau, g = args
    v = S / (tau + D + p)
    W = 1.0 / jnp.sqrt(1.0 - v * v)
    eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W)
    return (g - 1.0) * (D / W) * eps - p

def prims(p, D, S, tau):
    v = S / (tau + D + p)
    W = 1.0 / jnp.sqrt(1.0 - v * v)
    return D / W, W * v, p

def c2p_optx(U, g, solver):
    n = U.shape[0] // 3
    D, S, tau = U[:n], U[n:2 * n], U[2 * n:]
    pmin = jnp.maximum(jnp.abs(S) - tau - D, 0.0) + 1e-14
    def one(D, S, tau, pmin):
        args = (D, S, tau, g)
        if solver == 'newton':
            sol = optx.root_find(residual, optx.Newton(rtol=1e-14, atol=1e-30), jnp.maximum((g - 1.0) * tau, pmin), args,
                                 max_steps=60, throw=False, adjoint=optx.ImplicitAdjoint())
        else:
            hi = jnp.maximum(2.0 * (g - 1.0) * tau, 2.0 * pmin)
            sol = optx.root_find(residual, optx.Bisection(rtol=1e-15, atol=1e-30), 0.5 * (pmin + hi), args, max_steps=200,
                                 throw=False, options=dict(lower=pmin, upper=hi), adjoint=optx.ImplicitAdjoint())
        return prims(sol.value, D, S, tau)
    r, u, p = jax.vmap(one)(D, S, tau, pmin)
    return jnp.concatenate([r, u, p])

def c2p_unrolled(U, g, iters=40):
    n = U.shape[0] // 3
    D, S, tau = U[:n], U[n:2 * n], U[2 * n:]
    pmin = jnp.maximum(jnp.abs(S) - tau - D, 0.0) + 1e-14
    def body(_, p):
        v = S / (tau + D + p)
        W = 1.0 / jnp.sqrt(1.0 - v * v)
        eps = (tau + D * (1.0 - W) + p * (1.0 - W * W)) / (D * W)
        f = (g - 1.0) * (D / W) * eps - p
        cs2 = g * (g - 1.0) * eps / (1.0 + g * eps)
        return jnp.maximum(p - f / (v * v * cs2 - 1.0), pmin)
    p = jax.lax.fori_loop(0, iters, body, jnp.maximum((g - 1.0) * tau, pmin))
    r, u, p = prims(p, D, S, tau)
    return jnp.concatenate([r, u, p])

def cons(rho, u, p, G):
    W = np.sqrt(1 + u * u); v = u / W; h = 1 + G / (G - 1) * p / rho
    return np.concatenate([rho * W, rho * h * W * W * v, rho * h * W * W - p - rho * W])

host = None
if (out / 'srhd_adj_host.so').exists():
    host = ct.CDLL(str(out / 'srhd_adj_host.so'))
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    host.c2p_apply.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ptr]

methods = {
    'optx_newton': (lambda U, g: c2p_optx(U, g, 'newton'), 0),
    'optx_bisection': (lambda U, g: c2p_optx(U, g, 'bisection'), 1),
    'jax_unrolled40': (c2p_unrolled, 0),
}
rng = np.random.default_rng(11)
G = 5 / 3
rows = []
print('jax', jax.__version__, jax.devices(), flush=True)
for n in (1 << 12, 1 << 16, 1 << 20, 1 << 22):
    rho = rng.uniform(0.5, 2, n); u = rng.uniform(-3, 3, n); p = rng.uniform(0.01, 10, n)
    U = cons(rho, u, p, G); Pb = rng.normal(size=3 * n)
    Uj, Pbj = jnp.asarray(U), jnp.asarray(Pb)
    for name, (fn, host_solver) in methods.items():
        loss = jax.jit(jax.grad(lambda U_, g_: jnp.dot(fn(U_, g_), Pbj), argnums=(0, 1)))
        try:
            t0 = time.perf_counter(); Ub, gb = loss(Uj, jnp.float64(G)); Ub.block_until_ready(); first = time.perf_counter() - t0
            ts = []
            for r in range(REPS):
                t0 = time.perf_counter(); Ub, gb = loss(Uj, jnp.float64(G)); Ub.block_until_ready(); ts.append(time.perf_counter() - t0)
            row = dict(n=n, method=name, median_ms=1e3 * float(np.median(ts)), first_call_s=first, status='ok')
            if n <= 4096 and host is not None:
                dU = rng.normal(size=3 * n) * np.abs(U) * 1e-2
                def phi(t):
                    Q = np.zeros(3 * n); host.c2p_apply(host_solver, n, U + t * dU, np.array([G]), Q); return Q @ Pb
                fd = (phi(1e-5) - phi(-1e-5)) / 2e-5
                row['dot_test_rel_error'] = float(abs(np.asarray(Ub) @ dU - fd) / abs(fd))
        except Exception as e:  # e.g. out of memory for the unrolled tape at large n
            row = dict(n=n, method=name, status='failed: ' + type(e).__name__ + ': ' + str(e)[:200])
        rows.append(row); print(json.dumps(row), flush=True)
(out / 'jax-c2p-baseline.json').write_text(json.dumps(dict(jax=jax.__version__, rows=rows), indent=2) + '\n')
