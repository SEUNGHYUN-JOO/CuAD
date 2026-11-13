"""Figures of the manuscript (manuscript/implicit_srhd/figures/) from the raw results in research/srhd/out/.
Each figure reads only files written by the scripts named in its comment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter
import numpy as np

out = Path(__file__).parent / 'out'
figs = Path(__file__).resolve().parents[2] / 'manuscript' / 'implicit_srhd' / 'figures'
figs.mkdir(parents=True, exist_ok=True)

# Times New Roman when it is installed (also from a Windows font folder under WSL), else a metric clone.
for d in (Path('/mnt/c/Windows/Fonts'), Path('C:/Windows/Fonts'), Path('/usr/share/fonts/truetype/msttcorefonts')):
    for name in ('times.ttf', 'timesi.ttf', 'timesbd.ttf', 'timesbi.ttf', 'Times_New_Roman.ttf'):
        if (d / name).exists():
            font_manager.fontManager.addfont(str(d / name))
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.columnspacing': 1.2,
    'axes.titlelocation': 'left', 'figure.dpi': 200, 'savefig.dpi': 600,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
C = {'cuad': '#1f4e9a', 'enzyme': '#c0392b', 'tangent': '#7f8c8d', 'value': '#2c3e50', 'manual': '#27ae60',
     'jax': '#8e44ad', 'primal': '#000000'}
TWO, ONE = 6.69, 3.37   # AIP figure widths in inches (two columns, one column)


def new(w, h, **kw):
    return plt.subplots(figsize=(w, h), layout='constrained', **kw)


def top_legend(fig, axs, ncol):
    """One legend for all panels above them, built from the handles of every panel without repetition."""
    seen = {}
    for ax in np.ravel(axs):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc='outside upper center', ncol=ncol)


def nice_ticks(lo, hi, nbins=6, steps=(1, 2, 2.5, 5, 10)):
    """Linear ticks whose first and last values enclose [lo, hi]; the axis then starts and ends on a labelled tick."""
    t = MaxNLocator(nbins=nbins, steps=list(steps)).tick_values(lo, hi)
    i0 = max(i for i in range(len(t)) if t[i] <= lo + 1e-12 * abs(hi - lo))
    i1 = min(i for i in range(len(t)) if t[i] >= hi - 1e-12 * abs(hi - lo))
    return t[i0:i1 + 1]


def log_ticks(lo, hi):
    """Decade ticks enclosing [lo, hi], every decade or every few decades with both ends labelled."""
    e0, e1 = int(np.floor(np.log10(lo) + 1e-9)), int(np.ceil(np.log10(hi) - 1e-9))
    step = 1 if e1 - e0 <= 7 else 2 if e1 - e0 <= 12 else 3
    e0, e1 = step * int(np.floor(e0 / step)), step * int(np.ceil(e1 / step))
    return 10.0 ** np.arange(e0, e1 + 1, step), step


def snap(axs, which, lim=None):
    """Set the x or y range of the axes (shared range for a list) to start and end on labelled ticks."""
    axs = list(np.ravel(axs))
    if lim is None:
        iv = [getattr(ax.dataLim, 'intervalx' if which == 'x' else 'intervaly') for ax in axs]
        lim = (min(v[0] for v in iv), max(v[1] for v in iv))
    for ax in axs:
        axis = ax.xaxis if which == 'x' else ax.yaxis
        setlim = ax.set_xlim if which == 'x' else ax.set_ylim
        if axis.get_scale() == 'log':
            m = 1.15 if which == 'y' else 1.0   # markers never sit on the frame
            t, step = log_ticks(lim[0] / m, lim[1] * m)
            axis.set_major_locator(FixedLocator(t))
            minor = 10.0 ** np.arange(np.log10(t[0]), np.log10(t[-1]) + 1)
            axis.set_minor_locator(FixedLocator(minor) if step > 1 else LogLocator(subs=np.arange(2, 10)))
            axis.set_minor_formatter(NullFormatter())
        else:
            t = nice_ticks(*lim)
            axis.set_major_locator(FixedLocator(t))
        setlim(t[0], t[-1])


def snap_colorbar(im, cb, lo, hi):
    t = nice_ticks(lo, hi, nbins=6, steps=(1, 2, 5, 10))
    im.set_clim(t[0], t[-1])
    cb.set_ticks(t)


def save(fig, name):
    fig.savefig(figs / name)
    plt.close(fig)
    print('wrote', name)


# Fig. 3: derivative error against iteration count, analytic reference (tolerance_sweep.cpp)
rows = [json.loads(l) for l in (out / 'tolerance-sweep.jsonl').read_text().splitlines() if l.strip()]
fig, axs = new(TWO, 2.4, ncols=3, sharey=True)
clip = lambda v: np.maximum(np.array(v, dtype=float), 1e-17)
for ax, s, lab in zip(axs, ['newton', 'fixed_point', 'bisection'], ['(a) Newton', '(b) fixed point', '(c) bisection']):
    r = [x for x in rows if x['solver'] == s and x['t0'] == 1]
    K = np.array([x['K'] for x in r])
    ax.semilogy(K, clip([x['value_error'] for x in r]), '-', color=C['value'], label=r'value $|t_K-t^\ast|/t^\ast$')
    ax.semilogy(K, clip([x['cuad_implicit'] for x in r]), 'o-', color=C['cuad'], label='CuAD implicit adjoint')
    if s != 'bisection':
        ax.semilogy(K, clip([x['cuad_tangent_unrolled'] for x in r]), 's--', color=C['tangent'], mfc='none',
                    label='CuAD tangent of the iterations')
    ax.semilogy(K, clip([x['enzyme_unrolled'] for x in r]), 'x:', color=C['enzyme'], label='Enzyme (iterations)')
    ax.set_title(lab)
    ax.set_xlabel('iterations $K$')
    snap(ax, 'x', (0, 60 if s != 'newton' else 12))
snap(axs, 'y', (1e-17, max(ax.dataLim.intervaly[1] for ax in axs)))
axs[0].set_ylabel('relative error')
top_legend(fig, axs, 4)
save(fig, 'fig_iteration_sweep.pdf')

# Fig. 4: C2P Jacobian error over the state space (c2p_robustness.py)
z = np.load(out / 'c2p-robustness.npz')
fig, axs = new(TWO, 2.5, ncols=2)
lt, lw = np.log10(z['theta']), np.log10(z['W'])
edges = lambda c: np.concatenate([[1.5 * c[0] - 0.5 * c[1]], 0.5 * (c[1:] + c[:-1]), [1.5 * c[-1] - 0.5 * c[-2]]])
excluded = z['rec_bisect'] >= 1e-8   # primal less accurate than 1e-8: not in Table I
print('Fig. 4: states', excluded.size, 'excluded', int(excluded.sum()))
for ax, key, title, cl in ((axs[0], 'err_bisect', '(a) error of the CuAD adjoint of the bisection C2P',
                            r'$\log_{10}$ of the relative error'),
                           (axs[1], 'cond', r'(b) condition number of $\partial U/\partial P$',
                            r'$\log_{10}$ of the condition number')):
    v = np.log10(np.maximum(z[key], 1e-17))
    im = ax.pcolormesh(edges(lt), edges(lw), v, cmap='viridis', shading='flat', rasterized=True)
    ax.set_xlabel(r'$\log_{10}(p/\rho)$'); ax.set_ylabel(r'$\log_{10} W$'); ax.set_title(title)
    ax.tick_params(top=False, right=False)
    snap(ax, 'x', (lt[0], lt[-1])); snap(ax, 'y', (lw[0], lw[-1]))   # cell centres sit on the sampled states
    if key == 'err_bisect':
        T, Wg = np.meshgrid(lt, lw)
        ax.plot(T[excluded], Wg[excluded], 'x', color='white', ms=3.5, mew=0.8, clip_on=False)
    cb = plt.colorbar(im, ax=ax, label=cl)
    snap_colorbar(im, cb, float(np.floor(v.min())), float(np.ceil(v.max())))
save(fig, 'fig_c2p_robustness.pdf')

# Fig. 5: GPU cost of the C2P adjoint against cell count (compare_c2p_gpu.py, jax_c2p_baseline.py)
g = json.loads((out / 'c2p-gpu-compare.json').read_text())
jx = json.loads((out / 'jax-c2p-baseline.json').read_text())['rows']
fig, axs = new(TWO, 2.7, ncols=2, sharey=True)
for ax, solver, title in ((axs[0], 'newton', '(a) Newton C2P'), (axs[1], 'bisection', '(b) bisection C2P')):
    for tool, mk, lab in (('primal', '-', 'primal'), ('cuad', 'o-', 'CuAD implicit adjoint'),
                          ('manual', 'd-', 'hand-written implicit adjoint'), ('enzyme', 'x:', 'Enzyme (iterations)')):
        r = sorted([x for x in g if x['c2p'] == solver and x['tool'] == tool and x['status'] == 0], key=lambda x: x['n'])
        if r:
            ax.loglog([x['n'] for x in r], [x['median_ms'] for x in r], mk, color=C.get(tool, 'k'), label=lab,
                      mfc='none' if tool == 'manual' else None)
    jm = 'optx_newton' if solver == 'newton' else 'optx_bisection'
    r = sorted([x for x in jx if x['method'] == jm and x['status'] == 'ok'], key=lambda x: x['n'])
    ax.loglog([x['n'] for x in r], [x['median_ms'] for x in r], 's--', color=C['jax'], mfc='none',
              label='JAX with Optimistix implicit adjoint')
    if solver == 'newton':
        r = sorted([x for x in g if x['c2p'] == 'newton_fixed40' and x['tool'] == 'cuad'], key=lambda x: x['n'])
        ax.loglog([x['n'] for x in r], [x['median_ms'] for x in r], 'v-.', color=C['tangent'],
                  label='CuAD with 40 fixed iterations')
    ax.set_xlabel('cells'); ax.set_title(title)
    snap(ax, 'x')
snap(axs, 'y')
axs[0].set_ylabel('time per call (ms)')
top_legend(fig, axs, 3)
save(fig, 'fig_gpu_cost.pdf')

# Fig. 6: Riemann problems against the exact solution (check_primal.py)
fig, axs = new(TWO, 2.4, ncols=2)
for ax, name, title in zip(axs, ['MM1', 'MM2'], ['(a) problem 1', '(b) problem 2']):
    f = out / f'primal-{name}-1600.npz'
    if f.exists():
        z = np.load(f); n = z['x'].size
        ax.plot(z['x'], z['exact'][:, 0], 'k-', lw=0.8, label='exact solution')
        ax.plot(z['x'], z['P'][:n], '--', color=C['cuad'], lw=0.9, label='solver with $n=1600$')
        ax.set_title(title); ax.set_xlabel('$x$'); ax.set_ylabel(r'$\rho$')
        snap(ax, 'x', (0, 1)); snap(ax, 'y', (0, 1.05 * float(np.max(z['exact'][:, 0]))))
top_legend(fig, axs, 2)
save(fig, 'fig_riemann.pdf')

# Fig. 7: Taylor test of the 2-D adjoint (smooth_and_taylor.py)
st = json.loads((out / 'smooth-and-taylor.json').read_text())
fig, ax = new(ONE, 2.5)
for r, mk, col in zip(st['taylor_2d'], ('o-', 's--'), (C['cuad'], C['enzyme'])):
    lab = {'newton': 'Newton', 'bisection': 'bisection'}.get(r['c2p'], r['c2p'])
    ax.loglog(r['h'], r['remainder'], mk, color=col, label=f'second order with {lab} C2P',
              mfc='none' if mk == 's--' else None)
    ax.loglog(r['h'], r['first_order_only_remainder'], ':', color='gray')
ax.plot([], [], ':', color='gray', label='first order')
h = np.array(st['taylor_2d'][0]['h'])
ax.loglog(h, st['taylor_2d'][0]['remainder'][0] * (h / h[0]) ** 2, 'k-', lw=0.6, label='slope 2')
snap(ax, 'x'); snap(ax, 'y')
ax.set_xlabel('$h$'); ax.set_ylabel('remainder'); ax.legend(loc='lower right')
save(fig, 'fig_taylor.pdf')

# Fig. 8: 1-D medium reconstruction, exact gradient against differentiated bisection (inverse_density.py)
fig, axs = new(TWO, 2.6, ncols=2)
names = {'cuad': 'exact gradient (CuAD)', 'enzyme': 'differentiated bisection (Enzyme)'}
for tool, mk in (('cuad', '-'), ('enzyme', '--')):
    f = out / f'inverse-density-256-1-{tool}-noise0-case0.npz'
    j = out / f'inverse-density-256-1-{tool}-noise0-case0.json'
    if f.exists():
        z = np.load(f)
        axs[0].plot(z['x'], z['rho_fit'] - z['rho_true'], mk, color=C[tool], lw=0.8, label=names[tool])
        h = [x['rel_l2_error_rho0'] for x in json.loads(j.read_text())['history']]
        axs[1].semilogy(np.arange(1, len(h) + 1), h, mk, color=C[tool], label=names[tool])
axs[0].axhline(0, color='k', lw=0.4)
snap(axs[0], 'x', (0, 1)); snap(axs[0], 'y'); snap(axs[1], 'x', (0, axs[1].dataLim.intervalx[1])); snap(axs[1], 'y')
axs[0].set_xlabel('$x$'); axs[0].set_ylabel(r'$\rho_0^{\rm fit}-\rho_0^{\rm true}$')
axs[0].set_title('(a) error of the reconstructed density')
axs[1].set_xlabel('L-BFGS-B iteration'); axs[1].set_ylabel(r'relative error of $\rho_0$')
axs[1].set_title('(b) convergence')
top_legend(fig, axs, 2)
save(fig, 'fig_inverse1d.pdf')

# Fig. 9: 2-D reconstruction (inverse_density2d.py)
for f in sorted(out.glob('inverse2d-*-noise0.npz')):
    z = np.load(f); n = int(z['n'])
    fig, axs = new(TWO, 1.95, ncols=4)
    vmin, vmax = float(z['rho_true'].min()), float(z['rho_true'].max())
    err = z['rho_fit'] - z['rho_true']
    e = float(np.max(np.abs(err))) or 1.0
    obs = z['P_true'][-4 * n * n:-3 * n * n]
    panels = ((z['rho_true'], r'(a) true $\rho_0$', 'magma', (vmin, vmax)),
              (z['rho_fit'], r'(b) reconstructed $\rho_0$', 'magma', (vmin, vmax)),
              (err, '(c) error', 'RdBu_r', (-e, e)),
              (obs, r'(d) observed $\rho$ at $t=0.45$', 'magma', (0.0, float(obs.max()))))
    for ax, (field, title, cmap, (lo, hi)) in zip(axs, panels):
        im = ax.imshow(field.reshape(n, n), origin='lower', extent=[0, 1, 0, 1], cmap=cmap)
        ax.set_title(title, fontsize=7); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_aspect('equal'); ax.set_xlabel('$x$'); ax.tick_params(top=False, right=False)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
        snap_colorbar(im, cb, lo, hi)
        if cmap == 'RdBu_r':
            cb.formatter.set_powerlimits((0, 0)); cb.formatter.set_useMathText(True)
            cb.ax.yaxis.get_offset_text().set_fontsize(6)
    axs[0].set_ylabel('$y$')
    save(fig, f'fig_inverse2d_{n}.pdf')
