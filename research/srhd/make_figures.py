"""Figures of the manuscript (manuscript/implicit_srhd/figures/) from the raw results in research/srhd/out/.
Each figure reads only files written by the scripts named in its comment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

out = Path(__file__).parent / 'out'
figs = Path(__file__).resolve().parents[2] / 'manuscript' / 'implicit_srhd' / 'figures'
figs.mkdir(parents=True, exist_ok=True)

# Style of the figures in APL Computational Physics articles: sans-serif text, closed black frame, dashed light grid,
# coloured lines with black-edged markers, centred panel titles, framed legends inside the axes.
INK, MUTED, GRID, AXIS = '#000000', '#555555', '#c8c8c8', '#000000'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': AXIS, 'ytick.color': AXIS,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.spines.top': True, 'axes.spines.right': True,
    'axes.grid': True, 'axes.grid.which': 'major', 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '--',
    'axes.axisbelow': True,
    'lines.linewidth': 1.2, 'lines.markersize': 4.5, 'lines.markeredgecolor': 'black', 'lines.markeredgewidth': 0.6,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'legend.frameon': True, 'legend.edgecolor': '0.6', 'legend.fancybox': False, 'legend.framealpha': 1.0,
    'legend.handlelength': 2.2, 'legend.columnspacing': 1.4, 'legend.handletextpad': 0.5,
    'axes.titlelocation': 'center', 'axes.titlepad': 5, 'figure.dpi': 200, 'savefig.dpi': 600,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
# one colour and one marker per method in every figure
C = {'cuad': '#1f77b4', 'enzyme': '#d62728', 'manual': '#2ca02c', 'jax': '#9467bd', 'fixed': '#17becf',
     'tangent': '#7f7f7f', 'value': INK, 'primal': INK}
TWO, ONE = 6.69, 3.37   # AIP figure widths in inches (two columns, one column)


def new(w, h, **kw):
    return plt.subplots(figsize=(w, h), layout='constrained', **kw)


def title(ax, letter, text):
    ax.set_title(f'({letter}) {text}')


def top_legend(fig, axs, ncol):
    """One legend for all panels above them, built from the handles of every panel without repetition."""
    seen = {}
    for ax in np.ravel(axs):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc='outside upper center', ncol=ncol, frameon=False)


def note(ax, xy, xytext, s, color=INK, ha='left', **kw):
    """A direct annotation with a thin leader line."""
    ax.annotate(s, xy=xy, xytext=xytext, fontsize=6.8, color=color, ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=color, shrinkA=1.5, shrinkB=2), **kw)


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
    cb.outline.set_linewidth(0.5)


def field_axes(ax):
    ax.grid(False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(True)


def save(fig, name):
    fig.savefig(figs / name)
    plt.close(fig)
    print('wrote', name)


if __name__ == '__main__' or __name__ == 'make_figures':
    # Fig. 3: derivative error against iteration count, analytic reference (tolerance_sweep.cpp)
    rows = [json.loads(l) for l in (out / 'tolerance-sweep.jsonl').read_text().splitlines() if l.strip()]
    fig, axs = new(TWO, 2.45, ncols=3, sharey=True)
    clip = lambda v: np.maximum(np.array(v, dtype=float), 1e-17)
    for ax, s, (letter, lab) in zip(axs, ['newton', 'fixed_point', 'bisection'],
                                    [('a', 'Newton'), ('b', 'fixed point'), ('c', 'bisection')]):
        r = [x for x in rows if x['solver'] == s and x['t0'] == 1]
        K = np.array([x['K'] for x in r])
        me = 1 if s == 'newton' else 5   # markers on every fifth iterate where K runs to 60
        ax.semilogy(K, clip([x['value_error'] for x in r]), '-', color=C['value'], lw=1.0,
                    label=r'value error $|t_K-t^\ast|/t^\ast$')
        if s != 'bisection':
            ax.semilogy(K, clip([x['cuad_tangent_unrolled'] for x in r]), 's-', color=C['tangent'], lw=1.0, ms=3.4,
                        markevery=me, label='CuAD tangent of the iterations')
        ax.semilogy(K, clip([x['enzyme_unrolled'] for x in r]), 'D-', color=C['enzyme'], lw=1.0, ms=3.2,
                    markevery=me, label='Enzyme reverse of the iterations')
        ax.semilogy(K, clip([x['cuad_implicit'] for x in r]), 'o-', color=C['cuad'], markevery=me,
                    label='CuAD implicit adjoint')
        title(ax, letter, lab)
        ax.set_xlabel('iterations $K$')
        snap(ax, 'x', (0, 60 if s != 'newton' else 12))
    snap(axs, 'y', (1e-17, max(ax.dataLim.intervaly[1] for ax in axs)))
    axs[0].set_ylabel('relative error of $dt/d(a,b)$')
    note(axs[2], (44, 1.0), (31, 1e-4), 'differentiated bisection:\nerror 1 for every $K$', color=C['enzyme'])
    note(axs[0], (3, 4e-12), (5.2, 1e-8), 'implicit error\n' r'$\approx$ (value error)$^2$', color=C['cuad'])
    top_legend(fig, axs, 4)
    save(fig, 'fig_iteration_sweep.pdf')

    # Fig. 4: C2P Jacobian error over the state space (c2p_robustness.py)
    z = np.load(out / 'c2p-robustness.npz')
    fig, axs = new(TWO, 2.5, ncols=2)
    lt, lw = np.log10(z['theta']), np.log10(z['W'])
    edges = lambda c: np.concatenate([[1.5 * c[0] - 0.5 * c[1]], 0.5 * (c[1:] + c[:-1]), [1.5 * c[-1] - 0.5 * c[-2]]])
    excluded = z['rec_bisect'] >= 1e-8   # primal less accurate than 1e-8: not in Table I
    print('Fig. 4: states', excluded.size, 'excluded', int(excluded.sum()))
    for ax, key, (letter, ttl), cl, cmap in (
            (axs[0], 'err_bisect', ('a', 'error of the CuAD adjoint of the bisection C2P'),
             r'$\log_{10}$ relative error', 'viridis'),
            (axs[1], 'cond', ('b', r'condition number of $\partial U/\partial P$'),
             r'$\log_{10}$ condition number', 'cividis')):
        v = np.log10(np.maximum(z[key], 1e-17))
        im = ax.pcolormesh(edges(lt), edges(lw), v, cmap=cmap, shading='flat', rasterized=True)
        field_axes(ax)
        ax.set_xlabel(r'$\log_{10}(p/\rho)$'); ax.set_ylabel(r'$\log_{10} W$'); title(ax, letter, ttl)
        snap(ax, 'x', (lt[0], lt[-1])); snap(ax, 'y', (lw[0], lw[-1]))   # cell centres sit on the sampled states
        if key == 'err_bisect':
            T, Wg = np.meshgrid(lt, lw)
            ax.plot(T[excluded], Wg[excluded], 'x', color='white', ms=3.5, mew=0.9, clip_on=False,
                    label='primal itself less accurate than $10^{-8}$')
            ax.legend(loc='lower right', fontsize=6.3, labelcolor='white', handletextpad=0.2, borderaxespad=0.4,
                      frameon=False)
        cb = plt.colorbar(im, ax=ax, label=cl, aspect=25)
        snap_colorbar(im, cb, float(np.floor(v.min())), float(np.ceil(v.max())))
    save(fig, 'fig_c2p_robustness.pdf')

    # Fig. 5: GPU cost of the C2P adjoint against cell count (compare_c2p_gpu.py, jax_c2p_baseline.py)
    g = json.loads((out / 'c2p-gpu-compare.json').read_text())
    jx = json.loads((out / 'jax-c2p-baseline.json').read_text())['rows']
    fig, axs = new(TWO, 2.75, ncols=2, sharey=True)
    N20 = 1 << 20
    for ax, solver, (letter, ttl) in ((axs[0], 'newton', ('a', 'Newton C2P')), (axs[1], 'bisection', ('b', 'bisection C2P'))):
        series = {}
        for tool, mk, lab in (('primal', '-', 'primal'), ('enzyme', 'D-', 'Enzyme reverse of the iterations'),
                              ('manual', '^-', 'hand-written implicit adjoint'), ('cuad', 'o-', 'CuAD implicit adjoint')):
            r = sorted([x for x in g if x['c2p'] == solver and x['tool'] == tool and x['status'] == 0], key=lambda x: x['n'])
            if r:
                n_, t_ = [x['n'] for x in r], [x['median_ms'] for x in r]
                series[tool] = dict(zip(n_, t_))
                ax.loglog(n_, t_, mk, color=C[tool], label=lab, lw=1.0 if tool == 'primal' else 1.4,
                          ms=0 if tool == 'primal' else (3.6 if tool == 'enzyme' else 4.2), zorder=4 if tool == 'cuad' else 3)
        jm = 'optx_newton' if solver == 'newton' else 'optx_bisection'
        r = sorted([x for x in jx if x['method'] == jm and x['status'] == 'ok'], key=lambda x: x['n'])
        ax.loglog([x['n'] for x in r], [x['median_ms'] for x in r], 's-', color=C['jax'], ms=3.6,
                  label='JAX with Optimistix implicit adjoint')
        if solver == 'newton':
            r = sorted([x for x in g if x['c2p'] == 'newton_fixed40' and x['tool'] == 'cuad'], key=lambda x: x['n'])
            ax.loglog([x['n'] for x in r], [x['median_ms'] for x in r], 'v-', color=C['fixed'], ms=3.8,
                      label='CuAD with 40 fixed iterations')
        ax.set_xlabel('cells'); title(ax, letter, ttl)
        snap(ax, 'x')
        # the gap between the differentiated iterations and the implicit adjoint at 2^20 cells
        te, tc = series['enzyme'][N20], series['cuad'][N20]
        ax.annotate('', xy=(N20, tc * 1.25), xytext=(N20, te / 1.25),
                    arrowprops=dict(arrowstyle='<->', lw=0.7, color=INK, shrinkA=0, shrinkB=0))
        ax.text(N20 / 1.35, 1.6e2 if solver == 'newton' else 3e2, rf'$\times{te / tc:.0f}$', fontsize=7.5,
                ha='right', va='center', color=INK,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
    snap(axs, 'y')
    axs[0].set_ylabel('time per gradient (ms)')
    top_legend(fig, axs, 3)
    save(fig, 'fig_gpu_cost.pdf')

    # Fig. 6: Riemann problems against the exact solution (check_primal.py)
    fig, axs = new(TWO, 2.45, ncols=2)
    for ax, name, (letter, ttl) in zip(axs, ['MM1', 'MM2'], [('a', 'problem 1'), ('b', 'problem 2')]):
        f = out / f'primal-{name}-1600.npz'
        if f.exists():
            z = np.load(f); n = z['x'].size
            ax.plot(z['x'], z['exact'][:, 0], '-', color=INK, lw=0.9, label='exact solution')
            ax.plot(z['x'], z['P'][:n], '-', color=C['cuad'], lw=1.2, alpha=0.9, label='solver with $n=1600$')
            title(ax, letter, ttl); ax.set_xlabel('$x$'); ax.set_ylabel(r'$\rho$')
            snap(ax, 'x', (0, 1)); snap(ax, 'y', (0, 1.05 * float(np.max(z['exact'][:, 0]))))
            if name == 'MM2':   # the thin shell, magnified
                ins = inset_axes(ax, width='34%', height='52%', loc='upper left', borderpad=1.0,
                                 bbox_to_anchor=(0.2, 0, 1, 1), bbox_transform=ax.transAxes)
                ins.plot(z['x'], z['exact'][:, 0], '-', color=INK, lw=0.9)
                ins.plot(z['x'], z['P'][:n], '-', color=C['cuad'], lw=1.2)
                ins.set_xlim(0.875, 0.905); ins.set_ylim(0, 12); ins.set_xticks([0.88, 0.9]); ins.set_yticks([0, 5, 10])
                ins.tick_params(labelsize=6); ins.grid(False)
                for s in ('top', 'right'):
                    ins.spines[s].set_visible(True)
                ins.set_title('thin shell', fontsize=6.5, loc='center', pad=2, color=MUTED,
                              bbox=dict(fc='white', ec='none', pad=0.6))
                ax.add_patch(plt.Rectangle((0.875, 0), 0.03, 11.4, fc='none', ec=MUTED, lw=0.6, ls='--'))
    top_legend(fig, axs, 2)
    save(fig, 'fig_riemann.pdf')

    # Fig. 7: Taylor test of the 2-D adjoint (smooth_and_taylor.py)
    st = json.loads((out / 'smooth-and-taylor.json').read_text())
    fig, ax = new(ONE, 2.55)
    h = np.array(st['taylor_2d'][0]['h'])
    for r, mk, col in zip(st['taylor_2d'], ('o', 's'), (C['cuad'], C['enzyme'])):
        lab = {'newton': 'Newton', 'bisection': 'bisection'}.get(r['c2p'], r['c2p'])
        ax.loglog(r['h'], r['first_order_only_remainder'], mk + ':', color=col, ms=3.4, lw=0.8, alpha=0.45)
        ax.loglog(r['h'], r['remainder'], mk + '-', color=col, label=f'{lab} C2P', ms=4.4 if mk == 'o' else 3.6)
    ref = st['taylor_2d'][0]['remainder'][0] * (h / h[0]) ** 2
    ax.loglog(h, ref, '-', color=MUTED, lw=0.6, zorder=1)
    snap(ax, 'x'); snap(ax, 'y')
    ax.text(3e-4, 2.2e-12, 'second order\n' r'$|J(W_0{+}hd)-J-h\nabla J\cdot d|$', fontsize=6.6, color=INK, ha='left')
    ax.text(1.3e-6, 5e-7, 'first order\n' r'$|J(W_0{+}hd)-J|$', fontsize=6.6, color=MUTED, ha='left')
    ax.set_xlabel('$h$'); ax.set_ylabel('remainder'); ax.legend(loc='lower right')
    save(fig, 'fig_taylor.pdf')

    # Fig. 8: 1-D medium reconstruction, exact gradient against differentiated bisection (inverse_density.py)
    fig, axs = new(TWO, 2.65, ncols=2)
    names = {'cuad': 'exact gradient (CuAD)', 'enzyme': 'differentiated bisection (Enzyme)'}
    hist = {}
    for tool in ('enzyme', 'cuad'):
        f = out / f'inverse-density-256-1-{tool}-noise0-case0.npz'
        j = out / f'inverse-density-256-1-{tool}-noise0-case0.json'
        if f.exists():
            z = np.load(f)
            d = z['rho_fit'] - z['rho_true']
            axs[0].plot(z['x'], d, '-', color=C[tool], lw=0.9, label=names[tool])
            hist[tool] = [x['rel_l2_error_rho0'] for x in json.loads(j.read_text())['history']]
            axs[1].semilogy(np.arange(1, len(hist[tool]) + 1), hist[tool], '-', color=C[tool], label=names[tool])
    he = hist['enzyme']; hc = hist['cuad']
    axs[1].plot([len(he)], [he[-1]], 'o', color=C['enzyme'], ms=5)
    axs[1].plot([len(hc)], [hc[-1]], 'o', color=C['cuad'], ms=5)
    snap(axs[0], 'x', (0, 1)); snap(axs[0], 'y'); snap(axs[1], 'x', (0, len(hc))); snap(axs[1], 'y')
    note(axs[1], (len(he), he[-1]), (70, 0.12), f'stops after {len(he)} iterations\nand reports convergence',
         color=C['enzyme'])
    note(axs[1], (len(hc), hc[-1]), (215, 3e-2), f'{100 * hc[-1]:.2f}% after {len(hc)} iterations', color=C['cuad'])
    axs[0].set_xlabel('$x$'); axs[0].set_ylabel(r'$\rho_0^{\rm fit}-\rho_0^{\rm true}$')
    title(axs[0], 'a', 'error of the reconstructed density')
    axs[1].set_xlabel('L-BFGS-B iteration'); axs[1].set_ylabel(r'relative error of $\rho_0$')
    title(axs[1], 'b', 'convergence')
    top_legend(fig, axs, 2)
    save(fig, 'fig_inverse1d.pdf')

    # Fig. 9: 2-D reconstruction (inverse_density2d.py)
    for f in sorted(out.glob('inverse2d-*-noise0.npz')):
        z = np.load(f); n = int(z['n'])
        fig, axs = new(TWO, 2.0, ncols=4)
        vmin, vmax = float(z['rho_true'].min()), float(z['rho_true'].max())
        err = z['rho_fit'] - z['rho_true']
        e = float(np.max(np.abs(err))) or 1.0
        obs = z['P_true'][-4 * n * n:-3 * n * n]
        panels = ((z['rho_true'], ('a', r'true $\rho_0$'), 'magma', (vmin, vmax)),
                  (z['rho_fit'], ('b', r'reconstructed $\rho_0$'), 'magma', (vmin, vmax)),
                  (err, ('c', 'error'), 'RdBu_r', (-e, e)),
                  (obs, ('d', r'observed $\rho$ at $t=0.45$'), 'magma', (0.0, float(obs.max()))))
        for ax, (field, (letter, ttl), cmap, (lo, hi)) in zip(axs, panels):
            im = ax.imshow(field.reshape(n, n), origin='lower', extent=[0, 1, 0, 1], cmap=cmap, interpolation='bicubic')
            field_axes(ax)
            ax.set_title(f'({letter}) ' + ttl, fontsize=7)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.set_aspect('equal'); ax.set_xlabel('$x$')
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
            snap_colorbar(im, cb, lo, hi)
            if cmap == 'RdBu_r':
                cb.formatter.set_powerlimits((0, 0)); cb.formatter.set_useMathText(True)
                cb.ax.yaxis.get_offset_text().set_fontsize(6)
        axs[0].set_ylabel('$y$')
        save(fig, f'fig_inverse2d_{n}.pdf')
