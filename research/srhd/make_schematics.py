"""Schematic figures of the manuscript (no raw data): the overview of the transformation and the derivative of
bisection against the implicit derivative on t^3 + a t - b = 0."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import make_figures as mf   # shared style, palette and axis helpers (also writes the data figures)

figs = mf.figs
C = mf.C
INK, MUTED, RULE = '#1d2733', '#5b6675', '#9aa3ad'
FILL = {'in': '#eef2f7', 'ana': '#e8eef8', 'c1': '#e6eef9', 'c2': '#fbeee6', 'c3': '#eaf5ec', 'chk': '#f7f7f7',
        'adj': '#ffffff', 'rej': '#fbeaea'}


def box(ax, x, y, w, h, fc, ec=RULE, lw=0.7, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def text(ax, x, y, s, size=7, **kw):
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center'); kw.setdefault('color', INK)
    ax.text(x, y, s, fontsize=size, **kw)


def arrow(ax, p, q, color=INK, lw=0.8, style='-|>', rad=0.0):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=lw, color=color, mutation_scale=7,
                                                    shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


# Fig. 1: overview of the transformation
fig = plt.figure(figsize=(mf.TWO, 3.15))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis('off')
heads = ((10.5, 'CUDA kernel'), (31, 'static analysis'), (52, 'class'), (72.5, 'run-time check'), (91.5, 'implicit adjoint'))
for x, s in heads:
    text(ax, x, 45.2, s, size=7.5, style='italic', color=MUTED)
ax.plot([1, 99], [43.4, 43.4], color=RULE, lw=0.5)

# input kernel
box(ax, 1, 9, 19.5, 32.5, FILL['in'])
code = ('p = guess(U);\n'
        'for (it = 0; it < 60; ++it) {\n'
        '  f  = R(p, U);\n'
        '  dp = f / R_p(p, U);\n'
        '  p  = fmax(p - dp, pmin);\n'
        '  if (fabs(dp) <= tol*p)\n'
        '    break;\n'
        '}\n'
        'P[i] = prims(p, U);')
text(ax, 2.2, 25.5, code, size=5.0, family='monospace', ha='left', linespacing=1.5)
text(ax, 10.5, 4.6, 'forward sweep runs the loop\nas written and stores nothing', size=6, style='italic', color=MUTED)

# analysis
box(ax, 23.5, 16, 15, 17, FILL['ana'], ec=C['cuad'], lw=0.9)
text(ax, 31, 29.6, 'loop with top-level', size=6.5)
text(ax, 31, 26.9, r'$\mathtt{if\ (c)\ break;}$', size=6.5)
text(ax, 31, 22.2, r'state $x$   inputs $p$', size=7)
text(ax, 31, 19.0, r'does the update of $x$' '\n' r'depend on $p$?', size=6.3)
box(ax, 23.5, 2, 15, 8.5, FILL['rej'], ec=C['enzyme'], lw=0.7)
text(ax, 31, 6.25, 'memory write, nested loop\nor call: rejected with\nthe reason', size=5.8, color='#7b241c')
arrow(ax, (20.2, 25), (23.3, 25))
arrow(ax, (31, 15.8), (31, 10.7), color=C['enzyme'])

# classes, checks and adjoints in three rows
rows = (
    (36.0, 'c1', 'I   smooth update', r'$x\leftarrow G(x,p)$', 'Newton, fixed point',
     'one more iteration\nmoves less than ' r'$\epsilon_s$',
     r'$\lambda=\bar x+G_x^{T}\lambda$' '\n' r'$\bar p\leftarrow\bar p+G_p^{T}\lambda$'),
    (23.0, 'c2', 'II   bracketing', r'branch on $r(m,p)$', 'bisection',
     'bounds collapsed\nand ' r'$r(m)=0$', r'$\bar p\leftarrow\bar p-r_p\,s/r_m$'),
    (10.0, 'c3', 'III   several states', r'residual $R(t,p)$ found', 'secant, Illinois, Brent',
     'residual vanishes\ncandidates agree', r'$\bar p\leftarrow\bar p-R_p\,s/R_t$'),
)
ecol = {'c1': C['cuad'], 'c2': '#b9770e', 'c3': '#1e8449'}
for y, key, title, rel, ex, chk, adj in rows:
    box(ax, 42, y - 5, 20, 10, FILL[key], ec=ecol[key], lw=0.9)
    text(ax, 52, y + 2.6, title, size=7, weight='bold', color=ecol[key])
    text(ax, 52, y - 0.2, rel, size=6.8)
    text(ax, 52, y - 3.0, ex, size=5.9, style='italic', color=MUTED)
    box(ax, 65.5, y - 5, 14, 10, FILL['chk'])
    text(ax, 72.5, y + 1.0, chk, size=6.0)
    text(ax, 72.5, y - 3.4, 'otherwise trap', size=5.8, color=C['enzyme'], style='italic')
    box(ax, 83, y - 5, 16, 10, FILL['adj'], ec=ecol[key], lw=0.9)
    text(ax, 91, y, adj, size=7)
    arrow(ax, (38.7, 24.5 + (y - 23) * 0.12), (41.8, y))
    arrow(ax, (62.2, y), (65.3, y))
    arrow(ax, (79.7, y), (82.8, y))
fig.savefig(figs / 'fig_overview.pdf'); plt.close(fig); print('wrote fig_overview.pdf')

# Fig. 2: what differentiating bisection computes, on t^3 + a t - b = 0 with b = 5
b = 5.0
a = np.linspace(1.0, 3.0, 1201)
def root(av):
    q = -b; D = q * q / 4 + av ** 3 / 27
    return np.cbrt(-q / 2 + np.sqrt(D)) + np.cbrt(-q / 2 - np.sqrt(D))
def bisect(av, K, lo=0.0, hi=4.0):
    lo = np.full_like(av, lo); hi = np.full_like(av, hi)
    for _ in range(K):
        m = 0.5 * (lo + hi); up = m ** 3 + av * m - b > 0
        hi = np.where(up, m, hi); lo = np.where(up, lo, m)
    return 0.5 * (lo + hi)
def newton(av, K, t0=2.0):
    """K Newton steps and the derivative of the iterate with respect to a (forward mode through the iterations)."""
    tk = np.full_like(av, t0); dk = np.zeros_like(av)
    for _ in range(K):
        f, ft = tk ** 3 + av * tk - b, 3 * tk ** 2 + av
        df, dft = ft * dk + tk, 6 * tk * dk + 1
        tk, dk = tk - f / ft, dk - (df * ft - f * dft) / ft ** 2
    return tk, dk
t = root(a); dt = -t / (3 * t * t + a)
fig, axs = mf.new(mf.TWO, 2.5, ncols=2)
ax = axs[0]
for K, col in ((4, '#c8c8c8'), (6, '#9a9a9a'), (8, '#5c5c5c')):
    ax.plot(a, bisect(a, K), '-', color=col, lw=0.9, drawstyle='steps-mid', label=f'bisection after $K={K}$ steps')
ax.plot(a, t, '-', color=C['cuad'], lw=1.4, label=r'root $t^\ast(a)$', zorder=3)
ax.set_xlabel('$a$'); ax.set_ylabel('$t$'); ax.set_title('(a) value')
mf.snap(ax, 'x', (1, 3)); mf.snap(ax, 'y', (1.2, 1.6))
ax.legend(loc='upper right')
ax = axs[1]
for K, col in ((1, '#c8c8c8'), (2, '#9a9a9a'), (3, '#5c5c5c')):
    ax.plot(a, newton(a, K)[1], '-', color=col, lw=0.9, label=f'differentiated Newton, $K={K}$')
ax.plot(a, dt, '-', color=C['cuad'], lw=1.4, label=r'implicit $-t^\ast/(3t^{\ast2}+a)$', zorder=3)
ax.plot(a, np.zeros_like(a), '--', color=C['enzyme'], lw=1.5, label='differentiated bisection (every $K$)')
ax.set_xlabel('$a$'); ax.set_ylabel(r'$dt/da$'); ax.set_title('(b) derivative')
mf.snap(ax, 'x', (1, 3)); mf.snap(ax, 'y', (-0.4, 0.05))
ax.legend(loc='lower right')
mf.save(fig, 'fig_bisection.pdf')
