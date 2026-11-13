"""Schematic figures of the manuscript (no raw data): Fig. 1, the overview of the transformation (TikZ source
fig_overview.tex, compiled with pdflatex), and Fig. 2, the derivative of bisection against the implicit derivative on
t^3 + a t - b = 0. Importing make_figures also writes every data figure."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import make_figures as mf   # shared style, palette and axis helpers

here = Path(__file__).resolve().parent

# Fig. 1: overview (TikZ)
with tempfile.TemporaryDirectory() as tmp:
    shutil.copy(here / 'fig_overview.tex', tmp)
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'fig_overview.tex'], cwd=tmp, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copy(Path(tmp) / 'fig_overview.pdf', mf.figs / 'fig_overview.pdf')
print('wrote fig_overview.pdf')

# Fig. 2: what differentiating bisection computes, on t^3 + a t - b = 0 with b = 5
b = 5.0
a = np.linspace(1.0, 3.0, 2001)


def root(av):
    D = b * b / 4 + av ** 3 / 27
    return np.cbrt(b / 2 + np.sqrt(D)) + np.cbrt(b / 2 - np.sqrt(D))


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
C, INK, MUTED = mf.C, mf.INK, mf.MUTED
shade = ['#f7c3ac', '#f09a76', '#eb6834']          # light to full orange: more bisection steps
fig, axs = mf.new(mf.TWO, 2.6, ncols=2)
ax = axs[0]
for K, col in zip((4, 6, 8), shade):
    ax.plot(a, bisect(a, K), '-', color=col, lw=1.1, drawstyle='steps-mid', label=f'bisection after $K={K}$ steps')
ax.plot(a, t, '-', color=C['cuad'], lw=1.8, label=r'root $t^\ast(a)$', zorder=4)
mf.title(ax, 'a', 'value')
ax.set_xlabel('$a$'); ax.set_ylabel('$t$')
mf.snap(ax, 'x', (1, 3)); mf.snap(ax, 'y', (1.2, 1.6))
ax.legend(loc='upper right')
mf.note(ax, (1.85, float(bisect(np.array([1.85]), 8)[0])), (1.08, 1.26), 'flat between jumps:\nslope zero almost everywhere',
        color='#b4461c')
ax = axs[1]
for K, col in zip((1, 2, 3), ('#c9d9f0', '#86b6ef', '#4f8fdc')):
    ax.plot(a, newton(a, K)[1], '-', color=col, lw=1.1, label=f'differentiated Newton, $K={K}$')
ax.plot(a, dt, '-', color=C['cuad'], lw=1.8, label=r'implicit $-t^\ast/(3t^{\ast 2}+a)$', zorder=4)
ax.plot(a, np.zeros_like(a), '-', color=C['enzyme'], lw=1.8, label='differentiated bisection (every $K$)')
ax.fill_between(a, dt, 0, color=C['enzyme'], alpha=0.07, lw=0)
mf.title(ax, 'b', 'derivative')
ax.set_xlabel('$a$'); ax.set_ylabel(r'$dt/da$')
mf.snap(ax, 'x', (1, 3)); mf.snap(ax, 'y', (-0.4, 0.05))
ax.legend(loc='lower right')
ax.text(2.0, -0.055, 'derivative missed by the differentiated bisection', fontsize=6.8, color='#b4461c',
        ha='center', va='center')
mf.save(fig, 'fig_bisection.pdf')
