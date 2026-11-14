"""Reliability of the run-time checks of the implicit adjoint: builds check_reliability.cpp (adjoints generated from
check_reliability.cu by `./cuad -r`), runs every case and writes out/reliability.json. Host only, no GPU.
The same cases run once more with -DCUAD_SOLVE_CHECK=2, which never skips the second derivative evaluation of a
smooth update; out/reliability-every-check.json holds those runs and the summary compares the two.
Run from the repository root after ./build.sh:  python3 research/srhd/check_reliability.py"""
import json
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent.parent
out = here / 'out'
out.mkdir(exist_ok=True)
gen = subprocess.run([str(root / 'cuad'), '-r', str(here / 'check_reliability.cu')], capture_output=True, text=True,
                     check=True)
(out / 'check_reliability_b.cuh').write_text(gen.stdout)


def run(check, name):
    exe = out / 'check_reliability'
    subprocess.run(['g++', '-O2', '-std=c++17', '-fno-fast-math', f'-I{root / "test"}', f'-I{here}',
                    '-DCUAD_SOLVE_TOL=g_tol', '-DCUAD_SOLVE_ATOL=g_atol', '-DCUAD_SOLVE_MAXIT=g_maxit',
                    f'-DCUAD_SOLVE_CHECK={check}', str(here / 'check_reliability.cpp'), '-o', str(exe)], check=True)
    lines = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [json.loads(l) for l in lines]
    (out / name).write_text(json.dumps(rows, indent=1) + '\n')
    return rows


def fmt(r):
    g = 'TRAP' if r['trapped'] else (f"{r['grad_rel_error']:.1e}" if r['grad_rel_error'] is not None else 'n/a')
    return (f"{r['case']:22s} {r['label']}={r['param']:<8g} tolp={r['tolp']:.0e} eps={r['eps_s']:.0e} "
            f"atol={r['atol']:.0e} maxit={r['maxit']:<7d} grad={g:>8s} value={r['value_rel_error']:.1e} it={r['iterations']}")


rows = run(1, 'reliability.json')
every = run(2, 'reliability-every-check.json')
for r in rows:
    print(fmt(r))
differ = [(a, b) for a, b in zip(rows, every) if a['trapped'] != b['trapped']]
print(f'wrote {out / "reliability.json"} ({len(rows)} runs)')
print(f'with every check: {len(differ)} of {len(rows)} runs change between pass and trap')
for a, b in differ:
    print('  default:', fmt(a))
    print('  every:  ', fmt(b))
