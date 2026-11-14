"""Copy of a generated adjoint without its run-time checks, used only to measure what the checks cost.
Removes the blocks that test the exit state of a fixed-point loop ("the loop must have stopped at a fixed point"),
the bracket-collapse tests and the trap after a non-converged adjoint iteration. Usage: strip_checks.py in.cuh out.cuh"""
import re
import sys

src = open(sys.argv[1]).read().split('\n')
out, i, removed = [], 0, 0
while i < len(src):
    line = src[i]
    if '// cuad: the loop must have stopped at a fixed point' in line:
        depth = line.count('{') - line.count('}')
        i += 1
        while depth > 0:
            depth += src[i].count('{') - src[i].count('}')
            i += 1
        removed += 1
        continue
    if re.match(r'\s*if \(.*\) cuad_solve_fail\(\);\s*$', line):
        removed += 1
        i += 1
        continue
    out.append(line)
    i += 1
open(sys.argv[2], 'w').write('\n'.join(out))
print(f'{sys.argv[2]}: {removed} checks removed')
