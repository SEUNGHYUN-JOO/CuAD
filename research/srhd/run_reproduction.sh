#!/usr/bin/env bash
# Noise-free medium reconstruction with exact (CuAD) and Enzyme gradients, bisection C2P: several true media and
# resolutions. Results: research/srhd/out/inverse-density-<n>-1-<tool>-noise0-case<k>.json, summary reproduction.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH="/tmp/cuad-comparators-20261113/sdk/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
py=research/build/venv/bin/python
for cfg in "128 0" "256 0" "512 0" "256 1" "256 2" "256 3"; do
  set -- $cfg
  for tool in cuad enzyme; do
    echo "== n=$1 case=$2 $tool"
    $py research/srhd/inverse_density.py $1 1 $tool 0 1e-9 $2 > /dev/null || echo "FAILED"
  done
done
# the same reconstruction with the Newton C2P, whose iterations Enzyme differentiates correctly (Table I)
for tool in cuad enzyme; do
  echo "== n=256 case=0 $tool newton"
  $py research/srhd/inverse_density.py 256 0 $tool 0 1e-9 0 > /dev/null || echo "FAILED"
done
$py - <<'EOF'
import json, glob
rows = []
for f in sorted(glob.glob('research/srhd/out/inverse-density-*-[01]-*-noise0-case*.json')):
    d = json.load(open(f)); s = d['summary']
    rows.append({k: s[k] for k in ('case', 'n', 'c2p', 'gradient', 'unknowns', 'iterations', 'message', 'objective_final',
                                   'rel_l2_error_rho0_final', 'wall_s')})
    print(json.dumps(rows[-1]))
json.dump(rows, open('research/srhd/out/reproduction.json', 'w'), indent=2)
EOF
