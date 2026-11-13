#!/usr/bin/env bash
# Regenerate tangent code for every test case and (optionally) check it against
# golden output. Run ./build.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

CUAD=./cuad
[ -x "$CUAD" ] || { echo "build cuad first (./build.sh)" >&2; exit 1; }

mkdir -p test/out
fail=0
for f in test/cases/*.cu; do
  base=$(basename "$f" .cu)
  "$CUAD"    "$f" > "test/out/${base}_d.cuh"   # forward / tangent
  "$CUAD" -r "$f" > "test/out/${base}_b.cuh"   # reverse / adjoint
  echo "=== $f ==="
  echo "  -> test/out/${base}_d.cuh (tangent), test/out/${base}_b.cuh (adjoint)"
done

# Numerical checks: analytic derivatives vs central finite difference.
echo
echo "=== FORWARD (tangent) finite-difference verification ==="
g++ -O2 -std=c++17 -Itest test/verify.cpp -o test/out/verify
test/out/verify || fail=1

echo
echo "=== REVERSE (adjoint) finite-difference verification ==="
g++ -O2 -std=c++17 -Itest test/verify_reverse.cpp -o test/out/verify_reverse
test/out/verify_reverse || fail=1

echo
echo "=== CONDITIONAL INDEX: inactive out-of-bounds branch under ASan ==="
g++ -O1 -g -std=c++17 -fsanitize=address -fno-omit-frame-pointer -Itest test/verify_conditional_index.cpp -o test/out/verify_conditional_index
test/out/verify_conditional_index || fail=1

echo
echo "=== TAPE BOUNDS: valid LIFO and fatal overflow/underflow ==="
g++ -O2 -std=c++17 -Itest test/verify_tape_bounds.cpp -o test/out/verify_tape_bounds
test/out/verify_tape_bounds || fail=1

echo
echo "=== NESTED VALUE-RETURNING CALLS + MATH RULES: tangent FD + adjoint vs tangent ==="
g++ -O2 -std=c++17 -Itest test/verify_extra.cpp -o test/out/verify_extra
test/out/verify_extra || fail=1

echo
echo "=== LOOP-CARRIED SCALARS in recompute loops: adjoint vs FD ==="
g++ -O2 -std=c++17 -Itest test/verify_loops.cpp -o test/out/verify_loops
test/out/verify_loops || fail=1

echo
echo "=== LOOP-CARRIED LOCAL ARRAYS (recurrences, repeated indices, overwrites, LSTM, 2-D): adjoint vs FD, ASan ==="
g++ -O1 -g -std=c++17 -fsanitize=address -fno-omit-frame-pointer -Itest test/verify_carried.cpp -o test/out/verify_carried
test/out/verify_carried || fail=1

echo
echo "=== CONVERGENCE LOOPS (Newton, fixed point, 2-D Newton, bisection): implicit adjoint vs FD, ASan ==="
"$CUAD" -r test/reverse_only/bracketing.cu > test/out/bracketing_b.cuh
if "$CUAD" test/reverse_only/bracketing.cu > /dev/null 2> test/out/bracketing_fwd.err; then
  echo "  [FAIL] forward mode accepted a bracketing loop"; fail=1
else
  n=$(grep -c "error: cuad (forward mode) cannot differentiate a loop that exits with" test/out/bracketing_fwd.err || true)
  ok=FAIL; [ "$n" -eq 2 ] && ok=PASS
  echo "  forward mode rejects the bracketing loops: $n located errors (expected 2)  [$ok]"
  [ "$ok" = PASS ] || fail=1
fi
g++ -O1 -g -std=c++17 -fsanitize=address -fno-omit-frame-pointer -Itest test/verify_convergence.cpp -o test/out/verify_convergence
test/out/verify_convergence || fail=1

echo
echo "=== MULTI-STATE ROOT FINDERS (secant, Illinois, Brent): residual found in the loop, adjoint vs exact ==="
"$CUAD" -r test/reverse_only/rootfinders.cu > test/out/rootfinders_b.cuh
g++ -O2 -std=c++17 -Itest test/verify_rootfinders.cpp -o test/out/verify_rootfinders
test/out/verify_rootfinders || fail=1

echo
echo "=== CONVERGENCE-LOOP POLICY: loops that stop away from a fixed point trap; non-solve loops are rejected ==="
"$CUAD" -r test/reverse_only/solve_policy.cu > test/out/solve_policy_b.cuh
g++ -O2 -std=c++17 -Itest test/verify_solve_policy.cpp -o test/out/verify_solve_policy
test/out/verify_solve_policy || fail=1
if "$CUAD" -r test/unsupported/solve_loops.cu > /dev/null 2> test/out/solve_loops.err; then
  echo "  [FAIL] reverse mode accepted loops it cannot differentiate"; fail=1
else
  n=$(grep -c "not a convergence loop" test/out/solve_loops.err || true)
  ok=FAIL; [ "$n" -eq 4 ] && ok=PASS
  echo "  rejected with the reason: $n located errors (expected 4)  [$ok]"
  [ "$ok" = PASS ] || fail=1
fi

echo
echo "=== GLOBAL TAPE (-DCUAD_GLOBAL_TAPE): concurrent threads on one tape, nested frames, bounds; ASan ==="
g++ -O1 -g -std=c++17 -fsanitize=address -fno-omit-frame-pointer -pthread -Itest test/verify_global_tape.cpp -o test/out/verify_global_tape
test/out/verify_global_tape || fail=1

echo
echo "=== UNSUPPORTED CONSTRUCTS must be rejected with located errors (exit 3) ==="
for m in "" "-r"; do
  want=1; [ "$m" = "-r" ] && want=3        # forward: erf; reverse: while, break, erf
  if "$CUAD" $m test/unsupported/bad.cu > /dev/null 2> test/out/unsupported$m.err; then
    echo "  [FAIL] mode ${m:-fwd}: accepted"; fail=1
  else
    rc=$?; n=$(grep -c "error: cuad" test/out/unsupported$m.err || true)
    ok=FAIL; [ "$rc" -eq 3 ] && [ "$n" -eq "$want" ] && ok=PASS
    echo "  mode ${m:-fwd}: exit $rc, $n located errors (expected $want)  [$ok]"
    [ "$ok" = PASS ] || fail=1
  fi
done

echo
echo "=== REAL CuDAFLOW kernel (c2pKernel): tangent FD + adjoint dot-test ==="
./cuad    examples/c2p.cu > examples/c2p_d.cuh
./cuad -r examples/c2p.cu > examples/c2p_b.cuh
g++ -O2 -std=c++17 -Itest -Iexamples examples/verify_c2p.cpp -o test/out/verify_c2p
test/out/verify_c2p || fail=1

echo
echo "=== HEAVY CuDAFLOW kernel (volKernel->faceI->faceVec): tangent FD + dot-test ==="
./cuad    examples/vol.cu > examples/vol_d.cuh
./cuad -r examples/vol.cu > examples/vol_b.cuh
g++ -O2 -std=c++17 -Itest -Iexamples examples/verify_vol.cpp -o test/out/verify_vol
test/out/verify_vol || fail=1

echo
echo "=== REUSED-ARRAY kernel (real volKernel, S/C reused 6x -> array tape) ==="
./cuad    examples/vol_reuse.cu > examples/vol_reuse_d.cuh
./cuad -r examples/vol_reuse.cu > examples/vol_reuse_b.cuh
g++ -O2 -std=c++17 -Itest -Iexamples examples/verify_vol_reuse.cpp -o test/out/verify_vol_reuse
test/out/verify_vol_reuse || fail=1

echo
echo "=== SHAPE OPTIMIZATION (column mesh, cuad volKernel_b gradient) ==="
g++ -O2 -std=c++17 -Itest -Iexamples examples/shape_opt.cpp -o test/out/shape_opt
test/out/shape_opt || fail=1

echo
echo "=== FIML turbulence (SA source): reverse + field inversion (recover beta) ==="
./cuad    examples/sa_fiml.cu > examples/sa_fiml_d.cuh
./cuad -r examples/sa_fiml.cu > examples/sa_fiml_b.cuh
g++ -O2 -std=c++17 -Itest -Iexamples examples/fiml_inversion.cpp -o test/out/fiml
test/out/fiml || fail=1

echo
echo "=== NACA 0012 shape design (inverse, cuad adjoint gradient) ==="
g++ -O2 -std=c++17 -Itest -Iexamples examples/naca_shape.cpp -o test/out/naca_shape
test/out/naca_shape || fail=1

exit $fail
