#!/usr/bin/env bash
# Reproduce every table and figure of the manuscript "Automatic implicit adjoints of convergence loops in CUDA kernels for
# relativistic hydrodynamics" (see research/srhd/README.md).
#
# Requirements: Linux (tested on WSL2 Ubuntu 22.04), CUDA 12.x and an NVIDIA GPU with fp64, g++ (C++17), the CuAD build
# dependencies (Clang/LLVM 14 dev files), and for the comparisons:
#   COMPARATORS  directory with Clang 19.1.7 (sdk/usr/lib/llvm-19), ClangEnzyme-19.so of Enzyme v0.0.307
#                (enzyme-build/Enzyme/) and Clad v2.5 (clad-build/, clad/<src>/include)
#   PY           Python 3 with numpy, scipy, mpmath, matplotlib (and pdflatex with TikZ and newtx for Fig. 1)
#   JAXPY        Python 3 with jax[cuda12] 0.6.2, optimistix 0.0.11 (optional; skipped if missing)
# Run on an otherwise idle GPU (timings). Total time: about two hours on an RTX 5080.
set -uo pipefail
cd "$(dirname "$0")/../.."
COMPARATORS=${COMPARATORS:-/tmp/cuad-comparators-20261113}
PY=${PY:-research/build/venv/bin/python}
JAXPY=${JAXPY:-research/build/venv-jax/bin/python}
export LD_LIBRARY_PATH="$COMPARATORS/sdk/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
CL="$COMPARATORS/sdk/usr/lib/llvm-19/bin/clang++"
ENZ="-fplugin=$COMPARATORS/enzyme-build/Enzyme/ClangEnzyme-19.so"
CLADINC=$(ls -d "$COMPARATORS"/clad/*/include 2>/dev/null | head -1)
CUDA="-x cuda --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_90 -Wno-unknown-cuda-version -L/usr/local/cuda/lib64 -lcudart"
S=research/srhd; O=$S/out; E=examples/srhd
mkdir -p $O
step() { echo; echo "=== $*"; }
die() { echo "FAILED: $*"; exit 1; }

step "build CuAD and generate derivatives"
./build.sh > $O/build.log 2>&1 || die "CuAD build"
./cuad -r $E/srhd.cu > $E/srhd_b.cuh || die srhd_b
./cuad $E/srhd.cu > $E/srhd_d.cuh || die srhd_d
./cuad -r $E/srhd2d.cu > $E/srhd2d_b.cuh || die srhd2d_b
./cuad $E/srhd2d.cu > $E/srhd2d_d.cuh || die srhd2d_d
./cuad -r $E/cooling.cu > $E/cooling_b.cuh || die cooling_b
./cuad $E/cooling.cu > $E/cooling_d.cuh || die cooling_d
./cuad -r $S/c2p_fixed.cu > $O/c2p_fixed_b.cuh || die c2p_fixed_b
./cuad -r $S/c2p_variants.cu > $O/c2p_variants_b.cuh || die c2p_variants_b
./cuad -r $S/sweep_kernels.cu > $O/sweep_b.cuh || die sweep_b
./cuad $S/sweep_kernels.cu > $O/sweep_d.cuh || die sweep_d

step "host builds"
HF="-O2 -std=c++17 -shared -fPIC -Itest -I$E"
g++ $HF $E/srhd_host.cpp -o $O/srhd_host.so || die srhd_host
g++ $HF -DCUAD_TAPE_SIZE=256 $E/srhd_adj_host.cpp -o $O/srhd_adj_host.so || die srhd_adj_host
g++ $HF $E/srhd2d_host.cpp -o $O/srhd2d_host.so || die srhd2d_host
g++ $HF -I$S -I$O $S/c2p_variants_host.cpp -o $O/c2p_variants_cuad.so || die c2p_variants_cuad
g++ -O2 -std=c++17 -Itest -I$E $S/dot2d_kernels.cpp -o $O/dot2d || die dot2d
"$CL" -std=c++17 -O2 -fno-fast-math -ffp-contract=off -fPIC -shared $ENZ -I$E $S/enzyme_srhd_host.cpp -o $O/enzyme_srhd_host.so || die enzyme_host
"$CL" -std=c++17 -O2 -fno-fast-math -ffp-contract=off -fPIC -shared -DWITH_ENZYME $ENZ -I$E -I$S $S/c2p_variants_host.cpp -o $O/c2p_variants_enzyme.so || die enzyme_variants
"$CL" -std=c++17 -O2 -fno-fast-math -ffp-contract=off $ENZ -Itest -I$S -I$O -DCUAD_SOLVE_TOL=1e300 -DCUAD_SOLVE_AGREE=1e300 $S/tolerance_sweep.cpp -o $O/tolerance_sweep || die sweep
if [ -n "$CLADINC" ]; then
  "$CL" -std=c++17 -O2 -fPIC -shared -fplugin="$COMPARATORS/clad-build/lib/clad.so" -I"$CLADINC" -I"$COMPARATORS/clad-build/include" -I$E $S/clad_c2p_host.cpp -o $O/clad_c2p_host.so || die clad_host
  "$CL" -std=c++17 -O2 -fplugin="$COMPARATORS/clad-build/lib/clad.so" -I"$CLADINC" -I"$COMPARATORS/clad-build/include" $S/clad_newton_min.cpp -o $O/clad_newton_min || die clad_min
fi

step "GPU builds"
GF="-std=c++17 -O3 -fno-fast-math -ffp-contract=off $CUDA -fPIC -shared -Itest -I$E"
"$CL" $GF $S/srhd_gpu.cu -o $O/srhd_gpu.so || die srhd_gpu
"$CL" $GF $S/srhd2d_gpu.cu -o $O/srhd2d_gpu.so || die srhd2d_gpu
"$CL" $GF -I$S -I$O $ENZ $S/c2p_gpu_compare.cu -o $O/c2p_gpu_compare.so || die c2p_gpu_compare
# cost breakdown: the CuAD adjoint without its run-time checks, Enzyme on kernels with __restrict__ pointers
sed -E 's/((const )?(double|int) )\*([A-Za-z_][A-Za-z0-9_]*)([,)])/\1*__restrict__ \4\5/g' $E/srhd.cu > $O/srhd_restrict.cu
"$CL" $GF -DWITH_ABLATION -I$S -I$O $ENZ $S/c2p_gpu_compare.cu -o $O/c2p_gpu_ablation.so || die c2p_gpu_ablation
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | tee $O/gpu-env.txt

step "verification (Sec. VI)"
$O/tolerance_sweep > $O/tolerance-sweep.jsonl || die tolerance_sweep            # Fig. 3
$PY $S/c2p_robustness.py || die robustness                                     # Fig. 4
$PY $S/compare_c2p_variants.py || die variants                                 # Table I
$PY $S/check_adjoint.py > /dev/null || die check_adjoint                       # C2P kernel and trajectory vs FD
$PY $S/check_enzyme_host.py > /dev/null || die enzyme_host
[ -n "$CLADINC" ] && { $PY $S/check_clad_host.py > /dev/null; $O/clad_newton_min > $O/clad-newton-min.txt; }
$PY $S/check_cooling.py || die cooling
$PY $S/check_reliability.py > /dev/null || die reliability                     # Table III
$O/dot2d > $O/dot2d.txt || die dot2d

step "performance (Sec. VII)"
$PY $S/compare_c2p_gpu.py || die compare_gpu                                   # Fig. 5, Table V
$PY $S/c2p_enzyme_heap.py || die heap
$PY $S/c2p_gpu_ablation.py > /dev/null || die ablation                         # Table VI
[ -x "$JAXPY" ] && $JAXPY $S/jax_c2p_baseline.py

step "application (Sec. VIII)"
$PY $S/check_primal.py || die primal                                           # Fig. 6
$PY $S/check_gpu.py || die check_gpu
$PY $S/smooth_and_taylor.py || die taylor                                      # Fig. 7
$PY $S/check2d.py || die check2d
bash $S/run_reproduction.sh || die reproduction                                # Table VIII, Fig. 8
$PY $S/inverse_density.py 512 1 cuad 0.01 1e-6 0 > /dev/null || die inverse_noise
$PY $S/inverse_density.py 512 1 enzyme 0.01 1e-6 0 > /dev/null || die inverse_noise_enzyme
$PY $S/inverse_density2d.py 64 1 0 || die inverse2d                            # Fig. 9
$PY $S/inverse_density2d.py 64 1 0.01 || die inverse2d_noise

step "figures"
$PY $S/make_schematics.py || die figures   # every figure: imports make_figures.py, then Figs. 1 and 2
echo "DONE"
