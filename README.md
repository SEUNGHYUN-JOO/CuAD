# CuAD — source-to-source automatic differentiation for CUDA

**CuAD** (CUDA Automatic Differentiation) reads CUDA kernels and writes their derivatives as new CUDA source, in the
style of Tapenade: a **forward (tangent)** companion `f_d` and a **reverse (adjoint)**
companion `f_b` for each `__global__` / `__device__` function. It is built on
Clang LibTooling and differentiates the kernel math directly, so the derivatives are
exact (no finite differences) and the generated code runs on the GPU like the original.

CuAD was developed for the exact discrete adjoint of the
[CuDAFLOW](https://github.com/SEUNGHYUN-JOO/CuDAFlow) GPU RANS/LES solver, where it
differentiates the flux, viscous and turbulence-model residual kernels.

## Example

Input (`test/cases/scalar.cu`):

```cuda
__global__ void apply_f(const double *x, double *y, double a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double xi = x[i];
        double t = sin(a * xi) + xi * xi;
        y[i] = t;
    }
}
```

`./cuad -r test/cases/scalar.cu` generates the adjoint kernel:

```cuda
__global__ void apply_f_b(const double * x, double *x_b, double * y, double *y_b, double a, int n)
{
  ...
  if (i < n) {
    double _cuad_b0 = y_b[i];
    y_b[i] = 0;
    t_b += _cuad_b0;
    xi_b += (a)*((cos(a * xi))*(t_b));
    xi_b += (t_b)*(xi);
    xi_b += (xi)*(t_b);
    atomicAdd(&x_b[i], xi_b);
  }
}
```

## Features

- **Forward and reverse mode** for `__global__` and `__device__` functions, including
  calls between user functions (each callee gets its own companion).
- **Activity analysis**: only variables that depend on active inputs and influence
  active outputs are differentiated.
- **Tape for overwritten values**: values that are overwritten before the reverse sweep
  are pushed to a per-thread tape (`cuad_rt.h`); pure accumulations (`v = v + e`) are
  recognised and need no tape.
- **GPU-safe adjoint accumulation**: adjoints of shared arrays use `atomicAdd`, while
  thread-local arrays and scalars use plain `+=`.
- **Loops**: grid-stride loops, loops with local declarations (recomputed per
  iteration in the reverse sweep), `#pragma unroll`, early exits (`continue`, `return`).
- **C++ constructs**: function templates, default arguments, `float &` output
  parameters, local pointers and arrays of pointers, `c ? a : b`, multi-declarator
  statements, bare `{ ... }` blocks.
- **Math library**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `tanh`, `fabs`, `pow`,
  `fmin`, `fmax`, `atan`, `atan2`, `asin`, `acos`, `copysign`, `hypot`, `cbrt` (float
  and double); `floor` / `ceil` / `round` / `trunc` are treated as piecewise constant.

### Conventions and limits

- A `__global__` kernel's scalar parameters are uniform across the launch and are held
  passive (no adjoint argument). Make a quantity an array if its derivative is needed.
- `if` conditions are re-evaluated in the reverse sweep, so a branch must not reassign a
  variable that its own condition reads.
- In-place updates of array elements inside loops must be per-element per iteration.
- Integer locals are not taped: give a reassigned integer local a new variable for each
  value it takes.
- Not supported: lambdas, `while` / `for (;;)` loops with `break`, and straight-line
  in-place updates of a local-array element that an earlier statement also reads.
- **CuAD does not yet stop on unsupported constructs.** Some are marked with a
  `// TODO(cuad)` comment in the generated code, others are passed through without a
  warning. Always verify generated derivatives (finite differences or a dot-product
  test, as in `test/`) before using them.

## Build

Requirements: Linux, `g++` with C++17, and Clang/LLVM 14 development files.

```bash
sudo apt-get install -y libclang-14-dev llvm-14-dev
./build.sh                  # or: cmake -S . -B build && cmake --build build
```

`build.sh` uses `llvm-config-14` when it is available, otherwise `LLVM_DIR`
(default `/usr/lib/llvm-14`).

## Usage

The command-line tool is `cuad`:

```bash
./cuad    kernel.cu [function] > kernel_d.cuh   # forward (tangent) mode
./cuad -r kernel.cu [function] > kernel_b.cuh   # reverse (adjoint) mode
```

Without `function`, every function defined in the file is differentiated. The generated
code includes `cuad_rt.h` (tape and `atomicAdd` helpers, in `test/`).

## Tests and examples

```bash
bash test/run_tests.sh
```

The suite checks the generated tangents and adjoints against central finite differences
and with adjoint dot-product tests (`<y_b, J x_d> = <J^T y_b, x_d>`). The kernels are
compiled for the host, so no GPU is needed to run it.

- `test/cases/` — small kernels: scalar math, loops and arrays, overwritten values,
  device-function calls, `atomicAdd` scatter.
- `examples/` — kernels from CuDAFLOW: conservative-to-primitive conversion and cell
  volumes, plus small applications driven by the generated adjoints: mesh shape
  optimisation, NACA 0012 inverse design, and Spalart–Allmaras field inversion (FIML).

## License

GPL-3.0. See [LICENSE](LICENSE).
