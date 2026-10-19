// cuad — CUDA source-to-source automatic differentiation (forward/tangent mode)
//
//   cuad <file.cu> [functionName]   > file_d.cuh
//
// Parses a CUDA source with Clang, then emits a tangent-mode companion for each
// (or one named) function defined in the file. CUDA is parsed as C++ with the
// device qualifiers mapped to annotate() attributes so we can recover them, and
// with the other CUDA-only tokens defined away. This avoids needing Clang's full
// CUDA/NVPTX toolchain just to read the kernel math.

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

#include "ForwardMode.h"
#include "ReverseMode.h"

using namespace clang;

namespace {

std::string gTarget; // empty => all functions defined in the main file
bool gReverse = false; // false => forward/tangent, true => reverse/adjoint

class ADVisitor : public RecursiveASTVisitor<ADVisitor> {
  ASTContext &Ctx;
  std::string &Out;

public:
  ADVisitor(ASTContext &C, std::string &O) : Ctx(C), Out(O) {}

  // A function is worth differentiating only if it has a floating return or at
  // least one active (floating scalar / pointer-or-array-to-floating) parameter.
  // Pure integer helpers (index math like nidx/cidx) are skipped.
  static bool hasActiveInterface(const FunctionDecl *FD) {
    if (FD->getReturnType()->isRealFloatingType())
      return true;
    for (const ParmVarDecl *p : FD->parameters()) {
      QualType qt = p->getType().getCanonicalType();
      if (qt->isPointerType() && qt->getPointeeType()->isRealFloatingType())
        return true;
      if (qt->isArrayType() &&
          qt->getAsArrayTypeUnsafe()->getElementType()->isRealFloatingType())
        return true;
      if (qt->isRealFloatingType())
        return true;
    }
    return false;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    if (!FD->isThisDeclarationADefinition() || !FD->hasBody())
      return true;
    if (!Ctx.getSourceManager().isInMainFile(FD->getLocation()))
      return true;
    if (!gTarget.empty() && FD->getNameAsString() != gTarget)
      return true;
    if (gTarget.empty() && !hasActiveInterface(FD))
      return true;
    if (gReverse) {
      cuad::ReverseMode rm(Ctx);
      Out += rm.genFunction(FD);
    } else {
      cuad::ForwardMode fm(Ctx);
      Out += fm.genFunction(FD);
    }
    Out += "\n";
    return true;
  }
};

class ADConsumer : public ASTConsumer {
  std::string &Out;

public:
  explicit ADConsumer(std::string &O) : Out(O) {}
  void HandleTranslationUnit(ASTContext &Ctx) override {
    ADVisitor(Ctx, Out).TraverseDecl(Ctx.getTranslationUnitDecl());
  }
};

class ADAction : public ASTFrontendAction {
  std::string &Out;

public:
  explicit ADAction(std::string &O) : Out(O) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override {
    return std::make_unique<ADConsumer>(Out);
  }
};

} // namespace

int main(int argc, char **argv) {
  std::string file;
  std::vector<std::string> positional;
  bool allowUnsupported = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-r" || a == "--reverse")
      gReverse = true;
    else if (a == "-f" || a == "--forward")
      gReverse = false;
    else if (a == "--allow-unsupported")
      allowUnsupported = true;
    else
      positional.push_back(a);
  }
  if (positional.empty()) {
    std::cerr << "usage: cuad [-r|--reverse] [--allow-unsupported] <file.cu> [functionName]\n"
              << "  -r/--reverse          : emit reverse/adjoint mode (default: "
                 "forward/tangent)\n"
              << "  --allow-unsupported   : report unsupported constructs as warnings "
                 "and exit 0\n";
    return 1;
  }
  file = positional[0];
  if (positional.size() >= 2)
    gTarget = positional[1];

  std::ifstream ifs(file);
  if (!ifs) {
    std::cerr << "cuad: cannot open " << file << "\n";
    return 1;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string code = ss.str();

  // Parse CUDA as C++: keep the math, recover device qualifiers via annotate().
  std::vector<std::string> args = {
      "-x", "c++", // parse .cu as C++ so Clang doesn't pull in the CUDA runtime
      "-std=c++17",
      "-fsyntax-only",
      "-w",
      "-D__global__=__attribute__((annotate(\"cuda_global\")))",
      "-D__device__=__attribute__((annotate(\"cuda_device\")))",
      "-D__host__=__attribute__((annotate(\"cuda_host\")))",
      "-D__forceinline__=inline",
      "-D__restrict__=__restrict__",
      "-D__shared__=",
      "-D__constant__=",
      "-DblockIdx=cuad_blockIdx",
      "-DthreadIdx=cuad_threadIdx",
      "-DblockDim=cuad_blockDim",
      "-DgridDim=cuad_gridDim",
      "-include", "cuad_builtins.h",
  };

  // Provide fake CUDA builtins + math prototypes via an in-memory header so we
  // don't depend on libstdc++ <cmath> being on Clang's search path. Declaring
  // the math functions lets calls parse as proper CallExprs we can differentiate.
  static const char *kBuiltins =
      "struct cuad_dim3 { unsigned x, y, z; };\n"
      "static cuad_dim3 cuad_blockIdx, cuad_threadIdx, cuad_blockDim, "
      "cuad_gridDim;\n"
      "double sin(double); double cos(double); double tan(double);\n"
      "double asin(double); double acos(double); double atan(double);\n"
      "double sinh(double); double cosh(double); double tanh(double);\n"
      "double exp(double); double log(double); double log2(double);\n"
      "double log10(double); double sqrt(double); double cbrt(double);\n"
      "double fabs(double); double fabs(double); double ceil(double);\n"
      "double floor(double); double pow(double, double);\n"
      "double atan2(double, double); double fmin(double, double);\n"
      "double fmax(double, double); double fmod(double, double);\n"
      "double copysign(double, double);\n"
      "float sinf(float); float cosf(float); float expf(float);\n"
      "float logf(float); float sqrtf(float); float tanhf(float);\n"
      "float fabsf(float); float powf(float, float);\n"
      "float fminf(float, float); float fmaxf(float, float);\n"
      "float atan2f(float, float); float atanf(float); float asinf(float);\n"
      "float acosf(float); float tanf(float);\n"
      "float sin(float); float cos(float); float exp(float);\n"
      "float log(float); float sqrt(float); float tanh(float);\n"
      "float fabs(float); float pow(float, float);\n"
      "float fmin(float, float); float fmax(float, float);\n"
      "double atomicAdd(double*, double); float atomicAdd(float*, float);\n"
      "int atomicAdd(int*, int);\n"
      // piecewise-constant float math (zero derivative) and the barrier
      "float floorf(float); float ceilf(float); float roundf(float); float truncf(float);\n"
      "void __syncthreads(void);\n"
      // further float / double math and integer min / max
      "float copysignf(float, float); float cbrtf(float); float rsqrtf(float); float hypotf(float, float);\n"
      "double copysign(double, double); double cbrt(double); double hypot(double, double);\n"
      "double round(double); double trunc(double);\n"
      // declared so that calls parse; without a derivative rule they are reported as unsupported
      "double erf(double); double erfc(double); double log1p(double); double expm1(double);\n"
      "float erff(float); float erfcf(float); float log1pf(float); float expm1f(float);\n"
      "int min(int, int); int max(int, int); long min(long, long); long max(long, long);\n";

  std::string out;
  auto action = std::make_unique<ADAction>(out);
  bool ok = tooling::runToolOnCodeWithArgs(
      std::move(action), code, args, file, "cuad",
      std::make_shared<PCHContainerOperations>(),
      tooling::FileContentMappings{{"cuad_builtins.h", kBuiltins}});

  if (gReverse) {
    std::cout << "// Generated by cuad — reverse (adjoint) mode. DO NOT EDIT.\n";
    std::cout << "// source: " << file << "\n";
    std::cout << "// Output adjoints (y_b) are seeds; input adjoints (x_b) are\n";
    std::cout << "// accumulated:  x_b += (df/dx)^T * y_b.\n";
    std::cout << "#include \"cuad_rt.h\"  // tape + atomicAdd support\n\n";
  } else {
    std::cout << "// Generated by cuad — forward (tangent) mode. DO NOT EDIT.\n";
    std::cout << "// source: " << file << "\n";
    std::cout << "// Each active floating input x has a tangent partner x_d;\n";
    std::cout << "// the routine propagates directional derivatives (Jv).\n";
    std::cout << "#include \"cuad_rt.h\"  // atomicAdd support\n\n";
  }
  std::cout << out;

  if (!ok) {
    std::cerr << "cuad: warning — parse reported errors (output may be "
                 "incomplete)\n";
    return 2;
  }
  // Constructs that could not be differentiated: the output above still compiles but its
  // derivatives are wrong where they occur, so this is a failure unless explicitly allowed.
  const cuad::Diag &D = cuad::diag();
  for (const std::string &w : D.warnings)
    std::cerr << w << "\n";
  for (const std::string &e : D.errors)
    std::cerr << (allowUnsupported ? e + " [allowed]" : e) << "\n";
  if (!D.errors.empty() && !allowUnsupported) {
    std::cerr << "cuad: " << D.errors.size()
              << " unsupported construct(s); the generated derivatives are not valid "
                 "(use --allow-unsupported to emit them anyway)\n";
    return 3;
  }
  return 0;
}
