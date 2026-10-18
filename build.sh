#!/usr/bin/env bash
# Build cuad. Requires: libclang-14-dev (headers + static clang libs) and the
# llvm-14 runtime (libLLVM.so).
#   sudo apt-get install -y libclang-14-dev
# Uses llvm-config-14 when it is on PATH; otherwise LLVM_DIR (default /usr/lib/llvm-14).
set -euo pipefail
cd "$(dirname "$0")"

LLVM_CONFIG=${LLVM_CONFIG:-llvm-config-14}
if command -v "$LLVM_CONFIG" >/dev/null; then
  INC=$($LLVM_CONFIG --includedir)
  LIBDIR=$($LLVM_CONFIG --libdir)
  CXXFLAGS=$($LLVM_CONFIG --cxxflags)
  SYSLIBS=$($LLVM_CONFIG --system-libs)
  LLVMLIB=-lLLVM
else
  LLVM_DIR=${LLVM_DIR:-/usr/lib/llvm-14}
  INC=$LLVM_DIR/include
  LIBDIR=$LLVM_DIR/lib
  CXXFLAGS="-I$INC -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS"
  SYSLIBS="-lz -lrt -ldl -ltinfo -lpthread -lm"
  LLVMLIB=-lLLVM-14
fi

if [ ! -f "$INC/clang/Tooling/Tooling.h" ]; then
  echo "error: Clang dev headers missing. Run:" >&2
  echo "  sudo apt-get install -y libclang-14-dev" >&2
  exit 1
fi

# Links the static clang component libs + shared libLLVM directly (works where
# ClangConfig.cmake or the libclang-cpp.so symlink is absent).
g++ -O2 -fno-rtti src/main.cpp \
  $CXXFLAGS -std=c++17 \
  -L"$LIBDIR" \
  -Wl,--start-group \
    -lclangTooling -lclangFrontend -lclangDriver -lclangSerialization \
    -lclangParse -lclangSema -lclangAnalysis -lclangEdit -lclangASTMatchers \
    -lclangToolingCore -lclangAST -lclangLex -lclangBasic \
  -Wl,--end-group \
  $LLVMLIB \
  $SYSLIBS \
  -Wl,-rpath,"$LIBDIR" \
  -o cuad

echo "built -> $(pwd)/cuad"
