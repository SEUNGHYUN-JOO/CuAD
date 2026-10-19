// cuad diagnostics: every construct the generator cannot differentiate is recorded with
// its source location. main.cpp prints them as `file:line:col: error: ...` and exits
// with a non-zero status, unless --allow-unsupported is given.
#pragma once
#include "clang/Basic/SourceManager.h"
#include <string>
#include <vector>

namespace cuad {

struct Diag {
  std::vector<std::string> errors, warnings;

  void add(const clang::SourceManager &SM, clang::SourceLocation L, bool isError,
           const std::string &msg) {
    clang::PresumedLoc P = SM.getPresumedLoc(SM.getExpansionLoc(L));
    std::string where = P.isValid() ? std::string(P.getFilename()) + ":" +
                                          std::to_string(P.getLine()) + ":" +
                                          std::to_string(P.getColumn())
                                    : std::string("<unknown location>");
    (isError ? errors : warnings)
        .push_back(where + (isError ? ": error: " : ": warning: ") + msg);
  }
};

inline Diag &diag() {
  static Diag d;
  return d;
}

} // namespace cuad
