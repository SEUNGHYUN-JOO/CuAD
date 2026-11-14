// cuad — forward (tangent) mode source-to-source AD for CUDA
//
// Strategy (Tapenade-style): given a CUDA __global__/__device__ function, emit a
// companion function <name>_d whose active floating inputs each gain a "_d"
// (dot) partner. Each statement is transformed into a tangent statement followed
// by the original primal statement.
//
// Expression derivatives are built as *source text* over the Clang AST. The
// primal text of a subexpression is recovered verbatim from the original buffer
// (Lexer::getSourceText), so CUDA-isms like __shared__ / __restrict__ pass
// through untouched. Only the reconstructed signature and the synthesized
// tangent statements are generated from scratch.
//
// v0 scope: scalar/pointer floating args, arithmetic (+ - * /), unary +/-,
// intrinsics (sin cos exp log sqrt tanh fabs pow fmin fmax), assignments
// (= += -= *= /=), local VarDecls, for/if/while control flow, void kernels.
// Known gaps (flagged in output as TODO): value-returning tangents, local
// floating arrays, struct members, reductions/atomics.

#pragma once
#include "Activity.h"
#include "Diag.h"
#include "ReverseMode.h" // the analysis of convergence loops is shared with the reverse mode
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <functional>
#include <set>
#include <string>

namespace cuad {

using namespace clang;

class ForwardMode {
  ASTContext &Ctx;
  const SourceManager &SM;
  const LangOptions &LO;
  PrintingPolicy PP;
  Activity Act;
  std::string curRetDot_; // tangent out-param name of the function being emitted
  // Tangent calls hoisted out of the statement being emitted: a value-returning user
  // function nested in an expression (`0.5 * (f(a) + f(b))`) gets its _d companion
  // called just before the statement, and the expression uses the temporary.
  mutable std::string pre_;
  mutable std::string curInd_;
  mutable int callTmp_ = 0;
  ReverseMode An_;          // convergence-loop analysis (shared with the reverse mode)
  mutable int implicitId_ = 0;

public:
  explicit ForwardMode(ASTContext &C)
      : Ctx(C), SM(C.getSourceManager()), LO(C.getLangOpts()),
        PP(C.getPrintingPolicy()), An_(C) {
    PP.SuppressTagKeyword = true;
  }

  // -------- entry point --------
  std::string genFunction(const FunctionDecl *FD) {
    Act = Activity();
    Act.run(FD);
    An_.prepareSolveAnalysis(FD);
    std::string out;
    // CUDA qualifier recovered from the annotate() macro trick in main.cpp
    for (const auto *a : FD->specific_attrs<AnnotateAttr>()) {
      StringRef s = a->getAnnotation();
      if (s == "cuda_global")
        out += "__global__ ";
      else if (s == "cuda_device")
        out += "__device__ ";
      else if (s == "cuda_host")
        out += "__host__ ";
    }
    out += tstr(FD->getReturnType()) + " " + FD->getNameAsString() + "_d(";
    bool first = true;
    for (const ParmVarDecl *p : FD->parameters()) {
      if (!first)
        out += ", ";
      first = false;
      std::string ts = tstr(p->getType());
      std::string nm = p->getNameAsString();
      out += ts + " " + nm;
      if (isActiveType(p->getType()))
        out += ", " + ts + " " + nm + "_d";
    }
    // Value-returning functions get a trailing reference out-param that receives
    // the tangent of the return value (Tapenade-style tangent convention).
    curRetDot_.clear();
    if (FD->getReturnType()->isRealFloatingType()) {
      curRetDot_ = FD->getNameAsString() + "_dout";
      if (!first)
        out += ", ";
      out += tstr(FD->getReturnType()) + " &" + curRetDot_;
    }
    out += ")\n{\n";
    if (const auto *body = dyn_cast_or_null<CompoundStmt>(FD->getBody()))
      out += genStmt(body, 1);
    out += "}\n";
    return out;
  }

private:
  // ---------- small helpers ----------
  std::string tstr(QualType qt) const { return qt.getAsString(PP); }

  // Record an unsupported construct at its source location and return the marker text
  // that is still written into the generated code.
  std::string unsupported(SourceLocation L, const std::string &what,
                          const std::string &marker) const {
    diag().add(SM, L, true, "cuad (forward mode) cannot differentiate " + what);
    return marker;
  }

  std::string src(SourceRange r) const {
    return Lexer::getSourceText(CharSourceRange::getTokenRange(r), SM, LO).str();
  }

  // A loop that leaves with `break` and assigns floating variables declared before it, none of them from a
  // value that depends on the active inputs by data flow within one iteration, while a branch condition
  // in it does (a bisection). Its result converges to a function of the inputs, but the tangent of its
  // iterations only carries the tangent of the starting values. A bisection of the supported form receives an
  // implicit tangent (implicitTangent); one outside it is rejected through this test.
  static bool fwdHasBreak(const Stmt *s) {
    if (!s)
      return false;
    if (isa<BreakStmt>(s))
      return true;
    if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s) || isa<SwitchStmt>(s))
      return false;
    for (const Stmt *c : s->children())
      if (fwdHasBreak(c))
        return true;
    return false;
  }
  static void fwdRefs(const Stmt *s, std::set<const VarDecl *> &out) {
    if (!s)
      return;
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
        out.insert(vd);
    for (const Stmt *c : s->children())
      fwdRefs(c, out);
  }
  static bool fwdMentions(const Stmt *s, const std::set<const VarDecl *> &vs) {
    std::set<const VarDecl *> r;
    fwdRefs(s, r);
    for (const VarDecl *v : r)
      if (vs.count(v))
        return true;
    return false;
  }
  bool controlOnlyLoop(const ForStmt *fs) const {
    if (!fs->getBody() || !fwdHasBreak(fs->getBody()))
      return false;
    std::set<const VarDecl *> inner, outerAssigned, dep, refs;
    std::function<void(const Stmt *)> decls = [&](const Stmt *s) {
      if (!s)
        return;
      if (const auto *ds = dyn_cast<DeclStmt>(s))
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            inner.insert(vd);
      for (const Stmt *c : s->children())
        decls(c);
    };
    decls(fs->getBody());
    decls(fs->getInit());
    fwdRefs(fs->getBody(), refs);
    std::function<void(const Stmt *)> assigned = [&](const Stmt *s) {
      if (!s)
        return;
      if (const auto *bo = dyn_cast<BinaryOperator>(s))
        if (bo->isAssignmentOp())
          if (const auto *d = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
            if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
              if (!inner.count(vd) && vd->getType()->isRealFloatingType())
                outerAssigned.insert(vd);
      for (const Stmt *c : s->children())
        assigned(c);
    };
    assigned(fs->getBody());
    if (outerAssigned.empty())
      return false;
    for (const VarDecl *vd : refs)
      if (!inner.count(vd) && !outerAssigned.count(vd) && Act.isVaried(vd))
        dep.insert(vd);
    bool stateDep = false, condDep = false;
    std::function<void(const Stmt *)> flow = [&](const Stmt *s) {
      if (!s)
        return;
      if (const auto *ds = dyn_cast<DeclStmt>(s)) {
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            if (vd->hasInit() && fwdMentions(vd->getInit(), dep))
              dep.insert(vd);
        return;
      }
      if (const auto *bo = dyn_cast<BinaryOperator>(s))
        if (bo->isAssignmentOp()) {
          if (const auto *d = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
            if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
              if (fwdMentions(bo->getRHS(), dep) || (bo->isCompoundAssignmentOp() && dep.count(vd))) {
                dep.insert(vd);
                if (outerAssigned.count(vd))
                  stateDep = true;
              }
          return;
        }
      if (const auto *is = dyn_cast<IfStmt>(s)) {
        if (fwdMentions(is->getCond(), dep))
          condDep = true;
        flow(is->getThen());
        flow(is->getElse());
        return;
      }
      for (const Stmt *c : s->children())
        flow(c);
    };
    flow(fs->getBody());
    return !stateDep && condDep;
  }
  std::string prim(const Expr *e) const { return src(e->getSourceRange()); }

  static bool isFloat(const Expr *e) {
    return e->getType()->isRealFloatingType();
  }

  static bool isActiveType(QualType qt) {
    qt = qt.getCanonicalType();
    if (qt->isPointerType())
      return qt->getPointeeType()->isRealFloatingType();
    if (qt->isArrayType())
      return qt->getAsArrayTypeUnsafe()->getElementType()->isRealFloatingType();
    return qt->isRealFloatingType();
  }

  static std::string addT(const std::string &a, const std::string &b) {
    if (a == "0")
      return b;
    if (b == "0")
      return a;
    return a + " + " + b;
  }
  static std::string subT(const std::string &a, const std::string &b) {
    if (b == "0")
      return a;
    if (a == "0")
      return "-(" + b + ")";
    return a + " - (" + b + ")";
  }
  static std::string mulT(const std::string &a, const std::string &b) {
    if (a == "0" || b == "0")
      return "0";
    if (a == "1")
      return b;
    if (b == "1")
      return a;
    return "(" + a + ")*(" + b + ")";
  }

  static std::string declName(const DeclRefExpr *d) {
    return d->getNameInfo().getName().getAsString();
  }

  // ---------- expression differentiation ----------
  // Tangent ACCESS for an lvalue chain (var, a[i], a[i][j], p->f, s.f): rebuild the
  // access with the ROOT variable renamed to root_d. Returns "" if the root is not a
  // varied variable. Handles arbitrarily-nested array subscripts (2D CFD arrays).
  std::string dotAccess(const Expr *e) const {
    e = e->IgnoreParenImpCasts();
    if (const auto *dre = dyn_cast<DeclRefExpr>(e))
      return Act.isVaried(dre->getDecl()) ? declName(dre) + "_d" : "";
    if (const auto *ase = dyn_cast<ArraySubscriptExpr>(e)) {
      std::string b = dotAccess(ase->getBase());
      return b.empty() ? "" : b + "[" + prim(ase->getIdx()) + "]";
    }
    if (const auto *me = dyn_cast<MemberExpr>(e)) {
      std::string b = dotAccess(me->getBase());
      return b.empty() ? "" : b + (me->isArrow() ? "->" : ".") +
                                  me->getMemberDecl()->getNameAsString();
    }
    if (const auto *uo = dyn_cast<UnaryOperator>(e))
      if (uo->getOpcode() == UO_Deref) {
        std::string b = dotAccess(uo->getSubExpr());
        return b.empty() ? "" : "(*" + b + ")";
      }
    return "";
  }

  // Returns the tangent (dot) of expression e as source text.
  std::string diff(const Expr *e) const {
    if (const auto *ce = dyn_cast<CastExpr>(e)) // implicit + explicit casts
      return diff(ce->getSubExpr());
    if (const auto *pe = dyn_cast<ParenExpr>(e)) {
      std::string d = diff(pe->getSubExpr());
      return d == "0" ? "0" : "(" + d + ")";
    }
    if (isa<FloatingLiteral>(e) || isa<IntegerLiteral>(e))
      return "0";
    // Anything whose value is not floating carries no derivative (int indices,
    // comparisons, bools, ...).
    if (!isFloat(e))
      return "0";

    if (const auto *dre = dyn_cast<DeclRefExpr>(e))
      return Act.isVaried(dre->getDecl()) ? declName(dre) + "_d" : "0";

    if (isa<ArraySubscriptExpr>(e)) {
      std::string acc = dotAccess(e);   // handles nested a[i][j], a[i] uniformly
      return acc.empty() ? "0" : acc;
    }

    if (const auto *uo = dyn_cast<UnaryOperator>(e)) {
      switch (uo->getOpcode()) {
      case UO_Minus: {
        std::string d = diff(uo->getSubExpr());
        return d == "0" ? "0" : "-(" + d + ")";
      }
      case UO_Plus:
        return diff(uo->getSubExpr());
      case UO_Deref: {
        const Expr *s = uo->getSubExpr()->IgnoreParenImpCasts();
        if (const auto *d = dyn_cast<DeclRefExpr>(s))
          return Act.isVaried(d->getDecl()) ? "(*" + declName(d) + "_d)" : "0";
        return unsupported(uo->getBeginLoc(), "a dereference of a computed pointer",
                           "0 /*cuad: deref of complex ptr*/");
      }
      default:
        return unsupported(uo->getBeginLoc(), "this unary operator", "0 /*cuad: unhandled unary op*/");
      }
    }

    if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
      std::string L = diff(bo->getLHS()), R = diff(bo->getRHS());
      std::string pl = prim(bo->getLHS()), pr = prim(bo->getRHS());
      switch (bo->getOpcode()) {
      case BO_Add:
        return addT(L, R);
      case BO_Sub:
        return subT(L, R);
      case BO_Mul:
        return addT(mulT(L, pr), mulT(pl, R));
      case BO_Div: {
        std::string num = subT(mulT(L, pr), mulT(pl, R));
        if (num == "0")
          return "0";
        return "(" + num + ")/((" + pr + ")*(" + pr + "))";
      }
      default:
        return unsupported(e->getBeginLoc(), "this binary operator", "0 /*cuad: unhandled binary op*/");
      }
    }

    if (const auto *call = dyn_cast<CallExpr>(e))
      return diffCall(call);

    // ternary  cond ? a : b  ->  cond ? a' : b'  (derivative follows the taken branch;
    // the condition is a control decision, not differentiated). Real kernels use ?: as
    // heavily as if/else, so this must be handled, not flagged.
    if (const auto *co = dyn_cast<ConditionalOperator>(e)) {
      std::string dt = diff(co->getTrueExpr()), df = diff(co->getFalseExpr());
      if (dt == "0" && df == "0")
        return "0";
      return "((" + prim(co->getCond()) + ") ? (" + dt + ") : (" + df + "))";
    }

    // struct/union field: p->rho -> p_d->rho, q.rho -> q_d.rho (active base).
    if (const auto *me = dyn_cast<MemberExpr>(e)) {
      const Expr *base = me->getBase()->IgnoreParenImpCasts();
      if (const auto *d = dyn_cast<DeclRefExpr>(base))
        if (Act.isVaried(d->getDecl()))
          return declName(d) + (me->isArrow() ? "_d->" : "_d.") +
                 me->getMemberDecl()->getNameAsString();
      return "0";
    }

    return unsupported(e->getBeginLoc(), "the expression `" + prim(e) + "`",
                       "0 /*cuad: unhandled expr: " + prim(e) + "*/");
  }

  std::string diffCall(const CallExpr *call) const {
    const FunctionDecl *fd = call->getDirectCallee();
    std::string fn = fd ? fd->getNameAsString() : "";
    unsigned n = call->getNumArgs();

    // A value-returning user function inside a larger expression: its tangent comes from
    // the _d companion, called before the statement (see pre_). The arguments are read
    // before the statement executes, so they hold the same values as in the primal.
    if (isUserDiffFn(fd) && fd->getReturnType()->isRealFloatingType()) {
      std::string t = "_cuad_c" + std::to_string(callTmp_++) + "_d";
      pre_ += curInd_ + tstr(fd->getReturnType()) + " " + t + " = 0;\n" + curInd_ +
              buildDiffCall(call, t) + ";\n";
      return t;
    }

    // Normalize single-precision CUDA intrinsics (sinf, powf, ...) to their
    // double-named rule; the primal text keeps the original spelling.
    static const char *kF[] = {"sinf",  "cosf",  "tanf",   "expf",  "logf",
                               "sqrtf", "tanhf", "fabsf",  "powf",  "fminf",
                               "fmaxf", "asinf", "acosf",  "atanf", "atan2f",
                               "floorf", "ceilf", "roundf", "truncf", "copysignf",
                               "hypotf", "cbrtf"};
    for (const char *g : kF)
      if (fn == g) {
        fn.pop_back();
        break;
      }

    // floor / ceil / round / trunc are piecewise constant: zero derivative almost everywhere
    if (fn == "floor" || fn == "ceil" || fn == "round" || fn == "trunc")
      return "0";

    if (n >= 1) {
      const Expr *a0 = call->getArg(0);
      std::string pa = prim(a0), da = diff(a0);

      // copysign(a, b) = |a| sgn(b): d/da = sgn(a) sgn(b), d/db = 0
      if (n == 2 && fn == "copysign") {
        if (da == "0")
          return "0";
        return mulT("(((" + pa + ") >= 0) == ((" + prim(call->getArg(1)) + ") >= 0) ? 1.0 : -1.0)", da);
      }
      // hypot(a, b) = sqrt(a^2 + b^2): d = (a a' + b b') / hypot(a, b)
      if (n == 2 && fn == "hypot") {
        const Expr *a1 = call->getArg(1);
        std::string pb = prim(a1), db = diff(a1);
        std::string num = addT(mulT("(" + pa + ")", da), mulT("(" + pb + ")", db));
        if (num == "0")
          return "0";
        return "(" + num + ")/hypot(" + pa + ", " + pb + ")";
      }

      // atan2(y, x)' = (x y' - y x')/(x^2 + y^2)
      if (n == 2 && fn == "atan2") {
        const Expr *a1 = call->getArg(1);
        std::string pb = prim(a1), db = diff(a1);
        std::string den = "((" + pa + ")*(" + pa + ")+(" + pb + ")*(" + pb + "))";
        std::string t1 = mulT("(" + pb + ")", da), t2 = mulT("(" + pa + ")", db);
        return "((" + t1 + ")-(" + t2 + "))/" + den;
      }
      // two-arg selectors: derivative follows the chosen branch
      if (n == 2 && (fn == "fmax" || fn == "max")) {
        const Expr *a1 = call->getArg(1);
        return "((" + pa + ") >= (" + prim(a1) + ") ? (" + da + ") : (" +
               diff(a1) + "))";
      }
      if (n == 2 && (fn == "fmin" || fn == "min")) {
        const Expr *a1 = call->getArg(1);
        return "((" + pa + ") <= (" + prim(a1) + ") ? (" + da + ") : (" +
               diff(a1) + "))";
      }
      if (fn == "pow" && n == 2) {
        const Expr *a1 = call->getArg(1);
        std::string pb = prim(a1), db = diff(a1);
        // d/dx u^v = v*u^(v-1)*u' + u^v*ln(u)*v'
        std::string t1 =
            mulT(mulT("(" + pb + ")", "pow(" + pa + ", (" + pb + ") - 1)"), da);
        if (db == "0")
          return t1;
        std::string t2 = mulT("pow(" + pa + ", " + pb + ")*log(" + pa + ")", db);
        return addT(t1, t2);
      }

      if (da == "0")
        return "0"; // single-arg intrinsic of a constant
      if (fn == "sin")
        return mulT("cos(" + pa + ")", da);
      if (fn == "cos")
        return mulT("-sin(" + pa + ")", da);
      if (fn == "tan")
        return mulT("1.0/(cos(" + pa + ")*cos(" + pa + "))", da);
      if (fn == "exp")
        return mulT("exp(" + pa + ")", da);
      if (fn == "log")
        return "(" + da + ")/(" + pa + ")";
      if (fn == "sqrt")
        return "(" + da + ")/(2*sqrt(" + pa + "))";
      if (fn == "cbrt")
        return "(" + da + ")/(3*cbrt(" + pa + ")*cbrt(" + pa + "))";
      if (fn == "tanh")
        return mulT("(1 - tanh(" + pa + ")*tanh(" + pa + "))", da);
      if (fn == "fabs" || fn == "abs")
        return mulT("((" + pa + ") >= 0 ? 1.0 : -1.0)", da);
      if (fn == "atan")
        return "(" + da + ")/(1+(" + pa + ")*(" + pa + "))";
      if (fn == "asin")
        return "(" + da + ")/sqrt(1-(" + pa + ")*(" + pa + "))";
      if (fn == "acos")
        return "-(" + da + ")/sqrt(1-(" + pa + ")*(" + pa + "))";
    }
    return unsupported(call->getBeginLoc(), "a call to `" + fn + "` (no derivative rule)",
                       "0 /*cuad: unhandled call: " + fn + "*/");
  }

  // ---------- lvalue tangent (assignment target) ----------
  // Returns the dot form of an assignable expression, or "" if inactive.
  std::string diffLValue(const Expr *lhs) const {
    const Expr *e = lhs->IgnoreParenImpCasts();
    if (!isFloat(e))
      return "";
    return dotAccess(e);   // var, a[i], a[i][j], p->f, s.f, *p -- nested chains handled
  }

  // ---------- differentiable user calls ----------
  // A call target we generate a _d companion for: any function with a body in
  // this TU (math intrinsics come from prototypes with no body).
  static bool isUserDiffFn(const FunctionDecl *fd) {
    // has a body and a plain name (excludes implicit operator=/ctors, whose
    // getName() would assert).
    return fd && fd->hasBody() && fd->getDeclName().isIdentifier();
  }
  static bool isNamed(const FunctionDecl *fd, StringRef n) {
    return fd && fd->getDeclName().isIdentifier() && fd->getName() == n;
  }

  static const CallExpr *asCall(const Stmt *s) {
    if (const auto *ewc = dyn_cast<ExprWithCleanups>(s))
      s = ewc->getSubExpr();
    if (const auto *e = dyn_cast<Expr>(s))
      return dyn_cast<CallExpr>(e->IgnoreImpCasts());
    return nullptr;
  }

  // Build a call to <fn>_d: each arg is passed primal, and after each active
  // parameter its tangent. retDot (if non-empty) is appended as the trailing
  // return-tangent argument.
  std::string buildDiffCall(const CallExpr *call,
                            const std::string &retDot) const {
    const FunctionDecl *fd = call->getDirectCallee();
    std::string out = fd->getNameAsString() + "_d(";
    bool first = true;
    for (unsigned i = 0; i < call->getNumArgs(); ++i) {
      const Expr *arg = call->getArg(i);
      const ParmVarDecl *p =
          i < fd->getNumParams() ? fd->getParamDecl(i) : nullptr;
      if (!first)
        out += ", ";
      first = false;
      out += prim(arg);
      if (p && isActiveType(p->getType())) {
        out += ", ";
        out += p->getType().getCanonicalType()->isPointerType() ? diffPtr(arg)
                                                                 : diff(arg);
      }
    }
    if (!retDot.empty()) {
      if (!first)
        out += ", ";
      out += retDot;
    }
    return out + ")";
  }

  std::string genAtomicAdd(const CallExpr *call, const std::string &ind) const {
    const Expr *a0 = call->getArg(0)->IgnoreParenImpCasts();
    const Expr *lval = nullptr;
    if (const auto *uo = dyn_cast<UnaryOperator>(a0))
      if (uo->getOpcode() == UO_AddrOf)
        lval = uo->getSubExpr();
    std::string dl = lval ? diffLValue(lval) : "";
    std::string dv = diff(call->getArg(1));
    std::string out;
    if (!dl.empty() && dv != "0")
      out += ind + "atomicAdd(&" + dl + ", " + dv + ");\n";
    out += ind + prim(call) + ";\n";
    return out;
  }

  // ---------- statement transformation ----------
  // Leaf statements collect hoisted tangent calls (pre_) and emit them first.
  // ---------- implicit tangents of convergence loops ----------
  // The loop runs as written. Its tangent follows from the equation it solves (see ReverseMode::SolveInfo):
  //  - smooth update x* = G(x*, p): (I - G_x) x' = G_p p', with G_x and G_p p' from tangent sweeps of one iteration;
  //  - bracketing loop: the root t of r(m, p) has t' = -r_p p' / r_m (or the tangent of a bound that never moved);
  //  - several iterated variables: the root of the residual found in the loop, t' = -R_p p' / R_t.
  // The exit state must lie within CUAD_SOLVE_TOL of a solution, and the tangent must not change across the
  // interval in which the solution is known (CUAD_SOLVE_AGREE); otherwise the kernel traps.
  std::string primalOnly(const Stmt *s, const std::string &ind) const {
    std::string t = src(s->getSourceRange());
    while (!t.empty() && (t.back() == ' ' || t.back() == '\n'))
      t.pop_back();
    if (!t.empty() && t.back() != ';' && t.back() != '}')
      t += ";";
    return ind + t + "\n";
  }
  // the tangent of expression e as a statement `into = ...;` (calls in it hoisted before)
  std::string tangentInto(const Expr *e, const std::string &into, const std::string &ind) const {
    std::string savedPre = pre_, savedInd = curInd_;
    pre_.clear();
    curInd_ = ind;
    std::string d = diff(e);
    std::string out = pre_ + ind + into + " = " + d + ";\n";
    pre_ = savedPre;
    curInd_ = savedInd;
    return out;
  }
  std::string declOf(const VarDecl *v) const { return tstr(v->getType().getNonReferenceType().getUnqualifiedType()); }

  std::string implicitTangent(const Stmt *loop, const ReverseMode::SolveInfo &si, int depth) const {
    const std::string ind(depth * 2, ' '), i1 = ind + "  ", i2 = i1 + "  ", i3 = i2 + "  ";
    const std::string id = std::to_string(implicitId_++);
    const std::string line = std::to_string(SM.getSpellingLineNumber(loop->getBeginLoc()));
    std::string out = ind + src(loop->getSourceRange());
    if (isa<DoStmt>(loop) || !isa<CompoundStmt>(ReverseMode::loopBody(loop)))
      out += ";";
    out += "\n";
    std::string shadowP0;
    for (const VarDecl *p : si.params)
      shadowP0 += i2 + declOf(p) + " " + p->getNameAsString() + "_d = 0;\n";
    if (!si.bracket && !si.residual) {
      const size_t K = si.comps.size();
      const std::string Ks = std::to_string(K);
      const std::string X = "_cuad_x" + id, R = "_cuad_r" + id, J = "_cuad_J" + id, B = "_cuad_B" + id,
                        M = "_cuad_M" + id, D = "_cuad_dd" + id, W = "_cuad_w" + id, W2 = "_cuad_v" + id;
      out += ind + "{ // cuad: implicit tangent of the convergence loop at line " + line +
             ": (I - G_x) x' = G_p p' solved directly\n";
      std::string xInit;
      for (size_t k = 0; k < K; ++k)
        xInit += std::string(k ? ", " : "") + "(double)" + si.comps[k].x;
      out += i1 + "double " + X + "[" + Ks + "] = {" + xInit + "}, " + R + "[" + Ks + "], " + J + "[" + Ks + "][" + Ks +
             "], " + B + "[" + Ks + "], " + M + "[" + Ks + "][" + Ks + "], " + D + "[" + Ks + "], " + W + "[" + Ks + "], " +
             W2 + "[" + Ks + "];\n";
      // one more iteration from the exit state (primal only): G(x) - x (checks only)
      out += i1 + "if (CUAD_SOLVE_CHECK) {\n";
      for (const Stmt *s : si.iter)
        out += primalOnly(s, i2);
      for (size_t k = 0; k < K; ++k)
        out += i2 + R + "[" + std::to_string(k) + "] = (double)" + si.comps[k].x + " - " + X + "[" + std::to_string(k) +
               "];\n";
      out += i1 + "}\n";
      for (size_t k = 0; k < K; ++k)
        out += i1 + si.comps[k].x + " = (" + si.comps[k].T + ")" + X + "[" + std::to_string(k) + "];\n";
      // a tangent sweep of one iteration from the state `at` with state tangent e_j (j < K) or 0 (j == K, the
      // parameters' tangents then flow in); the result goes to column j of J or to B
      auto sweep = [&](size_t j, const std::string &at) {
        std::string o = i1 + "{\n";
        size_t k = 0;
        for (const VarDecl *v : si.state) {
          std::string x = v->getNameAsString();
          if (v->getType()->isRealFloatingType()) {
            o += i2 + declOf(v) + " " + x + " = (" + declOf(v) + ")" + at + "[" + std::to_string(k) + "];\n";
            o += i2 + declOf(v) + " " + x + "_d = " + (k == j ? "1" : "0") + ";\n";
            ++k;
            continue;
          }
          QualType elem;
          std::string dims = arrayDims(v->getType(), elem), vi, di;
          size_t n = 1;
          for (QualType qt = v->getType().getCanonicalType(); const auto *cat = Ctx.getAsConstantArrayType(qt);
               qt = cat->getElementType())
            n *= cat->getSize().getZExtValue();
          for (size_t e = 0; e < n; ++e, ++k) {
            vi += std::string(e ? ", " : "") + "(" + tstr(elem) + ")" + at + "[" + std::to_string(k) + "]";
            di += std::string(e ? ", " : "") + (k == j ? "1" : "0");
          }
          o += i2 + tstr(elem) + " " + x + dims + " = {" + vi + "};\n" + i2 + tstr(elem) + " " + x + "_d" + dims + " = {" +
               di + "};\n";
        }
        if (j < K)
          o += shadowP0;
        for (const Stmt *s : si.iter)
          o += genStmt(s, depth + 2);
        for (size_t c = 0; c < K; ++c)
          o += i2 + (j < K ? J + "[" + std::to_string(c) + "][" + std::to_string(j) + "]" : B + "[" + std::to_string(c) + "]") +
               " = " + si.comps[c].d + ";\n";
        return o + i1 + "}\n";
      };
      auto tangent = [&](const std::string &at, const std::string &into, bool distance) {
        std::string o;
        for (size_t j = 0; j <= K; ++j)
          o += sweep(j, at);
        if (distance) {
          o += i1 + "if (CUAD_SOLVE_CHECK) {\n";
          o += i1 + "for (int _cuad_i = 0; _cuad_i < " + Ks + "; ++_cuad_i) {\n" + i2 + D + "[_cuad_i] = " + R +
               "[_cuad_i];\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) " + M +
               "[_cuad_i][_cuad_k] = (_cuad_i == _cuad_k) - " + J + "[_cuad_i][_cuad_k];\n" + i1 + "}\n";
          o += i1 + "if (!cuad_lsolve(" + M + ", " + D + ")) cuad_solve_fail();\n";
          o += i1 + "{\n" + i2 + "double _cuad_dn = 0, _cuad_xn = 0;\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks +
               "; ++_cuad_k) { _cuad_dn = fmax(_cuad_dn, fabs(" + D + "[_cuad_k])); _cuad_xn = fmax(_cuad_xn, fabs(" + X +
               "[_cuad_k])); }\n" + i2 +
               "if (_cuad_dn > CUAD_SOLVE_TOL * _cuad_xn + CUAD_SOLVE_ATOL) cuad_solve_fail();\n" + i1 + "}\n" + i1 + "}\n";
        }
        o += i1 + "for (int _cuad_i = 0; _cuad_i < " + Ks + "; ++_cuad_i) {\n" + i2 + into + "[_cuad_i] = " + B +
             "[_cuad_i];\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) " + M +
             "[_cuad_i][_cuad_k] = (_cuad_i == _cuad_k) - " + J + "[_cuad_i][_cuad_k];\n" + i1 + "}\n";
        o += i1 + "if (!cuad_lsolve(" + M + ", " + into + ")) cuad_solve_fail();\n";
        return o;
      };
      out += tangent(X, W, true);
      out += i1 + "double _cuad_bn" + id + " = 0;\n" + i1 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) _cuad_bn" +
             id + " = fmax(_cuad_bn" + id + ", fmax(fabs(" + W + "[_cuad_k]), fabs(" + B + "[_cuad_k])));\n";
      // the same at the far side of the interval in which the solution is known (skipped where G_x is negligible:
      // Newton's method at a simple root converges quadratically and its exit state is exact to rounding)
      out += i1 + "double _cuad_jm" + id + " = 0;\n" + i1 + "for (int _cuad_i = 0; _cuad_i < " + Ks +
             "; ++_cuad_i) for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) _cuad_jm" + id + " = fmax(_cuad_jm" + id +
             ", fabs(" + J + "[_cuad_i][_cuad_k]));\n";
      out += i1 + "if (CUAD_SOLVE_CHECK && (_cuad_jm" + id + " > 0.25 || !(_cuad_bn" + id + " < 1e300))) {\n";
      out += i1 + "double _cuad_y" + id + "[" + Ks + "];\n" + i1 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) _cuad_y" +
             id + "[_cuad_k] = " + X + "[_cuad_k] + " + D + "[_cuad_k] + copysign(4.440892098500626e-16 * fabs(" + X +
             "[_cuad_k]), " + D + "[_cuad_k]);\n";
      out += tangent("_cuad_y" + id, W2, false);
      out += i1 + "{\n" + i2 + "double _cuad_wd = 0;\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks +
             "; ++_cuad_k) _cuad_wd = fmax(_cuad_wd, fabs(" + W + "[_cuad_k] - " + W2 + "[_cuad_k]));\n" + i2 +
             "if (!(_cuad_wd <= CUAD_SOLVE_AGREE * _cuad_bn" + id + ") || !(_cuad_bn" + id +
             " < 1e300)) cuad_solve_fail();\n" + i1 + "}\n";
      out += i1 + "}\n";
      for (size_t k = 0; k < K; ++k)
        out += i1 + si.comps[k].d + " = (" + si.comps[k].T + ")" + W + "[" + std::to_string(k) + "];\n";
      return out + ind + "}\n";
    }
    if (si.bracket) {
      const auto *cmp = cast<BinaryOperator>(si.sel->getCond()->IgnoreParenImpCasts());
      const std::string m = si.trial->getNameAsString(), Tm = declOf(si.trial);
      std::string lo, hi;
      for (const VarDecl *v : si.state) {
        std::string x = "(double)" + v->getNameAsString();
        lo = lo.empty() ? x : "fmin(" + lo + ", " + x + ")";
        hi = hi.empty() ? x : "fmax(" + hi + ", " + x + ")";
      }
      out += ind + "{ // cuad: implicit tangent of the bracketing loop at line " + line + ": the root of r = (" +
             prim(cmp->getLHS()) + ") - (" + prim(cmp->getRHS()) + ") in " + m + " moves by -r_p p' / r_" + m + "\n";
      out += i1 + "if (CUAD_SOLVE_CHECK && " + hi + " - " + lo + " > CUAD_SOLVE_TOL * fmax(fabs(" + hi + "), fabs(" + lo +
             ")) + CUAD_SOLVE_ATOL) cuad_solve_fail();\n";
      // the residual, r_m and r_p p' at the point `at`
      auto eval = [&](const std::string &at, const std::string &r, const std::string &rm, const std::string &rp) {
        std::string o;
        for (int pass = 0; pass < 2; ++pass) {
          o += i1 + "{\n";
          for (const VarDecl *v : si.state)
            o += i2 + declOf(v) + " " + v->getNameAsString() + " = (" + declOf(v) + ")(" + at + ");\n";
          o += primalOnly(si.iter.front(), i2) + i2 + Tm + " " + m + "_d = " + (pass == 0 ? "1" : "0") + ";\n";
          if (pass == 0)
            o += shadowP0;
          for (size_t i = 1; i < si.iter.size(); ++i)
            if (si.iter[i] != si.sel)
              o += genStmt(si.iter[i], depth + 2);
          o += i2 + "double _cuad_t;\n" + tangentInto(cmp->getLHS(), "_cuad_t", i2) + i2 + (pass == 0 ? rm : rp) +
               " = _cuad_t;\n" + tangentInto(cmp->getRHS(), "_cuad_t", i2) + i2 + (pass == 0 ? rm : rp) + " -= _cuad_t;\n";
          if (pass == 1 && !r.empty())
            o += i2 + r + " = (double)((" + prim(cmp->getLHS()) + ") - (" + prim(cmp->getRHS()) + "));\n";
          o += i1 + "}\n";
        }
        return o;
      };
      out += i1 + "double _cuad_r" + id + ", _cuad_rm" + id + ", _cuad_rp" + id + ", _cuad_m" + id + " = 0.5 * ((" + lo +
             ") + (" + hi + "));\n";
      out += eval("_cuad_m" + id, "_cuad_r" + id, "_cuad_rm" + id, "_cuad_rp" + id);
      out += i1 + "if (fabs(_cuad_r" + id + ") <= fabs(_cuad_rm" + id + ") * (CUAD_SOLVE_TOL * fabs(_cuad_m" + id +
             ") + CUAD_SOLVE_ATOL)) {\n";
      out += i2 + "double _cuad_e" + id + "[2], _cuad_a, _cuad_b;\n" + i2 + "const double _cuad_lo" + id + " = " + lo +
             ", _cuad_hi" + id + " = " + hi + ";\n";
      out += i2 + "const double _cuad_t" + id + " = -_cuad_rp" + id + " / _cuad_rm" + id + ";\n";
      out += i2 + "if (CUAD_SOLVE_CHECK) {\n" + i2 + "_cuad_e" + id + "[0] = -_cuad_rp" + id + " / _cuad_rm" + id + "; // at the midpoint\n";
      out += eval("_cuad_hi" + id, "", "_cuad_a", "_cuad_b") + i2 + "_cuad_e" + id + "[1] = -_cuad_b / _cuad_a;\n";
      out += i2 + "{\n" + i3 + "double _cuad_en = fmax(fabs(_cuad_e" + id + "[0]), fabs(_cuad_e" + id + "[1])), _cuad_ed = fabs(_cuad_e" +
             id + "[0] - _cuad_e" + id + "[1]);\n" + i3 +
             "if (!(_cuad_ed <= CUAD_SOLVE_AGREE * _cuad_en) || !(_cuad_en < 1e300)) cuad_solve_fail();\n" + i2 + "}\n" + i2 + "}\n";
      for (const VarDecl *v : si.state)
        out += i2 + v->getNameAsString() + "_d = _cuad_t" + id + ";\n";
      // the root is outside the bracket: the result is the bound that never moved, with its own tangent
      std::set<const VarDecl *> thenSet, elseSet;
      for (int armI = 0; armI < 2; ++armI) {
        const Stmt *arm = armI == 0 ? si.sel->getThen() : si.sel->getElse();
        std::vector<const Stmt *> as;
        if (const auto *cs = dyn_cast<CompoundStmt>(arm))
          as.assign(cs->body_begin(), cs->body_end());
        else
          as.push_back(arm);
        for (const Stmt *a : as)
          if (const auto *bo = dyn_cast<BinaryOperator>(a))
            if (const auto *l = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
              if (const auto *lv = dyn_cast<VarDecl>(l->getDecl()))
                (armI == 0 ? thenSet : elseSet).insert(lv);
      }
      auto pinned = [&](const std::set<const VarDecl *> &moving) {
        std::string sum;
        int n = 0;
        for (const VarDecl *v : si.state)
          if (!moving.count(v)) {
            sum += (sum.empty() ? "" : " + ") + v->getNameAsString() + "_d";
            ++n;
          }
        std::string o = i3 + "const double _cuad_f = (" + (sum.empty() ? std::string("0") : sum) + ") / " +
                        std::to_string(std::max(n, 1)) + ".0;\n";
        for (const VarDecl *v : si.state)
          o += i3 + v->getNameAsString() + "_d = _cuad_f;\n";
        return o;
      };
      out += i1 + "} else {\n";
      out += i2 + "const double _cuad_c = _cuad_m" + id + ";\n" + i2 + "bool _cuad_up;\n" + i2 + "{\n";
      for (const VarDecl *v : si.state)
        out += i3 + declOf(v) + " " + v->getNameAsString() + " = (" + declOf(v) + ")_cuad_c;\n";
      for (size_t i = 0; i < si.iter.size(); ++i)
        if (si.iter[i] != si.sel)
          out += primalOnly(si.iter[i], i3);
      out += i3 + "_cuad_up = (" + prim(si.sel->getCond()) + ");\n" + i2 + "}\n";
      out += i2 + "if (_cuad_up) {\n" + pinned(thenSet) + i2 + "} else {\n" + pinned(elseSet) + i2 + "}\n";
      out += i1 + "}\n";
      return out + ind + "}\n";
    }
    // several iterated variables: the residual that vanishes at the exit state
    const size_t K = si.res.size();
    const std::string Ks = std::to_string(K);
    out += ind + "{ // cuad: implicit tangent of the root-finding loop at line " + line +
           ": t' = -R_p p' / R_t for the residual that vanishes at the exit state, among";
    for (const auto &r : si.res)
      out += " " + r.var->getNameAsString() + "(" + r.trialVar->getNameAsString() + ")";
    out += "\n";
    const std::string V = "_cuad_v" + id, XK = "_cuad_xk" + id, RK = "_cuad_rk" + id, DK = "_cuad_dk" + id,
                      RV = "_cuad_rv" + id, use = "_cuad_use" + id;
    out += i1 + "int " + use + " = -1, " + V + "[" + Ks + "];\n" + i1 + "double " + XK + "[" + Ks + "], " + RK + "[" + Ks +
           "], " + DK + "[" + Ks + "], " + RV + "[" + Ks + "];\n";
    // R, R_t and R_p p' of candidate r at its trial point shifted by `shift`
    auto eval = [&](const ReverseMode::SolveInfo::Residual &r, const std::string &shift, const std::string &R,
                    const std::string &Rt, const std::string &Rp, const std::string &tAt) {
      std::string o;
      const std::string t = r.trialVar->getNameAsString(), Tt = declOf(r.trialVar);
      for (int pass = 0; pass < 2; ++pass) {
        o += i2 + "{\n";
        for (const Stmt *s : r.trialDecls)
          o += primalOnly(s, i3);
        o += i3 + "const double _cuad_tt = (double)" + t + shift + ";\n";
        if (pass == 0 && !tAt.empty())
          o += i3 + tAt + " = _cuad_tt;\n";
        o += i3 + "{\n" + i3 + "  " + Tt + " " + t + " = (" + Tt + ")_cuad_tt;\n" + i3 + "  " + Tt + " " + t + "_d = " +
             (pass == 0 ? "1" : "0") + ";\n";
        if (pass == 0)
          for (const VarDecl *p : si.params)
            o += i3 + "  " + declOf(p) + " " + p->getNameAsString() + "_d = 0;\n";
        for (const Stmt *s : r.slice)
          o += genStmt(s, depth + 4);
        o += tangentInto(r.expr, pass == 0 ? Rt : Rp, i3 + "  ");
        if (pass == 1 && !R.empty())
          o += i3 + "  " + R + " = (double)(" + prim(r.expr) + ");\n";
        o += i3 + "}\n" + i2 + "}\n";
      }
      return o;
    };
    for (size_t k = 0; k < K; ++k) {
      const std::string kk = std::to_string(k);
      out += i1 + "{ // candidate " + si.res[k].var->getNameAsString() + "(" + si.res[k].trialVar->getNameAsString() + ")\n";
      out += i2 + "double _cuad_R, _cuad_Rt, _cuad_Rp, _cuad_X;\n";
      out += eval(si.res[k], "", "_cuad_R", "_cuad_Rt", "_cuad_Rp", "_cuad_X");
      out += i2 + V + "[" + kk + "] = _cuad_Rt != 0 && fabs(_cuad_R) <= fabs(_cuad_Rt) * (CUAD_SOLVE_TOL * fabs(_cuad_X) + "
             "CUAD_SOLVE_ATOL);\n";
      out += i2 + XK + "[" + kk + "] = _cuad_X; " + RK + "[" + kk + "] = _cuad_Rt; " + RV + "[" + kk + "] = _cuad_R; " + DK +
             "[" + kk + "] = -_cuad_Rp / _cuad_Rt;\n" + i1 + "}\n";
    }
    out += i1 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) {\n" + i2 + "if (!" + V + "[_cuad_k]) continue;\n" +
           i2 + "if (" + use + " < 0) { " + use + " = _cuad_k; continue; }\n" + i2 + "if (fabs(" + XK + "[_cuad_k] - " + XK +
           "[" + use + "]) > CUAD_SOLVE_AGREE * fmax(fabs(" + XK + "[_cuad_k]), fabs(" + XK + "[" + use +
           "])) + CUAD_SOLVE_ATOL || fabs(" + DK + "[_cuad_k] - " + DK + "[" + use + "]) > CUAD_SOLVE_AGREE * fmax(fabs(" +
           DK + "[_cuad_k]), fabs(" + DK + "[" + use + "]))) cuad_solve_fail();\n" + i1 + "}\n";
    out += i1 + "if (" + use + " < 0) cuad_solve_fail();\n";
    // conditioning of the chosen root, then the tangents of the iterated variables
    for (size_t k = 0; k < K; ++k) {
      const auto &r = si.res[k];
      const std::string kk = std::to_string(k);
      out += i1 + "if (" + use + " == " + kk + ") {\n" + i2 + "if (CUAD_SOLVE_CHECK) {\n" + i2 + "double _cuad_R, _cuad_Rt, _cuad_Rp;\n";
      out += eval(r, " + fabs(" + RV + "[" + kk + "] / " + RK + "[" + kk + "]) + 4.440892098500626e-16 * fabs((double)" +
                         r.trialVar->getNameAsString() + ")",
                  "", "_cuad_Rt", "_cuad_Rp", "");
      out += i2 + "{\n" + i3 + "double _cuad_e = -_cuad_Rp / _cuad_Rt, _cuad_en = fmax(fabs(_cuad_e), fabs(" + DK + "[" + kk +
             "]));\n" + i3 + "if (!(fabs(_cuad_e - " + DK + "[" + kk +
             "]) <= CUAD_SOLVE_AGREE * _cuad_en) || !(_cuad_en < 1e300)) cuad_solve_fail();\n" + i2 + "}\n" + i2 + "}\n";
      for (const VarDecl *v : si.state) {
        std::string x = v->getNameAsString(), held;
        for (size_t c = 0; c < K; ++c)
          if (si.res[c].var == v)
            held += (held.empty() ? "" : " || ") + V + "[" + std::to_string(c) + "]";
        out += i2 + "if (fabs((double)" + x + " - " + XK + "[" + kk + "]) <= CUAD_SOLVE_TOL * fabs(" + XK + "[" + kk +
               "]) + CUAD_SOLVE_ATOL) " + x + "_d = " + DK + "[" + kk + "];\n";
        if (!held.empty())
          out += i2 + "else if (" + held + ") " + x + "_d = 0;\n";
        out += i2 + "else " + x + "_d = __builtin_nan(\"\"); // not a function of the root: its tangent is undefined\n";
      }
      out += i1 + "}\n";
    }
    return out + ind + "}\n";
  }

  std::string genStmt(const Stmt *s, int depth) const {
    if (isa<CompoundStmt>(s) || isa<ForStmt>(s) || isa<IfStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s))
      return genStmtInner(s, depth);
    std::string savedPre = pre_, savedInd = curInd_;
    pre_.clear();
    curInd_ = std::string(depth * 2, ' ');
    std::string body = genStmtInner(s, depth);
    std::string out = pre_ + body;
    pre_ = savedPre;
    curInd_ = savedInd;
    return out;
  }

  std::string genStmtInner(const Stmt *s, int depth) const {
    std::string ind(depth * 2, ' ');

    if (const auto *cs = dyn_cast<CompoundStmt>(s)) {
      std::string out;
      for (const Stmt *c : cs->body())
        out += genStmt(c, depth);
      return out;
    }

    // atomicAdd(&loc, v): tangent scatters the value's tangent into loc's dot.
    if (const CallExpr *call = asCall(s))
      if (isNamed(call->getDirectCallee(), "atomicAdd") &&
          call->getNumArgs() == 2)
        return genAtomicAdd(call, ind);

    // Non-differentiable operator/ctor call (e.g. struct copy `B = bb[i]`):
    // reproduce the primal, no tangent, no warning.
    if (const CallExpr *call = asCall(s))
      if (const FunctionDecl *fd = call->getDirectCallee())
        if (!fd->getDeclName().isIdentifier())
          return ind + prim(call) + ";\n";

    // Standalone call to a differentiable function: the _d companion computes
    // both primal and tangent, so it replaces the primal call.
    if (const CallExpr *call = asCall(s))
      if (isUserDiffFn(call->getDirectCallee()))
        return ind + buildDiffCall(call, "") + ";\n";

    if (const auto *ds = dyn_cast<DeclStmt>(s))
      return genDeclStmt(ds, ind);

    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        return genAssign(bo, ind);

    if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s)) {
      const ReverseMode::SolveInfo &si = An_.convergenceLoop(s);
      if (si.ok)
        return implicitTangent(s, si, depth);
    }
    if (const auto *dw = dyn_cast<DoStmt>(s)) {
      std::string out = ind + "do {\n";
      if (dw->getBody())
        out += genStmt(dw->getBody(), depth + 1);
      return out + ind + "} while (" + prim(dw->getCond()) + ");\n";
    }
    if (const auto *fs = dyn_cast<ForStmt>(s)) {
      if (controlOnlyLoop(fs))
        unsupported(fs->getBeginLoc(),
                    "a loop that exits with `break` and refines floating variables that reach the active inputs only "
                    "through its branch conditions (a bracketing root solve: the tangent of its iterations is the "
                    "tangent of the initial bracket, not of the root; reverse mode differentiates it implicitly)",
                    "");
      std::string init = fs->getInit() ? src(fs->getInit()->getSourceRange()) : "";
      if (!init.empty() && init.back() == ';') // normalize: we re-add the ';'
        init.pop_back();
      std::string cond = fs->getCond() ? prim(fs->getCond()) : "";
      std::string inc = fs->getInc() ? prim(fs->getInc()) : "";
      std::string out = ind + "for (" + init + "; " + cond + "; " + inc + ") {\n";
      if (fs->getBody())
        out += genStmt(fs->getBody(), depth + 1);
      out += ind + "}\n";
      return out;
    }

    if (const auto *is = dyn_cast<IfStmt>(s)) {
      std::string out = ind + "if (" + prim(is->getCond()) + ") {\n";
      if (is->getThen())
        out += genStmt(is->getThen(), depth + 1);
      out += ind + "}";
      if (is->getElse()) {
        out += " else {\n" + genStmt(is->getElse(), depth + 1) + ind + "}";
      }
      out += "\n";
      return out;
    }

    if (const auto *ws = dyn_cast<WhileStmt>(s)) {
      std::string out = ind + "while (" + prim(ws->getCond()) + ") {\n";
      if (ws->getBody())
        out += genStmt(ws->getBody(), depth + 1);
      out += ind + "}\n";
      return out;
    }

    if (const auto *rs = dyn_cast<ReturnStmt>(s)) {
      if (rs->getRetValue()) {
        if (!curRetDot_.empty())
          return ind + curRetDot_ + " = " + diff(rs->getRetValue()) + ";\n" +
                 ind + "return " + prim(rs->getRetValue()) + ";\n";
        return ind + "return " + prim(rs->getRetValue()) + ";\n";
      }
      return ind + "return;\n";
    }

    if (isa<NullStmt>(s))
      return ind + ";\n";
    // the tangent follows the primal control flow, so jumps are copied as they are
    if (isa<BreakStmt>(s))
      return ind + "break;\n";
    if (isa<ContinueStmt>(s))
      return ind + "continue;\n";

    // Unhandled: reproduce primal so the emitted code still compiles.
    unsupported(s->getBeginLoc(), "this statement (kept as primal only)", "");
    return ind + src(s->getSourceRange()) +
           "; // TODO(cuad): unhandled statement, primal only\n";
  }

  // "[N][M]" dimension suffix + innermost element type of an array.
  std::string arrayDims(QualType qt, QualType &elemOut) const {
    std::string d;
    qt = qt.getCanonicalType();
    while (const auto *cat = Ctx.getAsConstantArrayType(qt)) {
      d += "[" + std::to_string(cat->getSize().getZExtValue()) + "]";
      qt = cat->getElementType();
    }
    elemOut = qt;
    return d;
  }

  // Tangent of a pointer-valued expression (base±offset, &a[i], ...).
  std::string diffPtr(const Expr *e) const {
    if (const auto *c = dyn_cast<CastExpr>(e))
      return diffPtr(c->getSubExpr());
    if (const auto *p = dyn_cast<ParenExpr>(e))
      return "(" + diffPtr(p->getSubExpr()) + ")";
    if (const auto *dre = dyn_cast<DeclRefExpr>(e))
      return Act.isVaried(dre->getDecl()) ? declName(dre) + "_d" : "0";
    if (const auto *bo = dyn_cast<BinaryOperator>(e))
      if (bo->getOpcode() == BO_Add || bo->getOpcode() == BO_Sub) {
        std::string op = bo->getOpcode() == BO_Add ? " + " : " - ";
        if (bo->getLHS()->getType()->isPointerType())
          return diffPtr(bo->getLHS()) + op + prim(bo->getRHS());
        return prim(bo->getLHS()) + op + diffPtr(bo->getRHS());
      }
    if (const auto *uo = dyn_cast<UnaryOperator>(e))
      if (uo->getOpcode() == UO_AddrOf) {
        std::string lv = diffLValue(uo->getSubExpr());
        if (!lv.empty())
          return "&" + lv;
      }
    return unsupported(e->getBeginLoc(), "the tangent of this pointer expression",
                       "0 /*cuad: unhandled pointer tangent*/");
  }

  std::string genDeclStmt(const DeclStmt *ds, const std::string &ind) const {
    // Single-declarator init from a differentiable value-returning call:
    //   double v = f(u);  ->  double v_d;  double v = f_d(u, u_d, v_d);
    if (ds->isSingleDecl())
      if (const auto *vd = dyn_cast<VarDecl>(ds->getSingleDecl()))
        if (vd->hasInit() && Act.isVaried(vd) &&
            vd->getType()->isRealFloatingType())
          if (const CallExpr *call = asCall(vd->getInit()))
            if (isUserDiffFn(call->getDirectCallee())) {
              std::string ts = tstr(vd->getType());
              std::string nm = vd->getNameAsString();
              return ind + ts + " " + nm + "_d;\n" + ind + ts + " " + nm +
                     " = " + buildDiffCall(call, nm + "_d") + ";\n";
            }

    // Primal reproduced verbatim from source (keeps __shared__, array dims,
    // initializers, and multi-declarator forms). Dot companions are prepended
    // for each varied active declarator.
    std::string raw = src(ds->getSourceRange());
    if (!raw.empty() && raw.back() == ';') // DeclStmt range may include the ';'
      raw.pop_back();
    bool isShared = raw.find("__shared__") != std::string::npos;
    // `double W = sqrt(1 + u*u), v = u / W;` - a later declarator may read an earlier one, so all tangents
    // ahead of the whole statement read W before it exists. Scalar declarators are split, each tangent just
    // before its own primal.
    if (!ds->isSingleDecl() && !isShared) {
      bool scalars = true;
      for (const Decl *d : ds->decls()) {
        const auto *vd = dyn_cast<VarDecl>(d);
        if (!vd || Ctx.getAsConstantArrayType(vd->getType()))
          scalars = false;
      }
      if (scalars) {
        std::string out;
        for (const Decl *d : ds->decls()) {
          const auto *vd = cast<VarDecl>(d);
          std::string nm = vd->getNameAsString(), ts = tstr(vd->getType());
          if (Act.isVaried(vd)) {
            if (vd->getType().getCanonicalType()->isPointerType())
              out += ind + ts + " " + nm + "_d" + (vd->hasInit() ? " = " + diffPtr(vd->getInit()) : "") + ";\n";
            else
              out += ind + ts + " " + nm + "_d" + (vd->hasInit() ? " = " + diff(vd->getInit()) : "") + ";\n";
          }
          out += ind + ts + " " + nm + (vd->hasInit() ? " = " + prim(vd->getInit()) : "") + ";\n";
        }
        return out;
      }
    }
    std::string dots;
    for (const Decl *d : ds->decls()) {
      const auto *vd = dyn_cast<VarDecl>(d);
      if (!vd || !Act.isVaried(vd))
        continue;
      QualType qt = vd->getType();
      std::string nm = vd->getNameAsString();

      if (Ctx.getAsConstantArrayType(qt)) {
        QualType elem;
        std::string dims = arrayDims(qt, elem);
        if (elem->isRealFloatingType()) {
          std::string es = tstr(elem);
          dots += ind + (isShared ? "__shared__ " : "") + es + " " + nm + "_d" +
                  dims + (isShared ? "" : " = {0}") + ";\n";
        }
      } else if (qt.getCanonicalType()->isPointerType()) {
        std::string ts = tstr(qt);
        dots += ind + ts + " " + nm + "_d" +
                (vd->hasInit() ? " = " + diffPtr(vd->getInit()) : "") + ";\n";
      } else { // scalar floating
        std::string ts = tstr(qt);
        dots += ind + ts + " " + nm + "_d" +
                (vd->hasInit() ? " = " + diff(vd->getInit()) : "") + ";\n";
      }
    }
    return dots + ind + raw + ";\n";
  }

  std::string genAssign(const BinaryOperator *bo, const std::string &ind) const {
    std::string dl = diffLValue(bo->getLHS());
    std::string primal = prim(bo) + ";";
    if (dl.empty()) // inactive target: keep primal only
      return ind + primal + "\n";

    // lhs = f(args): a value-returning differentiable call. The _d companion
    // returns the primal (assigned to lhs) and writes the tangent into dl.
    if (bo->getOpcode() == BO_Assign)
      if (const CallExpr *call = asCall(bo->getRHS()))
        if (isUserDiffFn(call->getDirectCallee()))
          return ind + prim(bo->getLHS()) + " = " + buildDiffCall(call, dl) +
                 ";\n";

    std::string dr = diff(bo->getRHS());
    std::string tan;
    if (bo->getOpcode() == BO_Assign) {
      tan = dl + " = " + dr + ";";
    } else {
      std::string pl = prim(bo->getLHS()), pr = prim(bo->getRHS());
      switch (bo->getOpcode()) {
      case BO_AddAssign:
        tan = dl + " = " + addT(dl, dr) + ";";
        break;
      case BO_SubAssign:
        tan = dl + " = " + subT(dl, dr) + ";";
        break;
      case BO_MulAssign:
        tan = dl + " = " + addT(mulT(dl, pr), mulT(pl, dr)) + ";";
        break;
      case BO_DivAssign: {
        std::string num = subT(mulT(dl, pr), mulT(pl, dr));
        tan = dl + " = " +
              (num == "0" ? std::string("0")
                          : "(" + num + ")/((" + pr + ")*(" + pr + "))") +
              ";";
        break;
      }
      default:
        tan = unsupported(bo->getBeginLoc(), "this compound assignment",
                          "/* cuad: unsupported compound assignment */");
      }
    }
    // tangent first (uses old primal values), then primal update
    return ind + tan + "\n" + ind + primal + "\n";
  }
};

} // namespace cuad
