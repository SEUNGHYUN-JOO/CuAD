// cuad — reverse (adjoint) mode source-to-source AD for CUDA.
//
// For y = f(x) this generates f_b that propagates adjoints backward:
//   x_b += (df/dx)^T * y_b
// Output adjoints (y_b) are seeds; input adjoints (x_b) accumulate. The body is
// a forward *primal* sweep (recomputing intermediates and PUSHing values that
// get overwritten) followed by a reversed *adjoint* sweep (POPing to restore
// primal state and distributing each result's adjoint onto its operands).
//
// v1 scope:
//   - void kernels AND value-returning __device__ functions (return adjoint is
//     seeded through a trailing parameter);
//   - straight-line math, reduction/counted for-loops (reversed via a tape),
//     guard if-statements with non-active conditions;
//   - = , += , -= , *= , /=  (overwrites handled by the tape);
//   - cross-call adjoints: a call to another differentiable function becomes a
//     call to its _b companion.
// Tape: a per-thread LIFO stack (cuad_push / cuad_pop, see test/cuad_tape.h).
// Still TODO: struct/member fields, atomics, non-canonical loop reversal
// (kept in forward order with a warning), and calls whose active arguments are
// not simple lvalues.

#pragma once
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Activity.h"
#include "Diag.h"

namespace cuad {
using namespace clang;

class ReverseMode {
  ASTContext &Ctx;
  const SourceManager &SM;
  const LangOptions &LO;
  PrintingPolicy PP;
  Activity Act;
  std::set<const VarDecl *> taped_;  // scalar locals overwritten -> push/pop
  std::set<const VarDecl *> accum_;  // pure accumulators: overwritten, but no tape needed
  std::set<const VarDecl *> reused_; // local arrays written by >1 call -> tape
  std::string curRetSeed_;           // return-adjoint seed param (value-ret fns)
  mutable int tmp_ = 0;              // fresh snapshot-variable counter
  mutable bool noPush_ = false;      // emitting a push-free primal (a recompute loop's forward pass)
  mutable std::set<const VarDecl *> hoisted_; // branch-local declarations hoisted to their block
  mutable std::map<const ForStmt *, int> shadowId_; // loops whose in-place local arrays are saved

  // Record an unsupported construct (error) or a questionable one (warning) at its location.
  void unsupported(SourceLocation L, const std::string &what) const {
    diag().add(SM, L, true, "cuad (reverse mode) cannot differentiate " + what);
  }
  void caution(SourceLocation L, const std::string &what) const {
    diag().add(SM, L, false, "cuad (reverse mode): " + what);
  }
  // does the statement read or write any varied variable?
  bool hasActive(const Stmt *s) const {
    if (!s)
      return false;
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (Act.isVaried(d->getDecl()))
        return true;
    for (const Stmt *c : s->children())
      if (hasActive(c))
        return true;
    return false;
  }
  // a `break` that leaves this loop body (not one belonging to a nested loop or switch)
  static bool containsBreak(const Stmt *s) {
    if (!s)
      return false;
    if (isa<BreakStmt>(s))
      return true;
    if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s) || isa<SwitchStmt>(s))
      return false;
    for (const Stmt *c : s->children())
      if (containsBreak(c))
        return true;
    return false;
  }

public:
  explicit ReverseMode(ASTContext &C)
      : Ctx(C), SM(C.getSourceManager()), LO(C.getLangOpts()),
        PP(C.getPrintingPolicy()) {
    PP.SuppressTagKeyword = true;
  }

  std::string genFunction(const FunctionDecl *FD) {
    Act = Activity();
    Act.run(FD);
    taped_.clear();
    reused_.clear();
    hoisted_.clear();
    tmp_ = 0;
    collectTaped(FD->getBody());
    collectAccum(FD->getBody());
    { // local arrays written as active output args by >1 differentiable call
      std::map<const VarDecl *, int> cnt;
      collectReused(FD->getBody(), cnt);
      for (auto &kv : cnt)
        if (kv.second >= 2)
          reused_.insert(kv.first);
    }

    std::string out;
    // a function template's companion is a template with the same parameter list (the
    // pattern is what is differentiated; calls in it resolve through calleeOf, their companions keep the `<...>`)
    if (const FunctionTemplateDecl *ft = FD->getDescribedFunctionTemplate())
      out += src(ft->getTemplateParameters()->getSourceRange()) + "\n";
    bool isGlobal = false;
    for (const auto *a : FD->specific_attrs<AnnotateAttr>()) {
      StringRef s = a->getAnnotation();
      if (s == "cuda_global") {
        out += "__global__ ";
        isGlobal = true;
      } else if (s == "cuda_device")
        out += "__device__ ";
      else if (s == "cuda_host")
        out += "__host__ ";
    }
    QualType ret = FD->getReturnType();
    curRetSeed_.clear();
    if (!ret->isVoidType()) {
      if (ret->isRealFloatingType())
        curRetSeed_ = FD->getNameAsString() + "_seed_b";
      else {
        caution(FD->getBeginLoc(), "the non-floating return value of `" + FD->getNameAsString() +
                                       "` is dropped in the adjoint");
        out += "// TODO(cuad): reverse of non-floating return type; emitting "
               "void adjoint.\n";
      }
    }
    out += "void " + FD->getNameAsString() + "_b(";
    bool first = true;
    std::string localScalarAdj;   // a __global__ kernel cannot take `float &x_b`
    for (const ParmVarDecl *p : FD->parameters()) {
      if (!first)
        out += ", ";
      first = false;
      out += tstr(p->getType()) + " " + p->getNameAsString();
      // a passive parameter's default argument (`bool flag = false`) is kept - calls that
      // omit it must still match the companion (active parameters with defaults are not supported: the adjoint
      // parameter would follow a defaulted one)
      if (p->hasDefaultArg() && !isActiveType(p->getType()) && !p->hasUnparsedDefaultArg() &&
          !p->hasUninstantiatedDefaultArg())
        out += " = " + prim(p->getDefaultArg());
      else if (p->hasUninstantiatedDefaultArg())
        out += " = " + prim(p->getUninstantiatedDefaultArg());
      if (isActiveType(p->getType())) {
        bool scalar = !p->getType().getCanonicalType()->isPointerType() &&
                      !p->getType().getCanonicalType()->isArrayType();
        if (isGlobal && scalar)
          // a kernel's scalar parameters are uniform across the launch (ratios, radii, dt): held passive, their
          // adjoint is a per-thread local that is discarded
          localScalarAdj += "  " + tstr(p->getType().getCanonicalType().getUnqualifiedType()) + " " +
                            p->getNameAsString() + "_b = 0;\n";
        else
          out += ", " + adjParam(p);
      }
    }
    if (!curRetSeed_.empty()) {
      if (!first)
        out += ", ";
      out += tstr(ret) + " " + curRetSeed_;
    }
    out += ")\n{\n";
    out += localScalarAdj;
    if (!taped_.empty() || !reused_.empty()) // per-thread private tape
      out += "  double _cuad_tape[CUAD_TAPE_SIZE]; int _cuad_tp = 0;\n";
    if (const auto *body = dyn_cast_or_null<CompoundStmt>(FD->getBody())) {
      std::vector<const Stmt *> stmts(body->body_begin(), body->body_end());
      out += genScope(stmts, 1);
    }
    out += "}\n";
    return out;
  }

private:
  // ---------------- helpers ----------------
  std::string tstr(QualType qt) const { return qt.getAsString(PP); }
  std::string src(SourceRange r) const {
    return Lexer::getSourceText(CharSourceRange::getTokenRange(r), SM, LO).str();
  }
  std::string prim(const Expr *e) const { return src(e->getSourceRange()); }
  // inside a function template the expressions that involve the template parameter are
  // type-dependent; their floatness is read from their parts (a call: its callee's return type)
  static bool isFloat(const Expr *e) {
    if (!e)
      return false;
    QualType t = e->getType();
    if (!t.isNull() && !t->isDependentType())
      return t->isRealFloatingType();
    e = e->IgnoreParenImpCasts();
    if (const auto *ce = dyn_cast<CallExpr>(e)) {
      const FunctionDecl *fd = calleeOf(ce);
      return fd && fd->getReturnType()->isRealFloatingType();
    }
    if (const auto *bo = dyn_cast<BinaryOperator>(e))
      return isFloat(bo->getLHS()) || isFloat(bo->getRHS());
    if (const auto *uo = dyn_cast<UnaryOperator>(e))
      return isFloat(uo->getSubExpr());
    if (const auto *co = dyn_cast<ConditionalOperator>(e))
      return isFloat(co->getTrueExpr()) || isFloat(co->getFalseExpr());
    return false;
  }
  // the function a call resolves to - also inside a template, where a call with a
  // dependent template argument (`f<D>(...)`) or dependent arguments is an UnresolvedLookupExpr until instantiation
  static const FunctionDecl *calleeOf(const CallExpr *ce) {
    if (!ce)
      return nullptr;
    if (const FunctionDecl *fd = ce->getDirectCallee())
      return fd;
    const Expr *c = ce->getCallee()->IgnoreParenImpCasts();
    if (const auto *ule = dyn_cast<UnresolvedLookupExpr>(c))
      for (const NamedDecl *nd : ule->decls()) {
        if (const auto *ft = dyn_cast<FunctionTemplateDecl>(nd))
          return ft->getTemplatedDecl();
        if (const auto *fd = dyn_cast<FunctionDecl>(nd))
          return fd;
      }
    return nullptr;
  }
  // the explicit template arguments of a call's callee as written (`<D>`), or ""
  std::string calleeTArgs(const CallExpr *ce) const {
    const Expr *c = ce->getCallee()->IgnoreParenImpCasts();
    if (const auto *ule = dyn_cast<UnresolvedLookupExpr>(c))
      if (ule->hasExplicitTemplateArgs())
        return Lexer::getSourceText(CharSourceRange::getTokenRange(ule->getLAngleLoc(), ule->getRAngleLoc()),
                                    SM, LO).str();
    if (const auto *dre = dyn_cast<DeclRefExpr>(c))
      if (dre->hasExplicitTemplateArgs())
        return Lexer::getSourceText(CharSourceRange::getTokenRange(dre->getLAngleLoc(), dre->getRAngleLoc()),
                                    SM, LO).str();
    return "";
  }
  static std::string declName(const DeclRefExpr *d) {
    return d->getNameInfo().getName().getAsString();
  }
  static bool isActiveType(QualType qt) {
    // a `float &` parameter (an output by reference, e.g. reconstructed face states) is active too;
    // before, its adjoint was never passed and every derivative through it was lost
    qt = qt.getNonReferenceType().getCanonicalType();
    if (qt->isPointerType())
      return qt->getPointeeType()->isRealFloatingType();
    if (qt->isArrayType())
      return qt->getAsArrayTypeUnsafe()->getElementType()->isRealFloatingType();
    return qt->isRealFloatingType();
  }
  std::string adjParam(const ParmVarDecl *p) const {
    QualType qt = p->getType().getNonReferenceType().getCanonicalType();
    std::string nm = p->getNameAsString() + "_b";
    if (qt->isPointerType())
      return tstr(qt->getPointeeType().getUnqualifiedType()) + " *" + nm;
    return tstr(qt.getUnqualifiedType()) + " &" + nm;
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
  static bool isGuard(const Stmt *s) {
    // DISABLED. A "guard" (an if without else) split its scope into runs that genScope
    // processes one after the other, each with its own primal and reverse - valid only when the guard wraps the
    // whole rest of a kernel with nothing active before it (`if (tid < n) { ... }`). A mid-function if made the
    // reverse of the statements after it (the `return`'s seed!) run AFTER the reverse of those before it:
    // a device function then added its return seed last and every adjoint through it came out exactly 0. Every if
    // now goes through genBlock (primal with the branch, reverse re-testing it). Grid-stride kernels do not rely on
    // the guard form.
    (void)s;
    return false;
  }
  // A loop whose body declares active locals must recompute its primal inside
  // the backward loop (per-iteration), rather than reproduce it once outside.
  bool loopHasActiveLocals(const Stmt *s) const {
    if (!s)
      return false;
    // ANY local declared in the body forces the per-iteration re-materialisation - an
    // integer index local (`int ip = idx[4 * e + k]`) used by the adjoint statements was otherwise left undeclared
    // in the backward loop
    if (const auto *ds = dyn_cast<DeclStmt>(s))
      for (const Decl *d : ds->decls())
        if (isa<VarDecl>(d))
          return true;
    // a body with an early exit goes through genBlock too (its if/else split)
    if (isa<ContinueStmt>(s))
      return true;
    for (const Stmt *c : s->children())
      if (loopHasActiveLocals(c))
        return true;
    return false;
  }
  static bool isUserDiffFn(const FunctionDecl *fd) {
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

  // ---- tape set: local scalar-floating vars that are reassigned ----
  static const VarDecl *localScalarTarget(const Expr *lhs) {
    const Expr *e = lhs->IgnoreParenImpCasts();
    if (const auto *d = dyn_cast<DeclRefExpr>(e))
      if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
        if (vd->isLocalVarDecl() && vd->getType()->isRealFloatingType())
          return vd;
    return nullptr;
  }
  void collectTaped(const Stmt *s) {
    if (!s)
      return;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const VarDecl *v = localScalarTarget(bo->getLHS()))
          if (Act.isVaried(v))
            taped_.insert(v);
    for (const Stmt *c : s->children())
      collectTaped(c);
  }
  // ---- pure accumulators: v = v + e / v += e, which need no tape ----
  //
  // The tape exists so that the reverse sweep can restore a variable's value before the statement that
  // overwrote it. An accumulation needs neither half of that: d(v_new)/d(v_old) = 1, so the adjoint of
  // ``v = v + e`` is ``e_b += v_b`` with v_b untouched and the value of v never read. What is left is
  // whether some *other* statement's adjoint needs v's pre-accumulation value, and that is what the
  // conditions below rule out - a read outside the accumulators must follow all of them, and must not sit
  // inside a loop that contains one (there the value would have to be undone once per iteration, which is
  // exactly what a tape is for).
  //
  // Worth doing because a reduction loop is where the tape hurts: the tape is a per-thread local array, so
  // a kernel whose thread sums over N elements wants O(N) of it. A Biot-Savart segment-sum kernel, for
  // example, pushed five values per inner iteration, all five of them removable by this rule.
  static bool refersToVar(const Stmt *s, const VarDecl *v) {
    if (!s)
      return false;
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (d->getDecl() == v)
        return true;
    for (const Stmt *c : s->children())
      if (refersToVar(c, v))
        return true;
    return false;
  }
  // The increment of an accumulation of ``v``, or null if ``bo`` is not one. ``minus`` reports v = v - e.
  static const Expr *accumIncrement(const BinaryOperator *bo, const VarDecl *v,
                                    bool &minus) {
    minus = false;
    if (bo->getOpcode() == BO_AddAssign || bo->getOpcode() == BO_SubAssign) {
      if (localScalarTarget(bo->getLHS()) != v)
        return nullptr;
      if (refersToVar(bo->getRHS(), v))
        return nullptr;
      minus = bo->getOpcode() == BO_SubAssign;
      return bo->getRHS();
    }
    if (bo->getOpcode() != BO_Assign)
      return nullptr;
    if (localScalarTarget(bo->getLHS()) != v)
      return nullptr;
    const auto *rhs =
        dyn_cast<BinaryOperator>(bo->getRHS()->IgnoreParenImpCasts());
    if (!rhs)
      return nullptr;
    const bool add = rhs->getOpcode() == BO_Add;
    const bool sub = rhs->getOpcode() == BO_Sub;
    if (!add && !sub)
      return nullptr;
    if (localScalarTarget(rhs->getLHS()) == v && !refersToVar(rhs->getRHS(), v)) {
      minus = sub;               // v = v - e
      return rhs->getRHS();
    }
    if (add && localScalarTarget(rhs->getRHS()) == v &&
        !refersToVar(rhs->getLHS(), v))
      return rhs->getLHS();      // v = e + v
    return nullptr;              // v = e - v is not an accumulation
  }
  bool withinRange(SourceRange R, SourceLocation L) const {
    if (R.isInvalid() || L.isInvalid())
      return true;               // unknown provenance: assume the worst
    return !SM.isBeforeInTranslationUnit(L, R.getBegin()) &&
           !SM.isBeforeInTranslationUnit(R.getEnd(), L);
  }
  void collectAccumRefs(
      const Stmt *s,
      std::map<const VarDecl *, std::vector<const BinaryOperator *>> &asg,
      std::map<const VarDecl *, std::vector<const DeclRefExpr *>> &refs,
      std::vector<const Stmt *> &loops) const {
    if (!s)
      return;
    if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s))
      loops.push_back(s);
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const VarDecl *v = localScalarTarget(bo->getLHS()))
          asg[v].push_back(bo);
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
        refs[vd].push_back(d);
    for (const Stmt *c : s->children())
      collectAccumRefs(c, asg, refs, loops);
  }
  void collectAccum(const Stmt *body) {
    accum_.clear();
    std::map<const VarDecl *, std::vector<const BinaryOperator *>> asg;
    std::map<const VarDecl *, std::vector<const DeclRefExpr *>> refs;
    std::vector<const Stmt *> loops;
    collectAccumRefs(body, asg, refs, loops);
    for (const VarDecl *v : taped_) {
      auto it = asg.find(v);
      if (it == asg.end() || it->second.empty())
        continue;
      std::vector<const BinaryOperator *> acc;
      bool ok = true;
      for (const BinaryOperator *bo : it->second) {
        bool m = false;
        // Two kinds of write need no stored old value. An accumulation, because d(v_new)/d(v_old) = 1; and
        // a plain overwrite whose right-hand side does not mention v, because then d(v_new)/d(v_old) = 0 and
        // the generated adjoint zeroes v_b after distributing it. ``*=`` and ``/=`` are neither: their
        // adjoints multiply by the old value, so they keep the tape.
        const bool plain = bo->getOpcode() == BO_Assign &&
                           !refersToVar(bo->getRHS(), v);
        if (!plain && !accumIncrement(bo, v, m)) {
          ok = false;
          break;
        }
        acc.push_back(bo);
      }
      if (!ok)
        continue;
      for (const DeclRefExpr *r : refs[v]) {
        bool inAcc = false;
        for (const BinaryOperator *bo : acc)
          if (withinRange(bo->getSourceRange(), r->getBeginLoc())) {
            inAcc = true;
            break;
          }
        if (inAcc)
          continue;                      // the accumulator's own v operands
        for (const BinaryOperator *bo : acc)
          if (!SM.isBeforeInTranslationUnit(bo->getEndLoc(), r->getBeginLoc())) {
            ok = false;                  // read before (or within) an accumulation
            break;
          }
        if (!ok)
          break;
        for (const Stmt *L : loops) {
          // A loop that *declares* the accumulator starts it afresh every iteration, so nothing carries
          // across and the loop cannot be the reason a value would have to be undone. Without this the rule
          // rejects the common shape - a per-thread reduction declared and consumed inside an outer loop.
          if (withinRange(L->getSourceRange(), v->getBeginLoc()))
            continue;
          bool hasAcc = false;
          for (const BinaryOperator *bo : acc)
            if (withinRange(L->getSourceRange(), bo->getBeginLoc())) {
              hasAcc = true;
              break;
            }
          if (hasAcc && withinRange(L->getSourceRange(), r->getBeginLoc())) {
            ok = false;                  // read inside the accumulating loop
            break;
          }
        }
        if (!ok)
          break;
      }
      if (ok)
        accum_.insert(v);
    }
    for (const VarDecl *v : accum_)
      taped_.erase(v);                   // no push, no pop, no tape slot
  }

  const VarDecl *tapedAssignTarget(const Stmt *s) const {
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const VarDecl *v = localScalarTarget(bo->getLHS()))
          if (taped_.count(v))
            return v;
    return nullptr;
  }

  // ---- reused output arrays (e.g. faceI(...,S,C) called for several faces) ----
  void collectReused(const Stmt *s, std::map<const VarDecl *, int> &cnt) const {
    if (!s)
      return;
    if (const auto *ce = dyn_cast<CallExpr>(s)) {
      const FunctionDecl *fd = calleeOf(ce);
      if (isUserDiffFn(fd))
        for (unsigned i = 0; i < ce->getNumArgs(); ++i) {
          const ParmVarDecl *p =
              i < fd->getNumParams() ? fd->getParamDecl(i) : nullptr;
          if (!p || !isActiveType(p->getType()))
            continue;
          const Expr *arg = ce->getArg(i)->IgnoreParenImpCasts();
          if (const auto *d = dyn_cast<DeclRefExpr>(arg))
            if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
              if (vd->isLocalVarDecl() &&
                  Ctx.getAsConstantArrayType(vd->getType()))
                cnt[vd]++;
        }
    }
    for (const Stmt *c : s->children())
      collectReused(c, cnt);
  }
  // (array, element-count) pairs of a call's reused active output-array args
  std::vector<std::pair<const VarDecl *, uint64_t>>
  reusedOutArgs(const CallExpr *ce) const {
    std::vector<std::pair<const VarDecl *, uint64_t>> v;
    const FunctionDecl *fd = calleeOf(ce);
    if (!isUserDiffFn(fd))
      return v;
    for (unsigned i = 0; i < ce->getNumArgs(); ++i) {
      const ParmVarDecl *p =
          i < fd->getNumParams() ? fd->getParamDecl(i) : nullptr;
      if (!p || !isActiveType(p->getType()))
        continue;
      const Expr *arg = ce->getArg(i)->IgnoreParenImpCasts();
      if (const auto *d = dyn_cast<DeclRefExpr>(arg))
        if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
          if (reused_.count(vd))
            if (const auto *cat = Ctx.getAsConstantArrayType(vd->getType()))
              v.push_back({vd, cat->getSize().getZExtValue()});
    }
    return v;
  }
  // save reused output arrays before an overwriting call (forward sweep)
  std::string tapePushCall(const CallExpr *ce, const std::string &ind) const {
    std::string out;
    for (auto &ae : reusedOutArgs(ce))
      for (uint64_t e = 0; e < ae.second; ++e)
        out += ind + "cuad_push(" + ae.first->getNameAsString() + "[" +
               std::to_string(e) + "]);\n";
    return out;
  }
  // restore them in reverse (LIFO) after the reversed call
  std::string tapePopCall(const CallExpr *ce, const std::string &ind) const {
    std::string out;
    auto v = reusedOutArgs(ce);
    for (auto it = v.rbegin(); it != v.rend(); ++it)
      for (uint64_t e = it->second; e-- > 0;)
        out += ind + "cuad_pop(" + it->first->getNameAsString() + "[" +
               std::to_string(e) + "]);\n";
    return out;
  }

  // adjoint storage location of an active lvalue (or "" if inactive)
  std::string adjLoc(const Expr *e) const {
    e = e->IgnoreParenImpCasts();
    if (const auto *dre = dyn_cast<DeclRefExpr>(e))
      return Act.isVaried(dre->getDecl()) ? declName(dre) + "_b" : "";
    if (const auto *ase = dyn_cast<ArraySubscriptExpr>(e)) {
      const Expr *b = ase->getBase()->IgnoreParenImpCasts();
      if (const auto *d = dyn_cast<DeclRefExpr>(b))
        return Act.isVaried(d->getDecl())
                   ? declName(d) + "_b[" + prim(ase->getIdx()) + "]"
                   : "";
      // g[m][0], TT[d][l] - nested subscripts
      if (isa<ArraySubscriptExpr>(b)) {
        std::string bl = adjLoc(b);
        return bl.empty() ? "" : bl + "[" + prim(ase->getIdx()) + "]";
      }
    }
    if (const auto *uo = dyn_cast<UnaryOperator>(e))
      if (uo->getOpcode() == UO_Deref) {
        const Expr *s = uo->getSubExpr()->IgnoreParenImpCasts();
        if (const auto *d = dyn_cast<DeclRefExpr>(s))
          return Act.isVaried(d->getDecl()) ? "(*" + declName(d) + "_b)" : "";
      }
    if (const auto *me = dyn_cast<MemberExpr>(e)) {
      const Expr *base = me->getBase()->IgnoreParenImpCasts();
      if (const auto *d = dyn_cast<DeclRefExpr>(base))
        if (Act.isVaried(d->getDecl()))
          return declName(d) + (me->isArrow() ? "_b->" : "_b.") +
                 me->getMemberDecl()->getNameAsString();
    }
    return "";
  }

  // A shared adjoint (function-parameter storage) can be written by many
  // threads at once, so accumulate atomically; per-thread locals use plain +=.
  bool isSharedAdjoint(const Expr *e) const {
    e = e->IgnoreParenImpCasts();
    const ValueDecl *d = nullptr;
    if (const auto *dre = dyn_cast<DeclRefExpr>(e))
      d = dre->getDecl();
    else if (const auto *ase = dyn_cast<ArraySubscriptExpr>(e)) {
      const Expr *bb = ase->getBase()->IgnoreParenImpCasts();
      while (const auto *inner = dyn_cast<ArraySubscriptExpr>(bb))   // g[m][0], TT[d][l]: the root array
        bb = inner->getBase()->IgnoreParenImpCasts();
      if (const auto *b = dyn_cast<DeclRefExpr>(bb))
        d = b->getDecl();
    } else if (const auto *uo = dyn_cast<UnaryOperator>(e)) {
      if (uo->getOpcode() == UO_Deref)
        if (const auto *b =
                dyn_cast<DeclRefExpr>(uo->getSubExpr()->IgnoreParenImpCasts()))
          d = b->getDecl();
    } else if (const auto *me = dyn_cast<MemberExpr>(e)) {
      if (const auto *b = dyn_cast<DeclRefExpr>(me->getBase()->IgnoreParenImpCasts()))
        d = b->getDecl();
    }
    // a SCALAR parameter's adjoint is the caller's private variable (a __device__ helper
    // gets `float &x_b` from a thread-local; genReverseCall never passes shared storage for it), so it accumulates
    // with += - atomicAdd on a thread-local address faults on the device. Pointer / array parameters stay shared.
    if (d && isa<ParmVarDecl>(d) && isa<DeclRefExpr>(e) &&
        !d->getType().getCanonicalType()->isPointerType())
      return false;
    // a parameter DECLARED as an array (`float f[6]`, `float u[3]`) is, by convention, the caller's
    // thread-local array: plain += (atomicAdd on local memory faults). Pointer parameters (`const float *q`) stay
    // shared.
    if (const auto *pv = dyn_cast_or_null<ParmVarDecl>(d))
      if (pv->getOriginalType()->isArrayType())
        return false;
    // a local POINTER (`const float *T = P[d]`, `const float *Q = q + v * n`) aliases a shared array
    if (d && isa<VarDecl>(d) && !isa<ParmVarDecl>(d) && isFloatPtrType(d->getType()))
      return true;
    return d && isa<ParmVarDecl>(d);
  }
  static std::string addrOf(const std::string &loc) {
    if (loc.size() >= 3 && loc[0] == '(' && loc[1] == '*') // "(*p_b)" -> "p_b"
      return loc.substr(2, loc.size() - 3);
    return "&" + loc;
  }
  // Emit one adjoint accumulation: atomic for shared storage, += for private.
  void emitAccum(const Expr *e, const std::string &loc, const std::string &seed,
                 std::string &out, const std::string &ind) const {
    if (loc.empty())
      return;
    if (isSharedAdjoint(e))
      out += ind + "atomicAdd(" + addrOf(loc) + ", " + seed + ");\n";
    else
      out += ind + loc + " += " + seed + ";\n";
  }

  // ---------------- adjoint of an expression ----------------
  void pushAdj(const Expr *e, const std::string &seed, std::string &out,
               const std::string &ind) const {
    if (seed == "0")
      return;
    if (const auto *c = dyn_cast<CastExpr>(e))
      return pushAdj(c->getSubExpr(), seed, out, ind);
    if (const auto *p = dyn_cast<ParenExpr>(e))
      return pushAdj(p->getSubExpr(), seed, out, ind);
    if (isa<FloatingLiteral>(e) || isa<IntegerLiteral>(e))
      return;
    if (!isFloat(e))
      return;
    if (isa<DeclRefExpr>(e) || isa<ArraySubscriptExpr>(e) || isa<MemberExpr>(e))
      return emitAccum(e, adjLoc(e), seed, out, ind);
    if (const auto *uo = dyn_cast<UnaryOperator>(e)) {
      switch (uo->getOpcode()) {
      case UO_Minus:
        return pushAdj(uo->getSubExpr(), "-(" + seed + ")", out, ind);
      case UO_Plus:
        return pushAdj(uo->getSubExpr(), seed, out, ind);
      case UO_Deref:
        return emitAccum(e, adjLoc(e), seed, out, ind);
      default:
        return;
      }
    }
    if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
      const Expr *L = bo->getLHS(), *R = bo->getRHS();
      switch (bo->getOpcode()) {
      case BO_Add:
        pushAdj(L, seed, out, ind);
        pushAdj(R, seed, out, ind);
        return;
      case BO_Sub:
        pushAdj(L, seed, out, ind);
        pushAdj(R, "-(" + seed + ")", out, ind);
        return;
      case BO_Mul:
        pushAdj(L, mulT(seed, prim(R)), out, ind);
        pushAdj(R, mulT(prim(L), seed), out, ind);
        return;
      case BO_Div:
        pushAdj(L, "(" + seed + ")/(" + prim(R) + ")", out, ind);
        pushAdj(R,
                "-(" + mulT(seed, prim(L)) + ")/((" + prim(R) + ")*(" +
                    prim(R) + "))",
                out, ind);
        return;
      default:
        return;
      }
    }
    // c ? a : b - the seed goes to the branch the primal took (c re-tested on the primal
    // values, as an if statement's is); previously dropped silently
    if (const auto *co = dyn_cast<ConditionalOperator>(e)) {
      std::string c = prim(co->getCond());
      pushAdj(co->getTrueExpr(), "((" + c + ") ? (" + seed + ") : 0)", out, ind);
      pushAdj(co->getFalseExpr(), "((" + c + ") ? 0 : (" + seed + "))", out, ind);
      return;
    }
    if (const auto *call = dyn_cast<CallExpr>(e))
      return pushAdjCall(call, seed, out, ind);
  }

  void pushAdjCall(const CallExpr *call, const std::string &seed,
                   std::string &out, const std::string &ind) const {
    const FunctionDecl *fd = calleeOf(call);
    std::string fn = fd ? fd->getNameAsString() : "";
    unsigned n = call->getNumArgs();
    // a value-returning user function inside a larger expression (`mu(T1) + mu(T2)`): its
    // companion takes the seed as the return adjoint (before, only `v = f(...)` statements were reversed)
    if (isUserDiffFn(fd) && fd->getReturnType()->isRealFloatingType()) {
      out += genReverseCall(call, "(" + seed + ")", ind);
      return;
    }
    static const char *kF[] = {"sinf", "cosf",  "tanf",  "expf",  "logf",
                               "sqrtf", "tanhf", "fabsf", "powf",  "fminf",
                               "fmaxf", "atan2f", "atanf", "asinf", "acosf", "copysignf", "hypotf",
                               "cbrtf"};
    bool isF = false;           // a float function: its derivative stays in float
    for (const char *g : kF)
      if (fn == g) {
        fn.pop_back();
        isF = true;
        break;
      }
    auto F = [&](const std::string &f) { return isF ? f + "f" : f; };        // sqrt -> sqrtf
    auto L = [&](const std::string &x) { return isF ? x + "f" : x; };        // 1.0 -> 1.0f
    if (n == 0)
      return;
    // floor / ceil / round are piecewise constant - zero derivative almost everywhere (a table's angle
    // wrapping, an index)
    if (fn == "floorf" || fn == "ceilf" || fn == "roundf" || fn == "floor" || fn == "ceil" || fn == "round" ||
        fn == "truncf" || fn == "trunc")
      return;
    const Expr *a0 = call->getArg(0);
    std::string pa = prim(a0);
    // atan2(y, x): d/dy = x/(x^2+y^2), d/dx = -y/(x^2+y^2)
    if (n == 2 && fn == "atan2") {
      const Expr *a1 = call->getArg(1);
      std::string pb = prim(a1);
      std::string den = "((" + pa + ")*(" + pa + ")+(" + pb + ")*(" + pb + "))";
      pushAdj(a0, mulT("(" + pb + ")/" + den, seed), out, ind);
      pushAdj(a1, mulT("-(" + pa + ")/" + den, seed), out, ind);
      return;
    }
    // copysign(a, b) = |a| sgn(b): d/da = sgn(a) sgn(b), d/db = 0 (a switch
    // 0.5 + copysign(0.5, x) is then piecewise constant, as it should be)
    // hypot(a, b) (overflow-safe sqrt(a^2 + b^2)): d/da = a / hypot, d/db = b / hypot
    if (n == 2 && fn == "hypot") {
      std::string pb = prim(call->getArg(1));
      std::string h = F("hypot") + "(" + pa + ", " + pb + ")";
      pushAdj(a0, mulT("(" + pa + ")/" + h, seed), out, ind);
      pushAdj(call->getArg(1), mulT("(" + pb + ")/" + h, seed), out, ind);
      return;
    }
    if (n == 2 && fn == "copysign") {
      pushAdj(a0, mulT("(((" + pa + ") >= 0) == ((" + prim(call->getArg(1)) + ") >= 0) ? " + L("1.0") + " : -" +
                           L("1.0") + ")", seed), out, ind);
      return;
    }
    if (n == 2 && fn == "pow") {
      const Expr *a1 = call->getArg(1);
      std::string pb = prim(a1);
      pushAdj(a0, mulT("(" + pb + ")*" + F("pow") + "(" + pa + ",(" + pb + ")-" + L("1.0") + ")", seed), out,
              ind);
      pushAdj(a1, mulT(F("pow") + "(" + pa + "," + pb + ")*" + F("log") + "(" + pa + ")", seed), out,
              ind);
      return;
    }
    if (n == 2 && (fn == "fmax" || fn == "max" || fn == "fmin" || fn == "min")) {
      const Expr *a1 = call->getArg(1);
      std::string cmp = (fn == "fmax" || fn == "max") ? " >= " : " <= ";
      std::string cnd = "(" + pa + ")" + cmp + "(" + prim(a1) + ")";
      pushAdj(a0, "(" + cnd + " ? (" + seed + ") : 0)", out, ind);
      pushAdj(a1, "(" + cnd + " ? 0 : (" + seed + "))", out, ind);
      return;
    }
    std::string m;
    std::string one = L("1.0");
    if (fn == "sin")
      m = F("cos") + "(" + pa + ")";
    else if (fn == "cos")
      m = "-" + F("sin") + "(" + pa + ")";
    else if (fn == "tan")
      m = one + "/(" + F("cos") + "(" + pa + ")*" + F("cos") + "(" + pa + "))";
    else if (fn == "exp")
      m = F("exp") + "(" + pa + ")";
    else if (fn == "log")
      m = one + "/(" + pa + ")";
    else if (fn == "sqrt")
      m = one + "/(" + L("2.0") + "*" + F("sqrt") + "(" + pa + "))";
    else if (fn == "cbrt")
      m = one + "/(" + L("3.0") + "*" + F("cbrt") + "(" + pa + ")*" + F("cbrt") + "(" + pa + "))";
    else if (fn == "tanh")
      m = "(" + one + " - " + F("tanh") + "(" + pa + ")*" + F("tanh") + "(" + pa + "))";
    else if (fn == "fabs" || fn == "abs")
      m = "((" + pa + ") >= 0 ? " + one + " : -" + one + ")";
    else if (fn == "atan")
      m = one + "/(" + one + "+(" + pa + ")*(" + pa + "))";
    else if (fn == "asin")
      m = one + "/" + F("sqrt") + "(" + one + "-(" + pa + ")*(" + pa + "))";
    else if (fn == "acos")
      m = "-" + one + "/" + F("sqrt") + "(" + one + "-(" + pa + ")*(" + pa + "))";
    else {
      unsupported(call->getBeginLoc(), "a call to `" + fn + "` (no derivative rule)");
      out += ind + "// TODO(cuad): adjoint of call '" + fn + "' unhandled\n";
      return;
    }
    pushAdj(a0, mulT(m, seed), out, ind);
  }

  // ---- cross-call: call the callee's _b companion ----
  // Active arguments that are simple lvalues pass their adjoint location
  // directly; computed active arguments get a temporary adjoint that is
  // distributed back into the argument expression after the call.
  std::string genReverseCall(const CallExpr *call, const std::string &retSeed,
                             const std::string &ind) const {
    const FunctionDecl *fd = calleeOf(call);
    std::string pre, args, postSeeds;
    std::vector<std::pair<const Expr *, std::string>> post;
    bool first = true;
    for (unsigned i = 0; i < call->getNumArgs(); ++i) {
      const Expr *arg = call->getArg(i);
      const ParmVarDecl *p =
          i < fd->getNumParams() ? fd->getParamDecl(i) : nullptr;
      if (!first)
        args += ", ";
      first = false;
      args += prim(arg);
      if (!p || !isActiveType(p->getType()))
        continue;
      std::string loc = adjLoc(arg);
      const Expr *ae = arg->IgnoreParenImpCasts();
      bool scalarP = !p->getType().getCanonicalType()->isPointerType();
      if (!loc.empty() && scalarP && isSharedAdjoint(arg)) {
        // a scalar parameter bound to shared storage (an array element): the callee accumulates into a
        // private temporary (plain +=), distributed atomically after the call
        std::string t = "_cuad_a" + std::to_string(tmp_++);
        pre += ind + tstr(p->getType().getNonReferenceType().getUnqualifiedType()) + " " + t + " = 0;\n";
        args += ", " + t;
        post.push_back({arg, t});
      } else if (!loc.empty()) {
        args += ", " + loc;
      } else if (p->getType().getCanonicalType()->isPointerType()) {
        // a pointer expression (q + v * n, P[d], c ? a : b) gets the same expression
        // over the adjoint pointers
        args += ", " + adjPtrExpr(ae);
      } else { // computed scalar: temp adjoint, distribute after the call
        std::string t = "_cuad_a" + std::to_string(tmp_++);
        pre += ind + tstr(p->getType().getNonReferenceType().getUnqualifiedType()) + " " + t + " = 0;\n";
        args += ", " + t;
        post.push_back({arg, t});
      }
    }
    if (!retSeed.empty()) {
      if (!first)
        args += ", ";
      args += retSeed;
    }
    std::string out = pre + ind + fd->getNameAsString() + "_b" + calleeTArgs(call) + "(" + args + ");\n";
    for (auto &pr : post)
      pushAdj(pr.first, pr.second, out, ind);
    return out;
  }

  // ---------------- pointers, hoisting, early exits ----------------
  static const Stmt *unwrapAttr(const Stmt *s) {
    while (const auto *as = dyn_cast_or_null<AttributedStmt>(s))
      s = as->getSubStmt();           // `#pragma unroll` for (...): the loop itself
    return s;
  }
  static bool isFloatPtrType(QualType qt) {
    qt = qt.getCanonicalType();
    if (qt->isArrayType())
      qt = qt->getAsArrayTypeUnsafe()->getElementType().getCanonicalType();
    return qt->isPointerType() && qt->getPointeeType()->isRealFloatingType();
  }
  // the adjoint pointer type of a local float pointer (or array of them): `float *` (const dropped) + dims
  std::string adjPtrDecl(const VarDecl *vd) const {
    QualType qt = vd->getType().getCanonicalType();
    std::string dims;
    QualType elem = qt;
    if (qt->isArrayType())
      dims = arrayDims(qt, elem);
    QualType pt = elem->getPointeeType().getUnqualifiedType();
    return tstr(pt) + " *" + vd->getNameAsString() + "_b" + dims;
  }
  // the same pointer expression over the adjoint pointers: p -> p_b, c ? a : b -> c ? a_b : b_b, base + off ->
  // base_b + off, TT[d] -> TT_b[d], {a, b} -> {a_b, b_b}; a passive pointer has no adjoint (nullptr)
  std::string adjPtrExpr(const Expr *e) const {
    if (!e)
      return "nullptr";
    if (const auto *pe = dyn_cast<ParenExpr>(e))
      return "(" + adjPtrExpr(pe->getSubExpr()) + ")";
    if (const auto *ce = dyn_cast<CastExpr>(e))
      return adjPtrExpr(ce->getSubExpr());
    if (const auto *d = dyn_cast<DeclRefExpr>(e))
      return Act.isVaried(d->getDecl()) ? declName(d) + "_b" : "nullptr";
    if (const auto *co = dyn_cast<ConditionalOperator>(e))
      return "((" + prim(co->getCond()) + ") ? " + adjPtrExpr(co->getTrueExpr()) + " : " +
             adjPtrExpr(co->getFalseExpr()) + ")";
    if (const auto *as = dyn_cast<ArraySubscriptExpr>(e)) {
      std::string b = adjPtrExpr(as->getBase());
      return b.rfind("nullptr", 0) == 0 ? "nullptr" : b + "[" + prim(as->getIdx()) + "]";
    }
    if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
      if (bo->getOpcode() == BO_Add || bo->getOpcode() == BO_Sub) {
        bool lp = bo->getLHS()->getType()->isPointerType();
        std::string op = bo->getOpcode() == BO_Add ? " + " : " - ";
        if (lp)
          return adjPtrExpr(bo->getLHS()) + op + "(" + prim(bo->getRHS()) + ")";
        return "(" + prim(bo->getLHS()) + ")" + op + adjPtrExpr(bo->getRHS());
      }
    }
    if (const auto *il = dyn_cast<InitListExpr>(e)) {
      std::string s = "{";
      for (unsigned i = 0; i < il->getNumInits(); ++i)
        s += (i ? ", " : "") + adjPtrExpr(il->getInit(i));
      return s + "}";
    }
    if (isa<IntegerLiteral>(e) || isa<CXXNullPtrLiteralExpr>(e) || isa<GNUNullExpr>(e))
      return "nullptr";
    unsupported(e->getBeginLoc(), "the adjoint of the pointer expression `" + prim(e) + "`");
    return "nullptr /* TODO(cuad): pointer adjoint of '" + prim(e) + "' */";
  }
  // an `if (c) { ...; continue; }` / `if (c) { ...; return x; }` / bare `if (c) continue;` without else: the rest of
  // the block is its else (so the reverse of the rest runs only when the primal did not leave)
  static bool exitStmt(const Stmt *s) { return isa<ContinueStmt>(s) || isa<ReturnStmt>(s); }
  static bool isEarlyExit(const Stmt *s) {
    const auto *is = dyn_cast<IfStmt>(s);
    if (!is || is->getElse())
      return false;
    const Stmt *t = is->getThen();
    if (exitStmt(t))
      return true;
    if (const auto *cs = dyn_cast<CompoundStmt>(t))
      return cs->size() > 0 && exitStmt(cs->body_back());
    return false;
  }
  // the declarations inside the if-branches of these statements (not inside loops, which are scopes of their own
  // in the reverse): hoisted to the block, their adjoints declared with them
  void collectBranchDecls(const Stmt *s, bool inBranch, std::vector<const VarDecl *> &out) const {
    s = unwrapAttr(s);
    if (!s || isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s))
      return;
    if (const auto *ds = dyn_cast<DeclStmt>(s)) {
      if (inBranch)
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            out.push_back(vd);
      return;
    }
    if (const auto *is = dyn_cast<IfStmt>(s)) {
      collectBranchDecls(is->getThen(), true, out);
      if (is->getElse())
        collectBranchDecls(is->getElse(), true, out);
      return;
    }
    if (const auto *cs = dyn_cast<CompoundStmt>(s))
      for (const Stmt *c : cs->body())
        collectBranchDecls(c, inBranch, out);
  }
  std::string hoistDecls(const std::vector<const Stmt *> &stmts, const std::string &ind) const {
    std::vector<const VarDecl *> vds;
    for (const Stmt *s : stmts)
      collectBranchDecls(s, false, vds);
    std::string out;
    std::set<std::string> seen;
    for (const VarDecl *vd : vds) {
      hoisted_.insert(vd);
      std::string nm = vd->getNameAsString();
      QualType qt = vd->getType();
      std::string key = nm + "|" + tstr(qt.getCanonicalType());
      if (seen.count(key))
        continue;
      seen.insert(key);
      QualType elem;
      std::string dims = Ctx.getAsConstantArrayType(qt) ? arrayDims(qt, elem) : "";
      if (dims.empty())
        elem = qt;
      std::string base = tstr(elem.getUnqualifiedType());
      out += ind + base + " " + nm + dims + ";\n";
      if (Act.isVaried(vd)) {
        if (isFloatPtrType(qt))
          out += ind + adjPtrDecl(vd) + ";\n";
        else if (elem->isRealFloatingType())
          out += ind + base + " " + nm + "_b" + dims + (dims.empty() ? " = 0" : " = {0}") + ";\n";
      }
    }
    return out;
  }
  // a hoisted declaration's primal: assignments of its initialisers (+ the adjoint pointer's)
  std::string hoistedPrimal(const DeclStmt *ds, const std::string &ind) const {
    std::string out;
    for (const Decl *d : ds->decls())
      if (const auto *vd = dyn_cast<VarDecl>(d)) {
        if (vd->hasInit() && !isa<InitListExpr>(vd->getInit()->IgnoreImplicit()))
          out += ind + vd->getNameAsString() + " = " + prim(vd->getInit()) + ";\n";
        else if (vd->hasInit()) {
          unsupported(vd->getBeginLoc(), "the initialiser list of the branch-local array `" +
                                             vd->getNameAsString() + "`");
          out += ind + "// TODO(cuad): hoisted array initialiser of " + vd->getNameAsString() + "\n";
        }
        if (vd->hasInit() && Act.isVaried(vd) && isFloatPtrType(vd->getType()))
          out += ind + vd->getNameAsString() + "_b = " + adjPtrExpr(vd->getInit()) + ";\n";
      }
    return out;
  }

  // a bare `{ ... }` block is a scope only: its statements are spliced into the enclosing list (before, the whole block
  // was copied as one primal line and had no reverse at all - everything inside it was dropped)
  static void flattenInto(const Stmt *s, std::vector<const Stmt *> &out) {
    const Stmt *u = unwrapAttr(s);
    if (const auto *cs = dyn_cast_or_null<CompoundStmt>(u)) {
      for (const Stmt *c : cs->body())
        flattenInto(c, out);
      return;
    }
    out.push_back(s);
  }
  // the local 1-D float arrays a loop updates in place (`ql[m] += f(ql[m])`, `a[i] = a[i] * c`): a recompute loop's
  // reverse would otherwise re-run each iteration from the arrays' FINAL values. They are saved before the loop
  // (forward) and restored before and after its reverse - valid when an iteration updates only its own elements.
  void selfUpdated(const Stmt *s, std::set<const VarDecl *> &out) const {
    if (!s)
      return;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const auto *as = dyn_cast<ArraySubscriptExpr>(bo->getLHS()->IgnoreParenImpCasts()))
          if (const auto *d = dyn_cast<DeclRefExpr>(as->getBase()->IgnoreParenImpCasts()))
            if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
              if (vd->isLocalVarDecl() && !isa<ParmVarDecl>(vd))
                if (const auto *cat = Ctx.getAsConstantArrayType(vd->getType()))
                  if (cat->getElementType()->isRealFloatingType() &&
                      (bo->getOpcode() != BO_Assign || refersToVar(bo->getRHS(), vd)))
                    out.insert(vd);
    for (const Stmt *c : s->children())
      selfUpdated(c, out);
  }
  std::string shadowName(const ForStmt *fs, const VarDecl *vd) const {
    auto it = shadowId_.find(fs);
    int id = it == shadowId_.end() ? (shadowId_[fs] = (int)shadowId_.size()) : it->second;
    return vd->getNameAsString() + "_cuad_sv" + std::to_string(id);
  }
  std::string shadowCopy(const ForStmt *fs, bool save, const std::string &ind) const {
    std::set<const VarDecl *> arrs;
    selfUpdated(fs->getBody(), arrs);
    std::string out;
    for (const VarDecl *vd : arrs) {
      if (withinRange(fs->getSourceRange(), vd->getLocation()))
        continue;                       // declared inside the loop: fresh every iteration, nothing to save
      const auto *cat = Ctx.getAsConstantArrayType(vd->getType());
      std::string N = std::to_string(cat->getSize().getZExtValue()), a = vd->getNameAsString(),
                  sv = shadowName(fs, vd);
      if (save)
        out += ind + tstr(cat->getElementType().getUnqualifiedType()) + " " + sv + "[" + N + "];\n";
      out += ind + "for (int _cuad_k = 0; _cuad_k < " + N + "; ++_cuad_k) " +
             (save ? sv + "[_cuad_k] = " + a : a + "[_cuad_k] = " + sv) + "[_cuad_k];\n";
    }
    return out;
  }

  // ---------------- forward primal sweep (with PUSHes) ----------------
  std::string primalLine(const Stmt *s, const std::string &ind) const {
    std::string t = src(s->getSourceRange());
    if (!t.empty() && t.back() == ';')
      t.pop_back();
    return ind + t + ";\n";
  }
  std::string genPrimal(const Stmt *s, int depth) const {
    std::string ind(depth * 2, ' ');
    s = unwrapAttr(s);
    // value-returning: the return value isn't stored; its inputs are already
    // computed, so drop the return in the forward sweep.
    if (isa<ReturnStmt>(s))
      return "";
    if (const auto *ds = dyn_cast<DeclStmt>(s)) {
      bool hoisted = false;
      for (const Decl *d : ds->decls())
        if (const auto *vd = dyn_cast<VarDecl>(d))
          if (hoisted_.count(vd))
            hoisted = true;
      if (hoisted)
        return hoistedPrimal(ds, ind);
      std::string out = primalLine(s, ind);
      for (const Decl *d : ds->decls())
        if (const auto *vd = dyn_cast<VarDecl>(d))
          if (vd->hasInit() && Act.isVaried(vd) && isFloatPtrType(vd->getType()))
            out += ind + adjPtrDecl(vd) + " = " + adjPtrExpr(vd->getInit()) + ";\n";
      return out;
    }
    if (const VarDecl *v = tapedAssignTarget(s)) {
      if (noPush_)
        return primalLine(s, ind);
      return ind + "cuad_push(" + v->getNameAsString() + ");\n" +
             primalLine(s, ind);
    }
    // save reused output arrays before this call overwrites them
    if (const CallExpr *ce = asCall(s))
      if (!noPush_ && !reusedOutArgs(ce).empty())
        return tapePushCall(ce, ind) + primalLine(s, ind);
    if (const auto *fs = dyn_cast<ForStmt>(s)) {
      std::string init =
          fs->getInit() ? src(fs->getInit()->getSourceRange()) : "";
      if (!init.empty() && init.back() == ';')
        init.pop_back();
      std::string cond = fs->getCond() ? prim(fs->getCond()) : "";
      std::string inc = fs->getInc() ? prim(fs->getInc()) : "";
      std::string out = shadowCopy(fs, true, ind) + ind + "for (" + init + "; " + cond + "; " + inc + ") {\n";
      // a recompute loop re-materialises its primal, with its own pushes, inside the
      // backward loop - its forward pass must not push, NESTED ones included (genBlock only caught top-level ones).
      // Pushes left behind by a loop inside an `if` misaligned every pop of a value pushed before the loop: in
      // one kernel a running maximum popped an earlier loop's leftovers, and the
      // derivative went to the wrong element (invisible while nothing depended on that maximum).
      bool saved = noPush_;
      if (loopHasActiveLocals(fs->getBody()))
        noPush_ = true;
      out += genPrimalBody(fs->getBody(), depth + 1);
      noPush_ = saved;
      out += ind + "}\n";
      return out;
    }
    if (const auto *is = dyn_cast<IfStmt>(s)) {
      std::string out = ind + "if (" + prim(is->getCond()) + ") {\n";
      out += genPrimalBody(is->getThen(), depth + 1);
      out += ind + "}";
      if (is->getElse())
        out += " else {\n" + genPrimalBody(is->getElse(), depth + 1) + ind + "}";
      return out + "\n";
    }
    return primalLine(s, ind);
  }
  std::string genPrimalBody(const Stmt *body, int depth) const {
    std::string out;
    if (const auto *cs = dyn_cast<CompoundStmt>(body))
      for (const Stmt *c : cs->body())
        out += genPrimal(c, depth);
    else
      out += genPrimal(body, depth);
    return out;
  }

  // ---------------- reversed loop header ----------------
  // Reverse a canonical counted loop for (int i=A; i<B or i<=B; i++/++i).
  bool reverseForHeader(const ForStmt *fs, std::string &out) const {
    const auto *ds = dyn_cast_or_null<DeclStmt>(fs->getInit());
    if (!ds || !ds->isSingleDecl())
      return false;
    const auto *vd = dyn_cast<VarDecl>(ds->getSingleDecl());
    if (!vd || !vd->hasInit())
      return false;
    const auto *cond =
        fs->getCond() ? dyn_cast<BinaryOperator>(fs->getCond()->IgnoreImpCasts())
                      : nullptr;
    if (!cond)
      return false;
    std::string i = vd->getNameAsString();
    // A cast to the loop variable's type - `blockIdx.x * blockDim.x + threadIdx.x` is
    // UNSIGNED, and a tail thread's start value A - S (negative) compared with it as unsigned passed `i >= A` and ran
    // one iteration at a negative index (caught by compute-sanitizer as a read at index -32)
    std::string A = "(" + tstr(vd->getType()) + ")(" + prim(vd->getInit()) + ")";
    std::string B = prim(cond->getRHS());
    std::string start;
    if (cond->getOpcode() == BO_LT)
      start = "(" + B + ") - 1";
    else if (cond->getOpcode() == BO_LE)
      start = "(" + B + ")";
    else
      return false;
    // The **stride**. ``for (i = A; i < B; i += S)`` - a CUDA grid-stride loop is exactly this, with
    // ``A = blockIdx.x * blockDim.x + threadIdx.x`` and ``S = gridDim.x * blockDim.x`` - does not reverse as
    // ``i--``. That form makes every thread walk the whole range, so each element is processed once per
    // thread and every atomicAdd lands many times over. It is invisible in a single-threaded host
    // verification (there S is 1) and wrong on any real launch.
    std::string S = "1";
    if (const Expr *ie = fs->getInc()) {
      const Expr *e = ie->IgnoreParenImpCasts();
      if (const auto *u = dyn_cast<UnaryOperator>(e)) {
        if (!u->isIncrementOp())
          return false;
      } else if (const auto *b = dyn_cast<BinaryOperator>(e)) {
        if (b->getOpcode() != BO_AddAssign)
          return false;
        S = prim(b->getRHS());
      } else {
        return false;
      }
    }
    std::string T = tstr(vd->getType());
    if (S == "1") {
      out = T + " " + i + " = " + start + "; " + i + " >= (" + A + "); " + i + "--";
      return true;
    }
    // Start from the last value this thread actually visited, and step back by the stride. The guard is for
    // the case where the forward loop ran zero times (the tail threads of the last block): there
    // ``start < A`` and integer division truncates toward zero, which would otherwise give one spurious
    // iteration at ``i = A``.
    const std::string last = "((" + start + ") < (" + A + ") ? (" + A + ") - (" + S + ") : (" + A +
                             ") + (((" + start + ") - (" + A + ")) / (" + S + ")) * (" + S + "))";
    out = T + " " + i + " = " + last + "; " + i + " >= (" + A + "); " + i + " -= (" + S + ")";
    return true;
  }

  // ---------------- reversed adjoint sweep ----------------
  std::string reverseStmt(const Stmt *s, int depth) const {
    std::string ind(depth * 2, ' ');
    std::string out;
    s = unwrapAttr(s);

    if (const auto *rs = dyn_cast<ReturnStmt>(s)) {
      if (rs->getRetValue() && !curRetSeed_.empty())
        pushAdj(rs->getRetValue(), curRetSeed_, out, ind);
      return out;
    }

    if (const auto *ds = dyn_cast<DeclStmt>(s)) {
      // double v = f(u):  reverse via the callee's _b, seeding v_b.
      if (ds->isSingleDecl())
        if (const auto *vd = dyn_cast<VarDecl>(ds->getSingleDecl()))
          if (Act.isVaried(vd) && vd->hasInit() &&
              vd->getType()->isRealFloatingType())
            if (const CallExpr *call = asCall(vd->getInit()))
              if (isUserDiffFn(calleeOf(call)))
                return genReverseCall(call, vd->getNameAsString() + "_b", ind);
      // the declarators of one statement in REVERSE order - `float drho = f(rr, rl),
      // coef = g(drho)` must give coef's adjoint to drho before drho's is distributed (forward order lost it: errors of
      // up to 20 % in the affected derivatives)
      std::vector<const VarDecl *> vds;
      for (const Decl *d : ds->decls())
        if (const auto *vd = dyn_cast<VarDecl>(d))
          vds.push_back(vd);
      for (auto it = vds.rbegin(); it != vds.rend(); ++it) {
        const VarDecl *vd = *it;
        if (Act.isVaried(vd) && vd->hasInit() && vd->getType()->isRealFloatingType()) {
          if (const CallExpr *call = asCall(vd->getInit()))
            if (isUserDiffFn(calleeOf(call))) {
              out += genReverseCall(call, vd->getNameAsString() + "_b", ind);
              continue;
            }
          pushAdj(vd->getInit(), vd->getNameAsString() + "_b", out, ind);
        }
      }
      return out;
    }

    // atomicAdd(&loc, v): primal loc += v; adjoint distributes loc's adjoint.
    if (const CallExpr *call = asCall(s))
      if (isNamed(calleeOf(call), "atomicAdd") &&
          call->getNumArgs() == 2) {
          const Expr *a0 = call->getArg(0)->IgnoreParenImpCasts();
          const Expr *lval = nullptr;
          if (const auto *uo = dyn_cast<UnaryOperator>(a0))
            if (uo->getOpcode() == UO_AddrOf)
              lval = uo->getSubExpr();
          if (lval) {
            std::string loc = adjLoc(lval);
            if (!loc.empty())
              pushAdj(call->getArg(1), loc, out, ind);
          }
          return out;
        }

    if (const auto *call = asCall(s))
      if (isUserDiffFn(calleeOf(call)))
        // void call; restore any reused output arrays for earlier consumers
        return genReverseCall(call, "", ind) + tapePopCall(call, ind);

    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        return reverseAssign(bo, depth);

    if (const auto *fs = dyn_cast<ForStmt>(s)) {
      if (containsBreak(fs->getBody()) && hasActive(fs->getBody()))
        unsupported(fs->getBeginLoc(), "a loop that exits with `break` (its trip count is not known "
                                       "when the loop is reversed)");
      bool recompute = loopHasActiveLocals(fs->getBody());
      std::string hdr;
      if (!reverseForHeader(fs, hdr)) {
        std::string init =
            fs->getInit() ? src(fs->getInit()->getSourceRange()) : "";
        if (!init.empty() && init.back() == ';')
          init.pop_back();
        hdr = init + "; " + (fs->getCond() ? prim(fs->getCond()) : "") + "; " +
              (fs->getInc() ? prim(fs->getInc()) : "");
        if (!recompute) {
          caution(fs->getBeginLoc(), "a non-canonical loop is reversed in forward order "
                                     "(correct only if its iterations are independent)");
          out += ind + "// TODO(cuad): non-canonical loop kept in forward order "
                       "(correct only if the body is order-independent)\n";
        }
      }
      std::string restore = shadowCopy(fs, false, ind);
      out = restore + out + ind + "for (" + hdr + ") {\n";
      if (recompute) {
        // per-iteration: recompute the body's primal, then reverse it (its
        // locals are private to the iteration, so this needs no cross-iter tape)
        std::vector<const Stmt *> body;
        if (const auto *cs = dyn_cast_or_null<CompoundStmt>(fs->getBody()))
          body.assign(cs->body_begin(), cs->body_end());
        else if (fs->getBody())
          body.push_back(fs->getBody());
        out += genBlock(body, depth + 1);
      } else {
        out += reverseBody(fs->getBody(), depth + 1);
      }
      out += ind + "}\n" + restore;
      return out;
    }

    if (const auto *is = dyn_cast<IfStmt>(s)) {
      std::string thenB = reverseBody(is->getThen(), depth + 1);
      std::string elseB =
          is->getElse() ? reverseBody(is->getElse(), depth + 1) : "";
      if (thenB.empty() && elseB.empty())
        return ""; // no active content: nothing to reverse
      out += ind + "if (" + prim(is->getCond()) + ") {\n" + thenB + ind + "}";
      if (!elseB.empty())
        out += " else {\n" + elseB + ind + "}";
      return out + "\n";
    }

    // anything else that touches active data has no reverse here: say so instead of dropping it
    if ((isa<WhileStmt>(s) || isa<DoStmt>(s) || isa<SwitchStmt>(s) || isa<GotoStmt>(s) ||
         isa<CXXForRangeStmt>(s)) && hasActive(s))
      unsupported(s->getBeginLoc(), "this loop or jump (only counted `for` loops are reversed)");
    return out;
  }

  std::string reverseAssign(const BinaryOperator *bo, int depth) const {
    std::string ind(depth * 2, ' ');
    std::string loc = adjLoc(bo->getLHS());
    if (loc.empty())
      return "";
    std::string out;

    // lhs = f(args): value-returning cross-call; seed with lhs adjoint.
    if (bo->getOpcode() == BO_Assign)
      if (const CallExpr *call = asCall(bo->getRHS()))
        if (isUserDiffFn(calleeOf(call))) {
          const VarDecl *tv = tapedAssignTarget(bo);
          if (tv)
            out += ind + "cuad_pop(" + tv->getNameAsString() + ");\n";
          out += genReverseCall(call, loc, ind);
          out += ind + loc + " = 0;\n";
          return out;
        }

    // Pure accumulation (see collectAccum): no pop, no snapshot, no zeroing - the accumulator's adjoint
    // passes straight through and only the increment picks anything up.
    if (const VarDecl *av = localScalarTarget(bo->getLHS()))
      if (accum_.count(av)) {
        bool minus = false;
        if (const Expr *inc = accumIncrement(bo, av, minus)) {
          pushAdj(inc, minus ? ("-(" + loc + ")") : loc, out, ind);
          return out;
        }
      }

    const VarDecl *tv = tapedAssignTarget(bo);
    if (tv) { // taped local: restore old value, then op-specific adjoint
      std::string v = tv->getNameAsString();
      std::string pr = prim(bo->getRHS());
      out += ind + "cuad_pop(" + v + ");\n";
      std::string snap = "_cuad_b" + std::to_string(tmp_++);
      out += ind + tstr(tv->getType()) + " " + snap + " = " + loc + ";\n";
      switch (bo->getOpcode()) {
      case BO_Assign:
        out += ind + loc + " = 0;\n";
        pushAdj(bo->getRHS(), snap, out, ind);
        break;
      case BO_AddAssign:
        pushAdj(bo->getRHS(), snap, out, ind);
        break;
      case BO_SubAssign:
        pushAdj(bo->getRHS(), "-(" + snap + ")", out, ind);
        break;
      case BO_MulAssign: // v = v*rhs : d/dv_old = rhs, d/drhs = v_old
        out += ind + loc + " = " + mulT(pr, snap) + ";\n";
        pushAdj(bo->getRHS(), mulT(v, snap), out, ind);
        break;
      case BO_DivAssign: // v = v/rhs : d/dv_old = 1/rhs, d/drhs = -v_old/rhs^2
        out += ind + loc + " = (" + snap + ")/(" + pr + ");\n";
        pushAdj(bo->getRHS(),
                "-(" + v + ")*(" + snap + ")/((" + pr + ")*(" + pr + "))", out,
                ind);
        break;
      default:
        unsupported(bo->getBeginLoc(), "this compound assignment");
        out += ind + "// TODO(cuad): unhandled compound assignment\n";
      }
      return out;
    }

    // Non-taped target: an output write (y[i], *p). No stored old value needed. the seed is
    // SNAPSHOT before it is zeroed and propagated - an in-place update `B[i] = B[i] - acc` used to emit
    // `atomicAdd(&B_b[i], B_b[i]); acc_b += -(B_b[i]); B_b[i] = 0;` (seed doubled, then the pass-through lost).
    {
      std::string snap = "_cuad_b" + std::to_string(tmp_++);
      std::string ty = tstr(bo->getLHS()->getType().getCanonicalType().getUnqualifiedType());
      switch (bo->getOpcode()) {
      case BO_Assign:
        out += ind + ty + " " + snap + " = " + loc + ";\n";
        out += ind + loc + " = 0;\n";
        pushAdj(bo->getRHS(), snap, out, ind);
        return out;
      case BO_AddAssign:
        out += ind + ty + " " + snap + " = " + loc + ";\n";
        pushAdj(bo->getRHS(), snap, out, ind);
        return out;
      case BO_SubAssign:
        out += ind + ty + " " + snap + " = " + loc + ";\n";
        pushAdj(bo->getRHS(), "-(" + snap + ")", out, ind);
        return out;
      default:
        break;
      }
    }
    switch (bo->getOpcode()) {
    default:
      unsupported(bo->getBeginLoc(), "`*=` or `/=` on an output (it would need a tape)");
      return ind + "// TODO(cuad): *= or /= on a non-taped target needs a tape\n";
    }
  }

  // Adjoint of a nested body's statements, in reverse order (no primal re-run).
  std::string reverseBody(const Stmt *body, int depth) const {
    std::string out;
    if (const auto *cs = dyn_cast<CompoundStmt>(body)) {
      std::vector<const Stmt *> v(cs->body_begin(), cs->body_end());
      for (auto it = v.rbegin(); it != v.rend(); ++it)
        out += reverseStmt(*it, depth);
    } else {
      out += reverseStmt(body, depth);
    }
    return out;
  }

  // ---------------- scope handling ----------------
  std::string genScope(const std::vector<const Stmt *> &stmts, int depth) const {
    std::string ind(depth * 2, ' ');
    std::string out;
    size_t i = 0;
    while (i < stmts.size()) {
      if (isGuard(stmts[i])) {
        const auto *is = cast<IfStmt>(stmts[i]);
        out += ind + "if (" + prim(is->getCond()) + ") {\n";
        const auto *then = cast<CompoundStmt>(is->getThen());
        std::vector<const Stmt *> inner(then->body_begin(), then->body_end());
        out += genScope(inner, depth + 1);
        out += ind + "}\n";
        ++i;
      } else {
        std::vector<const Stmt *> run;
        while (i < stmts.size() && !isGuard(stmts[i]))
          run.push_back(stmts[i++]);
        out += genBlock(run, depth);
      }
    }
    return out;
  }

  std::string genBlock(const std::vector<const Stmt *> &stmtsIn, int depth) const {
    std::string ind(depth * 2, ' ');
    std::string out;
    std::vector<const Stmt *> stmts0;
    for (const Stmt *s : stmtsIn)
      flattenInto(s, stmts0);
    // an early exit (`if (c) { ...; continue; }`) splits the block - the statements before
    // it are this block's, the exit branch and the rest become the two arms of an if/else, each a block of its own
    std::vector<const Stmt *> stmts;
    const IfStmt *exitIf = nullptr;
    std::vector<const Stmt *> rest;
    for (size_t i = 0; i < stmts0.size(); ++i) {
      const Stmt *s = unwrapAttr(stmts0[i]);
      if (!exitIf && isEarlyExit(s)) {
        exitIf = cast<IfStmt>(s);
        rest.assign(stmts0.begin() + i + 1, stmts0.end());
        break;
      }
      stmts.push_back(stmts0[i]);
    }
    // branch-local declarations hoisted (their adjoints declared with them)
    out += hoistDecls(stmts, ind);
    for (const Stmt *s0 : stmts) { // forward primal (with pushes)
      const Stmt *s = unwrapAttr(s0);
      // recompute-loops regenerate their own primal (with pushes) inside the backward loop, per iteration.
      // They still run here, PUSH-FREE, so that what they leave behind (an accumulated sum) has its value when the
      // statements after them are recomputed - skipping them left an accumulated `v = sum M m` at 0, and every
      // derivative downstream of it was taken at the wrong state. Requirement on the
      // source: a recompute loop may change outer variables only by accumulation (x = x + ...).
      if (const auto *fs = dyn_cast<ForStmt>(s))
        if (loopHasActiveLocals(fs->getBody())) {
          // a grid-stride loop's iterations are independent elements (nothing carried between them): no primal
          // pass needed - the backward loop re-materialises each one
          std::string init = fs->getInit() ? src(fs->getInit()->getSourceRange()) : "";
          if (init.find("threadIdx") != std::string::npos)
            continue;
          bool saved = noPush_;
          noPush_ = true;
          out += genPrimal(s, depth);
          noPush_ = saved;
          continue;
        }
      out += genPrimal(s, depth);
    }
    for (const Stmt *s : stmts) // zero local adjoint accumulators
      if (const auto *ds = dyn_cast<DeclStmt>(s))
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            if (Act.isVaried(vd)) {
              QualType qt = vd->getType();
              std::string nm = vd->getNameAsString();
              if (Ctx.getAsConstantArrayType(qt)) {
                QualType elem;
                std::string dims = arrayDims(qt, elem);
                if (elem->isRealFloatingType())
                  out += ind + tstr(elem) + " " + nm + "_b" + dims + " = {0};\n";
              } else if (qt->isRealFloatingType()) {
                // (the adjoint of a `const float` local is not const)
                out += ind + tstr(qt.getUnqualifiedType()) + " " + nm + "_b = 0;\n";
              }
            }
    if (exitIf) {
      // the exit arm (its continue dropped, a return kept: its value seeds the return adjoint) and the rest, each
      // primal + reverse; then the reverse of the statements before
      std::vector<const Stmt *> arm;
      const Stmt *t = exitIf->getThen();
      if (const auto *cs = dyn_cast<CompoundStmt>(t))
        arm.assign(cs->body_begin(), cs->body_end());
      else
        arm.push_back(t);
      if (!arm.empty() && isa<ContinueStmt>(arm.back()))
        arm.pop_back();
      out += ind + "if (" + prim(exitIf->getCond()) + ") {\n" + genBlock(arm, depth + 1) + ind + "} else {\n" +
             genBlock(rest, depth + 1) + ind + "}\n";
    }
    for (auto it = stmts.rbegin(); it != stmts.rend(); ++it) // reversed adjoint
      out += reverseStmt(*it, depth);
    return out;
  }
};

} // namespace cuad
