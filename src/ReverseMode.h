#include <algorithm>
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
// Tape: a per-thread LIFO stack (CUAD_TAPE_DECL, cuad_push / cuad_pop, see test/cuad_rt.h);
// a function with a tape publishes its top around each call to another _b function.
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
#include <cctype>
#include <functional>
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
  std::set<const VarDecl *> arrayTaped_; // local arrays with self-dependent element writes
  std::string curRetSeed_;           // return-adjoint seed param (value-ret fns)
  bool hasTape_ = false;             // the function being generated declares a tape
  mutable int tmp_ = 0;              // fresh snapshot-variable counter
  mutable bool noPush_ = false;      // emitting a push-free primal (a recompute loop's forward pass)
  mutable const ForStmt *ckptLoop_ = nullptr; // the recompute loop whose own reverse will pop its checkpoints
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
    curBody_ = FD->getBody();
    taped_.clear();
    reused_.clear();
    arrayTaped_.clear();
    hoisted_.clear();
    tmp_ = 0;
    collectTaped(FD->getBody());
    collectAccum(FD->getBody());
    // every element write of a local floating array is taped (its old value may be read by an adjoint
    // statement after the reverse sweep has passed the write; before, only self-referencing writes were)
    writtenFloatArrays(FD->getBody(), arrayTaped_);
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
    hasTape_ = !taped_.empty() || !reused_.empty() || !arrayTaped_.empty() ||
               anyCheckpointedArray(FD->getBody());
    if (hasTape_) // per-thread tape: a local array, or a frame of the global tape (cuad_rt.h)
      out += "  CUAD_TAPE_DECL;\n";
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
    // ANY local declared in the body forces the per-iteration re-materialization - an
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
  // Scalars that a recompute loop carries from one iteration to the next: taped locals declared outside the loop
  // and reassigned inside it. Pure accumulators are not taped and so are not carried. The backward loop
  // recomputes an iteration from the values these hold when it starts, so the forward pass checkpoints them at
  // the top of every iteration and the backward loop restores them before recomputing.
  void collectCarried(const ForStmt *fs, const Stmt *s, std::vector<const VarDecl *> &out) const {
    if (!s)
      return;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const VarDecl *v = localScalarTarget(bo->getLHS()))
          if (taped_.count(v) && !withinRange(fs->getSourceRange(), v->getBeginLoc()) &&
              std::find(out.begin(), out.end(), v) == out.end())
            out.push_back(v);
    for (const Stmt *c : s->children())
      collectCarried(fs, c, out);
  }
  std::vector<const VarDecl *> carriedVars(const ForStmt *fs) const {
    std::vector<const VarDecl *> out;
    collectCarried(fs, fs->getBody(), out);
    return out;
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

  const ArraySubscriptExpr *tapedArrayTarget(const Stmt *s) const {
    const auto *bo = dyn_cast<BinaryOperator>(s);
    if (!bo || !bo->isAssignmentOp())
      return nullptr;
    const auto *as = dyn_cast<ArraySubscriptExpr>(bo->getLHS()->IgnoreParenImpCasts());
    if (!as)
      return nullptr;
    const auto *d = dyn_cast<DeclRefExpr>(subscriptRoot(as)); // a[i] and g[m][0] alike
    if (!d)
      return nullptr;
    const auto *vd = dyn_cast<VarDecl>(d->getDecl());
    return vd && arrayTaped_.count(vd) ? as : nullptr;
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
      out += ind + "if (" + c + ") {\n";
      pushAdj(co->getTrueExpr(), seed, out, ind + "  ");
      out += ind + "} else {\n";
      pushAdj(co->getFalseExpr(), seed, out, ind + "  ");
      out += ind + "}\n";
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
    if (hasTape_) // a global tape: the callee's frame starts above this function's entries
      out = pre + ind + "CUAD_TAPE_PUBLISH();\n" + ind + fd->getNameAsString() + "_b" + calleeTArgs(call) +
            "(" + args + ");\n" + ind + "CUAD_TAPE_RESTORE();\n";
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
  // a hoisted declaration's primal: assignments of its initializers (+ the adjoint pointer's)
  std::string hoistedPrimal(const DeclStmt *ds, const std::string &ind) const {
    std::string out;
    for (const Decl *d : ds->decls())
      if (const auto *vd = dyn_cast<VarDecl>(d)) {
        if (vd->hasInit() && !isa<InitListExpr>(vd->getInit()->IgnoreImplicit()))
          out += ind + vd->getNameAsString() + " = " + prim(vd->getInit()) + ";\n";
        else if (vd->hasInit()) {
          unsupported(vd->getBeginLoc(), "the initializer list of the branch-local array `" +
                                             vd->getNameAsString() + "`");
          out += ind + "// TODO(cuad): hoisted array initializer of " + vd->getNameAsString() + "\n";
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
  // ---------------- local arrays carried across loop iterations ----------------
  //
  // A loop's reverse must see, for every iteration it reverses, the values the iteration read in the forward sweep.
  // For a local array declared outside the loop and both written and read inside it, that is the array as it was
  // at the top of the iteration - which a later iteration may have overwritten (a recurrent state, `a[i] = a[i-1]`,
  // a read followed by an overwrite of the same element). Such an array is CHECKPOINTED: pushed whole at the top
  // of every forward iteration and popped before the backward loop re-runs that iteration (the loop is reversed as
  // a recompute loop). The one case where a single snapshot taken at loop entry suffices is proven, not assumed:
  // every access to the array in the loop uses one and the same index, injective in the induction variable, so
  // each iteration touches only its own element (`a[i] = f(a[i])`, `q[3*i+off] *= c`).
  static const VarDecl *localFloatArray(const Expr *base, ASTContext &C) {
    const auto *d = dyn_cast<DeclRefExpr>(base->IgnoreParenImpCasts());
    if (!d)
      return nullptr;
    const auto *vd = dyn_cast<VarDecl>(d->getDecl());
    if (!vd || !vd->isLocalVarDecl() || isa<ParmVarDecl>(vd))
      return nullptr;
    const auto *cat = C.getAsConstantArrayType(vd->getType());
    if (!cat)
      return nullptr;
    QualType el = cat->getElementType();
    while (const auto *in = C.getAsConstantArrayType(el))
      el = in->getElementType();
    return el->isRealFloatingType() ? vd : nullptr;
  }
  // the array an lvalue `a[i]` / `g[m][0]` writes into
  static const Expr *subscriptRoot(const Expr *e) {
    e = e->IgnoreParenImpCasts();
    while (const auto *as = dyn_cast<ArraySubscriptExpr>(e))
      e = as->getBase()->IgnoreParenImpCasts();
    return e;
  }
  void referencedLocalArrays(const Stmt *s, std::set<const VarDecl *> &out) const {
    if (!s)
      return;
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
        if (vd->isLocalVarDecl() && Ctx.getAsConstantArrayType(vd->getType()))
          out.insert(vd);
    for (const Stmt *c : s->children())
      referencedLocalArrays(c, out);
  }
  void writtenLocalArrays(const Stmt *s, std::set<const VarDecl *> &out) const {
    if (!s)
      return;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const auto *as = dyn_cast<ArraySubscriptExpr>(bo->getLHS()->IgnoreParenImpCasts()))
          if (const auto *d = dyn_cast<DeclRefExpr>(as->getBase()->IgnoreParenImpCasts()))
            if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
              if (vd->isLocalVarDecl() && Ctx.getAsConstantArrayType(vd->getType()))
                out.insert(vd);
    for (const Stmt *c : s->children())
      writtenLocalArrays(c, out);
  }
  // local floating arrays whose elements are assigned in s (the per-element tape handles exactly these)
  void writtenFloatArrays(const Stmt *s, std::set<const VarDecl *> &out) const {
    if (!s)
      return;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const auto *as = dyn_cast<ArraySubscriptExpr>(bo->getLHS()->IgnoreParenImpCasts()))
          if (const VarDecl *vd = localFloatArray(subscriptRoot(as), Ctx))
            out.insert(vd);
    for (const Stmt *c : s->children())
      writtenFloatArrays(c, out);
  }
  // Classify every reference to a local floating array inside s: written (assigned element, ++/--, or the array
  // passed whole - to a call, through a pointer - which may write it), read (any other use), and the subscripts
  // under which it is accessed. A whole-array use is recorded as an "escape".
  struct ArrayUse {
    bool written = false, read = false, escapes = false;
    std::vector<const Expr *> idx;
  };
  void arrayUses(const Stmt *s, std::map<const VarDecl *, ArrayUse> &u, std::vector<const VarDecl *> &order,
                 const Expr *writeTarget = nullptr) const {
    if (!s)
      return;
    auto touch = [&](const VarDecl *vd) -> ArrayUse & {
      if (!u.count(vd))
        order.push_back(vd);
      return u[vd];
    };
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp()) {
        const Expr *lhs = bo->getLHS()->IgnoreParenImpCasts();
        if (isa<ArraySubscriptExpr>(lhs)) {
          if (const VarDecl *vd = localFloatArray(subscriptRoot(lhs), Ctx)) {
            ArrayUse &a = touch(vd);
            a.written = true;
            if (bo->getOpcode() != BO_Assign)
              a.read = true;          // compound assignment reads the old element
          }
          arrayUses(lhs, u, order, lhs);
          arrayUses(bo->getRHS(), u, order);
          return;
        }
      }
    if (const auto *uo = dyn_cast<UnaryOperator>(s))
      if (uo->isIncrementDecrementOp())
        if (const VarDecl *vd = localFloatArray(subscriptRoot(uo->getSubExpr()), Ctx)) {
          ArrayUse &a = touch(vd);
          a.written = a.read = true;
        }
    if (const auto *as = dyn_cast<ArraySubscriptExpr>(s)) {
      const Expr *base = as->getBase()->IgnoreParenImpCasts();
      if (const VarDecl *vd = localFloatArray(base, Ctx)) {
        ArrayUse &a = touch(vd);
        if (s != writeTarget)
          a.read = true;
        a.idx.push_back(as->getIdx());
        arrayUses(as->getIdx(), u, order);
        return;
      }
      if (isa<ArraySubscriptExpr>(base)) { // g[m][0]: multi-dimensional - analyzed as one opaque array
        if (const VarDecl *vd = localFloatArray(subscriptRoot(base), Ctx)) {
          ArrayUse &a = touch(vd);
          if (s != writeTarget)
            a.read = true;
          a.escapes = true;
        }
        arrayUses(as->getIdx(), u, order);
        arrayUses(base, u, order, nullptr);
        return;
      }
    }
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (const VarDecl *vd = localFloatArray(d, Ctx)) { // the whole array (decayed to a pointer)
        ArrayUse &a = touch(vd);
        a.written = a.read = a.escapes = true;
        return;
      }
    for (const Stmt *c : s->children())
      arrayUses(c, u, order, nullptr);
  }
  // v is assigned (=, op=, ++/--) or declared somewhere in s
  static bool modifiedIn(const Stmt *s, const VarDecl *v) {
    if (!s)
      return false;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const auto *d = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
          if (d->getDecl() == v)
            return true;
    if (const auto *uo = dyn_cast<UnaryOperator>(s))
      if (uo->isIncrementDecrementOp() || uo->getOpcode() == UO_AddrOf)
        if (const auto *d = dyn_cast<DeclRefExpr>(uo->getSubExpr()->IgnoreParenImpCasts()))
          if (d->getDecl() == v)
            return true;
    if (const auto *ds = dyn_cast<DeclStmt>(s))
      for (const Decl *dd : ds->decls())
        if (dd == v)
          return true;
    for (const Stmt *c : s->children())
      if (modifiedIn(c, v))
        return true;
    return false;
  }
  // an integer expression whose value is the same in every iteration of fs: constants and variables that the loop
  // (body and increment) never modifies, combined by arithmetic - no calls, no memory reads
  bool loopInvariantInt(const Expr *e, const ForStmt *fs, const VarDecl *ind) const {
    e = e->IgnoreParenImpCasts();
    if (!e->getType()->isIntegerType())
      return false;
    if (isa<IntegerLiteral>(e))
      return true;
    if (const auto *d = dyn_cast<DeclRefExpr>(e)) {
      if (isa<EnumConstantDecl>(d->getDecl()))
        return true;
      const auto *v = dyn_cast<VarDecl>(d->getDecl());
      return v && v != ind && !withinRange(fs->getSourceRange(), v->getLocation()) &&
             !modifiedIn(fs->getBody(), v) && !modifiedIn(fs->getInc(), v);
    }
    if (const auto *u = dyn_cast<UnaryOperator>(e))
      return (u->getOpcode() == UO_Minus || u->getOpcode() == UO_Plus) &&
             loopInvariantInt(u->getSubExpr(), fs, ind);
    if (const auto *b = dyn_cast<BinaryOperator>(e))
      return (b->isAdditiveOp() || b->isMultiplicativeOp() || b->isShiftOp()) &&
             loopInvariantInt(b->getLHS(), fs, ind) && loopInvariantInt(b->getRHS(), fs, ind);
    if (const auto *p = dyn_cast<ParenExpr>(e))
      return loopInvariantInt(p->getSubExpr(), fs, ind);
    return false;
  }
  // idx = +-i, idx + inv, inv + idx, idx - inv, c * idx (c a nonzero constant): distinct iterations, distinct
  // elements. A reference to i alone is not enough - i % 2, i / 2 and i * i can revisit an element.
  bool injectiveArrayIndex(const Expr *idx, const VarDecl *ind, const ForStmt *fs) const {
    if (!idx || !ind)
      return false;
    idx = idx->IgnoreParenImpCasts();
    if (const auto *d = dyn_cast<DeclRefExpr>(idx))
      return d->getDecl() == ind;
    if (const auto *p = dyn_cast<ParenExpr>(idx))
      return injectiveArrayIndex(p->getSubExpr(), ind, fs);
    if (const auto *u = dyn_cast<UnaryOperator>(idx))
      if (u->getOpcode() == UO_Plus || u->getOpcode() == UO_Minus)
        return injectiveArrayIndex(u->getSubExpr(), ind, fs);
    if (const auto *b = dyn_cast<BinaryOperator>(idx)) {
      const Expr *l = b->getLHS(), *r = b->getRHS();
      if (b->getOpcode() == BO_Add)
        return (injectiveArrayIndex(l, ind, fs) && loopInvariantInt(r, fs, ind)) ||
               (injectiveArrayIndex(r, ind, fs) && loopInvariantInt(l, fs, ind));
      if (b->getOpcode() == BO_Sub)
        return injectiveArrayIndex(l, ind, fs) && loopInvariantInt(r, fs, ind);
      if (b->getOpcode() == BO_Mul) {
        const Expr *factor = injectiveArrayIndex(l, ind, fs) ? r : injectiveArrayIndex(r, ind, fs) ? l : nullptr;
        Expr::EvalResult v;
        return factor && factor->EvaluateAsInt(v, Ctx) && !v.Val.getInt().isZero();
      }
    }
    return false;
  }
  const VarDecl *inductionVar(const ForStmt *fs) const {
    if (const auto *ds = dyn_cast_or_null<DeclStmt>(fs->getInit()))
      if (ds->isSingleDecl())
        if (const auto *vd = dyn_cast<VarDecl>(ds->getSingleDecl()))
          if (vd->getType()->isIntegerType() && !modifiedIn(fs->getBody(), vd))
            return vd;
    return nullptr;
  }
  bool sameExpr(const Expr *a, const Expr *b) const {
    llvm::FoldingSetNodeID ia, ib;
    a->IgnoreParenImpCasts()->Profile(ia, Ctx, true);
    b->IgnoreParenImpCasts()->Profile(ib, Ctx, true);
    return ia == ib;
  }
  // each iteration touches only its own element of the array (see above)
  bool iterationPrivate(const ForStmt *fs, const ArrayUse &a) const {
    const VarDecl *ind = inductionVar(fs);
    if (!ind || a.escapes || a.idx.empty())
      return false;
    for (const Expr *e : a.idx)
      if (!sameExpr(e, a.idx.front()))
        return false;
    return injectiveArrayIndex(a.idx.front(), ind, fs);
  }
  // arrays fs must checkpoint per iteration (declared outside, written and read inside, not iteration-private),
  // and arrays it writes that need only a snapshot at loop entry (the rest of those written inside); source order
  void loopArrays(const ForStmt *fs, std::vector<const VarDecl *> &ckpt,
                  std::vector<const VarDecl *> *entry = nullptr) const {
    std::map<const VarDecl *, ArrayUse> u;
    std::vector<const VarDecl *> order;
    arrayUses(fs->getBody(), u, order);
    for (const VarDecl *vd : order) {
      const ArrayUse &a = u[vd];
      if (!a.written || withinRange(fs->getSourceRange(), vd->getLocation()))
        continue;                       // declared inside the loop: fresh every iteration
      if (a.read && !iterationPrivate(fs, a))
        ckpt.push_back(vd);
      else if (entry)
        entry->push_back(vd);
    }
  }
  std::vector<const VarDecl *> checkpointedArrays(const ForStmt *fs) const {
    std::vector<const VarDecl *> v;
    loopArrays(fs, v);
    return v;
  }
  bool anyCheckpointedArray(const Stmt *s) const {
    if (!s)
      return false;
    if (const auto *fs = dyn_cast<ForStmt>(s))
      if (!checkpointedArrays(fs).empty())
        return true;
    for (const Stmt *c : s->children())
      if (anyCheckpointedArray(c))
        return true;
    return false;
  }
  // a loop whose backward sweep re-runs each iteration's primal before reversing it
  bool isRecompute(const ForStmt *fs) const {
    if (solveInfo(fs).ok)
      return false;
    return loopHasActiveLocals(fs->getBody()) || !checkpointedArrays(fs).empty();
  }

  // ---------------- convergence loops: implicit adjoint ----------------
  // A `for`, `while` or `do` loop that leaves through `if (c) break;` or through a condition on what it iterates, and
  // iterates floating locals (scalars or local arrays) to convergence (a root solve, a fixed-point iteration), is not
  // reversed iteration by iteration. Its result is taken to be the converged point, which depends on the inputs only
  // through the equation that the loop solves:
  //  - smooth update (some state assignment depends on the inputs): x* = G(x*, p). G_x and G_p come from one reverse
  //    sweep of one iteration per state component; (I - G_x)^T l = s is solved directly and p_b += G_p^T l;
  //  - bracketing update (the inputs reach the state only through the branch that decides which bound the
  //    trial point m replaces): differentiating the iterations gives a derivative through the initial
  //    bracket only, which is wrong. x* is the root of the branch condition r(m, p) and p_b -= r_p s / r_m;
  //  - several iterated variables without a smooth map (secant, regula falsi, Brent): the residual found in the loop.
  // The exit state must lie within CUAD_SOLVE_TOL of a solution, and the derivative must not change across the
  // interval in which the solution is known (CUAD_SOLVE_AGREE): a double root or a degenerate fixed point traps. The
  // second evaluation is skipped for a smooth update whose G_x is negligible (Newton's method at a simple root).
  // The forward sweep runs the loop as written and stores nothing per iteration.
public:
  struct SolveInfo {
    bool ok = false, bracket = false;
    std::string why;                         // why a loop with `break` is not a convergence loop
    std::vector<const Stmt *> iter;          // the body without its exit tests and counter updates
    std::vector<const Expr *> exitConds;     // the exit tests: `if (c) break;` conditions and the loop condition
    struct Comp { std::string x, b, d, T; }; // one floating state component: value, adjoint, tangent, type
    std::vector<Comp> comps;
    std::vector<const VarDecl *> state, params;
    const IfStmt *sel = nullptr;             // bracketing: the branch that moves a bound
    const VarDecl *trial = nullptr;          // bracketing: the trial point that both arms assign
    // root-finding loops with several iterated variables (secant, regula falsi, Brent): candidate residuals, values
    // computed in one iteration from a single trial point and the inputs; at run time the first one that vanishes at
    // the exit state defines the root
    struct Residual {
      const VarDecl *var = nullptr;          // the variable the residual is stored in
      const Expr *expr = nullptr;            // its defining expression
      const VarDecl *trialVar = nullptr;     // the one iterated or trial variable it is evaluated at
      std::vector<const Stmt *> trialDecls;  // body declarations that recompute a trial local from the state
      std::vector<const Stmt *> slice;       // body declarations between the trial and the residual
    };
    bool residual = false;
    std::vector<Residual> res;
  };
  // the analysis of convergence loops for the forward mode, which shares it (Activity of the function FD)
  void prepareSolveAnalysis(const FunctionDecl *FD) {
    Act = Activity();
    Act.run(FD);
    curBody_ = FD->getBody();
  }
  const SolveInfo &convergenceLoop(const Stmt *loop) const { return solveInfo(loop); }
  static const Stmt *loopBody(const Stmt *s) {
    if (const auto *f = dyn_cast<ForStmt>(s))
      return f->getBody();
    if (const auto *w = dyn_cast<WhileStmt>(s))
      return w->getBody();
    if (const auto *d = dyn_cast<DoStmt>(s))
      return d->getBody();
    return nullptr;
  }
  static const Expr *loopCond(const Stmt *s) {
    if (const auto *f = dyn_cast<ForStmt>(s))
      return f->getCond();
    if (const auto *w = dyn_cast<WhileStmt>(s))
      return w->getCond();
    if (const auto *d = dyn_cast<DoStmt>(s))
      return d->getCond();
    return nullptr;
  }
  static const Stmt *loopInit(const Stmt *s) {
    if (const auto *f = dyn_cast<ForStmt>(s))
      return f->getInit();
    return nullptr;
  }

private:
  mutable std::map<const Stmt *, SolveInfo> solve_;

  static bool isBreakIf(const Stmt *s) {
    const auto *is = dyn_cast<IfStmt>(s);
    if (!is || is->getElse())
      return false;
    const Stmt *t = is->getThen();
    if (const auto *cs = dyn_cast<CompoundStmt>(t))
      return cs->size() == 1 && isa<BreakStmt>(cs->body_front());
    return isa<BreakStmt>(t);
  }
  static void solveDecls(const Stmt *s, std::set<const VarDecl *> &out) {
    if (!s)
      return;
    if (const auto *ds = dyn_cast<DeclStmt>(s))
      for (const Decl *d : ds->decls())
        if (const auto *vd = dyn_cast<VarDecl>(d))
          out.insert(vd);
    for (const Stmt *c : s->children())
      solveDecls(c, out);
  }
  static void solveRefs(const Stmt *s, std::set<const VarDecl *> &out) {
    if (!s)
      return;
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
        out.insert(vd);
    for (const Stmt *c : s->children())
      solveRefs(c, out);
  }
  static bool mentionsAny(const Stmt *s, const std::set<const VarDecl *> &vs) {
    std::set<const VarDecl *> r;
    solveRefs(s, r);
    for (const VarDecl *v : r)
      if (vs.count(v))
        return true;
    return false;
  }
  static bool isFloatScalar(const VarDecl *vd) {
    QualType t = vd->getType().getNonReferenceType();
    return t->isRealFloatingType();
  }

  // floating locals declared outside the loop that its body assigns (the candidates for its state)
  static void outerFloatAssigned(const Stmt *s, const std::set<const VarDecl *> &inner, std::set<const VarDecl *> &out) {
    if (!s)
      return;
    const Expr *lhs = nullptr;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        lhs = bo->getLHS();
    if (const auto *uo = dyn_cast<UnaryOperator>(s))
      if (uo->isIncrementDecrementOp())
        lhs = uo->getSubExpr();
    if (lhs) {
      const Expr *root = lhs->IgnoreParenImpCasts();
      while (const auto *as = dyn_cast<ArraySubscriptExpr>(root))
        root = as->getBase()->IgnoreParenImpCasts();
      if (const auto *d = dyn_cast<DeclRefExpr>(root))
        if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
          if (!inner.count(vd) && vd->getType()->getBaseElementTypeUnsafe()->isRealFloatingType())
            out.insert(vd);
    }
    for (const Stmt *c : s->children())
      outerFloatAssigned(c, inner, out);
  }
  // A top-level statement that only counts iterations: ++it, it++, it += 1, it = it + 1 on an integer variable.
  static const VarDecl *counterUpdate(const Stmt *s) {
    const Expr *lhs = nullptr;
    if (const auto *uo = dyn_cast<UnaryOperator>(s))
      if (uo->isIncrementDecrementOp())
        lhs = uo->getSubExpr();
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp() && !bo->getRHS()->getType()->isRealFloatingType()) {
        std::set<const VarDecl *> r;
        solveRefs(bo->getRHS(), r);
        bool floats = false;
        for (const VarDecl *v : r)
          floats |= v->getType()->isRealFloatingType();
        if (!floats)
          lhs = bo->getLHS();
      }
    if (!lhs)
      return nullptr;
    const auto *d = dyn_cast<DeclRefExpr>(lhs->IgnoreParenImpCasts());
    const auto *vd = d ? dyn_cast<VarDecl>(d->getDecl()) : nullptr;
    return vd && vd->getType()->isIntegerType() ? vd : nullptr;
  }

  // A user function called inside a convergence loop must be pure: its value may depend only on its arguments, and
  // calling it again must not change anything, because the derivative code evaluates the iteration again. It passes
  // its parameters by value and returns a floating value; its body refers to no variable other than its parameters
  // and its own locals (no global, __device__ or __constant__ variable), dereferences no pointer, writes only its own
  // locals, and calls only functions without a body (the math library) or pure user functions. Returns "" or why not.
  std::string pureFunction(const FunctionDecl *fd, int depth) const {
    if (depth > 8)
      return "calls user functions too deeply nested to check";
    if (!fd->getReturnType()->isRealFloatingType())
      return "returns no floating value";
    for (const ParmVarDecl *p : fd->parameters()) {
      QualType t = p->getType().getCanonicalType();
      if (t->isPointerType() || t->isReferenceType() || t->isArrayType())
        return "takes a pointer, a reference or an array";
    }
    const Stmt *body = fd->getBody();
    if (!body)
      return "";
    std::string why;
    std::function<void(const Stmt *)> scan = [&](const Stmt *s) {
      if (!s || !why.empty())
        return;
      if (const auto *d = dyn_cast<DeclRefExpr>(s))
        if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
          if (vd->hasGlobalStorage() || vd->getDeclContext() != fd)
            if (!(vd->isLocalVarDeclOrParm() && vd->getParentFunctionOrMethod() == fd)) {
              why = "reads the variable `" + vd->getNameAsString() + "` declared outside it";
              return;
            }
      if (const auto *uo = dyn_cast<UnaryOperator>(s))
        if (uo->getOpcode() == UO_Deref) {
          why = "dereferences a pointer";
          return;
        }
      if (const auto *as = dyn_cast<ArraySubscriptExpr>(s))
        if (as->getBase()->IgnoreParenImpCasts()->getType()->isPointerType()) {
          why = "indexes a pointer";
          return;
        }
      if (const auto *ce = dyn_cast<CallExpr>(s))
        if (const FunctionDecl *g = calleeOf(ce))
          if (g->hasBody() && g != fd) {
            std::string w = pureFunction(g, depth + 1);
            if (!w.empty()) {
              why = "calls `" + g->getNameAsString() + "`, which " + w;
              return;
            }
          }
      for (const Stmt *c : s->children())
        scan(c);
    };
    scan(body);
    return why;
  }

  const SolveInfo &solveInfo(const Stmt *loop) const {
    auto it = solve_.find(loop);
    if (it != solve_.end())
      return it->second;
    SolveInfo &si = solve_[loop];
    const Stmt *body = loopBody(loop);
    if (!body || !hasActive(body))
      return si; // not a convergence loop (why stays empty)
    // a convergence loop leaves through `if (c) break;` or through a loop condition that tests a floating variable
    // the body iterates; a counted `for` loop is neither
    std::set<const VarDecl *> inner, outer;
    solveDecls(body, inner);
    solveDecls(loopInit(loop), inner);
    outerFloatAssigned(body, inner, outer);
    const Expr *cond = loopCond(loop);
    bool condTests = cond && mentionsAny(cond, outer);
    if (isa<ForStmt>(loop) && !containsBreak(body) && !condTests)
      return si;
    si.why = analyseSolve(loop, si);
    si.ok = si.why.empty();
    return si;
  }

  std::string analyseSolve(const Stmt *loop, SolveInfo &si) const {
    const Stmt *body = loopBody(loop);
    std::vector<const Stmt *> top;
    if (const auto *cs = dyn_cast<CompoundStmt>(body))
      top.assign(cs->body_begin(), cs->body_end());
    else
      top.push_back(body);
    std::set<const VarDecl *> inner;
    solveDecls(body, inner);
    solveDecls(loopInit(loop), inner);
    bool exits = false;
    for (const Stmt *s0 : top) {
      const Stmt *s = unwrapAttr(s0);
      if (isBreakIf(s)) {
        exits = true;
        si.exitConds.push_back(cast<IfStmt>(s)->getCond());
        continue;
      }
      if (containsBreak(s))
        return "its `break` is not a top-level `if (c) break;`";
      if (const VarDecl *c = counterUpdate(s))
        if (!inner.count(c))
          continue; // an iteration counter declared outside the loop: not part of the iteration map
      si.iter.push_back(s);
    }
    if (const Expr *c = loopCond(loop)) {
      std::set<const VarDecl *> outer;
      outerFloatAssigned(body, inner, outer);
      if (mentionsAny(c, outer))
        exits = true;
      si.exitConds.push_back(c);
    }
    if (!exits)
      return "it has no top-level `if (c) break;` and its condition tests no iterated variable";
    std::string bad;
    std::set<const VarDecl *> assigned;
    std::set<const VarDecl *> stateArrays; // local floating arrays declared before the loop that it writes
    {
      std::set<const VarDecl *> w;
      for (const Stmt *s : si.iter)
        outerFloatAssigned(s, inner, w);
      for (const VarDecl *v : w)
        if (!isFloatScalar(v) && v->isLocalVarDecl() && Ctx.getAsConstantArrayType(v->getType()))
          stateArrays.insert(v);
    }
    std::function<void(const Stmt *)> walk = [&](const Stmt *s) {
      if (!s || !bad.empty())
        return;
      if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s) || isa<SwitchStmt>(s) || isa<ReturnStmt>(s) ||
          isa<ContinueStmt>(s) || isa<GotoStmt>(s)) {
        bad = "it contains a nested loop or another jump";
        return;
      }
      const Expr *lhs = nullptr;
      if (const auto *bo = dyn_cast<BinaryOperator>(s))
        if (bo->isAssignmentOp())
          lhs = bo->getLHS();
      if (const auto *uo = dyn_cast<UnaryOperator>(s))
        if (uo->isIncrementDecrementOp())
          lhs = uo->getSubExpr();
      if (lhs) {
        const Expr *root = lhs->IgnoreParenImpCasts();
        bool element = false;
        while (const auto *as = dyn_cast<ArraySubscriptExpr>(root)) {
          root = as->getBase()->IgnoreParenImpCasts();
          element = true;
        }
        const auto *d = dyn_cast<DeclRefExpr>(root);
        const auto *vd = d ? dyn_cast<VarDecl>(d->getDecl()) : nullptr;
        if (!vd || (element && !localFloatArray(root, Ctx))) {
          bad = "it writes memory other than its local variables (global or shared memory, or through a pointer)";
          return;
        }
        if (element && inner.count(vd)) {
          bad = "it writes the local array `" + vd->getNameAsString() + "` declared inside it";
          return;
        }
        if (!inner.count(vd)) {
          if (!element && !isFloatScalar(vd)) {
            bad = "it changes the outer non-floating variable `" + vd->getNameAsString() + "`";
            return;
          }
          assigned.insert(vd);
        }
      }
      if (const auto *ce = dyn_cast<CallExpr>(s))
        if (const FunctionDecl *fd = calleeOf(ce))
          if (isUserDiffFn(fd)) {
            // a call to a pure function is part of the iteration map: see pureFunction
            std::string why = pureFunction(fd, 0);
            if (!why.empty()) {
              bad = "it calls the user function `" + fd->getNameAsString() + "`, which " + why;
              return;
            }
          }
      if (const auto *as = dyn_cast<ArraySubscriptExpr>(s))
        if (hasActive(as->getBase())) {
          const Expr *root = as->getBase()->IgnoreParenImpCasts();
          while (const auto *in = dyn_cast<ArraySubscriptExpr>(root))
            root = in->getBase()->IgnoreParenImpCasts();
          const auto *d = dyn_cast<DeclRefExpr>(root);
          const auto *vd = d ? dyn_cast<VarDecl>(d->getDecl()) : nullptr;
          if (!vd || !stateArrays.count(vd)) {
            bad = "it reads an active array other than its own state (copy the element into a local before the loop)";
            return;
          }
        }
      for (const Stmt *c : s->children())
        walk(c);
    };
    for (const Stmt *s : si.iter)
      walk(s);
    if (!bad.empty())
      return bad;
    si.state.assign(assigned.begin(), assigned.end());
    if (si.state.empty())
      return "it iterates no floating local declared before it";
    std::set<const VarDecl *> refs;
    for (const Stmt *s : si.iter)
      solveRefs(s, refs);
    std::set<const VarDecl *> params;
    for (const VarDecl *vd : refs) {
      if (inner.count(vd) || assigned.count(vd) || !Act.isVaried(vd))
        continue;
      if (!isFloatScalar(vd) || !vd->isLocalVarDeclOrParm())
        return "it reads the active non-scalar `" + vd->getNameAsString() + "` (copy what it needs into locals)";
      params.insert(vd);
    }
    si.params.assign(params.begin(), params.end());
    // which values computed in one iteration depend on the inputs by data flow (branch conditions ignored)
    std::set<const VarDecl *> dep(params.begin(), params.end());
    std::function<void(const Stmt *)> flow = [&](const Stmt *s) {
      if (!s)
        return;
      if (const auto *ds = dyn_cast<DeclStmt>(s)) {
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            if (vd->hasInit() && mentionsAny(vd->getInit(), dep))
              dep.insert(vd);
        return;
      }
      if (const auto *bo = dyn_cast<BinaryOperator>(s))
        if (bo->isAssignmentOp()) {
          const Expr *root = bo->getLHS()->IgnoreParenImpCasts();
          bool element = false;
          while (const auto *as = dyn_cast<ArraySubscriptExpr>(root)) {
            root = as->getBase()->IgnoreParenImpCasts();
            element = true;
          }
          const auto *d = dyn_cast<DeclRefExpr>(root);
          if (const auto *vd = d ? dyn_cast<VarDecl>(d->getDecl()) : nullptr)
            if (mentionsAny(bo->getRHS(), dep) || ((bo->isCompoundAssignmentOp() || element) && dep.count(vd)))
              dep.insert(vd);
          return;
        }
      if (const auto *is = dyn_cast<IfStmt>(s)) {
        flow(is->getThen());
        flow(is->getElse());
        return;
      }
      for (const Stmt *c : s->children())
        flow(c);
    };
    for (const Stmt *s : si.iter)
      flow(s);
    bool smooth = false, arrays = false;
    for (const VarDecl *v : si.state) {
      if (dep.count(v))
        smooth = true;
      if (!isFloatScalar(v))
        arrays = true;
    }
    // the state components: every scalar and every element of a state array
    for (const VarDecl *v : si.state) {
      std::string x = v->getNameAsString();
      std::string T = tstr(v->getType().getNonReferenceType().getUnqualifiedType());
      if (isFloatScalar(v)) {
        si.comps.push_back({x, x + "_b", x + "_d", T});
        continue;
      }
      QualType elem;
      uint64_t n = arrayCount(v, elem);
      std::vector<uint64_t> dims;
      for (QualType qt = v->getType().getCanonicalType(); const auto *cat = Ctx.getAsConstantArrayType(qt);
           qt = cat->getElementType())
        dims.push_back(cat->getSize().getZExtValue());
      for (uint64_t k = 0; k < n; ++k) {
        std::string idx;
        uint64_t rem = k, stride = n;
        for (uint64_t d : dims) {
          stride /= d;
          idx += "[" + std::to_string(rem / stride) + "]";
          rem %= stride;
        }
        si.comps.push_back({x + idx, x + "_b" + idx, x + "_d" + idx, tstr(elem)});
      }
    }
    if (si.comps.size() > 16)
      return "its state has more than 16 floating components";
    if (smooth) {
      if (si.state.size() > 1 && !arrays)
        findResiduals(loop, si, inner, dep);
      si.residual = !si.res.empty();
      return "";
    }
    if (arrays)
      return "its bracketing update iterates an array (bracketing loops need scalar bounds)";
    // bracketing: one if/else whose arms only assign the same trial local to state variables
    si.bracket = true;
    std::set<const VarDecl *> stateSet(si.state.begin(), si.state.end()), seen;
    for (const Stmt *s : si.iter) {
      std::set<const VarDecl *> r;
      const auto *is = dyn_cast<IfStmt>(s);
      if (!is) {
        if (!isa<DeclStmt>(s))
          return "its bracketing update is not of the form `if (r(m) > 0) a = m; else b = m;`";
        continue;
      }
      if (si.sel || !is->getElse())
        return "its bracketing update is not a single if/else";
      si.sel = is;
      for (const Stmt *arm : {is->getThen(), is->getElse()}) {
        std::vector<const Stmt *> as;
        if (const auto *cs = dyn_cast<CompoundStmt>(arm))
          as.assign(cs->body_begin(), cs->body_end());
        else
          as.push_back(arm);
        for (const Stmt *a : as) {
          const auto *bo = dyn_cast<BinaryOperator>(a);
          const auto *l = bo && bo->getOpcode() == BO_Assign ? dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()) : nullptr;
          const auto *r2 = bo ? dyn_cast<DeclRefExpr>(bo->getRHS()->IgnoreParenImpCasts()) : nullptr;
          const auto *lv = l ? dyn_cast<VarDecl>(l->getDecl()) : nullptr;
          const auto *rv = r2 ? dyn_cast<VarDecl>(r2->getDecl()) : nullptr;
          if (!lv || !rv || !stateSet.count(lv) || !inner.count(rv) || (si.trial && si.trial != rv))
            return "an arm of its bracketing branch does other than assign one trial point to a bound";
          si.trial = rv;
          seen.insert(lv);
        }
      }
      const auto *cmp = dyn_cast<BinaryOperator>(is->getCond()->IgnoreParenImpCasts());
      if (!cmp || !cmp->isRelationalOp() || !mentionsAny(cmp, dep))
        return "its bracketing condition is not a comparison that depends on the inputs";
    }
    if (!si.sel || !si.trial || seen.size() != stateSet.size())
      return "its bracketing branch does not move every iterated variable";
    // the trial point is computed from the bracket alone, first; the residual only from it and the inputs
    const auto *first = dyn_cast<DeclStmt>(si.iter.front());
    if (!first || !first->isSingleDecl() || first->getSingleDecl() != si.trial || dep.count(si.trial))
      return "its trial point is not the first statement, computed from the bracket alone";
    for (size_t i = 1; i < si.iter.size(); ++i)
      if (si.iter[i] != si.sel && mentionsAny(si.iter[i], stateSet))
        return "its residual reads the bracket other than through the trial point";
    return "";
  }

  // Candidate residuals of a multi-state root-finding loop. A value computed in one iteration (a top-level declaration
  // or a top-level assignment) is a candidate if it depends on the inputs and, through the body's declarations, on
  // exactly one iterated variable or one trial local that is itself computed from iterated variables alone (the point
  // at which the function is evaluated). Body locals that are reassigned, or values read from other iterations
  // through anything but that one variable, disqualify it.
  void findResiduals(const Stmt *, SolveInfo &si, const std::set<const VarDecl *> &inner,
                     const std::set<const VarDecl *> &dep) const {
    std::set<const VarDecl *> stateSet(si.state.begin(), si.state.end()), paramSet(si.params.begin(), si.params.end());
    std::map<const VarDecl *, const DeclStmt *> declOf;
    std::set<const VarDecl *> reassigned;
    for (const Stmt *s : si.iter)
      if (const auto *ds = dyn_cast<DeclStmt>(s))
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            declOf[vd] = ds;
    std::function<void(const Stmt *)> assigns = [&](const Stmt *s) {
      if (!s)
        return;
      const Expr *lhs = nullptr;
      if (const auto *bo = dyn_cast<BinaryOperator>(s))
        if (bo->isAssignmentOp())
          lhs = bo->getLHS();
      if (const auto *uo = dyn_cast<UnaryOperator>(s))
        if (uo->isIncrementDecrementOp())
          lhs = uo->getSubExpr();
      if (lhs)
        if (const auto *d = dyn_cast<DeclRefExpr>(lhs->IgnoreParenImpCasts()))
          if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
            if (inner.count(vd))
              reassigned.insert(vd);
      for (const Stmt *c : s->children())
        assigns(c);
    };
    for (const Stmt *s : si.iter)
      assigns(s);
    // walk an expression's dependencies: inputs, iterated variables, and body locals followed through their
    // initializers; a body local computed from iterated variables alone is a frontier point (a trial local)
    struct Walk { bool inputs = false, bad = false; std::set<const VarDecl *> frontier, sliceLocals, trialLocals; };
    std::function<void(const Expr *, Walk &)> walk;
    std::function<bool(const VarDecl *)> pureState = [&](const VarDecl *vd) {
      // a body local whose initializer reaches iterated variables but no inputs
      if (!declOf.count(vd) || reassigned.count(vd) || !vd->hasInit())
        return false;
      Walk w;
      walk(vd->getInit(), w);
      return !w.bad && !w.inputs;
    };
    walk = [&](const Expr *e, Walk &w) {
      std::set<const VarDecl *> refs;
      solveRefs(e, refs);
      for (const VarDecl *vd : refs) {
        if (paramSet.count(vd) || (!inner.count(vd) && !stateSet.count(vd) && Act.isVaried(vd))) {
          w.inputs = true;
        } else if (stateSet.count(vd)) {
          w.frontier.insert(vd);
        } else if (inner.count(vd)) {
          if (!declOf.count(vd) || reassigned.count(vd) || !vd->hasInit()) {
            if (vd->getType()->isRealFloatingType())
              w.bad = true; // a floating body local without a single definition
            else if (reassigned.count(vd) || !declOf.count(vd))
              w.bad = true; // e.g. the loop counter: the function would change between iterations
            continue;
          }
          if (pureState(vd)) {
            // a body local computed from the iterated variables alone: one that reads a single iterated variable
            // is part of the residual's expression (e.g. `double u = x * x;`), one that reads several is a trial
            // point (e.g. a secant or interpolation step)
            Walk w2;
            walk(vd->getInit(), w2);
            if (w2.frontier.size() == 1 && w2.trialLocals.empty()) {
              if (!w.sliceLocals.count(vd)) {
                w.sliceLocals.insert(vd);
                walk(vd->getInit(), w);
              }
              continue;
            }
            w.frontier.insert(vd);
            w.trialLocals.insert(vd);
            continue;
          }
          if (!w.sliceLocals.count(vd)) {
            w.sliceLocals.insert(vd);
            walk(vd->getInit(), w);
          }
        }
      }
    };
    auto inOrder = [&](const std::set<const VarDecl *> &vars) {
      std::vector<const Stmt *> out;
      for (const Stmt *s : si.iter)
        if (const auto *ds = dyn_cast<DeclStmt>(s)) {
          bool any = false;
          for (const Decl *d : ds->decls())
            if (const auto *vd = dyn_cast<VarDecl>(d))
              if (vars.count(vd))
                any = true;
          if (any)
            out.push_back(ds);
        }
      return out;
    };
    auto consider = [&](const VarDecl *var, const Expr *e) {
      if (!var || !e || !isFloatScalar(var) || !dep.count(var))
        return;
      Walk w;
      walk(e, w);
      if (w.bad || !w.inputs || w.frontier.size() != 1)
        return;
      SolveInfo::Residual r;
      r.var = var;
      r.expr = e;
      r.trialVar = *w.frontier.begin();
      r.slice = inOrder(w.sliceLocals);
      if (w.trialLocals.count(r.trialVar)) {
        // the trial local and the pure-state locals its initializer uses, recomputed from the exit state
        std::set<const VarDecl *> need{r.trialVar};
        std::function<void(const Expr *)> grow = [&](const Expr *x) {
          std::set<const VarDecl *> refs;
          solveRefs(x, refs);
          for (const VarDecl *vd : refs)
            if (declOf.count(vd) && !need.count(vd)) {
              need.insert(vd);
              grow(vd->getInit());
            }
        };
        grow(r.trialVar->getInit());
        r.trialDecls = inOrder(need);
      }
      si.res.push_back(r);
    };
    for (const Stmt *s : si.iter) {
      if (const auto *ds = dyn_cast<DeclStmt>(s)) {
        for (const Decl *d : ds->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            if (vd->hasInit() && !reassigned.count(vd))
              consider(vd, vd->getInit());
      } else if (const auto *bo = dyn_cast<BinaryOperator>(s)) {
        if (bo->getOpcode() == BO_Assign)
          if (const auto *d = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
            consider(dyn_cast<VarDecl>(d->getDecl()), bo->getRHS());
      }
    }
    // try first the values the loop tests (in a branch or exit condition) or carries to the next iteration: an
    // intermediate quantity that happens to vanish at the exit state is then never mistaken for the residual
    std::set<const VarDecl *> tested;
    std::function<void(const Stmt *)> conds = [&](const Stmt *s) {
      if (!s)
        return;
      if (const auto *is = dyn_cast<IfStmt>(s))
        solveRefs(is->getCond(), tested);
      for (const Stmt *c : s->children())
        conds(c);
    };
    for (const Stmt *s : si.iter)
      conds(s);
    for (const Expr *c : si.exitConds)
      solveRefs(c, tested);
    // the equation a loop solves must steer it: a candidate is kept only if a condition tests it or the update of an
    // iterated variable reads it (directly or through body locals); a value computed beside the iteration and never
    // used by it is not a relation the loop drives to zero, even if it happens to vanish at the exit state
    std::set<const VarDecl *> steers(tested.begin(), tested.end());
    for (bool grew = true; grew;) {
      grew = false;
      auto add = [&](const Expr *e) {
        std::set<const VarDecl *> r;
        solveRefs(e, r);
        for (const VarDecl *x : r)
          grew |= steers.insert(x).second;
      };
      std::function<void(const Stmt *)> scan = [&](const Stmt *s) {
        if (!s)
          return;
        if (const auto *bo = dyn_cast<BinaryOperator>(s))
          if (bo->isAssignmentOp())
            if (const auto *d = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
              if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
                if (stateSet.count(vd) || steers.count(vd))
                  add(bo->getRHS());
        if (const auto *ds = dyn_cast<DeclStmt>(s))
          for (const Decl *d : ds->decls())
            if (const auto *vd = dyn_cast<VarDecl>(d))
              if (vd->hasInit() && steers.count(vd))
                add(vd->getInit());
        for (const Stmt *c : s->children())
          scan(c);
      };
      for (const Stmt *s : si.iter)
        scan(s);
    }
    si.res.erase(std::remove_if(si.res.begin(), si.res.end(),
                                [&](const SolveInfo::Residual &r) { return !steers.count(r.var); }),
                 si.res.end());
    std::stable_sort(si.res.begin(), si.res.end(), [&](const SolveInfo::Residual &x, const SolveInfo::Residual &y) {
      auto rank = [&](const SolveInfo::Residual &r) { return (tested.count(r.var) || stateSet.count(r.var)) ? 0 : 1; };
      return rank(x) < rank(y);
    });
  }

  // Adjoint of a multi-state root-finding loop through the residual found at run time (see findResiduals).
  std::string residualAdjoint(const Stmt *fs, const SolveInfo &si, int depth) const {
    const std::string ind(depth * 2, ' '), i1 = ind + "  ", i2 = i1 + "  ", i3 = i2 + "  ";
    const std::string id = std::to_string(tmp_++);
    const std::string line = std::to_string(SM.getSpellingLineNumber(fs->getBeginLoc()));
    auto T = [&](const VarDecl *v) { return tstr(v->getType().getNonReferenceType().getUnqualifiedType()); };
    std::string out = ind + "{ // cuad: implicit adjoint of the root-finding loop at line " + line +
                      ": the result is a root of the residual that vanishes at the exit state, among";
    for (const auto &r : si.res)
      out += " " + r.var->getNameAsString() + "(" + r.trialVar->getNameAsString() + ")";
    out += "\n";
    // every candidate is evaluated at the exit state: whether it vanishes, at which point, and the derivative of the
    // root it implies (-R_p / R_t for each input); several vanishing candidates must agree on both
    const size_t K = si.res.size(), P = std::max<size_t>(si.params.size(), 1);
    const std::string Ks = std::to_string(K), Ps = std::to_string(P);
    const std::string use = "_cuad_use" + id, X = "_cuad_x" + id, RT = "_cuad_rt" + id, V = "_cuad_v" + id,
                      XK = "_cuad_xk" + id, RK = "_cuad_rk" + id, D = "_cuad_d" + id;
    out += i1 + "int " + use + " = -1, " + V + "[" + Ks + "];\n" + i1 + "double " + X + " = 0, " + RT + " = 0, " + XK + "[" +
           Ks + "], " + RK + "[" + Ks + "], " + D + "[" + Ks + "][" + Ps + "], _cuad_rv" + id + "[" + Ks + "];\n";
    auto localsOf = [&](const std::vector<const Stmt *> &stmts) {
      std::vector<const VarDecl *> v;
      for (const Stmt *s : stmts)
        for (const Decl *d : cast<DeclStmt>(s)->decls())
          if (const auto *vd = dyn_cast<VarDecl>(d))
            if (Act.isVaried(vd) && isFloatScalar(vd))
              v.push_back(vd);
      return v;
    };
    // one candidate: recompute its trial and slice from the exit state; seed `seed` into the residual and reverse
    // the slice; the trial's adjoint is left in `trialOut` (or discarded)
    auto pass = [&](const SolveInfo::Residual &r, const std::string &seed, const std::string &trialOut, bool shadowParams,
                    const std::string &in, int dp) {
      std::string o;
      for (const Stmt *s : r.trialDecls)
        o += genPrimal(s, dp);
      for (const Stmt *s : r.slice)
        o += genPrimal(s, dp);
      if (shadowParams)
        o += shadowAdj(si.params, in);
      o += in + T(r.trialVar) + " " + r.trialVar->getNameAsString() + "_b = 0;\n";
      o += shadowAdj(localsOf(r.slice), in);
      pushAdj(r.expr, seed, o, in);
      for (auto it = r.slice.rbegin(); it != r.slice.rend(); ++it)
        o += reverseStmt(*it, dp);
      if (!trialOut.empty())
        o += in + trialOut + " = " + r.trialVar->getNameAsString() + "_b;\n";
      return o;
    };
    bool saved = noPush_;
    noPush_ = true;
    for (size_t k = 0; k < K; ++k) {
      const auto &r = si.res[k];
      const std::string kk = std::to_string(k), t = r.trialVar->getNameAsString();
      out += i1 + "{ // candidate " + r.var->getNameAsString() + "(" + t + ")\n";
      out += i2 + "double _cuad_R, _cuad_Rt;\n" + i2 + V + "[" + kk + "] = 0;\n" + i2 + "{\n";
      std::string o;
      for (const Stmt *s : r.trialDecls)
        o += genPrimal(s, depth + 3);
      for (const Stmt *s : r.slice)
        o += genPrimal(s, depth + 3);
      out += o + i3 + "_cuad_R = " + prim(r.expr) + ";\n" + i2 + "}\n" + i2 + "_cuad_rv" + id + "[" + kk +
             "] = _cuad_R;\n";
      // R_t and R_p: the residual's adjoint for the seed 1, with the inputs' adjoints shadowed
      std::string cp;
      for (size_t j = 0; j < si.params.size(); ++j)
        cp += i3 + D + "[" + kk + "][" + std::to_string(j) + "] = " + si.params[j]->getNameAsString() + "_b;\n";
      if (si.params.empty())
        cp += i3 + D + "[" + kk + "][0] = 0;\n";
      out += i2 + "{\n" + pass(r, "1", "_cuad_Rt", true, i3, depth + 3) + cp + i2 + "}\n";
      out += i2 + "{\n";
      for (const Stmt *s : r.trialDecls)
        out += genPrimal(s, depth + 3);
      out += i3 + "if (_cuad_Rt != 0 && fabs(_cuad_R) <= fabs(_cuad_Rt) * (CUAD_SOLVE_TOL * fabs((double)" + t +
             ") + CUAD_SOLVE_ATOL)) {\n";
      out += i3 + "  " + V + "[" + kk + "] = 1; " + XK + "[" + kk + "] = " + t + "; " + RK + "[" + kk + "] = _cuad_Rt;\n";
      out += i3 + "  for (int _cuad_j = 0; _cuad_j < " + Ps + "; ++_cuad_j) " + D + "[" + kk + "][_cuad_j] = -" + D + "[" +
             kk + "][_cuad_j] / _cuad_Rt;\n";
      out += i3 + "}\n" + i2 + "}\n" + i1 + "}\n";
    }
    // the first vanishing candidate defines the root; any other vanishing candidate must locate the same root and
    // imply the same derivative, or the relation the loop solved is ambiguous (e.g. x - a and x - b at a = b)
    out += i1 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) {\n";
    out += i2 + "if (!" + V + "[_cuad_k]) continue;\n";
    out += i2 + "if (" + use + " < 0) { " + use + " = _cuad_k; " + X + " = " + XK + "[_cuad_k]; " + RT + " = " + RK +
           "[_cuad_k]; continue; }\n";
    out += i2 + "double _cuad_dm = 0, _cuad_dd = 0;\n";
    out += i2 + "for (int _cuad_j = 0; _cuad_j < " + Ps + "; ++_cuad_j) {\n";
    out += i3 + "_cuad_dm = fmax(_cuad_dm, fmax(fabs(" + D + "[" + use + "][_cuad_j]), fabs(" + D + "[_cuad_k][_cuad_j])));\n";
    out += i3 + "_cuad_dd = fmax(_cuad_dd, fabs(" + D + "[" + use + "][_cuad_j] - " + D + "[_cuad_k][_cuad_j]));\n" + i2 + "}\n";
    out += i2 + "if (fabs(" + XK + "[_cuad_k] - " + X + ") > CUAD_SOLVE_AGREE * fmax(fabs(" + XK + "[_cuad_k]), fabs(" + X +
           ")) + CUAD_SOLVE_ATOL || _cuad_dd > CUAD_SOLVE_AGREE * _cuad_dm) cuad_solve_fail();\n";
    out += i1 + "}\n";
    out += i1 + "if (" + use + " < 0) cuad_solve_fail();\n";
    // the seeds of the iterated variables: those at the root are summed; a variable that holds a vanishing candidate
    // residual is zero along the solution (the candidates agree) and carries no derivative; any other seeded
    // variable, including one that only happens to be zero here, means the result is not a function of the root
    out += i1 + "double _cuad_s" + id + " = 0;\n";
    for (const VarDecl *v : si.state) {
      std::string x = v->getNameAsString(), held;
      for (size_t k = 0; k < K; ++k)
        if (si.res[k].var == v)
          held += (held.empty() ? "" : " || ") + V + "[" + std::to_string(k) + "]";
      out += i1 + "if (" + x + "_b != 0) {\n";
      out += i2 + "if (fabs((double)" + x + " - " + X + ") <= CUAD_SOLVE_TOL * fabs(" + X + ") + CUAD_SOLVE_ATOL) _cuad_s" +
             id + " += " + x + "_b;\n";
      if (!held.empty())
        out += i2 + "else if ((" + held + ") && fabs((double)" + x + ") <= fabs(" + RT + ") * (CUAD_SOLVE_TOL * fabs(" + X +
               ") + CUAD_SOLVE_ATOL)) {}\n";
      out += i2 + "else cuad_solve_fail();\n";
      out += i2 + x + "_b = 0;\n" + i1 + "}\n";
    }
    for (size_t k = 0; k < si.res.size(); ++k) {
      const auto &r = si.res[k];
      const std::string kk = std::to_string(k), t = r.trialVar->getNameAsString(), Tt = T(r.trialVar);
      out += i1 + "if (_cuad_use" + id + " == " + kk + ") {\n";
      // conditioning: -R_p / R_t once more at the far end of the interval |R / R_t| in which the root is known
      out += i2 + "if (CUAD_SOLVE_CHECK) {\n" + i3 + "double _cuad_e[" + Ps + "] = {0};\n";
      for (const Stmt *s : r.trialDecls)
        out += genPrimal(s, depth + 3);
      out += i3 + "const double _cuad_tt = (double)" + t + " + fabs(" + "_cuad_rv" + id + "[" + kk + "] / " + RK + "[" +
             kk + "]) + 4.440892098500626e-16 * fabs((double)" + t + ");\n";
      out += i3 + "{\n" + i3 + "  " + Tt + " " + t + " = (" + Tt + ")_cuad_tt;\n";
      for (const Stmt *s : r.slice)
        out += genPrimal(s, depth + 4);
      out += shadowAdj(si.params, i3 + "  ") + i3 + "  " + Tt + " " + t + "_b = 0;\n" +
             shadowAdj(localsOf(r.slice), i3 + "  ");
      {
        std::string o;
        pushAdj(r.expr, "1", o, i3 + "  ");
        for (auto it = r.slice.rbegin(); it != r.slice.rend(); ++it)
          o += reverseStmt(*it, depth + 4);
        out += o;
      }
      for (size_t q = 0; q < si.params.size(); ++q)
        out += i3 + "  _cuad_e[" + std::to_string(q) + "] = -" + si.params[q]->getNameAsString() + "_b / " + t + "_b;\n";
      out += i3 + "}\n" + i3 + "double _cuad_en = 0, _cuad_ed = 0;\n" + i3 + "for (int _cuad_q = 0; _cuad_q < " + Ps +
             "; ++_cuad_q) { _cuad_en = fmax(_cuad_en, fmax(fabs(_cuad_e[_cuad_q]), fabs(" + D + "[" + kk +
             "][_cuad_q]))); _cuad_ed = fmax(_cuad_ed, fabs(_cuad_e[_cuad_q] - " + D + "[" + kk + "][_cuad_q])); }\n" + i3 +
             "if (!(_cuad_ed <= CUAD_SOLVE_AGREE * _cuad_en) || !(_cuad_en < 1e300)) cuad_solve_fail();\n" + i2 + "}\n";
      out += pass(r, "-(_cuad_s" + id + ") / (_cuad_rt" + id + ")", "", false, i2, depth + 2) + i1 + "}\n";
    }
    noPush_ = saved;
    return out + ind + "}\n";
  }


  // declarations `T x_b = 0;` that hide the given adjoints inside a block
  std::string shadowAdj(const std::vector<const VarDecl *> &vs, const std::string &ind) const {
    std::string out;
    for (const VarDecl *v : vs)
      out += ind + tstr(v->getType().getNonReferenceType().getUnqualifiedType()) + " " + v->getNameAsString() +
             "_b = 0;\n";
    return out;
  }

  // One reverse sweep of one iteration with the seed e_j on the state: row j of G_x into J[j][.], of G_p into P[j][.].
  std::string jacobianSweep(const SolveInfo &si, size_t j, const std::string &J, const std::string &P,
                            int depth) const {
    const std::string in((depth + 1) * 2, ' ');
    std::string o = std::string(depth * 2, ' ') + "{\n" + shadowAdj(si.params, in);
    size_t k = 0;
    for (const VarDecl *v : si.state) {
      std::string x = v->getNameAsString();
      if (isFloatScalar(v)) {
        o += in + tstr(v->getType().getNonReferenceType().getUnqualifiedType()) + " " + x + "_b = " +
             (k == j ? "1" : "0") + ";\n";
        ++k;
        continue;
      }
      QualType elem;
      std::string dims = arrayDims(v->getType(), elem), init;
      uint64_t n = arrayCount(v, elem);
      for (uint64_t e = 0; e < n; ++e, ++k)
        init += std::string(e ? ", " : "") + (k == j ? "1" : "0");
      o += in + tstr(elem) + " " + x + "_b" + dims + " = {" + init + "};\n";
    }
    o += genBlock(si.iter, depth + 1);
    for (size_t c = 0; c < si.comps.size(); ++c)
      o += in + J + "[" + std::to_string(j) + "][" + std::to_string(c) + "] = " + si.comps[c].b + ";\n";
    for (size_t q = 0; q < si.params.size(); ++q)
      o += in + P + "[" + std::to_string(j) + "][" + std::to_string(q) + "] = " + si.params[q]->getNameAsString() +
           "_b;\n";
    return o + std::string(depth * 2, ' ') + "}\n";
  }

  std::string solveAdjoint(const Stmt *fs, const SolveInfo &si, int depth) const {
    if (si.residual)
      return residualAdjoint(fs, si, depth);
    const std::string ind(depth * 2, ' '), i1 = ind + "  ", i2 = i1 + "  ", i3 = i2 + "  ";
    const std::string id = std::to_string(tmp_++);
    const std::string line = std::to_string(SM.getSpellingLineNumber(fs->getBeginLoc()));
    auto T = [&](const VarDecl *v) { return tstr(v->getType().getNonReferenceType().getUnqualifiedType()); };
    bool saved = noPush_;
    noPush_ = false; // one iteration's primal must push what its own reverse pops
    std::string out;
    if (!si.bracket) {
      // x* = G(x*, p): with J = G_x and P = G_p from one reverse sweep per state component, the distance of the exit
      // state to the fixed point solves (I - J) d = G(x) - x and the adjoint solves (I - J)^T l = s; the inputs get
      // P^T l. The same is evaluated once more at x + d: the derivative must not change across the interval in
      // which the solution is known (a double root, a degenerate fixed point).
      const size_t K = si.comps.size(), NP = std::max<size_t>(si.params.size(), 1);
      const std::string Ks = std::to_string(K), Ps = std::to_string(NP);
      const std::string S = "_cuad_s" + id, X = "_cuad_x" + id, R = "_cuad_r" + id, J = "_cuad_J" + id,
                        P = "_cuad_P" + id, M = "_cuad_M" + id, D = "_cuad_d" + id, L = "_cuad_l" + id,
                        G = "_cuad_g" + id, G2 = "_cuad_h" + id;
      out += ind + "{ // cuad: implicit adjoint of the convergence loop at line " + line +
             ": (I - G_x)^T l = s solved directly, then the inputs get G_p^T l\n";
      std::string sInit, xInit;
      for (size_t k = 0; k < K; ++k) {
        sInit += std::string(k ? ", " : "") + "(double)" + si.comps[k].b;
        xInit += std::string(k ? ", " : "") + "(double)" + si.comps[k].x;
      }
      out += i1 + "double " + S + "[" + Ks + "] = {" + sInit + "}, " + X + "[" + Ks + "] = {" + xInit + "}, " + R + "[" +
             Ks + "], " + J + "[" + Ks + "][" + Ks + "], " + P + "[" + Ks + "][" + Ps + "], " + M + "[" + Ks + "][" + Ks +
             "], " + D + "[" + Ks + "], " + L + "[" + Ks + "], " + G + "[" + Ps + "] = {0}, " + G2 + "[" + Ps +
             "] = {0};\n";
      for (const auto &c : si.comps)
        out += i1 + c.b + " = 0;\n";
      // G(x) - x: one more iteration from the exit state, which is restored afterwards (checks only)
      bool np = noPush_;
      noPush_ = true;
      out += i1 + "if (CUAD_SOLVE_CHECK) {\n" + i1 + "{\n";
      for (const Stmt *s : si.iter)
        out += genPrimal(s, depth + 2);
      out += i1 + "}\n";
      noPush_ = np;
      for (size_t k = 0; k < K; ++k) {
        const std::string ks = std::to_string(k);
        out += i1 + R + "[" + ks + "] = (double)" + si.comps[k].x + " - " + X + "[" + ks + "]; " + si.comps[k].x + " = (" +
               si.comps[k].T + ")" + X + "[" + ks + "];\n";
      }
      out += i1 + "}\n";
      // J and P at the current state, then l and g = P^T l; `into` receives g
      auto derivative = [&](const std::string &into, bool distance) {
        std::string o;
        for (size_t j = 0; j < K; ++j)
          o += jacobianSweep(si, j, J, P, depth + 1);
        if (si.params.empty())
          for (size_t j = 0; j < K; ++j)
            o += i1 + P + "[" + std::to_string(j) + "][0] = 0;\n";
        if (distance) {
          o += i1 + "if (CUAD_SOLVE_CHECK) {\n";
          o += i1 + "for (int _cuad_i = 0; _cuad_i < " + Ks + "; ++_cuad_i) {\n" + i2 + D + "[_cuad_i] = " + R +
               "[_cuad_i];\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) " + M +
               "[_cuad_i][_cuad_k] = (_cuad_i == _cuad_k) - " + J + "[_cuad_i][_cuad_k];\n" + i1 + "}\n";
          o += i1 + "if (!cuad_lsolve(" + M + ", " + D + ")) cuad_solve_fail();\n";
          o += i1 + "{\n" + i2 + "double _cuad_dn = 0, _cuad_xn = 0;\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks +
               "; ++_cuad_k) { _cuad_dn = fmax(_cuad_dn, fabs(" + D + "[_cuad_k])); _cuad_xn = fmax(_cuad_xn, fabs(" + X +
               "[_cuad_k])); }\n" + i2 + "if (_cuad_dn > CUAD_SOLVE_TOL * _cuad_xn + CUAD_SOLVE_ATOL) cuad_solve_fail();\n" +
               i1 + "}\n" + i1 + "}\n";
        }
        o += i1 + "for (int _cuad_i = 0; _cuad_i < " + Ks + "; ++_cuad_i) {\n" + i2 + L + "[_cuad_i] = " + S +
             "[_cuad_i];\n" + i2 + "for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) " + M +
             "[_cuad_i][_cuad_k] = (_cuad_i == _cuad_k) - " + J + "[_cuad_k][_cuad_i];\n" + i1 + "}\n";
        o += i1 + "if (!cuad_lsolve(" + M + ", " + L + ")) cuad_solve_fail();\n";
        o += i1 + "for (int _cuad_q = 0; _cuad_q < " + Ps + "; ++_cuad_q) {\n" + i2 + into + "[_cuad_q] = 0;\n" + i2 +
             "for (int _cuad_j = 0; _cuad_j < " + Ks + "; ++_cuad_j) { " + into + "[_cuad_q] += " + L + "[_cuad_j] * " +
             P + "[_cuad_j][_cuad_q];" +
             (distance ? " _cuad_a" + id + " += fabs(" + L + "[_cuad_j] * " + P + "[_cuad_j][_cuad_q]);" : "") + " }\n" +
             i1 + "}\n";
        return o;
      };
      out += i1 + "double _cuad_a" + id + " = 0; // the scale of the terms that make up the derivative\n";
      out += derivative(G, true);
      // conditioning: the same at the far side of the interval in which the solution is known. Where the iteration
      // has a vanishing derivative (Newton's method at a simple root) it converges quadratically, the exit state is
      // exact to rounding and this second evaluation is skipped; at a double root Newton's G_x is 1/2.
      out += i1 + "double _cuad_jm" + id + " = 0;\n" + i1 + "for (int _cuad_i = 0; _cuad_i < " + Ks +
             "; ++_cuad_i) for (int _cuad_k = 0; _cuad_k < " + Ks + "; ++_cuad_k) _cuad_jm" + id + " = fmax(_cuad_jm" + id +
             ", fabs(" + J + "[_cuad_i][_cuad_k]));\n";
      out += i1 + "if (CUAD_SOLVE_CHECK && (CUAD_SOLVE_CHECK > 1 || _cuad_jm" + id + " > 0.25 || !(_cuad_a" + id + " < 1e300))) {\n";
      for (size_t k = 0; k < K; ++k) {
        const std::string ks = std::to_string(k);
        out += i1 + si.comps[k].x + " = (" + si.comps[k].T + ")(" + X + "[" + ks + "] + " + D + "[" + ks +
               "] + copysign(4.440892098500626e-16 * fabs(" + X + "[" + ks + "]), " + D + "[" + ks + "]));\n";
      }
      out += derivative(G2, false);
      for (size_t k = 0; k < K; ++k)
        out += i1 + si.comps[k].x + " = (" + si.comps[k].T + ")" + X + "[" + std::to_string(k) + "];\n";
      out += i1 + "{\n" + i2 + "double _cuad_gd = 0;\n" + i2 + "for (int _cuad_q = 0; _cuad_q < " + Ps +
             "; ++_cuad_q) _cuad_gd = fmax(_cuad_gd, fabs(" + G + "[_cuad_q] - " + G2 + "[_cuad_q]));\n" + i2 +
             "if (!(_cuad_gd <= CUAD_SOLVE_AGREE * _cuad_a" + id + ") || !(_cuad_a" + id + " < 1e300)) cuad_solve_fail();\n" + i1 + "}\n";
      out += i1 + "}\n";
      for (size_t q = 0; q < si.params.size(); ++q)
        out += i1 + si.params[q]->getNameAsString() + "_b += " + G + "[" + std::to_string(q) + "];\n";
      out += ind + "}\n";
      (void)T;
      noPush_ = saved;
      return out;
    }
    // bracketing
    const auto *cmp = cast<BinaryOperator>(si.sel->getCond()->IgnoreParenImpCasts());
    const std::string m = si.trial->getNameAsString(), Tm = T(si.trial);
    out += ind + "{ // cuad: implicit adjoint of the bracketing loop at line " + line + ": its result is the root of r = (" +
           prim(cmp->getLHS()) + ") - (" + prim(cmp->getRHS()) + ") in " + m + "; the inputs get -r_p s / r_" + m + "\n";
    std::string sum;
    for (const VarDecl *v : si.state)
      sum += (sum.empty() ? "" : " + ") + v->getNameAsString() + "_b";
    { // the loop must have left with a collapsed bracket: every bound equal to within CUAD_SOLVE_TOL
      std::string lo, hi;
      for (const VarDecl *v : si.state) {
        std::string x = "(double)" + v->getNameAsString();
        lo = lo.empty() ? x : "fmin(" + lo + ", " + x + ")";
        hi = hi.empty() ? x : "fmax(" + hi + ", " + x + ")";
      }
      out += i1 + "if (CUAD_SOLVE_CHECK && " + hi + " - " + lo + " > CUAD_SOLVE_TOL * fmax(fabs(" + hi + "), fabs(" + lo +
             ")) + CUAD_SOLVE_ATOL) cuad_solve_fail();\n";
    }
    out += i1 + Tm + " _cuad_s" + id + " = " + sum + ";\n";
    for (const VarDecl *v : si.state)
      out += i1 + v->getNameAsString() + "_b = 0;\n";
    std::vector<const VarDecl *> locals;
    for (const Stmt *s : si.iter)
      if (s != si.sel) {
        out += genPrimal(s, depth + 1);
        if (const auto *ds = dyn_cast<DeclStmt>(s))
          for (const Decl *d : ds->decls())
            if (const auto *vd = dyn_cast<VarDecl>(d))
              if (Act.isVaried(vd) && isFloatScalar(vd))
                locals.push_back(vd);
      }
    auto residualReverse = [&](const std::string &seed, const std::string &in, int dep) {
      std::string o;
      for (const VarDecl *v : locals)
        o += in + v->getNameAsString() + "_b = 0;\n";
      pushAdj(cmp->getLHS(), seed, o, in);
      pushAdj(cmp->getRHS(), "-(" + seed + ")", o, in);
      for (size_t i = si.iter.size(); i-- > 1;)
        if (si.iter[i] != si.sel)
          o += reverseStmt(si.iter[i], dep);
      return o;
    };
    out += shadowAdj(locals, i1);
    {
      const std::string Ps = std::to_string(std::max<size_t>(si.params.size(), 1));
      std::string rec;
      for (size_t q = 0; q < si.params.size(); ++q)
        rec += i2 + "_cuad_c" + id + "[" + std::to_string(q) + "] = -" + si.params[q]->getNameAsString() + "_b / " + m +
               "_b;\n";
      out += i1 + Tm + " _cuad_r" + id + ";\n" + i1 + "double _cuad_c" + id + "[" + Ps + "] = {0};\n" + i1 + "{\n" +
             shadowAdj(si.params, i2) + residualReverse("1", i2, depth + 2) + i2 + "_cuad_r" + id + " = " + m + "_b;\n" +
             rec + i1 + "}\n";
    }
    // A collapsed bracket is a root only if the residual vanishes there. If the root lies outside the initial bracket
    // (a bound then acts as a floor or a ceiling), the moving bounds close onto the one that never moved, and the
    // result is that bound: its derivative is the derivative of the program. The branch taken at the collapsed point
    // names the bounds that kept moving; the others receive the seed and pass it to their initial values.
    out += i1 + "if (fabs((double)((" + prim(cmp->getLHS()) + ") - (" + prim(cmp->getRHS()) +
           "))) <= fabs((double)_cuad_r" + id + ") * (CUAD_SOLVE_TOL * fabs((double)" + m + ") + CUAD_SOLVE_ATOL)) {\n";
    { // conditioning: the derivative -r_p / r_m at the upper end of the collapsed bracket must agree with the one at
      // its midpoint (a double root, where r_m vanishes, does not)
      const std::string Ps = std::to_string(std::max<size_t>(si.params.size(), 1));
      std::string lo, hi;
      for (const VarDecl *v : si.state) {
        std::string x = "(double)" + v->getNameAsString();
        lo = lo.empty() ? x : "fmin(" + lo + ", " + x + ")";
        hi = hi.empty() ? x : "fmax(" + hi + ", " + x + ")";
      }
      out += i2 + "if (CUAD_SOLVE_CHECK) {\n" + i2 + "double _cuad_e" + id + "[2][" + Ps + "] = {{0}};\n";
      out += i2 + "for (int _cuad_q = 0; _cuad_q < " + Ps + "; ++_cuad_q) _cuad_e" + id + "[0][_cuad_q] = _cuad_c" + id +
             "[_cuad_q];\n";
      for (int e = 1; e < 2; ++e) {
        out += i2 + "{\n" + i3 + "const double _cuad_at = " + (e ? hi : lo) + ";\n";
        for (const VarDecl *v : si.state)
          out += i3 + T(v) + " " + v->getNameAsString() + " = (" + T(v) + ")_cuad_at;\n";
        for (const Stmt *s : si.iter)
          if (s != si.sel)
            out += genPrimal(s, depth + 3);
        out += shadowAdj(locals, i3) + shadowAdj(si.params, i3) + residualReverse("1", i3, depth + 3);
        for (size_t q = 0; q < si.params.size(); ++q)
          out += i3 + "_cuad_e" + id + "[" + std::to_string(e) + "][" + std::to_string(q) + "] = -" +
                 si.params[q]->getNameAsString() + "_b / " + m + "_b;\n";
        out += i2 + "}\n";
      }
      out += i2 + "{\n" + i3 + "double _cuad_en = 0, _cuad_ed = 0;\n" + i3 + "for (int _cuad_q = 0; _cuad_q < " + Ps +
             "; ++_cuad_q) { _cuad_en = fmax(_cuad_en, fmax(fabs(_cuad_e" + id + "[0][_cuad_q]), fabs(_cuad_e" + id +
             "[1][_cuad_q]))); _cuad_ed = fmax(_cuad_ed, fabs(_cuad_e" + id + "[0][_cuad_q] - _cuad_e" + id +
             "[1][_cuad_q])); }\n" + i3 + "if (!(_cuad_ed <= CUAD_SOLVE_AGREE * _cuad_en) || !(_cuad_en < 1e300)) cuad_solve_fail();\n" + i2 +
             "}\n" + i2 + "}\n";
    }
    out += residualReverse("-(_cuad_s" + id + ") / (_cuad_r" + id + ")", i2, depth + 2);
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
      std::vector<const VarDecl *> fixed;
      for (const VarDecl *v : si.state)
        if (!moving.count(v))
          fixed.push_back(v);
      std::string o;
      for (const VarDecl *v : fixed)
        o += i3 + v->getNameAsString() + "_b = (_cuad_s" + id + ") / " + std::to_string(fixed.size()) + ".0;\n";
      return o;
    };
    out += i1 + "} else { // the root is outside the bracket: the result is the bound that never moved\n";
    out += i2 + "if (" + prim(si.sel->getCond()) + ") {\n" + pinned(thenSet) + i2 + "} else {\n" + pinned(elseSet) +
           i2 + "}\n" + i1 + "}\n";
    out += ind + "}\n";
    noPush_ = saved;
    return out;
  }
  // the element count of a (possibly multi-dimensional) local array and an lvalue for its flat element k
  uint64_t arrayCount(const VarDecl *vd, QualType &elem) const {
    uint64_t n = 1;
    QualType qt = vd->getType().getCanonicalType();
    while (const auto *cat = Ctx.getAsConstantArrayType(qt)) {
      n *= cat->getSize().getZExtValue();
      qt = cat->getElementType();
    }
    elem = qt;
    return n;
  }
  // ---- the region a loop iteration can write: one index interval per write site ----
  //
  // Checkpointing a carried array whole costs its declared size per iteration (an LSTM state declared
  // [256] of which 2*l*b elements are used). Only the elements the iteration may write need saving: the
  // others hold the same value at the top of every iteration and through every re-run in the reverse sweep.
  // For each write a[idx] the interval of idx is computed symbolically from the ranges of the variables in
  // it and evaluated at run time at the top of the iteration - identically in the forward and the backward
  // loop, because it may use only the loop's own induction variable, induction variables of loops enclosing
  // it (fixed while it runs and while it is reversed), variables the function never modifies, and the
  // induction variables (and initialized locals) of loops inside the body, through their bounds. Any other
  // index - an array read, a call, %, / - makes the whole array the region.
  using Interval = std::pair<std::string, std::string>; // [lo, hi] as C expressions
  const Stmt *curBody_ = nullptr;
  mutable bool rangeWhole_ = false; // range L's own counter over all its iterations (evaluated at loop entry)
  // v is assigned, incremented or has its address taken in s (its declaration does not count)
  static bool reassignedIn(const Stmt *s, const VarDecl *v) {
    if (!s)
      return false;
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp())
        if (const auto *d = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreParenImpCasts()))
          if (d->getDecl() == v)
            return true;
    if (const auto *uo = dyn_cast<UnaryOperator>(s))
      if (uo->isIncrementDecrementOp() || uo->getOpcode() == UO_AddrOf)
        if (const auto *d = dyn_cast<DeclRefExpr>(uo->getSubExpr()->IgnoreParenImpCasts()))
          if (d->getDecl() == v)
            return true;
    for (const Stmt *c : s->children())
      if (reassignedIn(c, v))
        return true;
    return false;
  }
  static bool isPoint(const Interval &r) { return r.first == r.second; }
  static bool asInt(const std::string &s, long long &v) {
    if (s.empty())
      return false;
    size_t i = s[0] == '-' ? 1 : 0;
    if (i == s.size())
      return false;
    for (size_t j = i; j < s.size(); ++j)
      if (s[j] < '0' || s[j] > '9')
        return false;
    v = std::stoll(s);
    return true;
  }
  // x op y as a C expression, folded when both are integer literals and simplified for 0 and 1
  static std::string foldOp(const std::string &x, char op, const std::string &y) {
    long long a, b;
    bool ka = asInt(x, a), kb = asInt(y, b);
    if (ka && kb)
      return std::to_string(op == '+' ? a + b : op == '-' ? a - b : a * b);
    if (op == '+' && ka && a == 0) return y;
    if ((op == '+' || op == '-') && kb && b == 0) return x;
    if (op == '*' && ((ka && a == 0) || (kb && b == 0))) return "0";
    if (op == '*' && ka && a == 1) return y;
    if (op == '*' && kb && b == 1) return x;
    auto wrap = [](const std::string &s) {
      long long t;
      bool simple = (asInt(s, t) && t >= 0) ||
                    (!s.empty() && std::all_of(s.begin(), s.end(), [](char ch) {
                       return std::isalnum((unsigned char)ch) || ch == '_';
                     }));
      if (!simple && s.size() > 2 && s.front() == '(' && s.back() == ')') {
        int depth = 0;
        simple = true;           // already one parenthesised group?
        for (size_t i = 0; i < s.size(); ++i) {
          depth += s[i] == '(' ? 1 : s[i] == ')' ? -1 : 0;
          if (depth == 0 && i + 1 < s.size()) { simple = false; break; }
        }
      }
      return simple ? s : "(" + s + ")";
    };
    return wrap(x) + " " + op + " " + wrap(y);
  }
  // the for-statement (within s) whose init declares v, or null
  static const ForStmt *loopDeclaring(const Stmt *s, const VarDecl *v) {
    if (!s)
      return nullptr;
    if (const auto *fs = dyn_cast<ForStmt>(s))
      if (const auto *ds = dyn_cast_or_null<DeclStmt>(fs->getInit()))
        for (const Decl *d : ds->decls())
          if (d == v)
            return fs;
    for (const Stmt *c : s->children())
      if (const ForStmt *r = loopDeclaring(c, v))
        return r;
    return nullptr;
  }
  // the declaration statement (within s) of v, or null
  static const VarDecl *declIn(const Stmt *s, const VarDecl *v) {
    if (!s)
      return nullptr;
    if (const auto *ds = dyn_cast<DeclStmt>(s))
      for (const Decl *d : ds->decls())
        if (d == v)
          return v;
    for (const Stmt *c : s->children())
      if (declIn(c, v))
        return v;
    return nullptr;
  }
  bool rangeOf(const Expr *e, const ForStmt *L, Interval &r, int depth = 0) const {
    if (!e || depth > 16)
      return false;
    e = e->IgnoreParenImpCasts();
    if (!e->getType()->isIntegerType())
      return false;
    Expr::EvalResult cv;
    if (e->EvaluateAsInt(cv, Ctx)) {
      r = {std::to_string(cv.Val.getInt().getExtValue()), std::to_string(cv.Val.getInt().getExtValue())};
      return true;
    }
    if (const auto *d = dyn_cast<DeclRefExpr>(e)) {
      const auto *v = dyn_cast<VarDecl>(d->getDecl());
      if (!v)
        return false;
      std::string nm = v->getNameAsString();
      const ForStmt *own = nullptr;     // L's own counter, ranged over the whole loop (entry snapshots)
      if (v == inductionVar(L)) {
        if (!rangeWhole_) {             // per-iteration checkpoint: its value at the top of the iteration
          r = {nm, nm};
          return true;
        }
        own = L;
      }
      if (withinRange(L->getSourceRange(), v->getLocation())) { // declared in the loop
        if (const ForStmt *inner = own ? own : loopDeclaring(L->getBody(), v)) {
          if (inductionVar(inner) != v)
            return false;
          const auto *cond =
              inner->getCond() ? dyn_cast<BinaryOperator>(inner->getCond()->IgnoreImpCasts()) : nullptr;
          if (!cond || (cond->getOpcode() != BO_LT && cond->getOpcode() != BO_LE))
            return false;
          const auto *cl = dyn_cast<DeclRefExpr>(cond->getLHS()->IgnoreParenImpCasts());
          if (!cl || cl->getDecl() != v)
            return false;
          const Expr *inc = inner->getInc() ? inner->getInc()->IgnoreParenImpCasts() : nullptr;
          bool up = false;
          if (const auto *u = dyn_cast_or_null<UnaryOperator>(inc))
            up = u->isIncrementOp();
          else if (const auto *b = dyn_cast_or_null<BinaryOperator>(inc)) {
            Expr::EvalResult s;
            up = b->getOpcode() == BO_AddAssign && b->getRHS()->EvaluateAsInt(s, Ctx) &&
                 s.Val.getInt().isStrictlyPositive();
          }
          Interval a, bnd;
          if (!up || !rangeOf(v->getInit(), L, a, depth + 1) || !rangeOf(cond->getRHS(), L, bnd, depth + 1))
            return false;
          r = {a.first, cond->getOpcode() == BO_LT ? foldOp(bnd.second, '-', "1") : bnd.second};
          return true;
        }
        // an initialized local of the body that is never reassigned: the range of its initializer
        if (v->hasInit() && !v->getType()->isReferenceType() && declIn(L->getBody(), v))
          return !reassignedIn(L->getBody(), v) && rangeOf(v->getInit(), L, r, depth + 1);
        return false;
      }
      if (v->getType()->isReferenceType())
        return false;
      // declared outside: the counter of an enclosing loop (fixed while L runs and while it is reversed) ...
      if (const ForStmt *outer = loopDeclaring(curBody_, v))
        if (inductionVar(outer) == v && withinRange(outer->getSourceRange(), L->getBeginLoc())) {
          r = {nm, nm};
          return true;
        }
      // ... or a parameter / initialized local that the function never reassigns
      if ((isa<ParmVarDecl>(v) || v->hasInit()) && !reassignedIn(curBody_, v)) {
        r = {nm, nm};
        return true;
      }
      return false;
    }
    if (const auto *u = dyn_cast<UnaryOperator>(e)) {
      Interval a;
      if (u->getOpcode() == UO_Plus)
        return rangeOf(u->getSubExpr(), L, r, depth + 1);
      if (u->getOpcode() == UO_Minus && rangeOf(u->getSubExpr(), L, a, depth + 1)) {
        r = {foldOp("0", '-', a.second), foldOp("0", '-', a.first)};
        return true;
      }
      return false;
    }
    if (const auto *b = dyn_cast<BinaryOperator>(e)) {
      Interval a, c;
      if (!rangeOf(b->getLHS(), L, a, depth + 1) || !rangeOf(b->getRHS(), L, c, depth + 1))
        return false;
      auto add = [](const std::string &x, const std::string &y) { return foldOp(x, '+', y); };
      auto sub = [](const std::string &x, const std::string &y) { return foldOp(x, '-', y); };
      auto mul = [](const std::string &x, const std::string &y) { return foldOp(x, '*', y); };
      switch (b->getOpcode()) {
      case BO_Add:
        r = {add(a.first, c.first), isPoint(a) && isPoint(c) ? "" : add(a.second, c.second)};
        if (r.second.empty())
          r.second = r.first;
        return true;
      case BO_Sub:
        r = {sub(a.first, c.second), isPoint(a) && isPoint(c) ? "" : sub(a.second, c.first)};
        if (r.second.empty())
          r.second = r.first;
        return true;
      case BO_Mul: {
        if (isPoint(c) && !isPoint(a))
          std::swap(a, c);              // a: the single value (if either is one)
        if (isPoint(a) && isPoint(c)) {
          r = {mul(a.first, c.first), ""};
          r.second = r.first;
          return true;
        }
        long long k;
        if (isPoint(a) && asInt(a.first, k)) { // a known constant: its sign orders the ends
          r = k >= 0 ? Interval{mul(a.first, c.first), mul(a.first, c.second)}
                     : Interval{mul(a.first, c.second), mul(a.first, c.first)};
          return true;
        }
        if (isPoint(a)) {               // a run-time value of unknown sign
          std::string p = mul(a.first, c.first), q = mul(a.first, c.second);
          r = {"cuad_imin(" + p + ", " + q + ")", "cuad_imax(" + p + ", " + q + ")"};
          return true;
        }
        // two intervals: the four corner products bound the product
        std::string p[4] = {mul(a.first, c.first), mul(a.first, c.second), mul(a.second, c.first),
                            mul(a.second, c.second)};
        r = {"cuad_imin(cuad_imin(" + p[0] + ", " + p[1] + "), cuad_imin(" + p[2] + ", " + p[3] + "))",
             "cuad_imax(cuad_imax(" + p[0] + ", " + p[1] + "), cuad_imax(" + p[2] + ", " + p[3] + "))"};
        return true;
      }
      default:
        return false;
      }
    }
    return false;
  }
  // the write intervals of vd in L's body, or false when the whole array must be saved
  bool writeIntervals(const ForStmt *L, const VarDecl *vd, std::vector<Interval> &out) const {
    if (!Ctx.getAsConstantArrayType(vd->getType()) ||
        Ctx.getAsConstantArrayType(Ctx.getAsConstantArrayType(vd->getType())->getElementType()))
      return false;                     // multi-dimensional: whole array
    std::map<const VarDecl *, ArrayUse> u;
    std::vector<const VarDecl *> order;
    arrayUses(L->getBody(), u, order);
    if (u[vd].escapes)
      return false;
    bool ok = true;
    std::set<std::string> seen;
    std::function<void(const Stmt *)> walk = [&](const Stmt *s) {
      if (!s || !ok)
        return;
      const Expr *target = nullptr;
      if (const auto *bo = dyn_cast<BinaryOperator>(s))
        if (bo->isAssignmentOp())
          target = bo->getLHS();
      if (const auto *uo = dyn_cast<UnaryOperator>(s))
        if (uo->isIncrementDecrementOp())
          target = uo->getSubExpr();
      if (target)
        if (const auto *as = dyn_cast<ArraySubscriptExpr>(target->IgnoreParenImpCasts()))
          if (const auto *d = dyn_cast<DeclRefExpr>(as->getBase()->IgnoreParenImpCasts()))
            if (d->getDecl() == vd) {
              Interval r;
              if (!rangeOf(as->getIdx(), L, r)) {
                ok = false;
                return;
              }
              if (seen.insert(r.first + "|" + r.second).second)
                out.push_back(r);
            }
      for (const Stmt *c : s->children())
        walk(c);
    };
    walk(L->getBody());
    return ok;
  }
  // Code visiting the elements of a set of index intervals of an N-element array, each clamped to [0, N).
  // Several intervals of one array may overlap (the write sites of an LSTM layer loop): when, at run time,
  // their lengths add up to more than their hull, the hull is visited once instead. The choice depends
  // only on the interval bounds, so a push and its pop make the same one. `reverse` visits the intervals
  // last-first (for pops; `emit` reverses the elements of one interval itself).
  std::string regionCode(std::vector<Interval> iv, long long N, const std::string &ind, bool reverse,
                         const std::function<std::string(const std::string &, const std::string &,
                                                         const std::string &)> &emit) const {
    for (Interval &r : iv) {
      long long a, b;
      r.first = asInt(r.first, a) ? std::to_string(std::max(a, 0LL)) : "cuad_imax(" + r.first + ", 0)";
      r.second = asInt(r.second, b) ? std::to_string(std::min(b, N - 1))
                                    : "cuad_imin(" + r.second + ", " + std::to_string(N - 1) + ")";
    }
    auto separate = [&](const std::string &in) {
      std::string s;
      if (reverse)
        for (auto it = iv.rbegin(); it != iv.rend(); ++it)
          s += emit(in, it->first, it->second);
      else
        for (const Interval &r : iv)
          s += emit(in, r.first, r.second);
      return s;
    };
    if (iv.size() == 1)
      return separate(ind);
    std::string lo = iv[0].first, hi = iv[0].second, sum;
    for (size_t j = 1; j < iv.size(); ++j) {
      lo = "cuad_imin(" + lo + ", " + iv[j].first + ")";
      hi = "cuad_imax(" + hi + ", " + iv[j].second + ")";
    }
    for (size_t j = 0; j < iv.size(); ++j)
      sum += (j ? " + " : "") + std::string("cuad_imax((") + iv[j].second + ") - (" + iv[j].first + ") + 1, 0)";
    return ind + "if (" + sum + " > (" + hi + ") - (" + lo + ") + 1) {\n" + emit(ind + "  ", lo, hi) + ind +
           "} else {\n" + separate(ind + "  ") + ind + "}\n";
  }
  std::string flatElem(const VarDecl *vd, const std::string &k) const {
    QualType elem;
    arrayCount(vd, elem);
    if (Ctx.getAsConstantArrayType(Ctx.getAsConstantArrayType(vd->getType())->getElementType()))
      return "((" + tstr(elem.getUnqualifiedType()) + " *)" + vd->getNameAsString() + ")[" + k + "]";
    return vd->getNameAsString() + "[" + k + "]";
  }
  // per-iteration checkpoint of fs's carried arrays (forward: push at the top of an iteration; backward: pop
  // before re-running it, last array first, last element first)
  std::string arrayCheckpoint(const ForStmt *fs, bool push, const std::string &ind) const {
    std::vector<const VarDecl *> v = checkpointedArrays(fs);
    std::string out;
    if (!push)
      std::reverse(v.begin(), v.end());
    for (const VarDecl *vd : v) {
      QualType elem;
      std::string n = std::to_string(arrayCount(vd, elem));
      std::vector<Interval> iv;
      if (writeIntervals(fs, vd, iv)) { // only the elements this iteration may write (clamped to the array)
        std::string a = vd->getNameAsString();
        out += regionCode(iv, std::stoll(n), ind, !push, [&](const std::string &in, const std::string &lo,
                                                             const std::string &hi) {
          return push ? in + "for (long long _cuad_k = " + lo + "; _cuad_k <= " + hi + "; ++_cuad_k) cuad_push(" +
                            a + "[_cuad_k]);\n"
                      : in + "for (long long _cuad_k = " + hi + "; _cuad_k >= " + lo + "; --_cuad_k) cuad_pop(" +
                            a + "[_cuad_k]);\n";
        });
        continue;
      }
      if (push)
        out += ind + "for (int _cuad_k = 0; _cuad_k < " + n + "; ++_cuad_k) cuad_push(" +
               flatElem(vd, "_cuad_k") + ");\n";
      else
        out += ind + "for (int _cuad_k = " + n + " - 1; _cuad_k >= 0; --_cuad_k) cuad_pop(" +
               flatElem(vd, "_cuad_k") + ");\n";
    }
    return out;
  }
  std::string shadowName(const ForStmt *fs, const VarDecl *vd) const {
    auto it = shadowId_.find(fs);
    int id = it == shadowId_.end() ? (shadowId_[fs] = (int)shadowId_.size()) : it->second;
    return vd->getNameAsString() + "_cuad_sv" + std::to_string(id);
  }
  // a recompute loop's forward pass runs without pushes, so the arrays it writes but does not checkpoint are
  // snapshotted at loop entry and restored before its reverse (each iteration then re-runs from its own elements'
  // entry values) and after it (for the statements before the loop)
  std::string shadowCopy(const ForStmt *fs, bool save, const std::string &ind) const {
    std::vector<const VarDecl *> ckpt, arrs;
    loopArrays(fs, ckpt, &arrs);
    std::string out;
    for (const VarDecl *vd : arrs) {
      QualType elem;
      std::string N = std::to_string(arrayCount(vd, elem)), sv = shadowName(fs, vd);
      if (save)
        out += ind + tstr(elem.getUnqualifiedType()) + " " + sv + "[" + N + "];\n";
      // only the region the whole loop can write (its bounds evaluated at loop entry, before and after)
      std::vector<Interval> iv;
      rangeWhole_ = true;
      bool regional = writeIntervals(fs, vd, iv);
      rangeWhole_ = false;
      if (regional) {
        std::string a = vd->getNameAsString();
        out += regionCode(iv, std::stoll(N), ind, false, [&](const std::string &in, const std::string &lo,
                                                            const std::string &hi) {
          return in + "for (long long _cuad_k = " + lo + "; _cuad_k <= " + hi + "; ++_cuad_k) " +
                 (save ? sv + "[_cuad_k] = " + a + "[_cuad_k]" : a + "[_cuad_k] = " + sv + "[_cuad_k]") + ";\n";
        });
        continue;
      }
      out += ind + "for (int _cuad_k = 0; _cuad_k < " + N + "; ++_cuad_k) " +
             (save ? sv + "[_cuad_k] = " + flatElem(vd, "_cuad_k")
                   : flatElem(vd, "_cuad_k") + " = " + sv + "[_cuad_k]") + ";\n";
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
    if (const ArraySubscriptExpr *as = tapedArrayTarget(s)) {
      if (noPush_)
        return primalLine(s, ind);
      return ind + "cuad_push(" + prim(as) + ");\n" + primalLine(s, ind);
    }
    // save reused output arrays before this call overwrites them
    if (const CallExpr *ce = asCall(s))
      if (!noPush_ && !reusedOutArgs(ce).empty())
        return tapePushCall(ce, ind) + primalLine(s, ind);
    if (isa<WhileStmt>(s) || isa<DoStmt>(s))
      if (solveInfo(s).ok) // a convergence loop written as `while` or `do`: it runs as written as well
        return ind + src(s->getSourceRange()) +
               (isa<DoStmt>(s) || !isa<CompoundStmt>(loopBody(s)) ? ";\n" : "\n");
    if (const auto *fs = dyn_cast<ForStmt>(s)) {
      // a convergence loop runs as written and stores nothing (its reverse starts from the converged point)
      if (solveInfo(fs).ok)
        return ind + src(fs->getSourceRange()) + (isa<CompoundStmt>(fs->getBody()) ? "\n" : ";\n");
      std::string init =
          fs->getInit() ? src(fs->getInit()->getSourceRange()) : "";
      if (!init.empty() && init.back() == ';')
        init.pop_back();
      std::string cond = fs->getCond() ? prim(fs->getCond()) : "";
      std::string inc = fs->getInc() ? prim(fs->getInc()) : "";
      const bool rec = isRecompute(fs);
      std::string out = (rec ? shadowCopy(fs, true, ind) : "") + ind + "for (" + init + "; " + cond + "; " +
                        inc + ") {\n";
      if (rec && (!noPush_ || fs == ckptLoop_)) {
        for (const VarDecl *v : carriedVars(fs))
          out += ind + "  cuad_push(" + v->getNameAsString() + ");\n";
        out += arrayCheckpoint(fs, true, ind + "  ");
      }
      // a recompute loop re-materializes its primal, with its own pushes, inside the
      // backward loop - its forward pass must not push, NESTED ones included (genBlock only caught top-level ones).
      // Pushes left behind by a loop inside an `if` misaligned every pop of a value pushed before the loop: in
      // one kernel a running maximum popped an earlier loop's leftovers, and the
      // derivative went to the wrong element (invisible while nothing depended on that maximum).
      bool saved = noPush_;
      if (rec)
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
      const SolveInfo &si = solveInfo(fs);
      if (si.ok)
        return solveAdjoint(fs, si, depth);
      if ((containsBreak(fs->getBody()) || !si.why.empty()) && hasActive(fs->getBody()))
        unsupported(fs->getBeginLoc(), "a loop that exits with `break` (its trip count is not known "
                                       "when the loop is reversed" +
                                           (si.why.empty() ? std::string(")")
                                                           : "; not a convergence loop: " + si.why + ")"));
      bool recompute = isRecompute(fs);
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
      std::string restore = recompute ? shadowCopy(fs, false, ind) : "";
      out = restore + out + ind + "for (" + hdr + ") {\n";
      if (recompute) {
        // per-iteration: recompute the body's primal, then reverse it (its
        // locals are private to the iteration, so this needs no cross-iter tape)
        std::vector<const Stmt *> body;
        if (const auto *cs = dyn_cast_or_null<CompoundStmt>(fs->getBody()))
          body.assign(cs->body_begin(), cs->body_end());
        else if (fs->getBody())
          body.push_back(fs->getBody());
        std::vector<const VarDecl *> carried = carriedVars(fs);
        out += arrayCheckpoint(fs, false, ind + "  ");
        for (auto it = carried.rbegin(); it != carried.rend(); ++it)
          out += ind + "  cuad_pop(" + (*it)->getNameAsString() + ");\n";
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

    if (isa<WhileStmt>(s) || isa<DoStmt>(s)) {
      const SolveInfo &si = solveInfo(s);
      if (si.ok)
        return solveAdjoint(s, si, depth);
      if (hasActive(s))
        unsupported(s->getBeginLoc(), "a `while` or `do` loop" +
                                          (si.why.empty() ? std::string(" (only counted `for` loops and convergence "
                                                                        "loops are reversed)")
                                                          : " (not a convergence loop: " + si.why + ")"));
      return out;
    }
    // anything else that touches active data has no reverse here: say so instead of dropping it
    if ((isa<SwitchStmt>(s) || isa<GotoStmt>(s) ||
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

    if (const ArraySubscriptExpr *as = tapedArrayTarget(bo)) {
      std::string v = prim(as);
      std::string snap = "_cuad_b" + std::to_string(tmp_++);
      out += ind + "cuad_pop(" + v + ");\n";
      out += ind + tstr(as->getType().getUnqualifiedType()) + " " + snap + " = " + loc + ";\n";
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
      case BO_MulAssign:
        out += ind + loc + " = " + mulT(prim(bo->getRHS()), snap) + ";\n";
        pushAdj(bo->getRHS(), mulT(v, snap), out, ind);
        break;
      case BO_DivAssign:
        out += ind + loc + " = (" + snap + ")/(" + prim(bo->getRHS()) + ");\n";
        pushAdj(bo->getRHS(), "-(" + v + ")*(" + snap + ")/((" + prim(bo->getRHS()) + ")*(" + prim(bo->getRHS()) + "))", out, ind);
        break;
      default:
        unsupported(bo->getBeginLoc(), "this compound local-array assignment");
        break;
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
    // A loop may read a local array whose elements are overwritten later in
    // this block. Its backward replay must see the array as it was on entry.
    std::map<const Stmt *, std::vector<std::pair<const VarDecl *, std::string>>> boundary;
    for (size_t i = 0; i < stmts.size(); ++i) {
      const Stmt *s = unwrapAttr(stmts[i]);
      const auto *fs = dyn_cast<ForStmt>(s);
      if (!fs)
        continue;
      std::set<const VarDecl *> refs, writes;
      referencedLocalArrays(fs->getBody(), refs);
      for (size_t j = i + 1; j < stmts.size(); ++j)
        writtenLocalArrays(stmts[j], writes);
      for (const VarDecl *vd : refs) {
        if (!writes.count(vd) || !Act.isVaried(vd))
          continue;
        const auto *cat = Ctx.getAsConstantArrayType(vd->getType());
        if (!cat || !cat->getElementType()->isRealFloatingType())
          continue;
        boundary[s].emplace_back(vd, vd->getNameAsString() + "_cuad_boundary" + std::to_string(tmp_++));
      }
    }
    for (const Stmt *s0 : stmts) { // forward primal (with pushes)
      const Stmt *s = unwrapAttr(s0);
      for (const auto &entry : boundary[s]) {
        const VarDecl *vd = entry.first;
        const auto *cat = Ctx.getAsConstantArrayType(vd->getType());
        std::string n = std::to_string(cat->getSize().getZExtValue());
        out += ind + tstr(cat->getElementType()) + " " + entry.second + "[" + n + "];\n";
        out += ind + "for (int _cuad_k = 0; _cuad_k < " + n + "; ++_cuad_k) " +
               entry.second + "[_cuad_k] = " + vd->getNameAsString() + "[_cuad_k];\n";
      }
      // recompute-loops regenerate their own primal (with pushes) inside the backward loop, per iteration.
      // They still run here, PUSH-FREE, so that what they leave behind (an accumulated sum) has its value when the
      // statements after them are recomputed - skipping them left an accumulated `v = sum M m` at 0, and every
      // derivative downstream of it was taken at the wrong state. Requirement on the
      // source: a recompute loop may change outer variables only by accumulation (x = x + ...).
      if (const auto *fs = dyn_cast<ForStmt>(s))
        if (isRecompute(fs)) {
          // a grid-stride loop's iterations are independent elements (nothing carried between them): no primal
          // pass needed - the backward loop re-materializes each one
          std::string init = fs->getInit() ? src(fs->getInit()->getSourceRange()) : "";
          if (init.find("threadIdx") != std::string::npos && carriedVars(fs).empty() &&
              checkpointedArrays(fs).empty())
            continue;
          bool saved = noPush_;
          const ForStmt *savedLoop = ckptLoop_;
          noPush_ = true;
          ckptLoop_ = fs;
          out += genPrimal(s, depth);
          noPush_ = saved;
          ckptLoop_ = savedLoop;
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
    for (auto it = stmts.rbegin(); it != stmts.rend(); ++it) { // reversed adjoint
      const Stmt *s = unwrapAttr(*it);
      for (const auto &entry : boundary[s]) {
        const auto *cat = Ctx.getAsConstantArrayType(entry.first->getType());
        std::string n = std::to_string(cat->getSize().getZExtValue());
        out += ind + "for (int _cuad_k = 0; _cuad_k < " + n + "; ++_cuad_k) " +
               entry.first->getNameAsString() + "[_cuad_k] = " + entry.second + "[_cuad_k];\n";
      }
      out += reverseStmt(*it, depth);
    }
    return out;
  }
};

} // namespace cuad
