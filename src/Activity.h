// cuad — forward activity analysis.
//
// A value is "varied" if it depends (transitively) on an active floating input.
// We seed the varied set with the active floating parameters (scalars and
// pointers/arrays to floating), then run an assignment fixpoint: a target
// becomes varied once it is ever assigned an expression that references a varied
// value. Reads of an active pointer (x[i], *x) reference the pointer's DeclRef,
// so they are covered automatically by seeding the pointer parameter.
//
// The analysis is conservative in the safe direction: anything we cannot prove
// constant stays varied, so we never drop a real derivative — we only prune
// provably-constant locals (their tangent is identically zero).

#pragma once
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include <set>

namespace cuad {
using namespace clang;

class Activity {
  std::set<const ValueDecl *> varied_;

  static bool isFloatingLike(QualType qt) {
    qt = qt.getNonReferenceType().getCanonicalType();   // `float &x` output parameters
    if (qt->isPointerType())
      return qt->getPointeeType()->isRealFloatingType();
    if (qt->isArrayType()) {
      QualType el = qt->getAsArrayTypeUnsafe()->getElementType().getCanonicalType();
      // an array of float pointers (`const float *P[3] = {a, b, c}`) too
      if (el->isPointerType())
        return el->getPointeeType()->isRealFloatingType();
      while (el->isArrayType())          // g[5][3]
        el = el->getAsArrayTypeUnsafe()->getElementType().getCanonicalType();
      return el->isRealFloatingType();
    }
    return qt->isRealFloatingType();
  }

  // Base declaration written by an lvalue (v, a[i], *p) — or null.
  static const ValueDecl *lvalueBase(const Expr *lhs) {
    const Expr *e = lhs->IgnoreParenImpCasts();
    if (const auto *d = dyn_cast<DeclRefExpr>(e))
      return d->getDecl();
    if (const auto *a = dyn_cast<ArraySubscriptExpr>(e)) {
      const Expr *b = a->getBase()->IgnoreParenImpCasts();
      while (const auto *in = dyn_cast<ArraySubscriptExpr>(b))   // g[m][0]
        b = in->getBase()->IgnoreParenImpCasts();
      if (const auto *d = dyn_cast<DeclRefExpr>(b))
        return d->getDecl();
    }
    if (const auto *u = dyn_cast<UnaryOperator>(e))
      if (u->getOpcode() == UO_Deref) {
        const Expr *s = u->getSubExpr()->IgnoreParenImpCasts();
        if (const auto *d = dyn_cast<DeclRefExpr>(s))
          return d->getDecl();
      }
    return nullptr;
  }

  bool refsVaried(const Stmt *s) const {
    if (!s)
      return false;
    if (const auto *d = dyn_cast<DeclRefExpr>(s))
      if (varied_.count(d->getDecl()))
        return true;
    for (const Stmt *c : s->children())
      if (refsVaried(c))
        return true;
    return false;
  }

  bool step(const Stmt *s) {
    if (!s)
      return false;
    bool changed = false;
    if (const auto *ds = dyn_cast<DeclStmt>(s)) {
      for (const Decl *d : ds->decls())
        if (const auto *vd = dyn_cast<VarDecl>(d))
          if (vd->hasInit() && isFloatingLike(vd->getType()) &&
              !varied_.count(vd) && refsVaried(vd->getInit())) {
            varied_.insert(vd);
            changed = true;
          }
    }
    if (const auto *bo = dyn_cast<BinaryOperator>(s))
      if (bo->isAssignmentOp()) {
        const ValueDecl *t = lvalueBase(bo->getLHS());
        if (t && isFloatingLike(t->getType()) && !varied_.count(t)) {
          bool rhsV = refsVaried(bo->getRHS()) ||
                      (bo->getOpcode() != BO_Assign && varied_.count(t));
          if (rhsV) {
            varied_.insert(t);
            changed = true;
          }
        }
      }

    // A call that receives any varied value taints the local variables passed as
    // its active pointer/array arguments (they are written as derivative-bearing
    // outputs, e.g. faceI(..., x, y, z, ..., S, C) makes S and C varied).
    if (const auto *ce = dyn_cast<CallExpr>(s)) {
      bool anyV = false;
      for (unsigned a = 0; a < ce->getNumArgs(); ++a)
        if (refsVaried(ce->getArg(a))) {
          anyV = true;
          break;
        }
      // only a call to a user function (with a body) writes outputs, and only through its
      // pointer / array / reference parameters - a local scalar passed BY VALUE to fmaxf(q[i], eps) is not
      // written (it was made varied, and its adjoint declared from its `const` type would not compile); an output
      // that is a row of a local 2-D array (`f(..., m[d])`) taints the array
      const FunctionDecl *fd = ce->getDirectCallee();
      if (!fd) {
        const Expr *c = ce->getCallee()->IgnoreParenImpCasts();
        if (const auto *ule = dyn_cast<UnresolvedLookupExpr>(c))
          for (const NamedDecl *nd : ule->decls()) {
            if (const auto *ft = dyn_cast<FunctionTemplateDecl>(nd)) { fd = ft->getTemplatedDecl(); break; }
            if (const auto *f2 = dyn_cast<FunctionDecl>(nd)) { fd = f2; break; }
          }
      }
      if (anyV && fd && fd->hasBody())
        for (unsigned a = 0; a < ce->getNumArgs(); ++a) {
          if (a < fd->getNumParams()) {
            QualType pt = fd->getParamDecl(a)->getType().getCanonicalType();
            if (!pt->isPointerType() && !pt->isArrayType() && !pt->isReferenceType())
              continue;
          }
          const Expr *arg = ce->getArg(a)->IgnoreParenImpCasts();
          while (const auto *as = dyn_cast<ArraySubscriptExpr>(arg))
            arg = as->getBase()->IgnoreParenImpCasts();
          if (const auto *d = dyn_cast<DeclRefExpr>(arg))
            if (const auto *vd = dyn_cast<VarDecl>(d->getDecl()))
              if (vd->isLocalVarDecl() && isFloatingLike(vd->getType()) &&
                  !varied_.count(vd)) {
                varied_.insert(vd);
                changed = true;
              }
        }
    }
    for (const Stmt *c : s->children())
      changed |= step(c);
    return changed;
  }

public:
  void run(const FunctionDecl *FD) {
    for (const ParmVarDecl *p : FD->parameters())
      if (isFloatingLike(p->getType()))
        varied_.insert(p);
    if (const Stmt *body = FD->getBody())
      while (step(body))
        ;
  }

  bool isVaried(const ValueDecl *d) const { return d && varied_.count(d); }
};

} // namespace cuad
