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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
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

public:
  explicit ForwardMode(ASTContext &C)
      : Ctx(C), SM(C.getSourceManager()), LO(C.getLangOpts()),
        PP(C.getPrintingPolicy()) {
    PP.SuppressTagKeyword = true;
  }

  // -------- entry point --------
  std::string genFunction(const FunctionDecl *FD) {
    Act = Activity();
    Act.run(FD);
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

  std::string src(SourceRange r) const {
    return Lexer::getSourceText(CharSourceRange::getTokenRange(r), SM, LO).str();
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
        return "0 /*cuad: deref of complex ptr*/";
      }
      default:
        return "0 /*cuad: unhandled unary op*/";
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
        return "0 /*cuad: unhandled binary op*/";
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

    return "0 /*cuad: unhandled expr: " + prim(e) + "*/";
  }

  std::string diffCall(const CallExpr *call) const {
    const FunctionDecl *fd = call->getDirectCallee();
    std::string fn = fd ? fd->getNameAsString() : "";
    unsigned n = call->getNumArgs();

    // Normalize single-precision CUDA intrinsics (sinf, powf, ...) to their
    // double-named rule; the primal text keeps the original spelling.
    static const char *kF[] = {"sinf",  "cosf",  "tanf",   "expf",  "logf",
                               "sqrtf", "tanhf", "fabsf",  "powf",  "fminf",
                               "fmaxf", "asinf", "acosf",  "atanf", "atan2f"};
    for (const char *g : kF)
      if (fn == g) {
        fn.pop_back();
        break;
      }

    if (n >= 1) {
      const Expr *a0 = call->getArg(0);
      std::string pa = prim(a0), da = diff(a0);

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
    return "0 /*cuad: unhandled call: " + fn + "*/";
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
  std::string genStmt(const Stmt *s, int depth) const {
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

    if (const auto *fs = dyn_cast<ForStmt>(s)) {
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

    // Unhandled: reproduce primal so the emitted code still compiles.
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
    return "0 /*cuad: unhandled pointer tangent*/";
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
        tan = "/* cuad: unsupported compound assignment */";
      }
    }
    // tangent first (uses old primal values), then primal update
    return ind + tan + "\n" + ind + primal + "\n";
  }
};

} // namespace cuad
