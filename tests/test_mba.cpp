#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "ring.h"
#include "matrix.h"
#include "solver.h"
#include "lattice.h"
#include "bitwise_expr.h"
#include "expr.h"
#include "linear_mba.h"
#include "poly.h"
#include "perm_poly.h"
#include "nonlinear.h"

using u8 = uint8_t;
using u32 = uint32_t;

static int checks = 0;
static int failures = 0;

#define CHECK(cond) do { \
    ++checks; \
    if (!(cond)) { \
        ++failures; \
        std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// solve_scalar_congruence returns (x, t) with a*x = b and a*t = 0, or nullopt.
static void test_scalar_congruence() {
    std::printf("solve_scalar_congruence\n");

    // a = 0: solvable only when b = 0
    CHECK(solve_scalar_congruence<u8>(0, 0).has_value());
    CHECK(!solve_scalar_congruence<u8>(0, 5).has_value());

    // odd a is a unit: unique solution, trivial kernel
    auto r = solve_scalar_congruence<u8>(3, 7);
    CHECK(r.has_value());
    if (r) {
        CHECK(static_cast<u8>(3 * r->first) == 7);
        CHECK(r->second == 0);
    }

    // even a, divisible b: solution plus a nontrivial kernel generator
    r = solve_scalar_congruence<u8>(4, 8);
    CHECK(r.has_value());
    if (r) {
        CHECK(static_cast<u8>(4 * r->first) == 8);
        CHECK(r->second != 0);
        CHECK(static_cast<u8>(4 * r->second) == 0);
    }

    // even a, non-divisible b: no solution
    CHECK(!solve_scalar_congruence<u8>(4, 6).has_value());
    CHECK(!solve_scalar_congruence<u8>(128, 64).has_value());

    // randomized sweep against brute force
    std::mt19937 rng(1234);
    for (int i = 0; i < 2000; ++i) {
        u8 a = static_cast<u8>(rng());
        u8 b = static_cast<u8>(rng());
        auto s = solve_scalar_congruence<u8>(a, b);
        bool any = false;
        for (int x = 0; x < 256 && !any; ++x)
            any = (static_cast<u8>(a * static_cast<u8>(x)) == b);
        CHECK(s.has_value() == any);
        if (s) {
            CHECK(static_cast<u8>(a * s->first) == b);
            CHECK(static_cast<u8>(a * s->second) == 0);
        }
    }
}

static void test_modular_diagonalize() {
    std::printf("modular_diagonalize\n");
    using R = BinaryRing<u8>;

    // every entry is even: no unit pivot exists anywhere
    Matrix<u8> A(2, 2);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 6; A(1, 1) = 2;
    Matrix<u8> orig = A;

    auto [S, T] = modular_diagonalize<u8>(A);

    // S * orig * T == A
    auto prod = S.mul(orig, R{}).mul(T, R{});
    CHECK(prod.data == A.data);

    // off-diagonal entries eliminated
    for (size_t i = 0; i < A.num_rows(); ++i)
        for (size_t j = 0; j < A.num_cols(); ++j)
            if (i != j) CHECK(A(i, j) == 0);

    // solve a system with no unit pivot: 2x + 2y = 4, 4x + 2y = 6
    Matrix<u8> M(2, 2);
    M(0, 0) = 2; M(0, 1) = 2;
    M(1, 0) = 4; M(1, 1) = 2;
    Vector<u8> b{4, 6};
    auto lat = solve_via_modular_diagonalize<u8>(M, b);
    CHECK(!lat.is_empty());
    if (!lat.is_empty()) {
        CHECK(M.mul_vec_post(lat.offset, R{}) == b);
        for (size_t i = 0; i < lat.lattice.rank(); ++i) {
            Vector<u8> row(lat.lattice.basis.num_cols());
            for (size_t j = 0; j < row.dim(); ++j)
                row[j] = lat.lattice.basis(i, j);
            auto kr = M.mul_vec_post(row, R{});
            for (size_t j = 0; j < kr.dim(); ++j)
                CHECK(kr[j] == 0);
        }
    }

    // 2x = 3 has no solution mod 256
    Matrix<u8> M2(1, 1);
    M2(0, 0) = 2;
    Vector<u8> b2{3};
    CHECK(solve_via_modular_diagonalize<u8>(M2, b2).is_empty());
}

static void test_expr_roundtrip() {
    std::printf("expr parse/print round-trip\n");
    using E = ExprOp<u32>;

    // parens: x-(y-z) is not (x-y)-z
    {
        auto e = E::parse("x - (y - z)");
        Valuation<u32> v;
        v.set("x", 10); v.set("y", 3); v.set("z", 7);
        CHECK(e->eval(v.as_fn()) == 14u);
        auto flat = E::parse("x - y - z");
        CHECK(flat->eval(v.as_fn()) == 0u);
        auto e2 = E::parse(e->to_string());
        CHECK(e2->eval(v.as_fn()) == 14u);
    }

    // '0' literal
    {
        auto e = E::parse("0");
        CHECK(e->to_string() == "0");
        Valuation<u32> v;
        CHECK(e->eval(v.as_fn()) == 0u);
        auto e2 = E::parse("x + 0");
        E::simplify(e2);
        CHECK(e2->to_string() == "x");
    }

    // repeated factors
    {
        auto e = E::parse("x * x * x");
        Valuation<u32> v;
        v.set("x", 5);
        CHECK(e->eval(v.as_fn()) == 125u);
        auto e2 = E::parse(e->to_string());
        CHECK(e2->eval(v.as_fn()) == 125u);
    }

    // precedence: & binds tighter than |
    {
        auto e = E::parse("x | y & z");
        Valuation<u32> v;
        v.set("x", 1); v.set("y", 2); v.set("z", 4);
        CHECK(e->eval(v.as_fn()) == (1u | (2u & 4u)));
    }

    // BExpr round trip
    {
        auto b = BExpr::parse("x & y | ~z");
        auto b2 = BExpr::parse(b.to_string());
        CHECK(b == b2);
    }

    // Poly: duplicate exponents accumulate
    {
        auto p = Poly<u32>::parse("x^2 + x^2 + 3x + x");
        CHECK(p.coeffs.size() == 3);
        CHECK(p.coeffs[2] == 2u);
        CHECK(p.coeffs[1] == 4u);
        auto q = Poly<u32>::parse(p.to_string());
        CHECK(q.coeffs == p.coeffs);
    }

    // expr_to_lbexpr: purely nonlinear input has no linear part
    CHECK(!expr_to_lbexpr<u32>(E::parse("x * y")).has_value());
    CHECK(expr_to_lbexpr<u32>(E::parse("x + y")).has_value());
}

static void test_simplify() {
    std::printf("simplify\n");
    using R = BinaryRing<u32>;

    // (x^y) + 2(x&y) == x + y; inflate it with
    // K * ((x&y) + (x^y) - (x|y)) == 0 to get huge coefficients.
    const u32 K = 0x12345678u;
    LBExpr<u32> big({
        {R::add(R::one(), K),          BExpr::bxor(BExpr::var("x"), BExpr::var("y"))},
        {R::add(R::from_usize(2), K),  BExpr::band(BExpr::var("x"), BExpr::var("y"))},
        {R::neg(K),                    BExpr::bor(BExpr::var("x"), BExpr::var("y"))},
    });

    std::vector<LBExpr<u32>> ops = {
        LBExpr<u32>::from_bexpr(BExpr::ones()),
        LBExpr<u32>::from_bexpr(BExpr::var("x")),
        LBExpr<u32>::from_bexpr(BExpr::var("y")),
        LBExpr<u32>::from_bexpr(BExpr::band(BExpr::var("x"), BExpr::var("y"))),
        LBExpr<u32>::from_bexpr(BExpr::bor(BExpr::var("x"), BExpr::var("y"))),
        LBExpr<u32>::from_bexpr(BExpr::bxor(BExpr::var("x"), BExpr::var("y"))),
        LBExpr<u32>::from_bexpr(BExpr::bnot(BExpr::var("x"))),
        LBExpr<u32>::from_bexpr(BExpr::bnot(BExpr::var("y"))),
    };
    std::vector<std::string> vars = {"x", "y"};

    auto out = simplify<u32>(big, ops, vars);
    CHECK(out.has_value());
    if (!out) return;

    std::printf("  %s\n  -> %s\n", big.to_string().c_str(), out->to_string().c_str());

    // still evaluates to x + y
    std::mt19937 rng(7);
    bool ok = true;
    for (int i = 0; i < 200 && ok; ++i) {
        Valuation<u32> v;
        v.set("x", R::random(rng));
        v.set("y", R::random(rng));
        if (out->eval(v.as_fn()) != R::add(v.get("x"), v.get("y"))) ok = false;
    }
    CHECK(ok);

    // coefficients collapsed: sum of min(c, -c) is small
    uint64_t norm = 0;
    for (const auto& [c, e] : out->terms)
        norm += std::min<uint64_t>(c, R::neg(c));
    CHECK(norm < 100);
}

static void test_obfuscate_nonlinear() {
    std::printf("obfuscate_nonlinear\n");

    std::mt19937 rng(99);
    ObfuscationConfig cfg;
    cfg.auxiliary_vars = 1;
    cfg.rewrite_expr_count = 12;
    cfg.rewrite_tries = 32;

    auto e = ExprOp<u32>::parse("x + y");
    auto obf = obfuscate_nonlinear<u32>(e, cfg, rng);
    CHECK(obf.has_value());
    if (obf) {
        std::uniform_int_distribution<u32> dist;
        bool ok = true;
        for (int i = 0; i < 500 && ok; ++i) {
            Valuation<u32> v;
            for (const auto& name : (*obf)->vars()) v.set(name, dist(rng));
            if ((*obf)->eval(v.as_fn()) != e->eval(v.as_fn())) ok = false;
        }
        CHECK(ok);

        // the perm-pair wrap introduces multiplication
        bool has_mul = false;
        std::function<void(const Expr<u32>&)> walk = [&](const Expr<u32>& n) {
            if (!n) return;
            if (n->tag == ExprOp<u32>::MUL) has_mul = true;
            walk(n->left);
            walk(n->right);
        };
        walk(*obf);
        CHECK(has_mul);
    }

    // a constant expression has no variables to wrap
    auto c = ExprOp<u32>::parse("5");
    CHECK(!obfuscate_nonlinear<u32>(c, cfg, rng).has_value());
}

int main() {
    test_scalar_congruence();
    test_modular_diagonalize();
    test_expr_roundtrip();
    test_simplify();
    test_obfuscate_nonlinear();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
