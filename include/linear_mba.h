#pragma once
#include "ring.h"
#include "matrix.h"
#include "lattice.h"
#include "bitwise_expr.h"
#include "expr.h"
#include "solver.h"
#include <vector>
#include <string>
#include <random>
#include <optional>
#include <functional>
#include <set>
#include <map>
#include <algorithm>
#include <cassert>

template<typename T>
struct Valuation {
    std::map<std::string, T> values;

    T get(const std::string& name) const {
        auto it = values.find(name);
        if (it != values.end()) return it->second;
        return BinaryRing<T>::zero();
    }

    void set(const std::string& name, T val) { values[name] = val; }

    std::function<T(const std::string&)> as_fn() const {
        return [this](const std::string& name) -> T { return get(name); };
    }

    void inc_bin(const std::vector<std::string>& vars) {
        using R = BinaryRing<T>;
        for (const auto& v : vars) {
            T cur = get(v);
            if (R::is_zero(cur)) {
                set(v, R::negative_one());
                return;
            }
            set(v, R::zero());
        }
    }

    void randomize(std::mt19937& rng, const std::vector<std::string>& vars) {
        using R = BinaryRing<T>;
        for (const auto& v : vars)
            set(v, R::random(rng));
    }
};

template<typename T>
AffineLattice<T> solve_linear_system(
    const LBExpr<T>& expr,
    const std::vector<LBExpr<T>>& ops,
    const std::vector<std::string>& vars);

template<typename T>
LBExpr<T> collect_solution(
    const Vector<T>& solution,
    const std::vector<LBExpr<T>>& ops);

template<typename T>
std::optional<LBExpr<T>> rewrite(
    const LBExpr<T>& expr,
    const std::vector<LBExpr<T>>& ops,
    std::mt19937* rng = nullptr);

struct ObfuscationConfig {
    size_t auxiliary_vars = 2;
    size_t rewrite_expr_depth = 3;
    size_t rewrite_expr_count = 24;
    size_t rewrite_tries = 128;
};

BExpr random_bool_expr(const std::vector<std::string>& vars, size_t max_depth, std::mt19937& rng);

template<typename T>
void obfuscate_expr(Expr<T>& e, const ObfuscationConfig& cfg, std::mt19937& rng);

template<typename T>
struct ExtractionResult {
    LBExpr<T> lbexpr;
    std::vector<std::pair<std::string, Expr<T>>> substitutions;
};

template<typename T>
std::optional<ExtractionResult<T>> expr_to_lbexpr(const Expr<T>& e);

template<typename T>
Expr<T> lbexpr_to_expr(const LBExpr<T>& lb);

template<typename T>
Expr<T> bexpr_to_expr(const BExpr& b);
