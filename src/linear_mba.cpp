#include "../include/linear_mba.h"

template<typename T>
Expr<T> bexpr_to_expr(const BExpr& b) {
    using R = BinaryRing<T>;
    using E = ExprOp<T>;
    switch (b.tag) {
        case BExpr::ONES: return E::make_const(R::negative_one());
        case BExpr::VAR:  return E::make_var(b.var_name);
        case BExpr::AND:  return E::make_binary(E::AND, bexpr_to_expr<T>(*b.left), bexpr_to_expr<T>(*b.right));
        case BExpr::OR:   return E::make_binary(E::OR, bexpr_to_expr<T>(*b.left), bexpr_to_expr<T>(*b.right));
        case BExpr::XOR:  return E::make_binary(E::XOR, bexpr_to_expr<T>(*b.left), bexpr_to_expr<T>(*b.right));
        case BExpr::NOT:  return E::make_unary(E::NOT, bexpr_to_expr<T>(*b.left));
    }
    return E::make_zero();
}

template<typename T>
Expr<T> lbexpr_to_expr(const LBExpr<T>& lb) {
    using R = BinaryRing<T>;
    using E = ExprOp<T>;

    Expr<T> result = nullptr;
    for (const auto& [coeff, bexpr] : lb.terms) {
        if (R::is_zero(coeff)) continue;

        Expr<T> term;
        if (bexpr.tag == BExpr::ONES) {
            term = E::make_const(R::neg(coeff));
        } else {
            term = E::make_binary(E::MUL, E::make_const(coeff), bexpr_to_expr<T>(bexpr));
        }

        if (!result) {
            result = std::move(term);
        } else {
            result = E::make_binary(E::ADD, std::move(result), std::move(term));
        }
    }
    if (!result) return E::make_zero();
    return result;
}

template<typename T>
AffineLattice<T> solve_linear_system(
    const LBExpr<T>& expr,
    const std::vector<LBExpr<T>>& ops,
    const std::vector<std::string>& vars)
{
    using R = BinaryRing<T>;

    assert(vars.size() < 20);

    size_t num_rows = size_t(1) << vars.size();
    size_t num_cols = ops.size();

    Matrix<T> A(num_rows, num_cols);
    Vector<T> b(num_rows);

    Valuation<T> val;
    for (const auto& v : vars) val.set(v, R::zero());

    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < vars.size(); ++j) {
            val.set(vars[j], ((i >> j) & 1) ? R::negative_one() : R::zero());
        }

        auto fn = val.as_fn();

        for (size_t j = 0; j < num_cols; ++j) {
            A(i, j) = ops[j].eval(fn);
        }

        b[i] = expr.eval(fn);
    }

    return solve_via_modular_diagonalize<T>(std::move(A), std::move(b));
}

template<typename T>
LBExpr<T> collect_solution(
    const Vector<T>& solution,
    const std::vector<LBExpr<T>>& ops)
{
    using R = BinaryRing<T>;
    LBExpr<T> result;

    for (size_t i = 0; i < solution.dim(); ++i) {
        T c = solution[i];
        if (R::is_zero(c)) continue;

        for (const auto& [d, e] : ops[i].terms) {
            bool found = false;
            for (auto& [f, re] : result.terms) {
                if (re == e) {
                    f = R::add(f, R::mul(c, d));
                    found = true;
                    break;
                }
            }
            if (!found) {
                result.terms.push_back({R::mul(c, d), e});
            }
        }
    }

    result.remove_zero_terms();
    return result;
}

template<typename T>
std::optional<LBExpr<T>> rewrite(
    const LBExpr<T>& expr,
    const std::vector<LBExpr<T>>& ops,
    std::mt19937* rng)
{
    std::set<std::string> var_set = expr.vars();
    for (const auto& op : ops) {
        auto v = op.vars();
        var_set.insert(v.begin(), v.end());
    }
    std::vector<std::string> vars(var_set.begin(), var_set.end());

    auto lat = solve_linear_system(expr, ops, vars);
    if (lat.is_empty()) return std::nullopt;

    Vector<T> solution;
    if (rng) {
        solution = lat.sample_point(*rng);
    } else {
        solution = lat.offset;
    }

    return collect_solution(solution, ops);
}

BExpr random_bool_expr(const std::vector<std::string>& vars, size_t max_depth, std::mt19937& rng) {
    assert(!vars.empty());

    if (max_depth == 0) {
        return BExpr::var(vars[rng() % vars.size()]);
    }

    switch (rng() % 5) {
        case 0: return BExpr::var(vars[rng() % vars.size()]);
        case 1: return BExpr::bnot(random_bool_expr(vars, max_depth - 1, rng));
        case 2: return BExpr::band(
            random_bool_expr(vars, max_depth - 1, rng),
            random_bool_expr(vars, max_depth - 1, rng));
        case 3: return BExpr::bor(
            random_bool_expr(vars, max_depth - 1, rng),
            random_bool_expr(vars, max_depth - 1, rng));
        case 4: return BExpr::bxor(
            random_bool_expr(vars, max_depth - 1, rng),
            random_bool_expr(vars, max_depth - 1, rng));
    }
    return BExpr::var(vars[0]);
}

template<typename T>
static std::optional<BExpr> expr_to_bexpr_impl(
    const Expr<T>& e,
    std::vector<std::pair<std::string, Expr<T>>>& subs,
    bool force)
{
    using E = ExprOp<T>;

    if (e->tag == E::VAR) return BExpr::var(e->var_name);

    auto new_sub = [&]() -> std::optional<BExpr> {
        if (!force) return std::nullopt;
        std::string name = "_sub_" + std::to_string(subs.size());
        subs.push_back({name, e});
        return BExpr::var(name);
    };

    switch (e->tag) {
        case E::AND:
            return BExpr::band(
                expr_to_bexpr_impl<T>(e->left, subs, true).value(),
                expr_to_bexpr_impl<T>(e->right, subs, true).value());
        case E::OR:
            return BExpr::bor(
                expr_to_bexpr_impl<T>(e->left, subs, true).value(),
                expr_to_bexpr_impl<T>(e->right, subs, true).value());
        case E::XOR:
            return BExpr::bxor(
                expr_to_bexpr_impl<T>(e->left, subs, true).value(),
                expr_to_bexpr_impl<T>(e->right, subs, true).value());
        case E::NOT:
            return BExpr::bnot(expr_to_bexpr_impl<T>(e->left, subs, true).value());
        default:
            return new_sub();
    }
}

template<typename T>
static std::optional<std::pair<T, BExpr>> parse_term(
    const Expr<T>& e,
    std::vector<std::pair<std::string, Expr<T>>>& subs,
    bool force)
{
    using R = BinaryRing<T>;
    using E = ExprOp<T>;

    if (e->tag == E::MUL) {
        if (e->left->tag == E::CONST) {
            auto b = expr_to_bexpr_impl<T>(e->right, subs, force);
            if (b) return std::make_pair(e->left->const_val, *b);
        } else if (e->right->tag == E::CONST) {
            auto b = expr_to_bexpr_impl<T>(e->left, subs, force);
            if (b) return std::make_pair(e->right->const_val, *b);
        }
    } else if (e->tag == E::CONST) {
        return std::make_pair(R::neg(e->const_val), BExpr::ones());
    }

    auto b = expr_to_bexpr_impl<T>(e, subs, force);
    if (b) return std::make_pair(R::one(), *b);
    return std::nullopt;
}

template<typename T>
static bool expr_to_lbexpr_impl(
    const Expr<T>& e,
    LBExpr<T>& lb,
    std::vector<std::pair<std::string, Expr<T>>>& subs,
    bool negate,
    bool force)
{
    using R = BinaryRing<T>;
    using E = ExprOp<T>;

    switch (e->tag) {
        case E::ADD:
            expr_to_lbexpr_impl<T>(e->left, lb, subs, negate, true);
            expr_to_lbexpr_impl<T>(e->right, lb, subs, negate, true);
            return true;
        case E::SUB:
            expr_to_lbexpr_impl<T>(e->left, lb, subs, negate, true);
            expr_to_lbexpr_impl<T>(e->right, lb, subs, !negate, true);
            return true;
        case E::NEG: {
            T c = negate ? R::one() : R::negative_one();
            auto b = expr_to_bexpr_impl<T>(e->left, subs, true);
            if (b) lb.terms.push_back({c, *b});
            return true;
        }
        default: {
            auto term = parse_term<T>(e, subs, force);
            if (!term) return false;
            auto& [f, u] = *term;
            if (negate) f = R::neg(f);
            lb.terms.push_back({f, u});
            return true;
        }
    }
}

template<typename T>
std::optional<ExtractionResult<T>> expr_to_lbexpr(const Expr<T>& e) {
    LBExpr<T> lb;
    std::vector<std::pair<std::string, Expr<T>>> subs;
    if (expr_to_lbexpr_impl<T>(e, lb, subs, false, false)) {
        return ExtractionResult<T>{std::move(lb), std::move(subs)};
    }
    return std::nullopt;
}

template<typename T>
void obfuscate_expr(Expr<T>& e, const ObfuscationConfig& cfg, std::mt19937& rng) {
    using R = BinaryRing<T>;
    using E = ExprOp<T>;

    auto var_set = e->vars();
    std::vector<std::string> vars(var_set.begin(), var_set.end());
    for (size_t i = 0; i < cfg.auxiliary_vars; ++i)
        vars.push_back("aux" + std::to_string(i));

    auto extracted = expr_to_lbexpr<T>(e);
    if (extracted) {
        for (size_t attempt = 0; attempt < cfg.rewrite_tries; ++attempt) {
            std::vector<LBExpr<T>> ops;
            ops.push_back(LBExpr<T>::from_bexpr(BExpr::ones()));
            for (size_t j = 0; j < cfg.rewrite_expr_count; ++j) {
                ops.push_back(LBExpr<T>::from_bexpr(
                    random_bool_expr(vars, cfg.rewrite_expr_depth, rng)));
            }

            auto result = rewrite<T>(extracted->lbexpr, ops, &rng);
            if (result) {
                e = lbexpr_to_expr<T>(*result);

                for (auto& [var, sub_expr] : extracted->substitutions) {
                    obfuscate_expr<T>(sub_expr, cfg, rng);
                    E::substitute(e, var, sub_expr);
                }
                return;
            }
        }
        e = lbexpr_to_expr<T>(extracted->lbexpr);
        for (auto& [var, sub_expr] : extracted->substitutions) {
            E::substitute(e, var, sub_expr);
        }
        return;
    }

    if (e->tag == E::MUL) {
        obfuscate_expr<T>(e->left, cfg, rng);
        obfuscate_expr<T>(e->right, cfg, rng);
    }
}

#define INSTANTIATE_LINEAR_MBA(T) \
    template AffineLattice<T> solve_linear_system<T>(const LBExpr<T>&, const std::vector<LBExpr<T>>&, const std::vector<std::string>&); \
    template LBExpr<T> collect_solution<T>(const Vector<T>&, const std::vector<LBExpr<T>>&); \
    template std::optional<LBExpr<T>> rewrite<T>(const LBExpr<T>&, const std::vector<LBExpr<T>>&, std::mt19937*); \
    template void obfuscate_expr<T>(Expr<T>&, const ObfuscationConfig&, std::mt19937&); \
    template std::optional<ExtractionResult<T>> expr_to_lbexpr<T>(const Expr<T>&); \
    template Expr<T> lbexpr_to_expr<T>(const LBExpr<T>&); \
    template Expr<T> bexpr_to_expr<T>(const BExpr&);

INSTANTIATE_LINEAR_MBA(uint8_t)
INSTANTIATE_LINEAR_MBA(uint16_t)
INSTANTIATE_LINEAR_MBA(uint32_t)
INSTANTIATE_LINEAR_MBA(uint64_t)
