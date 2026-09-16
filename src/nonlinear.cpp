#include "../include/nonlinear.h"
#include "../include/perm_poly.h"

static constexpr size_t NONLINEAR_PERM_DEGREE = 3;

template<typename T>
static Expr<T> poly_to_expr(const Poly<T>& p, const std::string& var) {
    using R = BinaryRing<T>;
    using E = ExprOp<T>;

    Expr<T> result = nullptr;
    Expr<T> power = nullptr;
    for (size_t i = 0; i < p.len(); ++i) {
        if (i == 0) power = E::make_const(R::one());
        else if (i == 1) power = E::make_var(var);
        else power = E::make_binary(E::MUL, std::move(power), E::make_var(var));

        if (R::is_zero(p.coeffs[i])) continue;

        Expr<T> term;
        if (i == 0) {
            term = E::make_const(p.coeffs[i]);
        } else {
            term = power;
            if (!R::is_one(p.coeffs[i]))
                term = E::make_binary(E::MUL, E::make_const(p.coeffs[i]), power);
        }

        result = result ? E::make_binary(E::ADD, std::move(result), std::move(term))
                        : term;
    }
    if (!result) return E::make_zero();
    return result;
}

template<typename T>
std::optional<Expr<T>> obfuscate_nonlinear(
    const Expr<T>& e,
    const ObfuscationConfig& cfg,
    std::mt19937& rng)
{
    using E = ExprOp<T>;

    if (!e) return std::nullopt;

    auto var_set = e->vars();
    if (var_set.empty()) return std::nullopt;

    auto zi = ZeroIdeal<T>::init();

    std::vector<std::pair<std::string, std::pair<Poly<T>, Poly<T>>>> pairs;
    pairs.reserve(var_set.size());
    for (const auto& v : var_set) {
        auto pq = perm_pair<T>(rng, zi, NONLINEAR_PERM_DEGREE);
        if (!pq) return std::nullopt;
        pairs.push_back({v, std::move(*pq)});
    }

    Expr<T> obf = E::deep_copy(e);
    for (const auto& [v, pq] : pairs)
        E::substitute(obf, v, poly_to_expr<T>(pq.first, v));

    obfuscate_expr<T>(obf, cfg, rng);

    // substitute() shares the replacement subtree across occurrences, so
    // obf may be a DAG; q_v contains v itself and must not be re-entered
    // through a second parent path. Work on a tree copy.
    obf = E::deep_copy(obf);
    for (const auto& [v, pq] : pairs)
        E::substitute(obf, v, poly_to_expr<T>(pq.second, v));

    E::simplify(obf);
    return obf;
}

#define INSTANTIATE_NONLINEAR(T) \
    template std::optional<Expr<T>> obfuscate_nonlinear<T>(const Expr<T>&, const ObfuscationConfig&, std::mt19937&);

INSTANTIATE_NONLINEAR(uint8_t)
INSTANTIATE_NONLINEAR(uint16_t)
INSTANTIATE_NONLINEAR(uint32_t)
INSTANTIATE_NONLINEAR(uint64_t)
