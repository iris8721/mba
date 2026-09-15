#include "../include/perm_poly.h"

template<typename T>
static bool parity_fold(const std::vector<T>& coeffs, size_t start, size_t step) {
    using R = BinaryRing<T>;
    bool acc = false;
    for (size_t i = start; i < coeffs.size(); i += step) {
        if (R::is_odd(coeffs[i])) acc = !acc;
    }
    return acc;
}

template<typename T>
ZeroIdeal<T> ZeroIdeal<T>::init() {
    using R = BinaryRing<T>;
    ZeroIdeal<T> zi;
    uint32_t bits = R::bits();

    uint32_t div = 0;
    for (uint32_t i = 2; ; i += 2) {
        uint32_t tz = 0;
        { uint32_t tmp = i; while ((tmp & 1) == 0) { ++tz; tmp >>= 1; } }
        div += tz;

        uint32_t e = (bits > div) ? (bits - div) : 0;

        Poly<T> p = Poly<T>::one();
        for (uint32_t j = 0; j < i; ++j)
            p.mul_linfac(R::from_usize(j));

        p.shl_coeff(e);
        p.truncate();

        zi.generators.push_back(std::move(p));

        if (e == 0) break;
    }

    return zi;
}

template<typename T>
bool is_perm_poly(const Poly<T>& f) {
    using R = BinaryRing<T>;
    if (f.len() < 2 || R::is_even(f.coeffs[1])) return false;
    if (parity_fold<T>(f.coeffs, 2, 2)) return false;
    if (parity_fold<T>(f.coeffs, 3, 2)) return false;
    return true;
}

template<typename T>
Poly<T> random_perm_poly(std::mt19937& rng, size_t degree) {
    using R = BinaryRing<T>;
    if (degree < 1) return Poly<T>::zero();

    std::vector<T> p(degree + 1);
    for (auto& c : p) c = R::random(rng);

    if (R::is_even(p[1])) p[1] = R::inc(p[1]);

    if (parity_fold<T>(p, 2, 2)) {
        std::uniform_int_distribution<size_t> dist(1, degree / 2);
        size_t i = dist(rng);
        p[2 * i] = R::inc(p[2 * i]);
    }

    if (degree >= 3 && parity_fold<T>(p, 3, 2)) {
        std::uniform_int_distribution<size_t> dist(1, (degree - 1) / 2);
        size_t i = dist(rng);
        p[2 * i + 1] = R::inc(p[2 * i + 1]);
    }

    Poly<T> poly(std::move(p));
    poly.truncate();
    return poly;
}

template<typename T>
void simplify_poly(Poly<T>& p, const ZeroIdeal<T>& zi) {
    using R = BinaryRing<T>;
    if (p.len() == 0) return;

    size_t coeff = p.len() - 1;

    for (auto it = zi.generators.rbegin(); it != zi.generators.rend(); ++it) {
        const auto& g = *it;
        size_t gen_len = g.len();

        while (coeff + 1 >= gen_len) {
            T m = R::euclidean_div(p.coeffs[coeff], g.coeffs[gen_len - 1]);
            if (!R::is_zero(m)) {
                for (size_t j = 0; j < gen_len; ++j) {
                    size_t idx = coeff + 1 - gen_len + j;
                    p.coeffs[idx] = R::sub(p.coeffs[idx], R::mul(m, g.coeffs[j]));
                }
            }
            if (coeff == 0) break;
            --coeff;
        }
    }

    p.truncate();
}

template<typename T>
void reduce_poly(Poly<T>& p, const ZeroIdeal<T>& zi) {
    using R = BinaryRing<T>;
    if (p.len() == 0 || zi.generators.empty()) return;

    const auto& g = zi.generators.back();
    size_t gen_len = g.len();

    while (p.len() >= gen_len) {
        size_t p_len = p.len();
        T c = p.coeffs[p_len - 1];
        if (!R::is_zero(c)) {
            for (size_t j = 0; j < gen_len - 1; ++j) {
                size_t idx = p_len - gen_len + j;
                p.coeffs[idx] = R::sub(p.coeffs[idx], R::mul(c, g.coeffs[j]));
            }
        }
        p.coeffs.pop_back();
    }
    p.truncate();
}

template<typename T>
Poly<T> compose(const Poly<T>& p, const Poly<T>& q, const ZeroIdeal<T>& zi) {
    using R = BinaryRing<T>;

    if (p.len() == 0) return Poly<T>::zero();

    Poly<T> result = Poly<T>::constant(p.coeffs.back());

    for (int i = static_cast<int>(p.len()) - 2; i >= 0; --i) {
        result.mul_assign(q);
        result.add_assign_const(p.coeffs[i]);
        reduce_poly(result, zi);
    }

    return result;
}

template<typename T>
std::optional<Poly<T>> compute_inverse(const Poly<T>& f, const ZeroIdeal<T>& zi) {
    using R = BinaryRing<T>;
    if (!is_perm_poly(f)) return std::nullopt;

    Poly<T> p = f;
    simplify_poly(p, zi);

    Poly<T> q({R::zero(), R::one()});

    uint32_t max_iter = R::bits() * 2;
    for (uint32_t it = 0; it <= max_iter; ++it) {
        Poly<T> comp = compose(p, q, zi);
        simplify_poly(comp, zi);

        if (comp.is_id()) return q;

        if (comp.len() < 2) comp.coeffs.resize(2, R::zero());
        comp.coeffs[1] = R::dec(comp.coeffs[1]);

        Poly<T> qd = q.derivative();
        Poly<T> correction = qd.mul(comp);
        q.sub_assign(correction);
        simplify_poly(q, zi);
    }

    return std::nullopt;
}

template<typename T>
std::optional<Poly<T>> compute_inverse_interpolation(const Poly<T>& f, const ZeroIdeal<T>& zi) {
    using R = BinaryRing<T>;
    if (!is_perm_poly(f)) return std::nullopt;

    size_t n = zi.generators.back().len();

    Matrix<T> A(n, n);
    Vector<T> b(n);

    for (size_t r = 0; r < n; ++r) {
        T x = R::from_usize(r);
        T fx = f.eval(x);
        T power = R::one();
        for (size_t c = 0; c < n; ++c) {
            A(r, c) = power;
            power = R::mul(power, fx);
        }
        b[r] = x;
    }

    auto lat = solve_via_modular_diagonalize<T>(std::move(A), std::move(b));
    if (lat.is_empty()) return std::nullopt;

    Poly<T> result(std::vector<T>(lat.offset.begin(), lat.offset.end()));
    simplify_poly(result, zi);
    return result;
}

template<typename T>
std::optional<std::pair<Poly<T>, Poly<T>>> perm_pair(std::mt19937& rng, const ZeroIdeal<T>& zi, size_t degree) {
    if (degree < 1) return std::nullopt;
    Poly<T> p = random_perm_poly<T>(rng, degree);
    auto q = compute_inverse(p, zi);
    if (!q) return std::nullopt;
    return std::make_pair(std::move(p), std::move(*q));
}

#define INSTANTIATE_PERM_POLY(T) \
    template struct ZeroIdeal<T>; \
    template bool is_perm_poly<T>(const Poly<T>&); \
    template Poly<T> random_perm_poly<T>(std::mt19937&, size_t); \
    template Poly<T> compose<T>(const Poly<T>&, const Poly<T>&, const ZeroIdeal<T>&); \
    template void simplify_poly<T>(Poly<T>&, const ZeroIdeal<T>&); \
    template void reduce_poly<T>(Poly<T>&, const ZeroIdeal<T>&); \
    template std::optional<Poly<T>> compute_inverse<T>(const Poly<T>&, const ZeroIdeal<T>&); \
    template std::optional<Poly<T>> compute_inverse_interpolation<T>(const Poly<T>&, const ZeroIdeal<T>&); \
    template std::optional<std::pair<Poly<T>, Poly<T>>> perm_pair<T>(std::mt19937&, const ZeroIdeal<T>&, size_t);

INSTANTIATE_PERM_POLY(uint8_t)
INSTANTIATE_PERM_POLY(uint16_t)
INSTANTIATE_PERM_POLY(uint32_t)
INSTANTIATE_PERM_POLY(uint64_t)
