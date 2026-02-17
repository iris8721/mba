#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <random>
#include "ring.h"

template<typename T>
struct Poly {
    std::vector<T> coeffs;

    Poly() = default;
    explicit Poly(std::vector<T> c) : coeffs(std::move(c)) {}

    static Poly zero() { return Poly(); }
    static Poly one() { return Poly({BinaryRing<T>::one()}); }
    static Poly constant(T c) { return Poly({c}); }

    static Poly random(std::mt19937& rng, size_t degree) {
        using R = BinaryRing<T>;
        std::vector<T> c(degree + 1);
        for (auto& v : c) v = R::random(rng);
        Poly p(std::move(c));
        p.truncate();
        return p;
    }

    size_t degree() const { return coeffs.empty() ? 0 : coeffs.size() - 1; }
    size_t len() const { return coeffs.size(); }
    bool is_zero() const { return coeffs.empty(); }

    bool is_id() const {
        using R = BinaryRing<T>;
        return coeffs.size() == 2 && R::is_zero(coeffs[0]) && R::is_one(coeffs[1]);
    }

    T eval(T a) const {
        using R = BinaryRing<T>;
        if (coeffs.empty()) return R::zero();
        T result = coeffs.back();
        for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
            result = R::add(R::mul(result, a), coeffs[i]);
        }
        return result;
    }

    void truncate() {
        while (!coeffs.empty() && BinaryRing<T>::is_zero(coeffs.back()))
            coeffs.pop_back();
    }

    Poly truncated() const { Poly p = *this; p.truncate(); return p; }

    Poly add(const Poly& rhs) const {
        using R = BinaryRing<T>;
        size_t n = std::max(len(), rhs.len());
        Poly result;
        result.coeffs.resize(n);
        for (size_t i = 0; i < n; ++i) {
            T a = (i < len()) ? coeffs[i] : R::zero();
            T b = (i < rhs.len()) ? rhs.coeffs[i] : R::zero();
            result.coeffs[i] = R::add(a, b);
        }
        return result;
    }

    void add_assign(const Poly& rhs) {
        using R = BinaryRing<T>;
        if (rhs.len() > len()) coeffs.resize(rhs.len(), R::zero());
        for (size_t i = 0; i < rhs.len(); ++i)
            coeffs[i] = R::add(coeffs[i], rhs.coeffs[i]);
    }

    void add_assign_const(T c) {
        using R = BinaryRing<T>;
        if (coeffs.empty()) coeffs.push_back(c);
        else coeffs[0] = R::add(coeffs[0], c);
    }

    Poly sub(const Poly& rhs) const {
        using R = BinaryRing<T>;
        size_t n = std::max(len(), rhs.len());
        Poly result;
        result.coeffs.resize(n);
        for (size_t i = 0; i < n; ++i) {
            T a = (i < len()) ? coeffs[i] : R::zero();
            T b = (i < rhs.len()) ? rhs.coeffs[i] : R::zero();
            result.coeffs[i] = R::sub(a, b);
        }
        return result;
    }

    void sub_assign(const Poly& rhs) {
        using R = BinaryRing<T>;
        if (rhs.len() > len()) coeffs.resize(rhs.len(), R::zero());
        for (size_t i = 0; i < rhs.len(); ++i)
            coeffs[i] = R::sub(coeffs[i], rhs.coeffs[i]);
    }

    void sub_assign_const(T c) {
        using R = BinaryRing<T>;
        if (coeffs.empty()) coeffs.push_back(R::neg(c));
        else coeffs[0] = R::sub(coeffs[0], c);
    }

    Poly mul(const Poly& rhs) const {
        using R = BinaryRing<T>;
        if (is_zero() || rhs.is_zero()) return zero();
        Poly result;
        result.coeffs.resize(len() + rhs.len() - 1, R::zero());
        for (size_t i = 0; i < rhs.len(); ++i)
            for (size_t j = 0; j < len(); ++j)
                result.coeffs[i + j] = R::add(result.coeffs[i + j], R::mul(rhs.coeffs[i], coeffs[j]));
        return result;
    }

    void mul_assign(const Poly& rhs) { *this = mul(rhs); }

    void mul_assign_const(T c) {
        using R = BinaryRing<T>;
        for (auto& v : coeffs) v = R::mul(v, c);
    }

    void mul_linfac(T a) {
        using R = BinaryRing<T>;
        coeffs.insert(coeffs.begin(), R::zero());
        for (size_t i = 0; i + 1 < coeffs.size(); ++i) {
            T m = R::mul(a, coeffs[i + 1]);
            coeffs[i] = R::sub(coeffs[i], m);
        }
    }

    Poly derivative() const {
        using R = BinaryRing<T>;
        if (len() <= 1) return zero();
        Poly result;
        result.coeffs.resize(len() - 1);
        for (size_t i = 1; i < len(); ++i)
            result.coeffs[i - 1] = R::mul(R::from_usize(i), coeffs[i]);
        return result;
    }

    void shl_coeff(uint32_t m) {
        using R = BinaryRing<T>;
        for (auto& c : coeffs) c = R::shl(c, m);
    }

    std::string to_string(const std::string& var = "x") const {
        using R = BinaryRing<T>;
        std::ostringstream os;
        bool first = true;
        for (int i = static_cast<int>(len()) - 1; i >= 0; --i) {
            if (R::is_zero(coeffs[i])) continue;
            if (!first) os << " + ";
            if (i == 0) {
                os << coeffs[i];
            } else {
                if (!R::is_one(coeffs[i])) os << coeffs[i];
                os << var;
                if (i > 1) os << "^" << i;
            }
            first = false;
        }
        if (first) os << "0";
        return os.str();
    }

    static Poly parse(const std::string& str) {
        using R = BinaryRing<T>;
        std::string s = str;

        bool has_x = false;
        for (char c : s) if (c == 'x' || c == 'X') { has_x = true; break; }

        if (!has_x) {
            std::istringstream iss(s);
            std::vector<T> c;
            std::string token;
            while (iss >> token) {
                auto v = R::parse(token);
                c.push_back(v.value_or(R::zero()));
            }
            std::reverse(c.begin(), c.end());
            return Poly(std::move(c)).truncated();
        }

        std::vector<T> c;
        s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
        for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

        size_t pos = 0;
        while (pos < s.size()) {
            bool neg = false;
            if (s[pos] == '+') { ++pos; continue; }
            if (s[pos] == '-') { neg = true; ++pos; }

            T coeff = R::one();
            bool has_coeff = false;
            if (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
                T val = R::zero();
                while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
                    val = R::add(R::mul(val, R::from_usize(10)), R::from_usize(s[pos] - '0'));
                    ++pos;
                }
                coeff = val;
                has_coeff = true;
                if (pos < s.size() && s[pos] == '*') ++pos;
            }
            if (neg) coeff = R::neg(coeff);

            size_t exp = 0;
            if (pos < s.size() && s[pos] == 'x') {
                ++pos;
                exp = 1;
                if (pos < s.size() && s[pos] == '^') {
                    ++pos;
                    exp = 0;
                    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
                        exp = exp * 10 + (s[pos] - '0');
                        ++pos;
                    }
                }
            } else if (!has_coeff) {
                break;
            }

            if (exp >= c.size()) c.resize(exp + 1, R::zero());
            c[exp] = coeff;
        }

        return Poly(std::move(c)).truncated();
    }
};
