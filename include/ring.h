#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <random>
#include <optional>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cassert>

namespace detail {
    constexpr uint8_t compute_mod_inv_u8(uint8_t e) {
        uint8_t x = 1;
        for (int i = 0; i < 3; ++i) {
            x = static_cast<uint8_t>(x * (2 - x * e));
        }
        return x;
    }

    struct ModInvTable {
        uint8_t data[256]{};
        constexpr ModInvTable() {
            for (int i = 1; i < 256; i += 2) {
                data[i] = compute_mod_inv_u8(static_cast<uint8_t>(i));
            }
        }
    };

    inline constexpr ModInvTable MOD_INV_TABLE{};

    constexpr uint32_t ceil_ilog2(uint32_t e) {
        if (e <= 1) return 0;
        uint32_t r = 0;
        uint32_t v = e - 1;
        while (v > 0) { v >>= 1; ++r; }
        return r;
    }
}

template<typename T>
struct BinaryRing {
    static_assert(std::is_unsigned_v<T>, "T must be unsigned");

    using Element = T;

    static constexpr uint32_t bits() { return static_cast<uint32_t>(sizeof(T) * 8); }

    static constexpr T zero() { return T(0); }
    static constexpr T one()  { return T(1); }
    static constexpr T negative_one() { return static_cast<T>(~T(0)); }

    static constexpr bool is_zero(T e) { return e == 0; }
    static constexpr bool is_one(T e)  { return e == 1; }
    static constexpr bool is_even(T e) { return (e & 1) == 0; }
    static constexpr bool is_odd(T e)  { return (e & 1) != 0; }
    static constexpr bool is_unit(T e) { return is_odd(e); }
    static constexpr bool is_zero_divisor(T e) { return is_even(e); }

    static constexpr T neg(T e) { return static_cast<T>(~e + 1); }
    static constexpr T add(T a, T b) { return static_cast<T>(a + b); }
    static constexpr T sub(T a, T b) { return static_cast<T>(a - b); }
    static constexpr T mul(T a, T b) { return static_cast<T>(a * b); }
    static constexpr T inc(T e) { return static_cast<T>(e + 1); }
    static constexpr T dec(T e) { return static_cast<T>(e - 1); }
    static constexpr T square(T a) { return mul(a, a); }

    static constexpr T mul_add(T acc, T a, T b) { return add(acc, mul(a, b)); }

    static constexpr T mul_sub(T acc, T a, T b) { return sub(acc, mul(a, b)); }

    static constexpr T bit_not(T e)    { return static_cast<T>(~e); }
    static constexpr T bit_and(T a, T b) { return static_cast<T>(a & b); }
    static constexpr T bit_or(T a, T b)  { return static_cast<T>(a | b); }
    static constexpr T bit_xor(T a, T b) { return static_cast<T>(a ^ b); }
    static constexpr T shl(T e, uint32_t n) { return static_cast<T>(e << n); }
    static constexpr T shr(T e, uint32_t n) { return static_cast<T>(e >> n); }

    static constexpr bool bit(T e, uint32_t i) { return (e & (T(1) << i)) != 0; }

    static constexpr uint32_t count_ones(T e) {
        uint32_t c = 0;
        while (e) { c += (e & 1); e >>= 1; }
        return c;
    }

    static constexpr uint32_t min_bits(T e) {
        if (e == 0) return 0;
        uint32_t n = bits();
        T mask = T(1) << (n - 1);
        while (mask && !(e & mask)) { --n; mask >>= 1; }
        return n;
    }

    static T inverse(T e) {
        assert(is_odd(e));

        T x = static_cast<T>(detail::MOD_INV_TABLE.data[static_cast<uint8_t>(e & 0xFF)]);
        T two = T(2);

        uint32_t iters = detail::ceil_ilog2(bits());
        uint32_t start = (iters > 3) ? (iters - 3) : 0;
        for (uint32_t i = 0; i < start; ++i) {
            x = mul(x, sub(two, mul(x, e)));
        }
        return x;
    }

    static std::optional<T> try_inverse(T e) {
        if (!is_odd(e)) return std::nullopt;
        return inverse(e);
    }

    static constexpr T euclidean_div(T a, T b) {
        if (b == 0) return 0;
        return static_cast<T>(a / b);
    }

    static constexpr T euclidean_rem(T a, T b) {
        if (b == 0) return a;
        return static_cast<T>(a % b);
    }

    static constexpr T rounded_div(T a, T b) {
        T q = euclidean_div(a, b);
        T rem = euclidean_rem(a, b);
        T half = static_cast<T>((b >> 1) + (b & 1));
        if (rem >= half) q = inc(q);
        return q;
    }

    static constexpr int cmp(T a, T b) {
        return (a < b) ? -1 : (a > b) ? 1 : 0;
    }

    static constexpr int cmp_abs(T a, T b) { return cmp(a, b); }

    static constexpr bool is_negative(T) { return false; }
    static constexpr bool is_positive(T e) { return e != 0; }

    static constexpr T abs(T e) { return e; }

    static constexpr T from_usize(size_t n) { return static_cast<T>(n); }

    static constexpr uint64_t to_representative(T e) { return static_cast<uint64_t>(e); }
    static constexpr size_t to_usize(T e) { return static_cast<size_t>(e); }

    static int64_t to_signed(T e) {
        using Signed = std::make_signed_t<T>;
        Signed s;
        std::memcpy(&s, &e, sizeof(T));
        return static_cast<int64_t>(s);
    }

    static T random(std::mt19937& rng) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(rng());
        } else {
            return static_cast<T>((static_cast<uint64_t>(rng()) << 32) | rng());
        }
    }

    static std::optional<T> parse(const std::string& s) {
        T result = zero();
        T ten = from_usize(10);
        for (char c : s) {
            if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
            result = add(mul(result, ten), from_usize(c - '0'));
        }
        return result;
    }

    static T element_from_usize(size_t n) { return static_cast<T>(n); }
};

using Ring8  = BinaryRing<uint8_t>;
using Ring16 = BinaryRing<uint16_t>;
using Ring32 = BinaryRing<uint32_t>;
using Ring64 = BinaryRing<uint64_t>;
