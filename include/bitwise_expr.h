#pragma once
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <sstream>
#include <functional>
#include <cassert>
#include <cctype>
#include "ring.h"

struct BExpr {
    enum Tag { ONES, ZERO, VAR, AND, OR, XOR, NOT };

    Tag tag;
    std::string var_name;
    std::shared_ptr<BExpr> left;
    std::shared_ptr<BExpr> right;

    static BExpr ones() { return {ONES, "", nullptr, nullptr}; }
    static BExpr zero() { return {ZERO, "", nullptr, nullptr}; }
    static BExpr var(const std::string& name) { return {VAR, name, nullptr, nullptr}; }

    static BExpr band(BExpr l, BExpr r) {
        return {AND, "", std::make_shared<BExpr>(std::move(l)), std::make_shared<BExpr>(std::move(r))};
    }
    static BExpr bor(BExpr l, BExpr r) {
        return {OR, "", std::make_shared<BExpr>(std::move(l)), std::make_shared<BExpr>(std::move(r))};
    }
    static BExpr bxor(BExpr l, BExpr r) {
        return {XOR, "", std::make_shared<BExpr>(std::move(l)), std::make_shared<BExpr>(std::move(r))};
    }
    static BExpr bnot(BExpr e) {
        return {NOT, "", std::make_shared<BExpr>(std::move(e)), nullptr};
    }

    template<typename T>
    T eval(const std::function<T(const std::string&)>& valuation) const {
        using R = BinaryRing<T>;
        switch (tag) {
            case ONES: return R::negative_one();
            case ZERO: return R::zero();
            case VAR:  return valuation(var_name);
            case AND:  return R::bit_and(left->eval<T>(valuation), right->eval<T>(valuation));
            case OR:   return R::bit_or(left->eval<T>(valuation), right->eval<T>(valuation));
            case XOR:  return R::bit_xor(left->eval<T>(valuation), right->eval<T>(valuation));
            case NOT:  return R::bit_not(left->eval<T>(valuation));
        }
        return R::zero();
    }

    void vars(std::set<std::string>& out) const {
        switch (tag) {
            case ONES: case ZERO: break;
            case VAR: out.insert(var_name); break;
            case AND: case OR: case XOR:
                left->vars(out);
                right->vars(out);
                break;
            case NOT:
                left->vars(out);
                break;
        }
    }

    std::set<std::string> vars() const {
        std::set<std::string> v;
        vars(v);
        return v;
    }

    uint32_t complexity() const {
        switch (tag) {
            case ONES: case ZERO: case VAR: return 1;
            case AND: case OR: case XOR: return left->complexity() + right->complexity() + 1;
            case NOT: return left->complexity() + 1;
        }
        return 0;
    }

    bool operator==(const BExpr& o) const {
        if (tag != o.tag) return false;
        switch (tag) {
            case ONES: case ZERO: return true;
            case VAR: return var_name == o.var_name;
            case AND: case OR: case XOR:
                return *left == *o.left && *right == *o.right;
            case NOT:
                return *left == *o.left;
        }
        return false;
    }
    bool operator!=(const BExpr& o) const { return !(*this == o); }

    bool operator<(const BExpr& o) const {
        if (tag != o.tag) return tag < o.tag;
        switch (tag) {
            case ONES: case ZERO: return false;
            case VAR: return var_name < o.var_name;
            case AND: case OR: case XOR:
                if (*left != *o.left) return *left < *o.left;
                return *right < *o.right;
            case NOT:
                return *left < *o.left;
        }
        return false;
    }

    std::string to_string() const {
        switch (tag) {
            case ONES: return "(-1)";
            case ZERO: return "0";
            case VAR: return var_name;
            case AND: return "(" + left->to_string() + " & " + right->to_string() + ")";
            case OR:  return "(" + left->to_string() + " | " + right->to_string() + ")";
            case XOR: return "(" + left->to_string() + " ^ " + right->to_string() + ")";
            case NOT: {
                if (left->tag == VAR || left->tag == ONES || left->tag == ZERO)
                    return "~" + left->to_string();
                return "~(" + left->to_string() + ")";
            }
        }
        return "?";
    }

    static BExpr parse(const std::string& s) {
        std::string cleaned;
        for (char c : s) if (c != ' ' && c != '\t' && c != '\n') cleaned += c;
        size_t pos = 0;
        return parse_impl(cleaned, pos, 0);
    }

private:
    static BExpr parse_impl(const std::string& s, size_t& pos, int pre) {
        if (pos >= s.size()) return ones();

        BExpr e = ones();
        char c = s[pos];

        if (c == '(') {
            ++pos;
            e = parse_impl(s, pos, 0);
            if (pos < s.size() && s[pos] == ')') ++pos;
        } else if (c == '-' && pos + 1 < s.size() && s[pos + 1] == '1') {
            pos += 2;
            e = ones();
        } else if (c == '~' || c == '!') {
            ++pos;
            e = bnot(parse_impl(s, pos, 15));
        } else if (std::isalpha(static_cast<unsigned char>(c))) {
            std::string name;
            while (pos < s.size() && std::isalnum(static_cast<unsigned char>(s[pos]))) {
                name += s[pos++];
            }
            e = var(name);
        } else if (c == '1') {
            ++pos;
            e = ones();
        } else if (c == '0') {
            ++pos;
            e = zero();
        }

        while (pos < s.size()) {
            c = s[pos];
            int op_pre = 0;
            switch (c) {
                case '|': op_pre = 1; break;
                case '^': op_pre = 2; break;
                case '&': op_pre = 3; break;
                default: return e;
            }
            if (op_pre <= pre) return e;

            ++pos;
            BExpr rhs = parse_impl(s, pos, op_pre);
            switch (c) {
                case '&': e = band(std::move(e), std::move(rhs)); break;
                case '|': e = bor(std::move(e), std::move(rhs)); break;
                case '^': e = bxor(std::move(e), std::move(rhs)); break;
            }
        }
        return e;
    }
};

template<typename T>
struct LBExpr {
    std::vector<std::pair<T, BExpr>> terms;

    LBExpr() = default;
    explicit LBExpr(std::vector<std::pair<T, BExpr>> t) : terms(std::move(t)) {}

    static LBExpr zero() { return LBExpr(); }

    static LBExpr constant(T c) {
        using R = BinaryRing<T>;
        return LBExpr({{R::neg(c), BExpr::ones()}});
    }

    static LBExpr from_bexpr(BExpr e) {
        using R = BinaryRing<T>;
        return LBExpr({{R::one(), std::move(e)}});
    }

    T eval(const std::function<T(const std::string&)>& valuation) const {
        using R = BinaryRing<T>;
        T acc = R::zero();
        for (const auto& [coeff, expr] : terms) {
            T val = expr.template eval<T>(valuation);
            acc = R::add(acc, R::mul(coeff, val));
        }
        return acc;
    }

    std::set<std::string> vars() const {
        std::set<std::string> v;
        for (const auto& [_, expr] : terms) expr.vars(v);
        return v;
    }

    void remove_zero_terms() {
        using R = BinaryRing<T>;
        terms.erase(
            std::remove_if(terms.begin(), terms.end(),
                [](const auto& p) { return BinaryRing<T>::is_zero(p.first); }),
            terms.end()
        );
    }

    uint32_t complexity() const {
        using R = BinaryRing<T>;
        uint32_t total = 0;
        for (const auto& [c, e] : terms) {
            if (R::is_zero(c)) continue;
            total += R::count_ones(c) / 2 + R::min_bits(c) + e.complexity();
        }
        return total;
    }

    std::string to_string() const {
        using R = BinaryRing<T>;
        std::ostringstream os;
        bool first = true;
        for (const auto& [c, e] : terms) {
            if (R::is_zero(c)) continue;
            if (e.tag == BExpr::ONES) {
                T val = R::neg(c);
                int64_t sv = R::to_signed(val);
                if (first) {
                    os << sv;
                    first = false;
                } else {
                    if (sv >= 0) os << " + " << sv;
                    else os << " - " << -static_cast<uint64_t>(sv);
                }
            } else {
                int64_t sv = R::to_signed(c);
                if (first) {
                    if (sv == 1) os << e.to_string();
                    else if (sv == -1) os << "-" << e.to_string();
                    else os << sv << "*" << e.to_string();
                    first = false;
                } else {
                    if (sv > 0) {
                        os << " + ";
                        if (sv == 1) os << e.to_string();
                        else os << sv << "*" << e.to_string();
                    } else {
                        os << " - ";
                        if (sv == -1) os << e.to_string();
                        else os << -static_cast<uint64_t>(sv) << "*" << e.to_string();
                    }
                }
            }
        }
        if (first) os << "0";
        return os.str();
    }
};
