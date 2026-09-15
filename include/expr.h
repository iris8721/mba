#pragma once
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <sstream>
#include <functional>
#include <cassert>
#include <cctype>
#include "ring.h"

template<typename T>
struct ExprOp;

template<typename T>
using Expr = std::shared_ptr<ExprOp<T>>;

template<typename T>
struct ExprOp {
    enum Tag { CONST, VAR, ADD, SUB, MUL, NEG, AND, OR, XOR, NOT };

    Tag tag;
    T const_val{};
    std::string var_name;
    Expr<T> left;
    Expr<T> right;

    static Expr<T> make_const(T val) {
        auto e = std::make_shared<ExprOp<T>>();
        e->tag = CONST; e->const_val = val;
        return e;
    }
    static Expr<T> make_var(const std::string& name) {
        auto e = std::make_shared<ExprOp<T>>();
        e->tag = VAR; e->var_name = name;
        return e;
    }
    static Expr<T> make_binary(Tag t, Expr<T> l, Expr<T> r) {
        auto e = std::make_shared<ExprOp<T>>();
        e->tag = t; e->left = std::move(l); e->right = std::move(r);
        return e;
    }
    static Expr<T> make_unary(Tag t, Expr<T> inner) {
        auto e = std::make_shared<ExprOp<T>>();
        e->tag = t; e->left = std::move(inner);
        return e;
    }

    static Expr<T> make_zero() { return make_const(BinaryRing<T>::zero()); }

    bool is_zero() const { return tag == CONST && BinaryRing<T>::is_zero(const_val); }
    bool is_one() const { return tag == CONST && BinaryRing<T>::is_one(const_val); }

    void vars(std::set<std::string>& out) const {
        switch (tag) {
            case CONST: break;
            case VAR: out.insert(var_name); break;
            case NEG: case NOT: if (left) left->vars(out); break;
            default:
                if (left) left->vars(out);
                if (right) right->vars(out);
                break;
        }
    }

    std::set<std::string> vars() const {
        std::set<std::string> v;
        vars(v);
        return v;
    }

    T eval(const std::function<T(const std::string&)>& valuation) const {
        using R = BinaryRing<T>;
        switch (tag) {
            case CONST: return const_val;
            case VAR:   return valuation(var_name);
            case ADD:   return R::add(left->eval(valuation), right->eval(valuation));
            case SUB:   return R::sub(left->eval(valuation), right->eval(valuation));
            case MUL:   return R::mul(left->eval(valuation), right->eval(valuation));
            case NEG:   return R::neg(left->eval(valuation));
            case AND:   return R::bit_and(left->eval(valuation), right->eval(valuation));
            case OR:    return R::bit_or(left->eval(valuation), right->eval(valuation));
            case XOR:   return R::bit_xor(left->eval(valuation), right->eval(valuation));
            case NOT:   return R::bit_not(left->eval(valuation));
        }
        return R::zero();
    }

    static void simplify(Expr<T>& e) {
        if (!e) return;
        if (e->left) simplify(e->left);
        if (e->right) simplify(e->right);

        switch (e->tag) {
            case ADD:
                if (e->left->is_zero()) { e = e->right; return; }
                if (e->right->is_zero()) { e = e->left; return; }
                break;
            case SUB:
                if (e->left->is_zero()) { e = make_unary(NEG, e->right); simplify(e); return; }
                if (e->right->is_zero()) { e = e->left; return; }
                break;
            case MUL:
                if (e->left->is_zero() || e->right->is_zero()) { e = make_zero(); return; }
                if (e->left->is_one()) { e = e->right; return; }
                if (e->right->is_one()) { e = e->left; return; }
                break;
            case NEG:
                if (e->left->is_zero()) { e = make_zero(); return; }
                if (e->left->tag == NEG) { e = e->left->left; return; }
                break;
            case AND:
                if (e->left->is_zero() || e->right->is_zero()) { e = make_zero(); return; }
                break;
            case OR:
                if (e->left->is_zero()) { e = e->right; return; }
                if (e->right->is_zero()) { e = e->left; return; }
                break;
            case XOR:
                if (e->left->is_zero()) { e = e->right; return; }
                if (e->right->is_zero()) { e = e->left; return; }
                break;
            default: break;
        }
    }

    static void substitute(Expr<T>& e, const std::string& var, Expr<T> replacement) {
        if (!e) return;
        if (e->tag == VAR && e->var_name == var) {
            e = replacement;
            return;
        }
        if (e->left) substitute(e->left, var, replacement);
        if (e->right) substitute(e->right, var, replacement);
    }

    static Expr<T> deep_copy(const Expr<T>& e) {
        if (!e) return nullptr;
        auto copy = std::make_shared<ExprOp<T>>();
        copy->tag = e->tag;
        copy->const_val = e->const_val;
        copy->var_name = e->var_name;
        if (e->left) copy->left = deep_copy(e->left);
        if (e->right) copy->right = deep_copy(e->right);
        return copy;
    }

    int precedence() const {
        switch (tag) {
            case OR:  return 1;
            case XOR: return 2;
            case AND: return 3;
            case ADD: case SUB: return 5;
            case MUL: return 6;
            case NEG: case NOT: return 15;
            case CONST: case VAR: return 16;
        }
        return 0;
    }

    std::string to_string() const {
        using R = BinaryRing<T>;
        switch (tag) {
            case CONST: {
                int64_t sv = R::to_signed(const_val);
                return std::to_string(sv);
            }
            case VAR: return var_name;
            case ADD: return left->paren_string(5) + " + " + right->paren_string(5);
            case SUB: return left->paren_string(5) + " - " + right->paren_string(6);
            case MUL: return left->paren_string(6) + " * " + right->paren_string(6);
            case NEG: return "-" + left->paren_string(15);
            case AND: return left->paren_string(3) + " & " + right->paren_string(3);
            case OR:  return left->paren_string(1) + " | " + right->paren_string(1);
            case XOR: return left->paren_string(2) + " ^ " + right->paren_string(2);
            case NOT: return "~" + left->paren_string(15);
        }
        return "?";
    }

    std::string paren_string(int parent_prec) const {
        if (precedence() < parent_prec) return "(" + to_string() + ")";
        return to_string();
    }

    static Expr<T> parse(const std::string& s) {
        std::string cleaned;
        for (char c : s) if (c != ' ' && c != '\t' && c != '\n') cleaned += c;
        size_t pos = 0;
        return parse_impl(cleaned, pos, 0);
    }

private:
    static Expr<T> parse_impl(const std::string& s, size_t& pos, int pre) {
        using R = BinaryRing<T>;
        if (pos >= s.size()) return make_zero();

        Expr<T> e;
        char c = s[pos];

        if (c == '(') {
            ++pos;
            e = parse_impl(s, pos, 0);
            if (pos < s.size() && s[pos] == ')') ++pos;
        } else if (c == '~' || c == '!') {
            ++pos;
            e = make_unary(NOT, parse_impl(s, pos, 15));
        } else if (c == '-') {
            ++pos;
            e = make_unary(NEG, parse_impl(s, pos, 15));
        } else if (std::isalpha(static_cast<unsigned char>(c))) {
            std::string name;
            while (pos < s.size() && (std::isalnum(static_cast<unsigned char>(s[pos])) || s[pos] == '_'))
                name += s[pos++];
            e = make_var(name);
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            T val = R::zero();
            T ten = R::from_usize(10);
            while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
                val = R::add(R::mul(val, ten), R::from_usize(s[pos] - '0'));
                ++pos;
            }
            e = make_const(val);
        } else {
            return make_zero();
        }

        while (pos < s.size()) {
            c = s[pos];
            int op_pre = 0;
            Tag op_tag = CONST;
            switch (c) {
                case '|': op_pre = 1; op_tag = OR; break;
                case '^': op_pre = 2; op_tag = XOR; break;
                case '&': op_pre = 3; op_tag = AND; break;
                case '+': op_pre = 5; op_tag = ADD; break;
                case '-': op_pre = 5; op_tag = SUB; break;
                case '*': op_pre = 6; op_tag = MUL; break;
                case ')': return e;
                default: return e;
            }
            if (op_pre <= pre) return e;

            ++pos;
            auto rhs = parse_impl(s, pos, op_pre);
            e = make_binary(op_tag, std::move(e), std::move(rhs));
        }
        return e;
    }
};
