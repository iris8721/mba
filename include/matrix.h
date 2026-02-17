#pragma once
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "ring.h"

template<typename T>
struct Vector {
    std::vector<T> data;

    Vector() = default;
    explicit Vector(size_t n) : data(n, T(0)) {}
    Vector(size_t n, T val) : data(n, val) {}
    Vector(std::initializer_list<T> il) : data(il) {}
    explicit Vector(std::vector<T> d) : data(std::move(d)) {}

    static Vector zero(size_t n) { return Vector(n); }
    static Vector empty() { return Vector(); }

    size_t dim() const { return data.size(); }
    bool is_empty() const { return data.empty(); }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    typename std::vector<T>::iterator begin() { return data.begin(); }
    typename std::vector<T>::iterator end() { return data.end(); }
    typename std::vector<T>::const_iterator begin() const { return data.begin(); }
    typename std::vector<T>::const_iterator end() const { return data.end(); }

    void append(T val) { data.push_back(std::move(val)); }

    bool operator==(const Vector& o) const { return data == o.data; }
    bool operator!=(const Vector& o) const { return data != o.data; }

    template<typename Ring>
    Vector neg(const Ring&) const {
        Vector r(dim());
        for (size_t i = 0; i < dim(); ++i) r[i] = Ring::neg(data[i]);
        return r;
    }

    template<typename Ring>
    void mul_add_assign(T scalar, const T* other, size_t len, const Ring&) {
        for (size_t i = 0; i < len && i < dim(); ++i)
            data[i] = Ring::add(data[i], Ring::mul(scalar, other[i]));
    }

    template<typename Ring>
    T dot(const Vector& o, const Ring&) const {
        T acc = Ring::zero();
        for (size_t i = 0; i < dim(); ++i)
            acc = Ring::add(acc, Ring::mul(data[i], o.data[i]));
        return acc;
    }

    template<typename Ring>
    T norm_sqr(const Ring& ring) const { return dot(*this, ring); }

    template<typename Ring>
    Vector sub_rhs(const Vector& other, const Ring&) const {
        Vector r(dim());
        for (size_t i = 0; i < dim(); ++i)
            r[i] = Ring::sub(data[i], other.data[i]);
        return r;
    }

    template<typename U, typename F>
    Vector<U> transform(F f) const {
        Vector<U> r(dim());
        for (size_t i = 0; i < dim(); ++i) r[i] = f(data[i]);
        return r;
    }

    template<typename Ring>
    void reduce(const auto& basis_matrix, const Ring& ring) {
        for (size_t i = 0; i < basis_matrix.num_rows(); ++i) {
            if (Ring::is_zero(basis_matrix(i, i))) continue;
            T q = Ring::euclidean_div(data[i], basis_matrix(i, i));
            if (Ring::is_zero(q)) continue;
            q = Ring::neg(q);
            for (size_t j = 0; j < dim(); ++j) {
                data[j] = Ring::add(data[j], Ring::mul(q, basis_matrix(i, j)));
            }
        }
    }
};

template<typename T>
struct Matrix {
    std::vector<T> data;
    size_t rows_ = 0;
    size_t cols_ = 0;

    Matrix() = default;
    Matrix(size_t r, size_t c) : data(r * c, T(0)), rows_(r), cols_(c) {}
    Matrix(size_t r, size_t c, T val) : data(r * c, val), rows_(r), cols_(c) {}

    static Matrix zero(size_t r, size_t c) { return Matrix(r, c); }

    static Matrix identity(size_t n) {
        Matrix m(n, n);
        for (size_t i = 0; i < n; ++i) m(i, i) = T(1);
        return m;
    }

    size_t num_rows() const { return rows_; }
    size_t num_cols() const { return cols_; }
    size_t min_dim() const { return std::min(rows_, cols_); }

    T& operator()(size_t r, size_t c) { return data[r * cols_ + c]; }
    const T& operator()(size_t r, size_t c) const { return data[r * cols_ + c]; }

    T* row_ptr(size_t r) { return &data[r * cols_]; }
    const T* row_ptr(size_t r) const { return &data[r * cols_]; }

    void swap_rows(size_t a, size_t b) {
        if (a == b) return;
        T* ra = row_ptr(a);
        T* rb = row_ptr(b);
        for (size_t j = 0; j < cols_; ++j) std::swap(ra[j], rb[j]);
    }

    template<typename Ring>
    void row_multiply(size_t r, T scalar, const Ring&) {
        T* rp = row_ptr(r);
        for (size_t j = 0; j < cols_; ++j) rp[j] = Ring::mul(rp[j], scalar);
    }

    template<typename Ring>
    void row_multiply_add(size_t dst, size_t src, T scalar, const Ring&) {
        T* dp = row_ptr(dst);
        const T* sp = row_ptr(src);
        for (size_t j = 0; j < cols_; ++j)
            dp[j] = Ring::add(dp[j], Ring::mul(scalar, sp[j]));
    }

    template<typename Ring>
    void negate_row(size_t r, const Ring&) {
        T* rp = row_ptr(r);
        for (size_t j = 0; j < cols_; ++j) rp[j] = Ring::neg(rp[j]);
    }

    void swap_columns(size_t a, size_t b) {
        if (a == b) return;
        for (size_t i = 0; i < rows_; ++i)
            std::swap(data[i * cols_ + a], data[i * cols_ + b]);
    }

    template<typename Ring>
    void col_multiply(size_t c, T scalar, const Ring&) {
        for (size_t i = 0; i < rows_; ++i)
            data[i * cols_ + c] = Ring::mul(data[i * cols_ + c], scalar);
    }

    template<typename Ring>
    void col_multiply_add(size_t dst, size_t src, T scalar, const Ring&) {
        for (size_t i = 0; i < rows_; ++i) {
            data[i * cols_ + dst] = Ring::add(
                data[i * cols_ + dst],
                Ring::mul(scalar, data[i * cols_ + src])
            );
        }
    }

    template<typename Ring>
    Vector<T> mul_vec_post(const Vector<T>& v, const Ring&) const {
        assert(v.dim() == cols_);
        Vector<T> result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            T acc = Ring::zero();
            const T* rp = row_ptr(i);
            for (size_t j = 0; j < cols_; ++j)
                acc = Ring::add(acc, Ring::mul(rp[j], v[j]));
            result[i] = acc;
        }
        return result;
    }

    template<typename Ring>
    Matrix mul(const Matrix& other, const Ring&) const {
        assert(cols_ == other.rows_);
        Matrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                T a = data[i * cols_ + k];
                if (Ring::is_zero(a)) continue;
                for (size_t j = 0; j < other.cols_; ++j) {
                    result(i, j) = Ring::add(result(i, j), Ring::mul(a, other(k, j)));
                }
            }
        }
        return result;
    }

    template<typename Ring>
    Matrix mul_transpose(const Matrix& other, const Ring&) const {
        assert(cols_ == other.cols_);
        Matrix result(rows_, other.rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.rows_; ++j) {
                T acc = Ring::zero();
                for (size_t k = 0; k < cols_; ++k)
                    acc = Ring::add(acc, Ring::mul(data[i * cols_ + k], other(j, k)));
                result(i, j) = acc;
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                t(j, i) = data[i * cols_ + j];
        return t;
    }

    void append_zero_rows(size_t n) {
        data.resize((rows_ + n) * cols_, T(0));
        rows_ += n;
    }

    void remove_zero_rows() {
        size_t new_rows = rows_;
        while (new_rows > 0) {
            bool all_zero = true;
            const T* rp = row_ptr(new_rows - 1);
            for (size_t j = 0; j < cols_; ++j) {
                if (rp[j] != T(0)) { all_zero = false; break; }
            }
            if (!all_zero) break;
            --new_rows;
        }
        rows_ = new_rows;
        data.resize(rows_ * cols_);
    }

    T col(size_t c, size_t r) const { return data[r * cols_ + c]; }

    bool col_all_zero_from(size_t c, size_t from) const {
        for (size_t i = from; i < rows_; ++i)
            if (data[i * cols_ + c] != T(0)) return false;
        return true;
    }

    bool row_all_zero_from(size_t r, size_t from) const {
        const T* rp = row_ptr(r);
        for (size_t j = from; j < cols_; ++j)
            if (rp[j] != T(0)) return false;
        return true;
    }

    template<typename U, typename F>
    Matrix<U> transform(F f) const {
        Matrix<U> result(rows_, cols_);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = f(data[i]);
        return result;
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ":\n";
        for (size_t i = 0; i < rows_; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < cols_; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << data[i * cols_ + j];
            }
            std::cout << "]\n";
        }
    }
};

struct DoubleOps {
    static double zero() { return 0.0; }
    static double one() { return 1.0; }
    static bool is_zero(double e) { return e == 0.0; }
    static double neg(double e) { return -e; }
    static double add(double a, double b) { return a + b; }
    static double sub(double a, double b) { return a - b; }
    static double mul(double a, double b) { return a * b; }
    static double div(double a, double b) { return a / b; }
    static double square(double a) { return a * a; }
    static double sqrt_(double a) { return std::sqrt(a); }
    static double abs_(double a) { return std::abs(a); }
    static double round_(double a) { return std::round(a); }
    static int cmp_abs(double a, double b) {
        double aa = std::abs(a), ab = std::abs(b);
        return (aa < ab) ? -1 : (aa > ab) ? 1 : 0;
    }
    static double mul_add(double acc, double a, double b) { return acc + a * b; }
    static double mul_sub(double acc, double a, double b) { return acc - a * b; }
    static double infinity() { return std::numeric_limits<double>::infinity(); }
};
