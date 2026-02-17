#include "../include/solver.h"
#include "../include/lattice.h"

template<typename T>
std::optional<std::pair<T, T>> solve_scalar_congruence(T a, T b) {
    using R = BinaryRing<T>;

    if (R::is_zero(a)) {
        if (R::is_zero(b)) return std::make_pair(R::zero(), R::one());
        return std::nullopt;
    }

    T old_r = R::zero(), r = a;
    T old_t = R::zero(), t = R::one();
    T q = R::inc(R::euclidean_div(R::neg(a), a));

    while (true) {
        T new_r = R::sub(old_r, R::mul(q, r));
        T new_t = R::sub(old_t, R::mul(q, t));
        old_r = r; r = new_r;
        old_t = t; t = new_t;

        if (R::is_zero(r)) break;
        q = R::euclidean_div(old_r, r);
    }

    T gcd = old_r;

    T x = R::mul(R::euclidean_div(b, gcd), old_t);

    if (R::mul(a, x) != b) return std::nullopt;

    T neg_t = R::neg(t);
    if (neg_t < t) t = neg_t;

    return std::make_pair(x, t);
}

template<typename T>
std::pair<Matrix<T>, Matrix<T>> modular_diagonalize(Matrix<T>& A) {
    using R = BinaryRing<T>;

    size_t nrows = A.num_rows();
    size_t ncols = A.num_cols();

    Matrix<T> S = Matrix<T>::identity(nrows);
    Matrix<T> Tm = Matrix<T>::identity(ncols);

    size_t min_dim = std::min(nrows, ncols);

    for (size_t i = 0; i < min_dim; ++i) {
        while (true) {
            bool col_zero = A.col_all_zero_from(i, i + 1);

            if (!col_zero) {
                size_t pivot = i;
                bool is_one = false;

                for (size_t k = i; k < nrows; ++k) {
                    if (R::is_one(A(k, i))) { pivot = k; is_one = true; break; }
                }

                if (!is_one) {
                    for (size_t k = i; k < nrows; ++k) {
                        auto inv = R::try_inverse(A(k, i));
                        if (inv.has_value()) {
                            A.row_multiply(k, *inv, R{});
                            S.row_multiply(k, *inv, R{});
                            pivot = k;
                            is_one = true;
                            break;
                        }
                    }
                }

                if (!is_one) {
                    T min_val = R::zero();
                    bool found = false;
                    for (size_t k = i; k < nrows; ++k) {
                        T v = A(k, i);
                        if (!R::is_zero(v) && (!found || v < min_val)) {
                            min_val = v;
                            pivot = k;
                            found = true;
                        }
                    }
                }

                A.swap_rows(i, pivot);
                S.swap_rows(i, pivot);

                for (size_t k = i + 1; k < nrows; ++k) {
                    T e = A(k, i);
                    if (R::is_zero(e)) continue;

                    T m;
                    if (is_one) {
                        m = R::neg(e);
                    } else {
                        m = R::neg(R::euclidean_div(e, A(i, i)));
                    }

                    A.row_multiply_add(k, i, m, R{});
                    S.row_multiply_add(k, i, m, R{});
                }

                continue;
            }

            bool row_zero = A.row_all_zero_from(i, i + 1);

            if (row_zero) {
                break;
            }

            size_t pivot = i;
            bool is_one = false;

            for (size_t k = i; k < ncols; ++k) {
                if (R::is_one(A(i, k))) { pivot = k; is_one = true; break; }
            }

            if (!is_one) {
                for (size_t k = i; k < ncols; ++k) {
                    auto inv = R::try_inverse(A(i, k));
                    if (inv.has_value()) {
                        A.col_multiply(k, *inv, R{});
                        Tm.col_multiply(k, *inv, R{});
                        pivot = k;
                        is_one = true;
                        break;
                    }
                }
            }

            if (!is_one) {
                T min_val = R::zero();
                bool found = false;
                for (size_t k = i; k < ncols; ++k) {
                    T v = A(i, k);
                    if (!R::is_zero(v) && (!found || v < min_val)) {
                        min_val = v;
                        pivot = k;
                        found = true;
                    }
                }
            }

            A.swap_columns(i, pivot);
            Tm.swap_columns(i, pivot);

            for (size_t k = i + 1; k < ncols; ++k) {
                T e = A(i, k);
                if (R::is_zero(e)) continue;

                T m;
                if (is_one) {
                    m = R::neg(e);
                } else {
                    m = R::neg(R::euclidean_div(e, A(i, i)));
                }

                A.col_multiply_add(k, i, m, R{});
                Tm.col_multiply_add(k, i, m, R{});
            }
        }
    }

    return {S, Tm};
}

template<typename T>
AffineLattice<T> solve_via_modular_diagonalize(Matrix<T> A, Vector<T> b) {
    using R = BinaryRing<T>;

    assert(A.num_rows() == b.dim());
    size_t nrows = A.num_rows();
    size_t ncols = A.num_cols();

    auto [S, Tm] = modular_diagonalize(A);

    Vector<T> bp = S.mul_vec_post(b, R{});

    size_t min_dim = std::min(nrows, ncols);
    for (size_t i = min_dim; i < nrows; ++i) {
        if (!R::is_zero(bp[i])) {
            return AffineLattice<T>::empty(ncols);
        }
    }

    Vector<T> offset(ncols);
    Matrix<T> basis_rows(0, ncols);

    for (size_t i = 0; i < min_dim; ++i) {
        T diag = A(i, i);
        T rhs = bp[i];

        auto sol = solve_scalar_congruence<T>(diag, rhs);
        if (!sol.has_value()) {
            return AffineLattice<T>::empty(ncols);
        }

        offset[i] = sol->first;
        T kern = sol->second;

        if (!R::is_zero(kern)) {
            basis_rows.append_zero_rows(1);
            basis_rows(basis_rows.num_rows() - 1, i) = kern;
        }
    }

    for (size_t i = nrows; i < ncols; ++i) {
        basis_rows.append_zero_rows(1);
        basis_rows(basis_rows.num_rows() - 1, i) = R::one();
    }

    offset = Tm.mul_vec_post(offset, R{});
    basis_rows = basis_rows.mul_transpose(Tm, R{});

    Lattice<T> lat(std::move(basis_rows));
    return AffineLattice<T>(std::move(offset), std::move(lat));
}

template std::optional<std::pair<uint8_t, uint8_t>> solve_scalar_congruence<uint8_t>(uint8_t, uint8_t);
template std::optional<std::pair<uint16_t, uint16_t>> solve_scalar_congruence<uint16_t>(uint16_t, uint16_t);
template std::optional<std::pair<uint32_t, uint32_t>> solve_scalar_congruence<uint32_t>(uint32_t, uint32_t);
template std::optional<std::pair<uint64_t, uint64_t>> solve_scalar_congruence<uint64_t>(uint64_t, uint64_t);

template std::pair<Matrix<uint8_t>, Matrix<uint8_t>> modular_diagonalize<uint8_t>(Matrix<uint8_t>&);
template std::pair<Matrix<uint16_t>, Matrix<uint16_t>> modular_diagonalize<uint16_t>(Matrix<uint16_t>&);
template std::pair<Matrix<uint32_t>, Matrix<uint32_t>> modular_diagonalize<uint32_t>(Matrix<uint32_t>&);
template std::pair<Matrix<uint64_t>, Matrix<uint64_t>> modular_diagonalize<uint64_t>(Matrix<uint64_t>&);

template AffineLattice<uint8_t> solve_via_modular_diagonalize<uint8_t>(Matrix<uint8_t>, Vector<uint8_t>);
template AffineLattice<uint16_t> solve_via_modular_diagonalize<uint16_t>(Matrix<uint16_t>, Vector<uint16_t>);
template AffineLattice<uint32_t> solve_via_modular_diagonalize<uint32_t>(Matrix<uint32_t>, Vector<uint32_t>);
template AffineLattice<uint64_t> solve_via_modular_diagonalize<uint64_t>(Matrix<uint64_t>, Vector<uint64_t>);
