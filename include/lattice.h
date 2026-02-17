#pragma once
#include "ring.h"
#include "matrix.h"
#include <vector>
#include <optional>
#include <random>

template<typename T>
struct Lattice {
    Matrix<T> basis;

    Lattice() = default;
    explicit Lattice(Matrix<T> b) : basis(std::move(b)) {}

    static Lattice zero(size_t ambient_dim) {
        return Lattice(Matrix<T>(0, ambient_dim));
    }

    size_t rank() const { return basis.num_rows(); }
    size_t ambient_dim() const { return basis.num_cols(); }

    Vector<T> at(const Vector<T>& coeffs) const {
        using R = BinaryRing<T>;
        assert(coeffs.dim() == rank());
        Vector<T> result(ambient_dim());
        for (size_t i = 0; i < rank(); ++i) {
            const T* row = basis.row_ptr(i);
            for (size_t j = 0; j < ambient_dim(); ++j)
                result[j] = R::add(result[j], R::mul(coeffs[i], row[j]));
        }
        return result;
    }

    Vector<T> sample_point(std::mt19937& rng) const {
        using R = BinaryRing<T>;
        Vector<T> result(ambient_dim());
        for (size_t i = 0; i < rank(); ++i) {
            T coeff = R::random(rng);
            const T* row = basis.row_ptr(i);
            for (size_t j = 0; j < ambient_dim(); ++j)
                result[j] = R::add(result[j], R::mul(coeff, row[j]));
        }
        return result;
    }
};

template<typename T>
struct AffineLattice {
    Vector<T> offset;
    Lattice<T> lattice;

    AffineLattice() = default;
    AffineLattice(Vector<T> off, Lattice<T> lat)
        : offset(std::move(off)), lattice(std::move(lat)) {}

    static AffineLattice empty(size_t ambient_dim) {
        return AffineLattice(Vector<T>(), Lattice<T>::zero(ambient_dim));
    }

    size_t ambient_dim() const { return lattice.ambient_dim(); }

    bool is_empty() const { return offset.is_empty(); }

    Vector<T> sample_point(std::mt19937& rng) const {
        using R = BinaryRing<T>;
        Vector<T> result = offset;
        for (size_t i = 0; i < lattice.rank(); ++i) {
            T coeff = R::random(rng);
            const T* row = lattice.basis.row_ptr(i);
            for (size_t j = 0; j < lattice.ambient_dim(); ++j)
                result[j] = R::add(result[j], R::mul(coeff, row[j]));
        }
        return result;
    }

    Vector<T> at(const Vector<T>& coeffs) const {
        using R = BinaryRing<T>;
        Vector<T> result = offset;
        for (size_t i = 0; i < lattice.rank(); ++i) {
            const T* row = lattice.basis.row_ptr(i);
            for (size_t j = 0; j < lattice.ambient_dim(); ++j)
                result[j] = R::add(result[j], R::mul(coeffs[i], row[j]));
        }
        return result;
    }
};

template<typename T>
void lll_reduce(Matrix<T>& basis, double delta = 0.75);

template<typename T>
void size_reduce(Matrix<T>& basis);

template<typename T>
Vector<T> cvp_rounding(const Matrix<T>& basis, const Vector<T>& target);

template<typename T>
Vector<T> cvp_nearest_plane(const Matrix<T>& basis, const Vector<T>& target);

template<typename T>
std::optional<Vector<T>> cvp_planes(const Matrix<T>& basis, const Vector<T>& target,
                                     double rad_sqr = -1.0);

