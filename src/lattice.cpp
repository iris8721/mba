#include "../include/lattice.h"
#include <cmath>

template<typename T>
static Matrix<double> to_double_matrix(const Matrix<T>& m) {
    Matrix<double> result(m.num_rows(), m.num_cols());
    for (size_t i = 0; i < m.num_rows(); ++i)
        for (size_t j = 0; j < m.num_cols(); ++j)
            result(i, j) = static_cast<double>(m(i, j));
    return result;
}

template<typename T>
static T from_double(double d) {
    return static_cast<T>(static_cast<int64_t>(std::round(d)));
}

static Matrix<double> gram_schmidt(Matrix<double> a) {
    for (size_t i = 0; i < a.num_rows(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < a.num_cols(); ++k)
                dot += a(i, k) * a(j, k);

            double norm_sqr = 0.0;
            for (size_t k = 0; k < a.num_cols(); ++k)
                norm_sqr += a(j, k) * a(j, k);

            if (norm_sqr == 0.0) continue;

            double f = -dot / norm_sqr;

            for (size_t k = 0; k < a.num_cols(); ++k)
                a(i, k) += f * a(j, k);
        }
    }
    return a;
}

template<typename T>
Vector<T> cvp_nearest_plane(const Matrix<T>& basis, const Vector<T>& target) {
    auto bf = to_double_matrix(basis);
    auto q = gram_schmidt(bf);

    size_t rank = basis.num_rows();
    size_t dim = basis.num_cols();

    using R = BinaryRing<T>;
    Vector<T> off = target;

    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        double dot = 0.0, nsq = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            double qik = q(i, k);
            double ofk = static_cast<double>(off[k]);
            dot += ofk * qik;
            nsq += qik * qik;
        }
        if (nsq == 0.0) continue;
        double c_double = dot / nsq;
        T c = R::neg(from_double<T>(c_double));

        const T* row = basis.row_ptr(i);
        for (size_t k = 0; k < dim; ++k)
            off[k] = R::add(off[k], R::mul(c, row[k]));
    }

    Vector<T> result(dim);
    for (size_t k = 0; k < dim; ++k)
        result[k] = R::sub(target[k], off[k]);
    return result;
}

template Vector<uint8_t> cvp_nearest_plane<uint8_t>(const Matrix<uint8_t>&, const Vector<uint8_t>&);
template Vector<uint16_t> cvp_nearest_plane<uint16_t>(const Matrix<uint16_t>&, const Vector<uint16_t>&);
template Vector<uint32_t> cvp_nearest_plane<uint32_t>(const Matrix<uint32_t>&, const Vector<uint32_t>&);
template Vector<uint64_t> cvp_nearest_plane<uint64_t>(const Matrix<uint64_t>&, const Vector<uint64_t>&);

