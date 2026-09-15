#include "../include/lattice.h"
#include <cmath>
#include <limits>
#include <cassert>

template<typename T>
static Matrix<double> to_double_matrix(const Matrix<T>& m) {
    Matrix<double> result(m.num_rows(), m.num_cols());
    for (size_t i = 0; i < m.num_rows(); ++i)
        for (size_t j = 0; j < m.num_cols(); ++j)
            result(i, j) = static_cast<double>(m(i, j));
    return result;
}

template<typename T>
static Vector<double> to_double_vector(const Vector<T>& v) {
    Vector<double> result(v.dim());
    for (size_t i = 0; i < v.dim(); ++i)
        result[i] = static_cast<double>(v[i]);
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

static Matrix<double> gram_schmidt_orthonormal(Matrix<double> a) {
    a = gram_schmidt(std::move(a));
    for (size_t i = 0; i < a.num_rows(); ++i) {
        double norm = 0.0;
        for (size_t k = 0; k < a.num_cols(); ++k)
            norm += a(i, k) * a(i, k);
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (size_t k = 0; k < a.num_cols(); ++k)
                a(i, k) /= norm;
        }
    }
    return a;
}

static std::pair<Matrix<double>, Matrix<double>>
rq_decomposition(const Matrix<double>& a) {
    auto q = gram_schmidt_orthonormal(a);
    auto qt = q.transpose();
    Matrix<double> r(a.num_rows(), qt.num_cols());
    for (size_t i = 0; i < a.num_rows(); ++i)
        for (size_t j = 0; j < qt.num_cols(); ++j) {
            double s = 0.0;
            for (size_t k = 0; k < a.num_cols(); ++k)
                s += a(i, k) * qt(k, j);
            r(i, j) = s;
        }
    return {r, q};
}

static std::optional<Vector<double>> solve_linear_double(
    Matrix<double> a, Vector<double> b)
{
    size_t n = a.num_rows();
    assert(n == a.num_cols());

    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        double max_abs = std::abs(a(i, i));
        for (size_t k = i + 1; k < n; ++k) {
            double v = std::abs(a(k, i));
            if (v > max_abs) { max_abs = v; pivot = k; }
        }
        if (max_abs < 1e-15) return std::nullopt;

        if (pivot != i) {
            for (size_t j = 0; j < n; ++j) std::swap(a(i, j), a(pivot, j));
            std::swap(b[i], b[pivot]);
        }

        for (size_t k = i + 1; k < n; ++k) {
            double fac = -a(k, i) / a(i, i);
            for (size_t j = i + 1; j < n; ++j)
                a(k, j) += fac * a(i, j);
            b[k] += fac * b[i];
        }
    }

    Vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double s = b[i];
        for (size_t j = i + 1; j < n; ++j)
            s -= a(i, j) * x[j];
        x[i] = s / a(i, i);
    }
    return x;
}

template<typename T>
Vector<T> cvp_rounding(const Matrix<T>& basis, const Vector<T>& target) {
    auto af = to_double_matrix(basis);
    auto bf = to_double_vector(target);
    size_t rank = basis.num_rows();
    size_t dim = basis.num_cols();

    Matrix<double> at(dim, rank);
    for (size_t r = 0; r < dim; ++r)
        for (size_t c = 0; c < rank; ++c)
            at(r, c) = af(c, r);

    Vector<double> x_double;
    if (dim == rank) {
        auto sol = solve_linear_double(at, bf);
        if (!sol) return Vector<T>(rank);
        x_double = *sol;
    } else {
        Matrix<double> ata(rank, rank);
        for (size_t i = 0; i < rank; ++i)
            for (size_t j = 0; j < rank; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < dim; ++k)
                    s += at(k, i) * at(k, j);
                ata(i, j) = s;
            }
        Vector<double> atb(rank);
        for (size_t i = 0; i < rank; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < dim; ++k)
                s += at(k, i) * bf[k];
            atb[i] = s;
        }
        auto sol = solve_linear_double(ata, atb);
        if (!sol) return Vector<T>(rank);
        x_double = *sol;
    }

    Vector<T> result(rank);
    for (size_t i = 0; i < rank; ++i)
        result[i] = from_double<T>(x_double[i]);
    return result;
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

struct CvpResult {
    Vector<double> coords;
    Vector<double> point;
    double dist_sqr;
};

static std::optional<CvpResult> cvp_planes_impl(
    size_t i,
    const Matrix<double>& r,
    const Vector<double>& qt,
    double rad_sqr)
{
    if (i == 0) {
        double qtc = qt[0];
        double rc = r(0, 0);
        double m = std::round(qtc / rc);
        double plane = rc * m;
        double d = (plane - qtc) * (plane - qtc);
        if (d <= rad_sqr) {
            Vector<double> coords({m});
            Vector<double> point({plane});
            return CvpResult{coords, point, d};
        }
        return std::nullopt;
    }

    double qtc = qt[i];
    double rc = r(i, i);

    double start_fl = qtc / rc;
    double start_idx = std::round(start_fl);
    bool negate_offset = (start_fl < start_idx);

    int offset = 0;
    double min_dist = rad_sqr;
    std::optional<CvpResult> best;

    while (true) {
        double index = start_idx + offset;

        offset = -offset;
        if (negate_offset && offset <= 0) --offset;
        else if (!negate_offset && offset >= 0) ++offset;

        double d = (index * rc - qtc) * (index * rc - qtc);
        if (d > min_dist) break;

        double plane_dist = min_dist - d;

        Vector<double> sub_qt(i);
        for (size_t k = 0; k < i; ++k)
            sub_qt[k] = qt[k] - r(i, k) * index;

        auto sub = cvp_planes_impl(i - 1, r, sub_qt, plane_dist);
        if (!sub) continue;

        Vector<double> w = sub->point;
        for (size_t k = 0; k < i; ++k)
            w[k] += r(i, k) * index;
        w.append(index * rc);

        double total_d = 0.0;
        for (size_t k = 0; k <= i; ++k)
            total_d += (w[k] - qt[k]) * (w[k] - qt[k]);

        if (total_d <= min_dist) {
            sub->coords.append(index);
            min_dist = total_d;
            best = CvpResult{sub->coords, w, total_d};
        }
    }

    return best;
}

template<typename T>
std::optional<Vector<T>> cvp_planes(const Matrix<T>& basis, const Vector<T>& target,
                                     double rad_sqr)
{
    if (basis.num_rows() == 0) return std::nullopt;

    auto bf = to_double_matrix(basis);
    auto tf = to_double_vector(target);

    auto [r, q] = rq_decomposition(bf);

    Vector<double> qt(q.num_rows());
    for (size_t i = 0; i < q.num_rows(); ++i) {
        double s = 0.0;
        for (size_t k = 0; k < q.num_cols(); ++k)
            s += q(i, k) * tf[k];
        qt[i] = s;
    }

    double rad = (rad_sqr < 0) ? std::numeric_limits<double>::infinity() : rad_sqr;

    auto result = cvp_planes_impl(r.num_cols() - 1, r, qt, rad);
    if (!result) return std::nullopt;

    Vector<T> coeffs(result->coords.dim());
    for (size_t i = 0; i < result->coords.dim(); ++i)
        coeffs[i] = from_double<T>(result->coords[i]);

    return coeffs;
}


template Vector<uint8_t> cvp_rounding<uint8_t>(const Matrix<uint8_t>&, const Vector<uint8_t>&);
template Vector<uint16_t> cvp_rounding<uint16_t>(const Matrix<uint16_t>&, const Vector<uint16_t>&);
template Vector<uint32_t> cvp_rounding<uint32_t>(const Matrix<uint32_t>&, const Vector<uint32_t>&);
template Vector<uint64_t> cvp_rounding<uint64_t>(const Matrix<uint64_t>&, const Vector<uint64_t>&);

template Vector<uint8_t> cvp_nearest_plane<uint8_t>(const Matrix<uint8_t>&, const Vector<uint8_t>&);
template Vector<uint16_t> cvp_nearest_plane<uint16_t>(const Matrix<uint16_t>&, const Vector<uint16_t>&);
template Vector<uint32_t> cvp_nearest_plane<uint32_t>(const Matrix<uint32_t>&, const Vector<uint32_t>&);
template Vector<uint64_t> cvp_nearest_plane<uint64_t>(const Matrix<uint64_t>&, const Vector<uint64_t>&);

template std::optional<Vector<uint8_t>> cvp_planes<uint8_t>(const Matrix<uint8_t>&, const Vector<uint8_t>&, double);
template std::optional<Vector<uint16_t>> cvp_planes<uint16_t>(const Matrix<uint16_t>&, const Vector<uint16_t>&, double);
template std::optional<Vector<uint32_t>> cvp_planes<uint32_t>(const Matrix<uint32_t>&, const Vector<uint32_t>&, double);
template std::optional<Vector<uint64_t>> cvp_planes<uint64_t>(const Matrix<uint64_t>&, const Vector<uint64_t>&, double);

