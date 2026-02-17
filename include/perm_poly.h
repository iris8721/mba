#pragma once
#include "ring.h"
#include "poly.h"
#include "matrix.h"
#include "solver.h"
#include <vector>
#include <random>
#include <cassert>

template<typename T>
struct ZeroIdeal {
    std::vector<Poly<T>> generators;

    static ZeroIdeal init();
};

template<typename T>
bool is_perm_poly(const Poly<T>& f);

template<typename T>
Poly<T> random_perm_poly(std::mt19937& rng, size_t degree);

template<typename T>
Poly<T> compose(const Poly<T>& p, const Poly<T>& q, const ZeroIdeal<T>& zi);

template<typename T>
void simplify_poly(Poly<T>& p, const ZeroIdeal<T>& zi);

template<typename T>
void reduce_poly(Poly<T>& p, const ZeroIdeal<T>& zi);

template<typename T>
Poly<T> compute_inverse(const Poly<T>& f, const ZeroIdeal<T>& zi);

template<typename T>
Poly<T> compute_inverse_interpolation(const Poly<T>& f, const ZeroIdeal<T>& zi);

template<typename T>
std::pair<Poly<T>, Poly<T>> perm_pair(std::mt19937& rng, const ZeroIdeal<T>& zi, size_t degree);
