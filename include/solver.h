#pragma once
#include "ring.h"
#include "matrix.h"
#include "lattice.h"
#include <optional>
#include <utility>

template<typename T>
std::optional<std::pair<T, T>> solve_scalar_congruence(T a, T b);

template<typename T>
std::pair<Matrix<T>, Matrix<T>> modular_diagonalize(Matrix<T>& A);

template<typename T>
AffineLattice<T> solve_via_modular_diagonalize(Matrix<T> A, Vector<T> b);
