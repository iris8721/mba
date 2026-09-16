#pragma once
#include "expr.h"
#include "linear_mba.h"
#include <optional>
#include <random>

// Nonlinear MBA composition (Zhou et al. section 4): wrap each variable
// v of e in a permutation polynomial pair p_v(q_v(x)) = x, obfuscate the
// wrapped expression with the linear engine, then unwrap with q_v.
// Returns nullopt if a permutation pair cannot be generated; the caller
// should fall back to plain linear obfuscation.
template<typename T>
std::optional<Expr<T>> obfuscate_nonlinear(
    const Expr<T>& e,
    const ObfuscationConfig& cfg,
    std::mt19937& rng);
