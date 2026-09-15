# mba

A C++ toolkit for generating and verifying **linear MBA (Mixed Boolean-Arithmetic)
obfuscation identities** over the binary ring Z/2^n.

## What is MBA obfuscation?

MBA obfuscation rewrites arithmetic expressions into semantically equivalent
expressions that mix bitwise operators (`&`, `|`, `^`, `~`) with arithmetic
operators (`+`, `-`, `*`). For example, over 32-bit words:

```
x + y  ==  (x ^ y) + 2*(x & y)
x + y  ==  (x | y) + (x & y)
x ^ y  ==  (x | y) - (x & y)
```

Because the rewrite explodes expression size while preserving semantics, it is
used by software obfuscators to hide constants and operations from static
analysis and decompilers. Zhou et al. showed that linear MBA identities —
expressions of the form `sum_i c_i * f_i(x, y, ...)` where each `f_i` is a
bitwise expression — can be generated mechanically by solving a linear system
over Z/2^n, and that the solution space is an affine lattice, so arbitrarily
many distinct identities exist for the same target.

## What this toolkit does

- Generates linear MBA identities for targets like `x + y`, `x - y`, `x * 3`,
  `-x`, `x + 1`, and arbitrary constants, over Z/2^32.
- Solves the coefficient system via **modular diagonalization** (a Smith-normal-
  form-style decomposition over Z/2^n), yielding an affine lattice of all valid
  coefficient vectors — sampling it produces identities with large, random
  coefficients.
- Rewrites expression trees (`Expr`) into obfuscated equivalents by extracting
  the linear-MBA part, solving, and substituting back.
- Generates **permutation polynomial pairs** `(p, q)` over Z/2^n with
  `p(q(x)) = x`, via the zero ideal of polynomial functions — usable as
  invertible encodings.
- Verifies every generated identity against 1000 random inputs, plus the full
  4-point uniform truth table.

## Build

Requires a C++20 compiler (uses `auto` parameters and `if constexpr`).
No external dependencies.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```sh
./build/mba --demo   # non-interactive demo: generates identities for all
                     # targets, verifies each with 1000 random tests, and
                     # produces an 8-bit permutation polynomial pair
./build/mba          # interactive menu
```

Interactive options:

1. Generate an MBA identity with a hand-picked basis
2. Generate an obfuscated identity (random basis, large coefficients)
3. Obfuscate a constant value
4. Generate obfuscated identities for all built-in targets
5. Obfuscate a parsed expression tree
6. Generate a permutation polynomial pair

## Layout

| File | Contents |
|---|---|
| `include/ring.h` | `BinaryRing<T>`: arithmetic over Z/2^n for `uint8/16/32/64`, including modular inverse via Newton iteration seeded from a constexpr 8-bit table |
| `include/matrix.h` | `Vector<T>` / `Matrix<T>` over a generic ring, plus `DoubleOps` for the floating-point lattice machinery |
| `include/solver.h`, `src/solver.cpp` | Scalar congruences `a*x = b (mod 2^n)` and `modular_diagonalize`: Smith-style diagonalization over Z/2^n giving the full affine solution lattice of `A*x = b` |
| `include/lattice.h`, `src/lattice.cpp` | `Lattice` / `AffineLattice`, LLL reduction, and CVP solvers (rounding, nearest-plane, plane enumeration) |
| `include/bitwise_expr.h` | `BExpr` (purely bitwise expression trees) and `LBExpr` (linear combinations of `BExpr`s — the linear MBA normal form) |
| `include/expr.h` | `Expr<T>`: mixed arithmetic/bitwise expression trees with parser, evaluator, simplifier, substitution |
| `include/linear_mba.h`, `src/linear_mba.cpp` | The rewrite engine: `solve_linear_system`, `rewrite`, `obfuscate_expr`, `expr_to_lbexpr` / `lbexpr_to_expr` |
| `include/poly.h`, `src/poly.cpp` | `Poly<T>`: polynomials over Z/2^n (eval, compose, derivative, parse/print) |
| `include/perm_poly.h`, `src/perm_poly.cpp` | Permutation polynomials: zero-ideal construction, `is_perm_poly`, composition/inversion, `perm_pair` |
| `main.cpp` | CLI driver: `--demo` mode and the interactive menu |

## How it works

1. Fix a target function `t(x, y)` and a set of basis bitwise expressions
   `f_i(x, y)`.
2. Sample the functions on a uniform input set to build the linear system
   `sum_i c_i * f_i(x, y) = t(x, y)` over Z/2^32.
3. `modular_diagonalize` reduces the system matrix, returning one particular
   solution plus a basis for the homogeneous solution space — an affine lattice.
4. Sampling the lattice with random coefficients yields distinct, valid
   identities; large coefficients make the result look nothing like the target.
5. Every identity is verified by evaluating both sides on 1000 random inputs.

## Known limitations

- The "compile to binary" option in the interactive menu shells out to a
  hardcoded Windows clang path (`C:/Program Files/LLVM/bin/clang.exe`); it only
  works on that setup. The generated `.c` file is still written next to the
  requested output path and can be compiled manually.
- Verification is randomized testing, not a proof — 1000 samples plus the
  uniform truth table is strong evidence, not a guarantee.
- Only two-variable targets are wired into the CLI, though the solver and
  expression machinery are generic over word size and variable count.

## References

- Y. Zhou, A. Main, Y. X. Gu, H. Johnson — *Information Hiding in Software with
  Mixed Boolean-Arithmetic Transforms* (WISA 2007). The linear MBA framework
  this toolkit implements.
- N. Eyrolles — *Obfuscation with Mixed Boolean-Arithmetic Expressions:
  reconstruction, analysis and simplification tools* (PhD thesis, 2017).
  The deobfuscation side of the story.

## License

MIT — see [LICENSE](LICENSE).
