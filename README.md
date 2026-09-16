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
- **Nonlinear obfuscation**: wraps each variable in a permutation polynomial
  pair `p(q(x)) = x` before the linear rewrite, then unwraps — the result
  mixes `*` terms into the identity (Zhou et al. section 4).
- **Simplifies** obfuscated linear MBA expressions back to small-coefficient
  identities via nearest-plane CVP over the solution lattice.
- Generates **permutation polynomial pairs** `(p, q)` over Z/2^n with
  `p(q(x)) = x`, via the zero ideal of polynomial functions — usable as
  invertible encodings.
- Verifies every generated identity against 1000 random inputs; the
  interactive menu also prints the full 4-point uniform truth table.

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

Flags:

| Flag | Effect |
|---|---|
| `--demo` | Run the self-verifying demo and exit |
| `--target EXPR` | Generate one obfuscated identity for `EXPR`, verify it, print it, and exit 0 on success / 1 on failure |
| `--simplify EXPR` | Reduce an obfuscated MBA expression to a small-coefficient identity, verify it, and exit 0 on success / 1 on failure |
| `--seed N` | Seed the RNG; without it everything is time-seeded. Two runs with the same seed produce identical output |
| `--hex` | Print coefficients as `0x........` hex instead of signed decimal |
| `-h`, `--help` | Show usage |

```sh
./build/mba --target 'x + y' --vars 3 --hex --seed 1
```

Interactive options:

1. Generate an MBA identity with a hand-picked basis
2. Generate an obfuscated identity (random basis, large coefficients)
3. Obfuscate a constant value
4. Generate obfuscated identities for all built-in targets
5. Obfuscate a parsed expression tree — asks for a method: linear, or
   nonlinear (permutation polynomial wrap)
6. Generate a permutation polynomial pair
7. Simplify an obfuscated MBA expression

Options 1, 2, and 4 first ask for a variable count (2 or 3); the target list
grows accordingly (`x + y + z`, `x ^ y ^ z`, ... for 3 variables).

## Simplify

`simplify` (menu option 7, or `--simplify EXPR`) is the deobfuscation
direction: it takes a linear MBA expression — typically one with huge
coefficients produced by option 5 or `--target` — and finds the
smallest-coefficient identity for it over the standard basis. The
coefficient system is solved as usual, then a nearest-plane CVP step over
the solution lattice shortens the particular solution:

```sh
./build/mba --simplify '305419897*(x ^ y) + 305419898*(x & y) - 305419896*(x | y)'
# Simplified: (x & y) + (x | y)
```

Non-bitwise subtrees (e.g. a `x * y` factor inside a bitwise operand) are
carried through as opaque variables and substituted back afterwards. An
expression with no linear MBA part at all is rejected.

## Nonlinear obfuscation

The nonlinear method in option 5 implements the composition step from Zhou
et al. section 4: for each variable `v` it draws a permutation polynomial
pair `p_v(q_v(x)) = x`, substitutes `p_v(v)` into the expression, runs the
linear rewrite engine on the wrapped tree, then substitutes `q_v(v)` back.
The output is still semantically identical but no longer a linear MBA
expression — it contains `*` terms and is much larger. If a permutation
pair cannot be generated it falls back to the plain linear rewrite.

## Expression grammar

`--target`, `--simplify`, and interactive options 5 and 7 share the same
parser. An expression is:

```
expr     := or_expr
or_expr  := xor_expr  ('|' xor_expr)*
xor_expr := and_expr  ('^' and_expr)*
and_expr := add_expr  ('&' add_expr)*
add_expr := mul_expr  (('+' | '-') mul_expr)*
mul_expr := unary     ('*' unary)*
unary    := '~' unary | '!' unary | '-' unary | primary
primary  := '(' expr ')' | ident | integer
```

`ident` is `[A-Za-z][A-Za-z0-9_]*` (for `--target`, only `x`, `y`, `z`, `w`
are meaningful) and `integer` is a decimal constant taken mod 2^32. `~`/`!`
is bitwise not, `-` is unary minus. Precedence follows C: `|` lowest, then
`^`, `&`, `+`/`-`, `*`, then unary operators. Whitespace is ignored.

## Layout

| File | Contents |
|---|---|
| `include/ring.h` | `BinaryRing<T>`: arithmetic over Z/2^n for `uint8/16/32/64`, including modular inverse via Newton iteration seeded from a constexpr 8-bit table |
| `include/matrix.h` | `Vector<T>` / `Matrix<T>` over a generic ring, plus `DoubleOps` for the floating-point lattice machinery |
| `include/solver.h`, `src/solver.cpp` | Scalar congruences `a*x = b (mod 2^n)` and `modular_diagonalize`: Smith-style diagonalization over Z/2^n giving the full affine solution lattice of `A*x = b` |
| `include/lattice.h`, `src/lattice.cpp` | `Lattice` / `AffineLattice`, plus `cvp_nearest_plane` — the approximate CVP solver used by `simplify` |
| `include/bitwise_expr.h` | `BExpr` (purely bitwise expression trees) and `LBExpr` (linear combinations of `BExpr`s — the linear MBA normal form) |
| `include/expr.h` | `Expr<T>`: mixed arithmetic/bitwise expression trees with parser, evaluator, simplifier, substitution |
| `include/linear_mba.h`, `src/linear_mba.cpp` | The rewrite engine: `solve_linear_system`, `rewrite`, `simplify`, `obfuscate_expr`, `expr_to_lbexpr` / `lbexpr_to_expr` |
| `include/nonlinear.h`, `src/nonlinear.cpp` | `obfuscate_nonlinear`: perm-pair variable wrapping around the linear rewrite engine |
| `include/poly.h`, `src/poly.cpp` | `Poly<T>`: polynomials over Z/2^n (eval, compose, derivative, parse/print) |
| `include/perm_poly.h`, `src/perm_poly.cpp` | Permutation polynomials: zero-ideal construction, `is_perm_poly`, composition/inversion, `perm_pair` |
| `main.cpp` | CLI driver: `--demo`, `--target`/`--simplify`/`--vars`/`--seed`/`--hex` flags, and the interactive menu |
| `tests/test_mba.cpp` | Test suite covering the solver, diagonalization, expression round-trips, `simplify`, and `obfuscate_nonlinear` (run via ctest) |

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

- The "compile to binary" option in the interactive menu writes a `.c` file
  next to the requested output path and invokes `clang` from `PATH` (on
  Windows it falls back to `C:/Program Files/LLVM/bin/clang.exe`). Output
  paths containing shell metacharacters are rejected.
- Verification is randomized testing, not a proof — 1000 samples is strong
  evidence, not a guarantee.
- The interactive menu supports 2- and 3-variable targets; `--target` accepts
  up to 4 (`x`, `y`, `z`, `w`). The solver and expression machinery are
  generic over word size and variable count beyond that.
- `--target` finds a linear identity over the sampled basis; a non-linear
  target (e.g. `x * y`) may still produce a candidate that fails the random
  verification, in which case the exit code is 1.
- `simplify`'s CVP step runs nearest-plane in floating point over Z, which
  does not see wraparound — it approximates the true closest vector over
  Z/2^n, so the result is small but not guaranteed minimal.
- Nonlinear obfuscation output is large: each variable expands into a
  degree-3 polynomial before and after the linear rewrite.

## References

- Y. Zhou, A. Main, Y. X. Gu, H. Johnson — *Information Hiding in Software with
  Mixed Boolean-Arithmetic Transforms* (WISA 2007). The linear MBA framework
  this toolkit implements.
- N. Eyrolles — *Obfuscation with Mixed Boolean-Arithmetic Expressions:
  reconstruction, analysis and simplification tools* (PhD thesis, 2017).
  The deobfuscation side of the story.

## License

MIT — see [LICENSE](LICENSE).
