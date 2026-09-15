#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <random>
#include <numeric>
#include <sstream>
#include <fstream>
#include <limits>
#include <cctype>

#include "include/ring.h"
#include "include/matrix.h"
#include "include/solver.h"
#include "include/lattice.h"
#include "include/bitwise_expr.h"
#include "include/expr.h"
#include "include/linear_mba.h"
#include "include/poly.h"
#include "include/perm_poly.h"

using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using i32 = int32_t;

static constexpr u64 MOD      = static_cast<u64>(1) << 32;
static constexpr u32 ALL_ONES = 0xFFFFFFFFu;

static i32 to_signed(u64 v) {
    auto u = static_cast<u32>(v & 0xFFFFFFFFu);
    return static_cast<i32>(u);
}

static size_t parse_size(const std::string& s, size_t fallback) {
    if (s.empty()) return fallback;
    size_t v = 0;
    for (char ch : s) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) return fallback;
        v = v * 10 + static_cast<size_t>(ch - '0');
    }
    return v;
}

static void clear_cin() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

template<typename T>
static void print_perm_pair(std::mt19937& rng, size_t degree) {
    auto zi = ZeroIdeal<T>::init();
    auto pair = perm_pair<T>(rng, zi, degree);
    if (!pair) {
        std::cout << "  Failed to compute an inverse.\n";
        return;
    }
    const auto& [p, q] = *pair;
    std::cout << "  p(x) = " << p.to_string() << "\n";
    std::cout << "  q(x) = " << q.to_string() << "\n";

    auto comp = compose(p, q, zi);
    simplify_poly(comp, zi);
    std::cout << "  p(q(x)) = " << comp.to_string()
              << (comp.is_id() ? "  [identity - PASS]" : "  [FAIL]") << "\n";
}

struct BasisFunc {
    std::string                    name;
    std::string                    c_expr;
    std::function<u32(u32, u32)>   fn;
};

static std::vector<BasisFunc> make_basis_2var() {
    return {
        {"x & y",      "(x & y)",      [](u32 a, u32 b) -> u32 { return a & b; }},
        {"x | y",      "(x | y)",      [](u32 a, u32 b) -> u32 { return a | b; }},
        {"x ^ y",      "(x ^ y)",      [](u32 a, u32 b) -> u32 { return a ^ b; }},
        {"~(x & y)",   "(~(x & y))",   [](u32 a, u32 b) -> u32 { return ~(a & b); }},
        {"~(x | y)",   "(~(x | y))",   [](u32 a, u32 b) -> u32 { return ~(a | b); }},
        {"~(x ^ y)",   "(~(x ^ y))",   [](u32 a, u32 b) -> u32 { return ~(a ^ b); }},
        {"~x & y",     "(~x & y)",     [](u32 a, u32 b) -> u32 { return ~a & b; }},
        {"x & ~y",     "(x & ~y)",     [](u32 a, u32 b) -> u32 { return a & ~b; }},
        {"~x | y",     "(~x | y)",     [](u32 a, u32 b) -> u32 { return ~a | b; }},
        {"x | ~y",     "(x | ~y)",     [](u32 a, u32 b) -> u32 { return a | ~b; }},
        {"~x",         "(~x)",         [](u32 a, u32  ) -> u32 { return ~a; }},
        {"~y",         "(~y)",         [](u32  , u32 b) -> u32 { return ~b; }},
        {"x",          "(x)",          [](u32 a, u32  ) -> u32 { return a; }},
        {"y",          "(y)",          [](u32  , u32 b) -> u32 { return b; }},
    };
}

static std::vector<BasisFunc> make_all_basis_2var() {
    auto basis = make_basis_2var();
    basis.push_back({"0",  "((uint32_t)0)", [](u32, u32) -> u32 { return 0; }});
    basis.push_back({"-1", "(~(uint32_t)0)",[](u32, u32) -> u32 { return ALL_ONES; }});
    return basis;
}

struct TargetFunc {
    std::string                    name;
    std::function<u32(u32, u32)>   fn;
};

static std::vector<TargetFunc> make_targets() {
    return {
        {"x + y",  [](u32 a, u32 b) -> u32 { return a + b; }},
        {"x - y",  [](u32 a, u32 b) -> u32 { return a - b; }},
        {"x * 2",  [](u32 a, u32  ) -> u32 { return a * 2; }},
        {"x * 3",  [](u32 a, u32  ) -> u32 { return a * 3; }},
        {"-x",     [](u32 a, u32  ) -> u32 { return 0u - a; }},
        {"-y",     [](u32  , u32 b) -> u32 { return 0u - b; }},
        {"x + 1",  [](u32 a, u32  ) -> u32 { return a + 1; }},
        {"x - 1",  [](u32 a, u32  ) -> u32 { return a - 1; }},
    };
}

static constexpr u32 UNIFORM_INPUTS[4][2] = {
    {0,        0       },
    {0,        ALL_ONES},
    {ALL_ONES, 0       },
    {ALL_ONES, ALL_ONES},
};
static constexpr int NUM_INPUTS = 4;

struct LinearSystem {
    Matrix<u32> A;
    Vector<u32> b;
};

static LinearSystem build_system(const std::vector<BasisFunc>& basis,
                                 const TargetFunc& target)
{
    size_t cols = basis.size();
    Matrix<u32> A(NUM_INPUTS, cols);
    Vector<u32> b(NUM_INPUTS);

    for (int r = 0; r < NUM_INPUTS; ++r) {
        u32 x = UNIFORM_INPUTS[r][0];
        u32 y = UNIFORM_INPUTS[r][1];
        b[static_cast<size_t>(r)] = target.fn(x, y);
        for (size_t c = 0; c < cols; ++c) {
            A(static_cast<size_t>(r), c) = basis[c].fn(x, y);
        }
    }
    return {std::move(A), std::move(b)};
}

struct Solution {
    std::vector<u32> coeffs;
    bool             valid = false;
};

static Solution solve_modular_system(const LinearSystem& sys) {
    auto lat = solve_via_modular_diagonalize<u32>(
        Matrix<u32>(sys.A), Vector<u32>(sys.b));

    if (lat.is_empty()) return {{}, false};

    std::vector<u32> coeffs(lat.offset.data.begin(), lat.offset.data.end());
    return {coeffs, true};
}

struct MBAResult {
    std::string                target_name;
    std::vector<std::string>   basis_names;
    std::vector<std::string>   basis_c_exprs;
    std::vector<i32>           coeffs_signed;
    std::vector<u32>           coeffs_raw;
    std::string                expression;
    std::string                c_code;
    bool                       valid = false;
    std::string                error;
};

static MBAResult generate_mba(const TargetFunc& target,
                               const std::vector<BasisFunc>& basis)
{
    MBAResult res;
    res.target_name = target.name;

    if (basis.size() < 2) {
        res.error = "Need at least 2 basis functions.";
        return res;
    }

    LinearSystem sys = build_system(basis, target);

    Solution sol = solve_modular_system(sys);
    if (!sol.valid) {
        res.error = "No solution found. Try different/more basis functions.";
        return res;
    }

    res.basis_names.resize(basis.size());
    res.basis_c_exprs.resize(basis.size());
    res.coeffs_raw.resize(basis.size());
    res.coeffs_signed.resize(basis.size());

    for (std::size_t i = 0; i < basis.size(); ++i) {
        res.basis_names[i]   = basis[i].name;
        res.basis_c_exprs[i] = basis[i].c_expr;
        res.coeffs_raw[i]    = sol.coeffs[i];
        res.coeffs_signed[i] = static_cast<i32>(sol.coeffs[i]);
    }

    std::ostringstream expr;
    bool first = true;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        i32 c = res.coeffs_signed[i];
        if (c == 0) continue;

        if (first) {
            if (c == 1)       expr << basis[i].c_expr;
            else if (c == -1) expr << "-" << basis[i].c_expr;
            else              expr << c << " * " << basis[i].c_expr;
            first = false;
        } else {
            if (c > 0) {
                expr << " + ";
                if (c == 1) expr << basis[i].c_expr;
                else        expr << c << " * " << basis[i].c_expr;
            } else {
                expr << " - ";
                if (c == -1) expr << basis[i].c_expr;
                else         expr << (0u - static_cast<u32>(c)) << " * " << basis[i].c_expr;
            }
        }
    }
    if (first) expr << "0";
    res.expression = expr.str();

    std::string fname = target.name;
    for (auto& ch : fname) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) ch = '_';
    }

    std::ostringstream code;
    code << "// MBA expression equivalent to: " << target.name << "\n";
    code << "uint32_t mba_" << fname << "(uint32_t x, uint32_t y) {\n";
    code << "    return " << res.expression << ";\n";
    code << "}\n";
    res.c_code = code.str();

    res.valid = true;
    return res;
}

static MBAResult generate_obfuscated_mba(const TargetFunc& target,
                                          std::mt19937& rng)
{
    auto all_basis = make_all_basis_2var();

    MBAResult best;

    for (int attempt = 0; attempt < 20; ++attempt) {
        std::vector<std::size_t> indices(all_basis.size());
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        std::shuffle(indices.begin(), indices.end(), rng);

        auto count = static_cast<std::size_t>(6 + static_cast<int>(rng() % 7));
        if (count > all_basis.size()) count = all_basis.size();

        std::vector<std::size_t> used(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(count));
        std::sort(used.begin(), used.end());

        std::vector<BasisFunc> subset;
        subset.reserve(count);
        for (auto idx : used) subset.push_back(all_basis[idx]);

        MBAResult res = generate_mba(target, subset);
        if (!res.valid) continue;

        int non_zero  = 0;
        bool has_large = false;
        for (auto c : res.coeffs_signed) {
            if (c != 0) ++non_zero;
            if (std::abs(static_cast<i64>(c)) > 100) has_large = true;
        }

        if (non_zero >= 4 && has_large) return res;
        if (!best.valid || non_zero > 3) best = res;
    }

    return best;
}

static MBAResult obfuscate_constant(u32 constant_val, std::mt19937& rng) {
    TargetFunc target;
    target.name = std::to_string(static_cast<i32>(constant_val));
    target.fn   = [constant_val](u32, u32) -> u32 { return constant_val; };
    return generate_obfuscated_mba(target, rng);
}

static bool verify_mba(const MBAResult& res,
                        const TargetFunc& target,
                        const std::vector<BasisFunc>& all_basis,
                        int num_tests = 1000)
{
    if (!res.valid) return false;

    std::mt19937 rng_v(42);
    std::uniform_int_distribution<u32> dist;

    for (int t = 0; t < num_tests; ++t) {
        u32 x = dist(rng_v);
        u32 y = dist(rng_v);
        u32 expected = target.fn(x, y);

        u64 actual = 0;
        for (std::size_t i = 0; i < res.basis_names.size(); ++i) {
            if (res.coeffs_raw[i] == 0) continue;
            for (const auto& bf : all_basis) {
                if (bf.name == res.basis_names[i]) {
                    actual = (actual + static_cast<u64>(res.coeffs_raw[i]) * static_cast<u64>(bf.fn(x, y))) % MOD;
                    break;
                }
            }
        }

        if (static_cast<u32>(actual) != expected) {
            std::cerr << "  FAIL: x=" << x << " y=" << y
                      << " expected=" << expected
                      << " got=" << static_cast<u32>(actual) << "\n";
            return false;
        }
    }
    return true;
}

static bool shell_safe_path(const std::string& p) {
    if (p.empty()) return false;
    for (unsigned char ch : p) {
        if (!std::isalnum(ch) && ch != '.' && ch != '_' && ch != '-' &&
            ch != '/' && ch != '\\' && ch != ':' && ch != ' ' && ch != '~')
            return false;
    }
    return true;
}

#ifdef _WIN32
static std::string find_on_path(const char* name) {
    const char* path_env = std::getenv("PATH");
    if (!path_env) return "";
    std::istringstream iss(path_env);
    std::string dir;
    while (std::getline(iss, dir, ';')) {
        std::string full = dir + "\\" + name;
        std::ifstream f(full);
        if (f.good()) return full;
    }
    return "";
}
#endif

static std::string find_clang() {
#ifdef _WIN32
    std::string on_path = find_on_path("clang.exe");
    if (!on_path.empty()) return on_path;
    const char* fallback = "C:/Program Files/LLVM/bin/clang.exe";
    std::ifstream f(fallback);
    if (f.good()) return fallback;
    return "";
#else
    return "clang";
#endif
}

static void compile_to_binary(const std::string& expression,
                              const std::vector<std::string>& vars)
{
    std::cout << "\n  Compile to binary? (enter output path, or empty to skip): ";
    std::string out_path;
    std::getline(std::cin, out_path);
    if (out_path.empty()) std::getline(std::cin, out_path);
    if (out_path.empty()) return;

    std::string src_path = out_path + ".c";

    if (!shell_safe_path(out_path) || !shell_safe_path(src_path)) {
        std::cout << "  Path contains characters that are unsafe for a shell command.\n";
        std::cout << "  Source not written.\n";
        return;
    }

    {
        std::ofstream ofs(src_path);
        if (!ofs) {
            std::cout << "  Failed to create " << src_path << "\n";
            return;
        }

        ofs << "#include <stdint.h>\n#include <stdlib.h>\n\n";

        // build parameter list from variables
        std::ostringstream params;
        for (size_t i = 0; i < vars.size(); ++i) {
            if (i) params << ", ";
            params << "uint32_t " << vars[i];
        }

        ofs << "#ifdef _MSC_VER\n";
        ofs << "__declspec(noinline)\n";
        ofs << "#else\n";
        ofs << "__attribute__((noinline))\n";
        ofs << "#endif\n";
        ofs << "uint32_t obfuscated(" << params.str() << ") {\n";
        ofs << "    return (uint32_t)(" << expression << ");\n";
        ofs << "}\n\n";

        ofs << "int main(int argc, char** argv) {\n";
        for (size_t i = 0; i < vars.size(); ++i) {
            ofs << "    uint32_t " << vars[i]
                << " = (argc > " << (i + 1)
                << ") ? (uint32_t)strtoul(argv[" << (i + 1) << "], 0, 0)"
                << " : 0u;\n";
        }
        ofs << "    volatile uint32_t result = obfuscated(";
        for (size_t i = 0; i < vars.size(); ++i) {
            if (i) ofs << ", ";
            ofs << vars[i];
        }
        ofs << ");\n";
        ofs << "    return (int)result;\n";
        ofs << "}\n";
    }

    std::string clang = find_clang();
    if (clang.empty()) {
        std::cout << "  clang not found on PATH; source kept at " << src_path << "\n";
        return;
    }

    std::string cmd = "\"" + clang + "\" -O0 -g -std=c11 -o \""
                    + out_path + "\" \"" + src_path + "\"";
    std::cout << "  Compiling: " << cmd << "\n";
    int rc = std::system(cmd.c_str());
    if (rc == 0) {
        std::cout << "  Binary written to " << out_path << "\n";
        std::cout << "  Source kept at " << src_path << "\n";
    } else {
        std::cout << "  Compilation failed (exit code " << rc << ")\n";
    }
}

static void print_separator(char ch = '-', int width = 72) {
    std::cout << std::string(static_cast<std::size_t>(width), ch) << "\n";
}

static void print_truth_table(const MBAResult& res,
                               const TargetFunc& target,
                               const std::vector<BasisFunc>& all_basis)
{
    std::cout << "\n  Truth Table ({0, -1}^2 inputs - Fundamental Theorem):\n";
    std::cout << "  " << std::setw(12) << "x"
              << std::setw(12) << "y"
              << std::setw(14) << "target"
              << std::setw(14) << "MBA"
              << std::setw(8)  << "match" << "\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    for (int r = 0; r < NUM_INPUTS; ++r) {
        u32 x = UNIFORM_INPUTS[r][0];
        u32 y = UNIFORM_INPUTS[r][1];
        u32 tgt = target.fn(x, y);

        u64 mba = 0;
        for (std::size_t i = 0; i < res.basis_names.size(); ++i) {
            if (res.coeffs_raw[i] == 0) continue;
            for (const auto& bf : all_basis) {
                if (bf.name == res.basis_names[i]) {
                    mba = (mba + static_cast<u64>(res.coeffs_raw[i]) * static_cast<u64>(bf.fn(x, y))) % MOD;
                    break;
                }
            }
        }

        bool match = (static_cast<u32>(mba) == tgt);
        std::cout << "  " << std::setw(12) << static_cast<i32>(x)
                  << std::setw(12) << static_cast<i32>(y)
                  << std::setw(14) << static_cast<i32>(tgt)
                  << std::setw(14) << to_signed(mba)
                  << std::setw(8)  << (match ? "OK" : "FAIL") << "\n";
    }
}

static void print_result_short(const MBAResult& res) {
    if (!res.valid) { std::cout << "  Error: " << res.error << "\n"; return; }
    std::cout << "  " << res.target_name << "  =  " << res.expression << "\n";
}

static void print_result(const MBAResult& res) {
    if (!res.valid) { std::cout << "  Error: " << res.error << "\n"; return; }

    std::cout << "  Target:     " << res.target_name << "\n";
    std::cout << "  Expression: " << res.expression << "\n\n";

    std::cout << "  Coefficients:\n";
    for (std::size_t i = 0; i < res.basis_names.size(); ++i) {
        if (res.coeffs_signed[i] == 0) continue;
        std::cout << "    " << std::setw(12) << res.basis_names[i]
                  << "  ->  " << std::setw(12) << res.coeffs_signed[i]
                  << "  (0x" << std::hex << std::setw(8) << std::setfill('0')
                  << res.coeffs_raw[i] << std::dec << std::setfill(' ') << ")\n";
    }

    std::cout << "\n  C/C++ code:\n";
    std::istringstream stream(res.c_code);
    std::string line;
    while (std::getline(stream, line)) {
        std::cout << "    " << line << "\n";
    }
}

static void option_obfuscate_expr(std::mt19937& rng) {
    std::cout << "\n  Enter expression (e.g., x + y, x ^ y, x & y + 1):\n  > ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) std::getline(std::cin, line);
    if (line.empty()) { std::cout << "  No input.\n"; return; }

    auto expr = ExprOp<u32>::parse(line);
    std::cout << "  Parsed: " << expr->to_string() << "\n";

    ObfuscationConfig cfg;
    std::cout << "  Auxiliary vars [" << cfg.auxiliary_vars << "]: ";
    std::string tmp;
    std::getline(std::cin, tmp);
    if (!tmp.empty()) cfg.auxiliary_vars = parse_size(tmp, cfg.auxiliary_vars);

    std::cout << "  Rewrite depth [" << cfg.rewrite_expr_depth << "]: ";
    std::getline(std::cin, tmp);
    if (!tmp.empty()) cfg.rewrite_expr_depth = parse_size(tmp, cfg.rewrite_expr_depth);

    std::cout << "  Rewrite count [" << cfg.rewrite_expr_count << "]: ";
    std::getline(std::cin, tmp);
    if (!tmp.empty()) cfg.rewrite_expr_count = parse_size(tmp, cfg.rewrite_expr_count);

    std::cout << "\n  Obfuscating...\n";
    obfuscate_expr<u32>(expr, cfg, rng);
    ExprOp<u32>::simplify(expr);

    std::cout << "\n";
    print_separator();
    std::cout << "  Result: " << expr->to_string() << "\n";

    auto orig = ExprOp<u32>::parse(line);
    auto vars_set = orig->vars();
    auto vars2 = expr->vars();
    std::vector<std::string> orig_vars(vars_set.begin(), vars_set.end());

    std::uniform_int_distribution<u32> dist;
    bool pass = true;
    for (int t = 0; t < 1000 && pass; ++t) {
        Valuation<u32> val;
        for (const auto& v : orig_vars) val.set(v, dist(rng));
        for (const auto& v : vars2) {
            if (val.values.find(v) == val.values.end())
                val.set(v, dist(rng));
        }
        u32 expected = orig->eval(val.as_fn());
        u32 actual = expr->eval(val.as_fn());
        if (expected != actual) { pass = false; }
    }
    std::cout << "  Verification (1000 random tests): " << (pass ? "PASSED" : "FAILED") << "\n";

    std::cout << "\n  Save to file? (enter path, or empty to skip): ";
    std::string path;
    std::getline(std::cin, path);
    if (!path.empty()) {
        std::ofstream ofs(path);
        if (ofs) {
            ofs << expr->to_string() << "\n";
            std::cout << "  Written to " << path << "\n";
        } else {
            std::cout << "  Failed to open " << path << "\n";
        }
    }

    {
        auto all_vars = expr->vars();
        std::vector<std::string> var_list(all_vars.begin(), all_vars.end());
        std::sort(var_list.begin(), var_list.end());
        compile_to_binary(expr->to_string(), var_list);
    }

    print_separator();
}

static void option_perm_poly_pair(std::mt19937& rng) {
    std::cout << "\n  Bit width:\n"
              << "    [1] 8-bit\n"
              << "    [2] 16-bit\n"
              << "    [3] 32-bit\n"
              << "  Choice [1]: ";
    std::string tmp;
    std::getline(std::cin, tmp);
    if (tmp.empty()) std::getline(std::cin, tmp);
    int bw = static_cast<int>(parse_size(tmp, 1));

    std::cout << "  Polynomial degree [3]: ";
    std::getline(std::cin, tmp);
    size_t degree = parse_size(tmp, 3);
    if (degree < 1) degree = 1;

    std::cout << "\n  Computing...\n";
    print_separator();

    if (bw == 2) {
        print_perm_pair<uint16_t>(rng, degree);
    } else if (bw == 3) {
        print_perm_pair<uint32_t>(rng, degree);
    } else {
        print_perm_pair<uint8_t>(rng, degree);
    }

    print_separator();
}

static void run_interactive() {
    auto all_basis = make_all_basis_2var();
    auto targets   = make_targets();
    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));

    while (true) {
        std::cout << "\n";
        print_separator('=');
        std::cout << "  Linear MBA Generator\n";
        print_separator('=');
        std::cout << "\n  [1] Generate MBA (custom basis selection)\n"
                  << "  [2] Generate obfuscated MBA (random basis, large coefficients)\n"
                  << "  [3] Obfuscate a constant value\n"
                  << "  [4] Generate all targets (obfuscated)\n"
                  << "  [5] Obfuscate an expression (expression tree)\n"
                  << "  [6] Generate permutation polynomial pair\n"
                  << "  [0] Exit\n\n"
                  << "  Choice: ";

        int choice = 0;
        if (!(std::cin >> choice)) {
            if (std::cin.eof()) break;
            clear_cin();
            continue;
        }
        if (choice == 0) break;

        if (choice == 1) {
            std::cout << "\n  Available targets:\n";
            for (std::size_t i = 0; i < targets.size(); ++i) {
                std::cout << "    [" << i << "] " << targets[i].name << "\n";
            }
            std::cout << "  Target: ";
            int ti = 0;
            if (!(std::cin >> ti)) { clear_cin(); continue; }
            if (ti < 0 || ti >= static_cast<int>(targets.size())) continue;

            std::cout << "\n  Available basis functions:\n";
            for (std::size_t i = 0; i < all_basis.size(); ++i) {
                std::cout << "    [" << std::setw(2) << i << "] " << all_basis[i].name << "\n";
            }
            std::cout << "\n  Enter basis indices separated by spaces (e.g., 0 2 10 11 12 13).\n";
            std::cout << "  End with -1: ";

            std::vector<BasisFunc> selected;
            int idx = 0;
            while (std::cin >> idx && idx >= 0 && idx < static_cast<int>(all_basis.size())) {
                selected.push_back(all_basis[static_cast<std::size_t>(idx)]);
            }

            if (std::cin.fail()) clear_cin();

            if (selected.size() < 2) {
                std::cout << "  Need at least 2 basis functions.\n";
                continue;
            }

            std::cout << "\n";
            print_separator();
            MBAResult res = generate_mba(targets[static_cast<std::size_t>(ti)], selected);
            print_result(res);

            if (res.valid) {
                bool ok = verify_mba(res, targets[static_cast<std::size_t>(ti)], all_basis);
                std::cout << "\n  Verification (1000 random tests): "
                          << (ok ? "PASSED" : "FAILED") << "\n";
                print_truth_table(res, targets[static_cast<std::size_t>(ti)], all_basis);
                compile_to_binary(res.expression, {"x", "y"});
            }
            print_separator();
        }
        else if (choice == 2) {
            std::cout << "\n  Available targets:\n";
            for (std::size_t i = 0; i < targets.size(); ++i) {
                std::cout << "    [" << i << "] " << targets[i].name << "\n";
            }
            std::cout << "  Target: ";
            int ti = 0;
            if (!(std::cin >> ti)) { clear_cin(); continue; }
            if (ti < 0 || ti >= static_cast<int>(targets.size())) continue;

            std::cout << "\n";
            print_separator();
            MBAResult res = generate_obfuscated_mba(targets[static_cast<std::size_t>(ti)], rng);
            print_result(res);

            if (res.valid) {
                bool ok = verify_mba(res, targets[static_cast<std::size_t>(ti)], all_basis);
                std::cout << "\n  Verification (1000 random tests): "
                          << (ok ? "PASSED" : "FAILED") << "\n";
                print_truth_table(res, targets[static_cast<std::size_t>(ti)], all_basis);
                compile_to_binary(res.expression, {"x", "y"});
            }
            print_separator();
        }
        else if (choice == 3) {
            std::cout << "  Enter constant value (signed 32-bit): ";
            i32 val = 0;
            if (!(std::cin >> val)) { clear_cin(); continue; }

            std::cout << "\n";
            print_separator();
            MBAResult res = obfuscate_constant(static_cast<u32>(val), rng);
            print_result(res);

            if (res.valid) {
                TargetFunc ct;
                ct.name = std::to_string(val);
                ct.fn   = [val](u32, u32) -> u32 { return static_cast<u32>(val); };
                bool ok = verify_mba(res, ct, all_basis);
                std::cout << "\n  Verification (1000 random tests): "
                          << (ok ? "PASSED" : "FAILED") << "\n";
                print_truth_table(res, ct, all_basis);
                compile_to_binary(res.expression, {"x", "y"});
            }
            print_separator();
        }
        else if (choice == 4) {
            std::cout << "\n";
            for (auto& t : targets) {
                print_separator();
                MBAResult res = generate_obfuscated_mba(t, rng);
                print_result(res);
                if (res.valid) {
                    bool ok = verify_mba(res, t, all_basis);
                    std::cout << "\n  Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
                }
            }
            print_separator();
        }
        else if (choice == 5) {
            option_obfuscate_expr(rng);
        }
        else if (choice == 6) {
            option_perm_poly_pair(rng);
        }
    }
}

int main(int argc, char** argv) {

    if (argc > 1 && std::string(argv[1]) == "--demo") {
        auto all_basis = make_all_basis_2var();
        auto targets   = make_targets();
        std::mt19937 rng(12345);

        std::cout << "\n  Linear MBA Generator -- Demo\n";
        print_separator('=');

        {
            std::vector<BasisFunc> basis = {all_basis[2], all_basis[0]};
            MBAResult res = generate_mba(targets[0], basis);
            std::cout << "  Classic:  x + y  =  " << res.expression << "\n";
        }

        print_separator();

        std::cout << "  Obfuscated identities:\n\n";
        for (const auto& t : targets) {
            MBAResult res = generate_obfuscated_mba(t, rng);
            bool ok = verify_mba(res, t, all_basis);
            print_result_short(res);
            std::cout << "    " << (ok ? "[PASS]" : "[FAIL]") << "\n\n";
        }

        {
            MBAResult res = obfuscate_constant(42, rng);
            TargetFunc ct{"42", [](u32, u32) -> u32 { return 42; }};
            bool ok = verify_mba(res, ct, all_basis);
            print_result_short(res);
            std::cout << "    " << (ok ? "[PASS]" : "[FAIL]") << "\n";
        }

        print_separator();

        std::cout << "\n  Permutation polynomial pair (8-bit, degree 3):\n";
        print_perm_pair<uint8_t>(rng, 3);

        print_separator('=');
        return 0;
    }

    run_interactive();
    return 0;
}
