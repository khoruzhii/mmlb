#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "CLI11.hpp"
#include "cnpy.h"

using Matrix = std::uint32_t;

// Algorithm:
// 1. Reduce the first matrix to diag(I_r,0), so only its rank remains.
// 2. Enumerate the orbits of the other two matrices under the explicit block stabilizer.
// 3. For square tensors, merge the ordered orbits under the six S3 symmetries.

static int bit(Matrix x, int cols, int row, int col) {
    return static_cast<int>((x >> (row * cols + col)) & 1U);
}

static Matrix identity(int n) {
    Matrix x = 0;
    for (int i = 0; i < n; ++i) x |= Matrix{1} << (i * n + i);
    return x;
}

static Matrix transvection(int n, int row, int col) {
    Matrix x = identity(n);
    x ^= Matrix{1} << (row * n + col);
    return x;
}

static Matrix multiply(Matrix a, int rows_a, int cols_a,
                       Matrix b, int rows_b, int cols_b) {
    if (cols_a != rows_b) std::abort();
    Matrix c = 0;
    for (int i = 0; i < rows_a; ++i) {
        for (int k = 0; k < cols_a; ++k) {
            if (!bit(a, cols_a, i, k)) continue;
            for (int j = 0; j < cols_b; ++j) {
                if (bit(b, cols_b, k, j)) c ^= Matrix{1} << (i * cols_b + j);
            }
        }
    }
    return c;
}

static Matrix transpose(Matrix a, int rows, int cols) {
    Matrix t = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (bit(a, cols, i, j)) t |= Matrix{1} << (j * rows + i);
        }
    }
    return t;
}

static Matrix inverse(Matrix a, int n) {
    std::vector<Matrix> r(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (bit(a, n, i, j)) r[i] |= Matrix{1} << j;
        }
        r[i] |= Matrix{1} << (n + i);
    }

    int pivots = 0;
    for (int col = 0; col < n; ++col) {
        int pivot = pivots;
        while (pivot < n && ((r[pivot] >> col) & 1U) == 0) ++pivot;
        if (pivot == n) return 0;
        std::swap(r[pivots], r[pivot]);
        for (int i = 0; i < n; ++i) {
            if (i != pivots && ((r[i] >> col) & 1U)) r[i] ^= r[pivots];
        }
        ++pivots;
    }

    Matrix out = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((r[i] >> (n + j)) & 1U) out |= Matrix{1} << (i * n + j);
        }
    }
    return out;
}

static Matrix diagonal_rectangle(int r, int cols) {
    Matrix a = 0;
    for (int i = 0; i < r; ++i) a |= Matrix{1} << (i * cols + i);
    return a;
}

static std::string bits(Matrix a, int rows, int cols) {
    std::string s;
    s.reserve(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) s.push_back(bit(a, cols, i, j) ? '1' : '0');
    }
    return s;
}

struct Triple {
    Matrix a = 0;
    Matrix b = 0;
    Matrix c = 0;
};

using PackedTensor = std::array<std::uint64_t, 12>;

static void add_rank_one(PackedTensor& tensor, Matrix u, Matrix v, Matrix w) {
    for (; u != 0; u &= u - 1) {
        const int i = std::countr_zero(u);
        for (Matrix y = v; y != 0; y &= y - 1) {
            const int j = std::countr_zero(y);
            for (Matrix z = w; z != 0; z &= z - 1) {
                const int k = std::countr_zero(z);
                const int tensor_bit = (9 * i + j) * 9 + k;
                tensor[tensor_bit / 64] ^=
                    std::uint64_t{1} << (tensor_bit % 64);
            }
        }
    }
}

static PackedTensor matrix_multiplication_tensor(int a, int b, int c) {
    PackedTensor tensor{};
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < b; ++j) {
            for (int k = 0; k < c; ++k) {
                add_rank_one(tensor, Matrix{1} << (b * i + j),
                             Matrix{1} << (c * j + k),
                             Matrix{1} << (a * k + i));
            }
        }
    }
    return tensor;
}

static void save_children(const std::string& stem, int a, int b, int c,
                          const std::vector<Triple>& representatives) {
    const PackedTensor tensor = matrix_multiplication_tensor(a, b, c);
    std::vector<std::uint64_t> packed;
    packed.reserve(12 * representatives.size());
    for (const Triple& representative : representatives) {
        PackedTensor child = tensor;
        add_rank_one(child, representative.a, representative.b,
                     representative.c);
        packed.insert(packed.end(), child.begin(), child.end());
    }

    const std::filesystem::path path =
        std::filesystem::path("data/rk1step") / ("t" + stem + ".npy");
    cnpy::npy_save(path.string(), packed.data(),
                   {representatives.size(), PackedTensor{}.size()}, "w");
    std::cout << "  saved " << path.string() << " shape=("
              << representatives.size() << ",12) dtype=uint64\n";
}

struct ActionGenerator {
    Matrix p = 0;
    Matrix q = 0;
    Matrix r = 0;
};

static std::vector<Matrix> gl_generators(int n) {
    std::vector<Matrix> generators;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) generators.push_back(transvection(n, i, j));
        }
    }
    return generators;
}

// Stabilizer of A0 = diag(I_r,0):
// P = [S X; 0 U], Q = [S 0; Y V].
static std::vector<ActionGenerator> stabilizer_generators(int a, int b, int c, int r) {
    std::vector<ActionGenerator> generators;
    const Matrix ia = identity(a);
    const Matrix ib = identity(b);
    const Matrix ic = identity(c);

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            if (i == j) continue;
            Matrix p = ia ^ (Matrix{1} << (i * a + j));
            Matrix q = ib ^ (Matrix{1} << (i * b + j));
            generators.push_back({p, q, ic});
        }
    }

    for (int i = r; i < a; ++i) {
        for (int j = r; j < a; ++j) {
            if (i != j) generators.push_back({ia ^ (Matrix{1} << (i * a + j)), ib, ic});
        }
    }

    for (int i = r; i < b; ++i) {
        for (int j = r; j < b; ++j) {
            if (i != j) generators.push_back({ia, ib ^ (Matrix{1} << (i * b + j)), ic});
        }
    }

    for (int i = 0; i < r; ++i) {
        for (int j = r; j < a; ++j) {
            generators.push_back({ia ^ (Matrix{1} << (i * a + j)), ib, ic});
        }
    }

    for (int i = r; i < b; ++i) {
        for (int j = 0; j < r; ++j) {
            generators.push_back({ia, ib ^ (Matrix{1} << (i * b + j)), ic});
        }
    }

    for (Matrix x : gl_generators(c)) generators.push_back({ia, ib, x});
    return generators;
}

static std::pair<Matrix, Matrix> act_on_pair(Matrix b_matrix, Matrix c_matrix,
                                             const ActionGenerator& g,
                                             int a, int b, int c) {
    // Every generator used here is an involution, so g.p⁻¹ = g.p and g.r⁻¹ = g.r.
    Matrix new_b = multiply(multiply(g.q, b, b, b_matrix, b, c), b, c, g.r, c, c);
    Matrix new_c = multiply(multiply(g.r, c, c, c_matrix, c, a), c, a, g.p, a, a);
    return {new_b, new_c};
}

static int pair_index(Matrix b_matrix, Matrix c_matrix, int c_count) {
    return static_cast<int>(b_matrix - 1) * c_count + static_cast<int>(c_matrix - 1);
}

static std::pair<Matrix, Matrix> decode_pair(int index, int c_count) {
    return {static_cast<Matrix>(index / c_count + 1),
            static_cast<Matrix>(index % c_count + 1)};
}

struct RankLayer {
    int r = 0;
    std::vector<int> orbit_of_pair;
};

struct OrderedOrbits {
    int a = 0;
    int b = 0;
    int c = 0;
    std::vector<Triple> representatives;
    std::vector<RankLayer> layers;
};

static OrderedOrbits find_ordered_orbits(int a, int b, int c) {
    OrderedOrbits result;
    result.a = a;
    result.b = b;
    result.c = c;

    const int b_count = (1 << (b * c)) - 1;
    const int c_count = (1 << (c * a)) - 1;
    const int pair_count = b_count * c_count;

    for (int r = 1; r <= std::min(a, b); ++r) {
        RankLayer layer;
        layer.r = r;
        layer.orbit_of_pair.assign(pair_count, -1);
        const Matrix a0 = diagonal_rectangle(r, b);
        const auto generators = stabilizer_generators(a, b, c, r);

        for (Matrix bm = 1; bm <= static_cast<Matrix>(b_count); ++bm) {
            for (Matrix cm = 1; cm <= static_cast<Matrix>(c_count); ++cm) {
                const int start = pair_index(bm, cm, c_count);
                if (layer.orbit_of_pair[start] != -1) continue;

                const int orbit_id = static_cast<int>(result.representatives.size());
                result.representatives.push_back({a0, bm, cm});
                layer.orbit_of_pair[start] = orbit_id;

                std::queue<int> queue;
                queue.push(start);
                while (!queue.empty()) {
                    const int current = queue.front();
                    queue.pop();
                    const auto [current_b, current_c] = decode_pair(current, c_count);
                    for (const auto& generator : generators) {
                        const auto [next_b, next_c] = act_on_pair(current_b, current_c, generator, a, b, c);
                        const int next = pair_index(next_b, next_c, c_count);
                        if (layer.orbit_of_pair[next] == -1) {
                            layer.orbit_of_pair[next] = orbit_id;
                            queue.push(next);
                        }
                    }
                }
            }
        }
        result.layers.push_back(std::move(layer));
    }
    return result;
}

struct Canonicalizer {
    int r = 0;
    Matrix q = 0;
    Matrix p_inverse = 0;
};

static void swap_rows(Matrix& a, int cols, int x, int y) {
    if (x == y) return;
    for (int j = 0; j < cols; ++j) {
        const int bx = bit(a, cols, x, j);
        const int by = bit(a, cols, y, j);
        if (bx != by) {
            a ^= Matrix{1} << (x * cols + j);
            a ^= Matrix{1} << (y * cols + j);
        }
    }
}

static void add_row(Matrix& a, int cols, int target, int source) {
    for (int j = 0; j < cols; ++j) {
        if (bit(a, cols, source, j)) a ^= Matrix{1} << (target * cols + j);
    }
}

static void swap_columns(Matrix& a, int rows, int cols, int x, int y) {
    if (x == y) return;
    for (int i = 0; i < rows; ++i) {
        const int bx = bit(a, cols, i, x);
        const int by = bit(a, cols, i, y);
        if (bx != by) {
            a ^= Matrix{1} << (i * cols + x);
            a ^= Matrix{1} << (i * cols + y);
        }
    }
}

static void add_column(Matrix& a, int rows, int cols, int target, int source) {
    for (int i = 0; i < rows; ++i) {
        if (bit(a, cols, i, source)) a ^= Matrix{1} << (i * cols + target);
    }
}

// Find P and H with P A H = diag(I_r,0). In the sandwich action H = Q⁻¹.
static Canonicalizer canonicalize_square(Matrix original, int n) {
    Matrix a = original;
    Matrix p = identity(n);
    Matrix h = identity(n);
    int r = 0;

    while (r < n) {
        int pivot_row = -1;
        int pivot_col = -1;
        for (int i = r; i < n && pivot_row == -1; ++i) {
            for (int j = r; j < n; ++j) {
                if (bit(a, n, i, j)) {
                    pivot_row = i;
                    pivot_col = j;
                    break;
                }
            }
        }
        if (pivot_row == -1) break;

        swap_rows(a, n, r, pivot_row);
        swap_rows(p, n, r, pivot_row);
        swap_columns(a, n, n, r, pivot_col);
        swap_columns(h, n, n, r, pivot_col);

        for (int i = 0; i < n; ++i) {
            if (i != r && bit(a, n, i, r)) {
                add_row(a, n, i, r);
                add_row(p, n, i, r);
            }
        }
        for (int j = 0; j < n; ++j) {
            if (j != r && bit(a, n, r, j)) {
                add_column(a, n, n, j, r);
                add_column(h, n, n, j, r);
            }
        }
        ++r;
    }

    if (a != diagonal_rectangle(r, n)) std::abort();
    return {r, inverse(h, n), inverse(p, n)};
}

static std::vector<Canonicalizer> square_canonicalizers(int n) {
    std::vector<Canonicalizer> result(1U << (n * n));
    for (Matrix a = 1; a < (Matrix{1} << (n * n)); ++a) {
        result[a] = canonicalize_square(a, n);
    }
    return result;
}

static int ordered_orbit_of(const Triple& x, const OrderedOrbits& ordered,
                            const std::vector<Canonicalizer>& canonicalizers) {
    const int n = ordered.a;
    const auto& k = canonicalizers[x.a];
    Matrix b = multiply(k.q, n, n, x.b, n, n);
    Matrix c = multiply(x.c, n, n, k.p_inverse, n, n);
    const int c_count = (1 << (n * n)) - 1;
    return ordered.layers[k.r - 1].orbit_of_pair[pair_index(b, c, c_count)];
}

struct DisjointSet {
    std::vector<int> parent;

    explicit DisjointSet(int n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y) parent[y] = x;
    }
};

static std::vector<Triple> merge_square_s3(const OrderedOrbits& ordered) {
    const int n = ordered.a;
    const auto canonicalizers = square_canonicalizers(n);
    DisjointSet sets(static_cast<int>(ordered.representatives.size()));

    for (int i = 0; i < static_cast<int>(ordered.representatives.size()); ++i) {
        const Triple x = ordered.representatives[i];
        const Matrix at = transpose(x.a, n, n);
        const Matrix bt = transpose(x.b, n, n);
        const Matrix ct = transpose(x.c, n, n);
        const std::vector<Triple> images = {
            x,
            {x.b, x.c, x.a},
            {x.c, x.a, x.b},
            {ct, bt, at},
            {bt, at, ct},
            {at, ct, bt}
        };
        for (const Triple& image : images) {
            sets.unite(i, ordered_orbit_of(image, ordered, canonicalizers));
        }
    }

    std::vector<int> first(ordered.representatives.size(), -1);
    for (int i = 0; i < static_cast<int>(ordered.representatives.size()); ++i) {
        const int root = sets.find(i);
        if (first[root] == -1) first[root] = i;
    }

    std::vector<Triple> result;
    for (int i : first) {
        if (i != -1) result.push_back(ordered.representatives[i]);
    }
    std::sort(result.begin(), result.end(), [](const Triple& x, const Triple& y) {
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        return x.c < y.c;
    });
    return result;
}

static std::string format(const Triple& x, int a, int b, int c) {
    return bits(x.a, a, b) + "x" + bits(x.b, b, c) + "x" + bits(x.c, c, a);
}

static void run_case(const std::string& name, int a, int b, int c,
                     bool use_square_s3, int expected_ordered,
                     int expected_full, bool save, const std::string& stem) {
    const auto start = std::chrono::steady_clock::now();
    const OrderedOrbits ordered = find_ordered_orbits(a, b, c);
    const std::vector<Triple> representatives =
        use_square_s3 ? merge_square_s3(ordered) : ordered.representatives;
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();

    if (static_cast<int>(ordered.representatives.size()) != expected_ordered ||
        static_cast<int>(representatives.size()) != expected_full) {
        std::cerr << name << ": orbit-count verification failed\n";
        std::exit(1);
    }

    std::cout << name
              << ": ordered=" << ordered.representatives.size()
              << ", Aut(T)=" << representatives.size()
              << ", time=" << std::fixed << std::setprecision(6) << seconds << " s\n";
    for (const Triple& representative : representatives) {
        std::cout << "  " << format(representative, a, b, c) << '\n';
    }
    if (save) save_children(stem, a, b, c, representatives);
    std::cout << '\n';
}

int main(int argc, char** argv) {
    CLI::App app{"Rank-one orbits of matrix-multiplication tensors"};
    bool save = false;
    app.add_flag("--save", save,
                 "Save packed T+z representatives to data/rk1step");
    CLI11_PARSE(app, argc, argv);

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (save) std::filesystem::create_directories("data/rk1step");

    const auto total_start = std::chrono::steady_clock::now();
    run_case("M222", 2, 2, 2, true, 29, 13, save, "222");
    run_case("M223", 2, 2, 3, false, 41, 41, save, "223");
    run_case("M233", 2, 3, 3, false, 79, 79, save, "233");
    run_case("M333", 3, 3, 3, true, 211, 67, save, "333");

    const auto total_stop = std::chrono::steady_clock::now();
    std::cout << "Total time: "
              << std::fixed << std::setprecision(6)
              << std::chrono::duration<double>(total_stop - total_start).count()
              << " s\n";
}
