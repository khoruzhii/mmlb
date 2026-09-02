#pragma once

#include "symmetries.h"
#include "substitution.h"
#include <bit>
#include <utility>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath>

// basic use of subspaces

using Subspace = std::vector<U16>; // Basis for a subspace

inline Subspace rref(Subspace basis) { // returns it in rref OR as {} if it isn't full rank
    Subspace rref_basis;
    for (U16 v : basis) {
        if (v==0) return {};
        for (U16 b : rref_basis) {
            int p = std::countr_zero(b);
            if ((v >> p) & 1) v ^= b;
        }
        if (v==0) return {};
        int p = std::countr_zero(v);
        for (U16& b : rref_basis) {
            if ((b >> p) & 1) b ^= v;
        }
        rref_basis.push_back(v);
        std::sort(rref_basis.begin(), rref_basis.end(), [](U16 a, U16 b){return std::countr_zero(a) < std::countr_zero(b);});
    }
    return rref_basis;
}

inline Subspace apply_symmetry(const std::vector<U64>& sym, const Subspace& S, size_t dim, int axis) {
    Subspace out;
    out.reserve(S.size());
    for (U16 v : S) out.push_back(apply_symmetry(sym, v, dim, axis));
    return rref(out);
}

// generates all 2^d-1 non-zero vectors in subspace S
inline std::vector<U16> get_subspace_elements(const Subspace& S) {
    std::vector<U16> elements;
    size_t k = S.size();
    if (k==0) return elements;
    elements.reserve((1 << k)-1);
    for (size_t mask = 1; mask < (1ULL << k); mask++) {
        U16 val = 0;
        for (size_t bit = 0; bit < k; bit++) {
            if ((mask >> bit) & 1) val ^= S[bit];
        }
        elements.push_back(val);
    }
    return elements;
}

// reduces w mod S, assuming S is in rref
inline U16 proj_quotient(U16 w, const Subspace& S) {
    for (U16 b : S) {
        int p = std::countr_zero(b);
        if ((w >> p) & 1) w ^= b;
    }
    return w;
}

// Fast stack-allocated matrix representation of a symmetry on axis
struct MatrixSym {
    U16 col[16];
    U16 row[16];
};

inline MatrixSym extract_sym_matrix(const std::vector<U64>& sym, size_t dim, int axis) {
    MatrixSym M{};
    for (size_t i = 0; i < dim; i++) {
        M.row[i] = (sym[axis * 4 + i / 4] >> (16 * (i % 4))) & 0xFFFF;
    }
    for (size_t j = 0; j < dim; j++) {
        U16 c = 0;
        for (size_t i = 0; i < dim; i++) {
            if ((M.row[i] >> j) & 1) c |= (1 << i);
        }
        M.col[j] = c;
    }
    return M;
}

inline U16 apply_mat(const MatrixSym& M, U16 v) {
    U16 out = 0;
    while (v) {
        int j = std::countr_zero(v);
        out ^= M.col[j];
        v &= v - 1;
    }
    return out;
}

// Stack-allocated Subspace representation for up to 16 basis vectors (zero heap allocs)
struct Subspace16 {
    U16 v[16];
    U8 size;

    U64 key0() const {
        return (U64)v[0] | ((U64)v[1] << 16) | ((U64)v[2] << 32) | ((U64)v[3] << 48);
    }
    U64 key1() const {
        return (U64)v[4] | ((U64)v[5] << 16) | ((U64)v[6] << 32) | ((U64)v[7] << 48);
    }
    U64 key2() const {
        return (U64)v[8] | ((U64)v[9] << 16) | ((U64)v[10] << 32) | ((U64)v[11] << 48);
    }
    U64 key3() const {
        return (U64)v[12] | ((U64)v[13] << 16) | ((U64)v[14] << 32) | ((U64)v[15] << 48);
    }
    bool operator==(const Subspace16& o) const {
        if (size != o.size) return false;
        for (int i = 0; i < size; i++) if (v[i] != o.v[i]) return false;
        return true;
    }
};

inline Subspace16 stack_rref(Subspace16 s) {
    Subspace16 res{};
    for (int i = 0; i < s.size; i++) {
        U16 vec = s.v[i];
        if (vec == 0) return Subspace16{{}, 0};
        for (int j = 0; j < res.size; j++) {
            int p = std::countr_zero(res.v[j]);
            if ((vec >> p) & 1) vec ^= res.v[j];
        }
        if (vec == 0) return Subspace16{{}, 0};
        int p = std::countr_zero(vec);
        for (int j = 0; j < res.size; j++) {
            if ((res.v[j] >> p) & 1) res.v[j] ^= vec;
        }
        if (res.size < 16) {
            res.v[res.size++] = vec;
        }
        for (int j = res.size - 1; j > 0; j--) {
            if (std::countr_zero(res.v[j]) < std::countr_zero(res.v[j - 1])) {
                std::swap(res.v[j], res.v[j - 1]);
            }
        }
    }
    return res;
}

struct SubKey256 {
    U64 k0, k1, k2, k3;
    bool operator==(const SubKey256& o) const {
        return k0 == o.k0 && k1 == o.k1 && k2 == o.k2 && k3 == o.k3;
    }
};

struct SubKey256Hash {
    size_t operator()(const SubKey256& k) const {
        size_t h = k.k0 ^ (k.k1 * 1315423911ULL);
        h ^= (k.k2 * 2654435761ULL) ^ (k.k3 * 1000000007ULL);
        return h;
    }
};

inline SubKey256 make_key256(const Subspace16& s) {
    return {s.key0(), s.key1(), s.key2(), s.key3()};
}

inline Subspace to_subspace(const Subspace16& s) {
    Subspace out(s.size);
    for (int i = 0; i < s.size; i++) out[i] = s.v[i];
    return out;
}

inline Subspace16 from_subspace(const Subspace& s) {
    Subspace16 out{};
    out.size = (U8)std::min<size_t>(s.size(), 16);
    for (size_t i = 0; i < out.size; i++) out.v[i] = s[i];
    return out;
}

// Counts the size of the Grassmannian. Used for heuristics, so we don't mind a small amount of float errors
inline double gaussian_binomial_2(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k;
    double res = 1.0;
    for (int i = 0; i < k; i++) {
        res *= (std::pow(2.0, n - i) - 1.0) / (std::pow(2.0, k - i) - 1.0);
    }
    return res;
}

// Fast inductive generation of d-dimensional subspace orbit representatives under group G
inline std::vector<Subspace> get_subspace_orbit_reps(
    const std::vector<std::vector<U64>>& syms,
    size_t dim, int axis, size_t target_d, size_t max_reps = 30)
{
    if (target_d == 0) return {{}};

    // Early mathematical orbit count lower bound: Total subspaces / |G|
    double total_subspaces = gaussian_binomial_2((int)dim, (int)target_d);
    double group_size = syms.empty() ? 1.0 : (double)syms.size();
    if (total_subspaces / group_size > (double)max_reps) { // too many subspaces to enumerate, abort
        return std::vector<Subspace>(max_reps + 1);
    }
    if (target_d == 1) {
        auto raw_orbits = get_orbits(syms, dim, axis);
        std::vector<Subspace> reps;
        for (auto& orb : raw_orbits) {
            for (U16 v : orb) {
                if (v != 0) {
                    reps.push_back({v});
                    break;
                }
            }
            if (reps.size() > max_reps) break;
        }
        return reps;
    }

    std::vector<MatrixSym> mat_syms;
    mat_syms.reserve(syms.size());
    for (const auto& sym : syms) {
        mat_syms.push_back(extract_sym_matrix(sym, dim, axis));
    }

    // Get parent representatives of dimension (target_d - 1)
    std::vector<Subspace> prev_reps = get_subspace_orbit_reps(syms, dim, axis, target_d - 1, max_reps);
    if (prev_reps.size() > max_reps) {
        return std::vector<Subspace>(max_reps + 1);
    }

    std::unordered_set<SubKey256, SubKey256Hash> visited;
    std::vector<Subspace> reps;

    for (const auto& parent_vec : prev_reps) {
        Subspace16 parent = from_subspace(parent_vec);
        int last_pivot = parent.size == 0 ? -1 : std::countr_zero(parent.v[parent.size - 1]);

        for (int p_d = last_pivot + 1; p_d < (int)dim; p_d++) {
            std::vector<int> free_bits;
            for (int j = p_d + 1; j < (int)dim; j++) {
                free_bits.push_back(j);
            }
            int num_free = free_bits.size();
            int max_mask = (1 << num_free);

            for (int mask = 0; mask < max_mask; mask++) {
                U16 v = (1 << p_d);
                for (int bit = 0; bit < num_free; bit++) {
                    if ((mask >> bit) & 1) v |= (1 << free_bits[bit]);
                }

                Subspace16 cand = parent;
                for (int i = 0; i < cand.size; i++) {
                    if ((cand.v[i] >> p_d) & 1) cand.v[i] ^= v;
                }
                if (cand.size < 16) {
                    cand.v[cand.size++] = v;
                }

                SubKey256 key = make_key256(cand);
                if (visited.count(key)) continue;

                reps.push_back(to_subspace(cand));
                if (reps.size() > max_reps) {
                    return reps; // Early exit!
                }

                // Compute orbit via BFS
                std::queue<Subspace16> q;
                q.push(cand);
                visited.insert(key);

                while (!q.empty()) {
                    Subspace16 curr = q.front();
                    q.pop();

                    for (const auto& M : mat_syms) {
                        Subspace16 next{};
                        next.size = curr.size;
                        for (int i = 0; i < curr.size; i++) {
                            next.v[i] = apply_mat(M, curr.v[i]);
                        }
                        next = stack_rref(next);
                        if (next.size == 0) continue;

                        SubKey256 next_key = make_key256(next);
                        if (visited.insert(next_key).second) {
                            q.push(next);
                        }
                    }
                }
            }
        }
    }
    return reps;
}

inline std::vector<Subspace> get_subspace_orbit_reps(Tensor& T, size_t dim, int axis, size_t d = 2, size_t max_reps = 30) {
    auto syms = symmetry_generators(T, {}, {}, {});
    return get_subspace_orbit_reps(syms, dim, axis, d, max_reps);
}

inline Tensor apply_subspace_substitution(const Tensor& T, const Subspace& S, int axis) {
    if (S.empty()) return T;
    Subspace S_rref = rref(S);
    size_t d = S_rref.size();
    size_t n = T.shape[axis];
    if (d >= n) {
        Tensor empty;
        empty.shape = T.shape;
        empty.shape[axis] = 0;
        return empty;
    }

    // Find pivot and non-pivot columns
    std::vector<int> pivots;
    std::vector<bool> is_pivot(n, false);
    for (U16 b : S_rref) {
        int p = std::countr_zero(b);
        pivots.push_back(p);
        is_pivot[p] = true;
    }
    std::vector<int> non_pivots;
    for (size_t i = 0; i < n; i++) {
        if (!is_pivot[i]) non_pivots.push_back(i);
    }

    // Orient tensor so substitution axis is axis 0
    Tensor T_rot = T;
    if (axis == 1) T_rot = T_rot.transpose_AB();
    else if (axis == 2) T_rot = T_rot.transpose_BC().transpose_AB();

    Tensor T_out;
    T_out.shape = T_rot.shape;
    T_out.shape[0] = n - d;

    // Base: non-pivot slices
    for (size_t k = 0; k < non_pivots.size(); k++) {
        int src = non_pivots[k];
        for (int w = 0; w < 4; w++) {
            T_out.mut_raw()[4 * k + w] = T_rot.raw()[4 * src + w];
        }
    }

    // Add pivot slices according to relations in S_rref
    for (size_t i = 0; i < d; i++) {
        int p = pivots[i];
        U16 b = S_rref[i];
        for (size_t k = 0; k < non_pivots.size(); k++) {
            int q = non_pivots[k];
            if ((b >> q) & 1) {
                for (int w = 0; w < 4; w++) {
                    T_out.mut_raw()[4 * k + w] ^= T_rot.raw()[4 * p + w];
                }
            }
        }
    }

    // Rotate back
    if (axis == 1) T_out = T_out.transpose_AB();
    else if (axis == 2) T_out = T_out.transpose_AB().transpose_BC();

    return T_out.nf();
}