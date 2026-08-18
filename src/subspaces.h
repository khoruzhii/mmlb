#pragma once

#include "symmetries.h"
#include <bit>
#include <utility>
#include <set>
#include <algorithm>

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

// IDEA: generate all d-dimensional subspaces up to symmetry by first creating a list of candidates
// then we check for symmetry between them all. We can do this recursively picking a new basis element
// and have each one be chosen uniquely up to symmetry. This stops us from checking exponentially large
// Grassmannians.

inline void get_subspace_candidates_rec(Tensor& T, size_t dim, int axis, size_t target_d, const Subspace& current_subspace, std::vector<Subspace>& candidates) {
    if (current_subspace.size() == target_d) {
        candidates.push_back(current_subspace);
        return;
    }
    // First get the symmetries
    std::vector<std::vector<U64>> syms_locked;
    if (axis == 0) syms_locked = symmetry_generators(T, current_subspace, {}, {});
    else if (axis == 1) syms_locked = symmetry_generators(T, {}, current_subspace, {});
    else syms_locked = symmetry_generators(T, {}, {}, current_subspace);
    // Now we search the space
    std::vector<bool> visited(1<<dim, false);
    for (U16 v = 1; v < (1 << dim); v++) {
        bool in_space = true;
        for (U16 b : current_subspace) {
            if ((v >> std::countr_zero(b))&1) {in_space = false; break;}
        }
        if (!in_space || visited[v]) continue;
        visited[v] = true;
        // now we need to check all the terms that might contain v, and then remove all choices equivalent to v
        Subspace next_subspace = current_subspace;
        next_subspace.push_back(v);
        next_subspace = rref(next_subspace);
        get_subspace_candidates_rec(T, dim, axis, target_d, next_subspace, candidates);
        std::vector<U16> v_orbit = {v};
        size_t head = 0;
        while (head < v_orbit.size()) {
            U16 current = v_orbit[head++];
            for (const auto& sym : syms_locked) {
                U16 transformed = apply_symmetry(sym, current, dim, axis);
                U16 next = proj_quotient(transformed, current_subspace);
                if (!visited[next]) {
                    visited[next] = true;
                    v_orbit.push_back(next);
                }
            }
        }
    }
}

// uses the above to find all d dimensional subspaces up to orbit equivalence
inline std::vector<Subspace> get_subspace_orbit_reps(Tensor& T, size_t dim, int axis, size_t d = 2) {
    if (d==0) return {{}};
    auto syms_all = symmetry_generators(T, {}, {}, {});
    std::vector<Subspace> candidates;
    get_subspace_candidates_rec(T, dim, axis, d, {}, candidates);
    std::vector<Subspace> reps;
    std::set<Subspace> visited;
    for (const auto& candidate : candidates) {
        if (visited.count(candidate)) continue;
        reps.push_back(candidate);
        visited.insert(candidate);
        std::vector<Subspace> orbit = {candidate};
        size_t head = 0;
        while (head < orbit.size()) {
            const Subspace current_rep = orbit[head++];
            for (const auto& sym : syms_all) {
                Subspace transformed_rep = apply_symmetry(sym, current_rep, dim, axis);
                Subspace next_rep = rref(transformed_rep);
                if (next_rep.empty()) continue;
                if (!visited.count(next_rep)) {
                    visited.insert(next_rep);
                    orbit.push_back(next_rep);
                }
            }
        }
    }
    return reps;
}

inline Tensor apply_subspace_substitution(const Tensor& T, const Subspace& S, int axis) {
    Tensor out = T;
    Subspace S_mut = S;
    S_mut = rref(S_mut);
    S_mut.erase(std::remove(S_mut.begin(), S_mut.end(), 0), S_mut.end());
    for (size_t i = 0; i < S_mut.size(); i++) {
        int idx = std::countr_zero(S_mut[i]);
        int last = out.shape[axis] - 1;
        out = apply_substitution_no_nf(out, S_mut[i], axis);
        if (idx != last) {
            U16 mask = (1 << idx) | (1 << last);
            for (size_t j = i+1; j < S_mut.size(); j++) {
                if ((S_mut[j] >> last) & 1) {
                    S_mut[j] ^= mask;
                }
            }
        }
    }
    return out.nf();
}