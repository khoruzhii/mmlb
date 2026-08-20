#pragma once
#include "tensor.h"
#include "subspaces.h"
#include <vector>
#include <map>
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include "matrix.h"

inline U64 lcm(U64 a, U64 b) {
    return (a / std::gcd(a, b)) * b;
}

struct ILP {
    int r;
    int num_vars;
    int num_constraints;
    std::vector<int> max_multiplicities; // k_i * |O_i|
    std::vector<U64> weights;            // flattened C[j * num_vars + i] = v[j][i] * (L / |O_i|)
    std::vector<U64> rhs;                // l[j] * L
};

inline std::vector<std::vector<int>> get_v(const std::vector<std::vector<U16>>& vector_orbits, const std::vector<Subspace>& subspace_reps) {
    size_t num_orbits = vector_orbits.size();
    size_t num_subspaces = subspace_reps.size();
    std::vector<std::vector<int>> v(num_subspaces, std::vector<int>(num_orbits, 0));
    for (size_t j = 0; j < num_subspaces; j++) {
        std::vector<U16> subspace_elements = get_subspace_elements(subspace_reps[j]);
        std::unordered_set<U16> sub_set(subspace_elements.begin(), subspace_elements.end());
        for (size_t i = 0; i < num_orbits; i++) {
            int count = 0;
            for (U16 element : vector_orbits[i]) {
                if (sub_set.count(element)) count++;
            }
            v[j][i] = count;
        }
    }
    return v;
}

inline ILP build_ilp(const std::vector<std::vector<U16>>& vector_orbits, const std::vector<Subspace>& subspace_orbits, int r, const std::vector<int>& k, const std::vector<int>& l) {
    ILP ilp;
    ilp.r = r;
    ilp.num_vars = (int)vector_orbits.size();
    ilp.num_constraints = (int)subspace_orbits.size();
    ilp.max_multiplicities.resize(ilp.num_vars);
    for (int i = 0; i < ilp.num_vars; i++) {
        ilp.max_multiplicities[i] = k[i] * (int)vector_orbits[i].size();
    }
    U64 L = 1;
    for (const auto& orb : vector_orbits) {
        L = lcm(L, (U64)orb.size());
    }
    auto raw_v = get_v(vector_orbits, subspace_orbits);
    ilp.weights.resize(ilp.num_constraints * ilp.num_vars);
    ilp.rhs.resize(ilp.num_constraints);
    for (int j = 0; j < ilp.num_constraints; j++) {
        ilp.rhs[j] = (U64)l[j] * L;
        for (int i = 0; i < ilp.num_vars; i++) {
            ilp.weights[j * ilp.num_vars + i] = (U64)raw_v[j][i] * (L / (U64)vector_orbits[i].size());
        }
    }
    return ilp;
}

inline bool solve_ilp_dfs(const ILP& ilp, int idx, int curr_sum, std::vector<U64>& curr_lhs, std::vector<int>& x) {
    if (idx == ilp.num_vars - 1) {
        int rem = ilp.r - curr_sum;
        if (rem < 0 || rem > ilp.max_multiplicities[idx]) return false;
        for (int j = 0; j < ilp.num_constraints; j++) {
            if (curr_lhs[j] + (U64)rem * ilp.weights[j * ilp.num_vars + idx] > ilp.rhs[j]) {
                return false;
            }
        }
        x.push_back(rem);
        return true;
    }

    int max_val = std::min(ilp.max_multiplicities[idx], ilp.r - curr_sum);
    for (int val = 0; val <= max_val; val++) {
        bool feasible = true;
        for (int j = 0; j < ilp.num_constraints; j++) {
            if (curr_lhs[j] + (U64)val * ilp.weights[j * ilp.num_vars + idx] > ilp.rhs[j]) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;

        for (int j = 0; j < ilp.num_constraints; j++) {
            curr_lhs[j] += (U64)val * ilp.weights[j * ilp.num_vars + idx];
        }
        x.push_back(val);

        if (solve_ilp_dfs(ilp, idx + 1, curr_sum + val, curr_lhs, x)) {
            return true;
        }

        x.pop_back();
        for (int j = 0; j < ilp.num_constraints; j++) {
            curr_lhs[j] -= (U64)val * ilp.weights[j * ilp.num_vars + idx];
        }
    }
    return false;
}

inline bool feasibility_rec(const ILP& ilp, std::vector<int>& x) {
    x.clear();
    if (ilp.num_vars == 0) return ilp.r == 0;
    std::vector<U64> curr_lhs(ilp.num_constraints, 0);
    return solve_ilp_dfs(ilp, 0, 0, curr_lhs, x);
}