#pragma once
#include "tensor.h"
#include "subspaces.h"
#include <vector>
#include <map>
#include <numeric>
#include "matrix.h"

// The idea here is outlined in docs/HK-ILP-lemma.md
// We use the subspace generation from subspaces.h which is infeasible

inline U64 lcm(U64 a, U64 b) {
    return (a / std::gcd(a, b)) * b;
}

struct ILP {
    int r;
    std::vector<int> orbit_sizes; // |O_i|
    std::vector<int> max_multiplicities; // k_i * |O_i|
    std::vector<std::vector<int>> v; // v_{ij}
    std::vector<int> l; // l_j
};

inline std::vector<std::vector<int>> get_v(const std::vector<std::vector<U16>>& vector_orbits, const std::vector<Subspace>& subspace_reps) {
    size_t num_orbits = vector_orbits.size();
    size_t num_subspaces = subspace_reps.size();
    std::vector<std::vector<int>> v(num_subspaces, std::vector<int>(num_orbits, 0));
    for (size_t j = 0; j < num_subspaces; j++) {
        std::vector<U16> subspace_elements = get_subspace_elements(subspace_reps[j]);
        for (size_t i = 0; i < num_orbits; i++) {
            int count = 0;
            for (U16 element : vector_orbits[i]) {
                for (U16 subspace_element : subspace_elements) {
                    if (element == subspace_element) {count++; break;}
                }
            }
            v[j][i] = count;
        }
    }
    return v;
}

inline ILP build_ilp(std::vector<std::vector<U16>>& vector_orbits, std::vector<Subspace>& subspace_orbits, int r, std::vector<int> k, std::vector<int> l) {
    ILP ilp;
    ilp.r = r;
    ilp.orbit_sizes = std::vector<int>(vector_orbits.size());
    for (size_t i = 0; i < vector_orbits.size(); i++) {
        ilp.orbit_sizes[i] = vector_orbits[i].size();
    }
    ilp.max_multiplicities = std::vector<int>(vector_orbits.size());
    for (size_t i = 0; i < vector_orbits.size(); i++) {
        ilp.max_multiplicities[i] = k[i] * vector_orbits[i].size();
    }
    ilp.v = get_v(vector_orbits, subspace_orbits);
    ilp.l = l;
    return ilp;
}

// There are smart ways to solve an ILP. I am not going to do them. I am going to check each possible input and check if any work.

inline bool check_ilp(ILP& ilp, std::vector<int>& x) {
    if (std::accumulate(x.begin(), x.end(), 0) != ilp.r) return false;
    for (size_t i = 0; i < ilp.orbit_sizes.size(); i++) {
        if (x[i] < 0 || x[i] > ilp.max_multiplicities[i]) return false;
    }
    U64 lcm_all_orbitsizes = 1; // is U64 big enough here...? Should I use lcm instead somehow?
    for (int size : ilp.orbit_sizes) {
        lcm_all_orbitsizes = lcm(lcm_all_orbitsizes, (U64)size);
    }
    for (size_t j = 0; j < ilp.v.size(); j++) {
        U64 sum = 0;
        for (size_t i = 0; i < ilp.v[j].size(); i++) {
            sum += (U64)ilp.v[j][i] * (U64)x[i] * (lcm_all_orbitsizes / (U64)ilp.orbit_sizes[i]);
        }
        if (sum > (U64)ilp.l[j] * lcm_all_orbitsizes) return false;
    }
    return true;
}

inline bool feasibility_rec(ILP& ilp, std::vector<int>& x) {
    if (x.size() == ilp.orbit_sizes.size()) {
        return check_ilp(ilp, x);
    }
    if (x.size() == ilp.orbit_sizes.size() - 1) {
        x.push_back(ilp.r - std::accumulate(x.begin(), x.end(), 0));
        bool out = check_ilp(ilp, x);
        x.pop_back();
        return out;
    }
    int current_sum = std::accumulate(x.begin(), x.end(), 0);
    for (int i = 0; i <= ilp.max_multiplicities[x.size()] && ilp.r >= current_sum + i; i++) {
        x.push_back(i);
        if (feasibility_rec(ilp, x)) return true;
        x.pop_back();
    }
    return false;
}