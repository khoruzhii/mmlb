#pragma once

#include "tensor.h"
#include "ub.h"
#include "flatten.h"
#include "symmetries.h"
#include "substitution.h"
#include "forced_product.h"
#include "rank_table.h"
#include <iostream>
#include <map>
#include <algorithm>

struct CacheEntry {
    int max_success = -1;
    int min_fail = 1000;
};
inline std::map<Tensor, CacheEntry> lb_cache;
inline void clear_cache() {
    lb_cache.clear();
}

inline std::vector<U8> rank_table;

inline std::string format_orbit(U16 u, int axis) {
    std::string axis_name = (axis == 0) ? "a" : (axis == 1) ? "b" : "c";
    std::string result = "";
    bool first = true;
    for (int i = 0; i < 16; i++) {
        if ((u >> i) & 1) {
            if (!first) result += "+";
            result += axis_name + std::to_string(i + 1);
            first = false;
        }
    }
    return result;
}

inline std::string tensor_to_string(const Tensor& T) {
    std::string result = "";
    bool first = true;
    for (size_t i = 0; i < T.shape[0]; i++) {
        for (size_t j = 0; j < T.shape[1]; j++) {
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (T.get_bit(i, j, k)) {
                    if (!first) result += " + ";
                    result += "a" + std::to_string(i + 1) + "*b" + std::to_string(j + 1) + "*c" + std::to_string(k + 1);
                    first = false;
                }
            }
        }
    }
    if (first) return "0";
    return result;
}

inline bool topdown_lb_internal(Tensor T, int target_lb, int conj_rank, int depth);

inline bool topdown_lb(Tensor T, int target_lb, int conj_rank, int depth = 0) {
    std::string indent = "";
    for(int i=0; i<depth; i++) indent += "│   ";
    
    if (std::max({T.shape[0],T.shape[1],T.shape[2]}) <= 3) {
        if (rank_table_lookup(T,rank_table) >= target_lb) {
            std::cout << indent << "├─ [LOOKUP] Rank >= " << target_lb << " is TRUE\n" << std::flush;
            return true;
        }
    }

    T = T.nf();

    if (lb_cache.count(T)) {
        if (lb_cache[T].max_success >= target_lb) {
            std::cout << indent << "├─ [CACHED] Rank >= " << target_lb << " is TRUE\n" << std::flush;
            return true;
        }
        if (lb_cache[T].min_fail <= target_lb) {
            std::cout << indent << "├─ [CACHED] Rank >= " << target_lb << " is FALSE\n" << std::flush;
            return false;
        }
    }

    std::cout << indent << "├─ Analyzing tensor: " << tensor_to_string(T) << "\n" << std::flush;
    
    bool res = topdown_lb_internal(T, target_lb, conj_rank, depth);
    if (res) {
        lb_cache[T].max_success = std::max(lb_cache[T].max_success, target_lb);
    } else {
        lb_cache[T].min_fail = std::min(lb_cache[T].min_fail, target_lb);
    }
    return res;
}

inline bool topdown_lb_internal(Tensor T, int target_lb, int conj_rank, int depth) {
    std::string indent = "";
    for(int i=0; i<depth; i++) indent += "│   ";

    // NOTE: these steps are not complete, nor necessarily in the right order yet

    // (0) Check if T is the zero tensor. Not yet implemented, but also probably not helpful

    // (1) Check if target_lb is more than the conj_rank. If so, return false
    if (target_lb > conj_rank) {
        std::cout << indent << "├─ Step 1: target_lb > conj_rank, returning false\n" << std::flush;
        return false;
    }

    // (2) Check if flatten_rank is at least target_lb. If so, return true
    if (flattening_bound(T, target_lb)) {
        std::cout << indent << "├─ Step 2: flatten_rank >= target_lb, returning true\n" << std::flush;
        return true;
    }

    // (3) Simple Forced Products
    std::cout << indent << "├─ Step 3: Checking simple forced products\n" << std::flush;
    auto fps = find_forced_products(T);
    
    for (int fp_ax = 0; fp_ax < 3; fp_ax++) {
        std::vector<ForcedProduct> axis_fps;
        for (auto& fp : fps) if (fp.axis == fp_ax) axis_fps.push_back(fp);
        
        for (int factor_ax = 0; factor_ax < 3; factor_ax++) {
            if (factor_ax == fp_ax) continue;
            
            std::map<U16, int> factor_counts;
            for (auto& fp : axis_fps) {
                factor_counts[fp.term[factor_ax]]++;
            }
            
            for (auto const& [f, m] : factor_counts) {
                if (f == 0) continue;
                Tensor sub = apply_substitution(T, f, factor_ax);
                int sub_rank = ub(sub);
                if (sub_rank >= target_lb - m) {
                    std::cout << indent << "├─ Step 3: Branching on factor " << f << " on axis " << factor_ax 
                              << " (drops rank by " << m << ", target_lb=" << target_lb - m << ")\n" << std::flush;
                    if (topdown_lb(sub, target_lb - m, sub_rank, depth + 1)) {
                        std::cout << indent << "├─ Step 3: Recursive call returned true\n" << std::flush;
                        return true;
                    }
                }
            }
        }
    }

    std::cout << indent << "├─ Step 4: computing substitutions (shape: " << (int)T.shape[0] << "x" << (int)T.shape[1] << "x" << (int)T.shape[2] << ", target_lb: " << target_lb << ", conj_rank: " << conj_rank << ")\n" << std::flush;
    auto symmetries_init = symmetry_generators(T);
    std::cout << indent << "├─ Step 4: symmetries computed\n" << std::flush;

    // (3) Loop through the spaces and then the orbits for that space
    int min_sub_rank[3] = {100,100,100};
    std::vector<std::pair<Tensor, int>> subs[3];
    for (int axis = 0; axis < 3; axis++) {
        std::vector<U16> orbits = get_orbit_reps(symmetries_init, T.shape[axis], axis);
        for (U16 u : orbits) {
            if (u == 0) continue;
            Tensor sub = apply_substitution(T, u, axis);
            if (std::max({sub.shape[0],sub.shape[1],sub.shape[2]}) <= 3) {
                std::cout << indent << "├─ Step 4: rank table lookup for orbit " << format_orbit(u, axis) << "=0 (axis " << axis << ")\n" << std::flush;
                if (rank_table_lookup(sub,rank_table) >= target_lb) {
                    std::cout << indent << "├─ Step 4: [LOOKUP] Rank(sub) >= " << target_lb << ", returning true\n" << std::flush;
                    return true;
                }
                else {
                    std::cout << indent << "├─ Step 4: [LOOKUP] Rank(sub) < " << target_lb << "\n" << std::flush;
                    continue;
                }
            }

            sub = sub.nf();
            if (lb_cache.count(sub)) {
                if (lb_cache[sub].max_success >= target_lb) {
                    std::cout << indent << "├─ Step 4: [CACHED] Rank(sub) >= " << target_lb << ", returning true\n" << std::flush;
                    return true;
                }
                if (lb_cache[sub].min_fail <= target_lb) {
                    std::cout << indent << "├─ Step 4: [CACHED] Rank(sub) < " << target_lb << "\n" << std::flush;
                    continue;
                }
            }

            std::cout << indent << "├─ Step 4: calling ub(sub) for orbit " << format_orbit(u, axis) << "=0 (axis " << axis << ")\n" << std::flush;
                        int sub_conj_rank = ub(sub);
            std::cout << indent << "├─ Step 4: ub(sub) = " << sub_conj_rank << "\n" << std::flush;
            if (sub_conj_rank >= target_lb) {
                std::cout << indent << "├─ Step 4: Calling recursive topdown_lb\n" << std::flush;
                if (topdown_lb(sub, target_lb, sub_conj_rank, depth + 1)) {
                    std::cout << indent << "├─ Step 4: Recursive call returned true, returning true\n" << std::flush;
                    return true;
                }
            }
            min_sub_rank[axis] = std::min(min_sub_rank[axis], sub_conj_rank);
            subs[axis].push_back({sub, sub_conj_rank});
        }
    }

    // (5) If minimum conj_rank across all substitutions for a space is at least target_lb - 1
    std::cout << indent << "├─ Step 5: checking min_sub_rank\n" << std::flush;
    for (int axis = 0; axis < 3; axis++) {
        if (min_sub_rank[axis] >= target_lb - 1 && !subs[axis].empty()) {
            std::cout << indent << "├─ Step 5: Branching on axis " << axis << " with " << subs[axis].size() << " subs\n" << std::flush;
            bool all_true = true;
            for (auto& info : subs[axis]) {
                if (!topdown_lb(info.first, target_lb - 1, info.second, depth + 1)) {
                    std::cout << indent << "├─ Step 5: Recursive call returned false, breaking\n" << std::flush;
                    all_true = false;
                    break;
                }
            }
            if (all_true) {
                std::cout << indent << "├─ Step 5: All recursive calls returned true, returning true\n" << std::flush;
                return true;
            }
        }
    }

    // (6) Branching Forced Products
    std::cout << indent << "├─ Step 6: Branching forced products\n" << std::flush;
    for (int ax = 0; ax < 3; ax++) {
        std::vector<ForcedProduct> axis_fps;
        for (auto& fp : fps) if (fp.axis == ax) axis_fps.push_back(fp);
        
        int N = axis_fps.size();
        if (N == 0) continue;
        
        Tensor T_sub = T;
        std::vector<int> indices;
        for (auto& fp : axis_fps) indices.push_back(fp.index);
        std::sort(indices.rbegin(), indices.rend());
        
        for (int idx : indices) {
            T_sub = apply_substitution_no_nf(T_sub, 1 << idx, ax);
        }
        
        int D_prime = T_sub.shape[ax];
        int num_combinations = 1 << (N * D_prime);

        if (num_combinations > 10000) {
            std::cout << indent << "├─ Step 6: Too many combinations (" << num_combinations << "), skipping this axis\n" << std::flush;
            continue;
        }
        
        std::cout << indent << "├─ Step 6: Axis " << ax << " has " << N << " FPs. D'=" << D_prime 
                  << ". Branching " << num_combinations << " combinations. target_lb=" << target_lb - N << "\n" << std::flush;
        // Before we branch over all possibilities, we should use symmetries to prune the search space.
        // First we list out every branch        
        std::vector<Tensor> raw_branches;
        for (int comb = 0; comb < num_combinations; comb++) {
            Tensor T_branch = T_sub;
            for (int i = 0; i < N; i++) {
                int guessed_vec = (comb >> (i * D_prime)) & ((1 << D_prime) - 1);
                
                U16 u = axis_fps[i].term[0];
                U16 v = axis_fps[i].term[1];
                U16 w = axis_fps[i].term[2];
                
                if (ax == 0) u = guessed_vec;
                else if (ax == 1) v = guessed_vec;
                else if (ax == 2) w = guessed_vec;
                
                for (size_t a = 0; a < T_branch.shape[0]; a++) {
                    if ((u >> a) & 1) {
                        for (size_t b = 0; b < T_branch.shape[1]; b++) {
                            if ((v >> b) & 1) {
                                for (size_t c = 0; c < T_branch.shape[2]; c++) {
                                    if ((w >> c) & 1) {
                                        bool current = T_branch.get_bit(a, b, c);
                                        T_branch.set_bit(a, b, c, !current);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            raw_branches.push_back(T_branch);
        }
        // Now we group them by orbits
        std::vector<Tensor> unique_branches;
        if (num_combinations > 1) {
            auto syms = symmetry_generators(T_sub);
            
            std::vector<bool> visited(num_combinations, false);
            for (int i = 0; i < num_combinations; i++) {
                if (visited[i]) continue;
                unique_branches.push_back(raw_branches[i]);
                std::vector<int> orbit = {i};
                visited[i] = true;
                size_t head = 0;
                while (head < orbit.size()) {
                    int curr_idx = orbit[head++];
                    Tensor curr_T = raw_branches[curr_idx];
                    for (const auto& sym : syms) {
                        Tensor next_T = apply_symmetry_to_tensor(curr_T, sym);
                        for (int j = 0; j < num_combinations; j++) {
                            if (!visited[j] && raw_branches[j] == next_T) { // if we find a symmetry of T_sub that actually maps to another branch, then we group these
                                visited[j] = true;
                                orbit.push_back(j);
                            }
                        }
                    }
                }
            }
        } else {
            unique_branches = raw_branches;
        }
        std::cout << indent << "├─ Step 6: Reduced " << num_combinations << " branches to " << unique_branches.size() << " branches up to symmetry\n" << std::flush;
        
        bool all_true = true;
        for (size_t b = 0; b < unique_branches.size(); b++) {
            std::cout << indent << "├─ Starting branch " << (b + 1) << "/" << unique_branches.size() << " (pruned from " << num_combinations << "):\n" << std::flush;
            Tensor T_branch = unique_branches[b];
            int branch_rank = ub(T_branch);
            if (!topdown_lb(T_branch, target_lb - N, branch_rank, depth + 1)) {
                all_true = false;
                break;
            }
        }
        
        if (all_true) {
            std::cout << indent << "├─ Step 6: All combinations for axis " << ax << " returned true!\n" << std::flush;
            return true;
        }
    }

    std::cout << indent << "├─ Step 6: some forced product branch didn't pan out, continuing\n" << std::flush;

    // (7) Generate all rank one tensors, construct T+t, and call topdown_lb
    std::vector<Term> rank1_orbits = get_rank1_orbits(T);
    std::cout << indent << "├─ Step 7: Generated " << rank1_orbits.size() << " rank1 orbits\n" << std::flush;
    if (rank1_orbits.size() > 150 || (depth >= 2 && rank1_orbits.size() > 50)) { // without this branching factor check, we can *always* use this to prove any valid lower bound. However, it will usually struggle to be fast...
        std::cout << indent << "├─ Step 7: Too many rank1 orbits, giving up\n" << std::flush;
        return false;
    }
    int branch_idx = 1;
    for (const Term& t : rank1_orbits) {
        std::cout << indent << "├─ Starting branch " << branch_idx++ << "/" << rank1_orbits.size() << ":\n" << std::flush;
        Tensor T_plus_t = T;
        for (size_t i = 0; i < T.shape[0]; i++) {
            if ((t[0] >> i) & 1) {
                for (size_t j = 0; j < T.shape[1]; j++) {
                    if ((t[1] >> j) & 1) {
                        for (size_t k = 0; k < T.shape[2]; k++) {
                            if ((t[2] >> k) & 1) {
                                bool current = T_plus_t.get_bit(i, j, k);
                                T_plus_t.set_bit(i, j, k, !current);
                            }
                        }
                    }
                }
            }
        }
        
        int sub_rank = ub(T_plus_t);
        if (!topdown_lb(T_plus_t, target_lb - 1, sub_rank, depth + 1)) {
            std::cout << indent << "├─ Step 7: Some rank1 orbit didn't pan out, continuing\n" << std::flush;
            return false;
        }
    }

    std::cout << indent << "├─ Step 7: some rank1 orbit didn't pan out, continuing\n" << std::flush;

    // If step 7 loop finished successfully, all branches returned true.
    return true;
}
