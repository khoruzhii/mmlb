#pragma once

#include "flatten.h"
#include "projection.h"
#include "ub.h"
#include "forced_product.h"
#include "support.h"

#include <vector>
#include <map>

#include <iostream>
#include <string>

namespace fgs {

inline std::string format_kernel(U16 kernel, int axis, int dim) {
    std::string s;
    char prefix = (axis == 0 ? 'a' : (axis == 1 ? 'b' : 'c'));
    int cols = 1;
    if (dim == 4) cols = 2;
    else if (dim == 9) cols = 3;
    
    bool first = true;
    for (int i = 0; i < 16; ++i) {
        if ((kernel & (1 << i)) != 0) {
            if (!first) s += "+";
            s += prefix;
            if (cols > 1) {
                s += std::to_string(i / cols) + std::to_string(i % cols);
            } else {
                s += std::to_string(i);
            }
            first = false;
        }
    }
    s += "=0";
    return s;
}

inline std::string get_indent(int depth) {
    return std::string(depth * 2, ' ');
}

inline std::map<std::pair<Tensor, U8>, bool> lb_cache;
inline std::map<Tensor, U8> ub_cache;

inline U8 get_ub(const Tensor& t, int flips) {
    Tensor t_norm = t;
    proj::normalize(t_norm);
    if (ub_cache.find(t_norm) != ub_cache.end()) {
        return ub_cache[t_norm];
    }
    U8 res = ub(t_norm, flips);
    ub_cache[t_norm] = res;
    return res;
}

inline bool topdown_lb(const Tensor &tensor, U8 conjectured_rank, U8 target_lb, int depth = 0) {
    if (target_lb == 0) return true;

    Tensor t = tensor;
    proj::normalize(t);
    const Shape shape = proj::natural_shape(t);
    if (shape[0] == 0 && shape[1] == 0 && shape[2] == 0) {
        return false; // rank is 0, which is < target_lb since target_lb > 0
    }

    auto key = std::make_pair(t, target_lb);
    if (lb_cache.find(key) != lb_cache.end()) {
        return lb_cache[key];
    }

    auto solve = [&]() -> bool {
        // 1. Flattening lower bound
        if (flat_rank(t, shape) >= target_lb) {
            return true;
        }

    // 2. Sub-orbits
    std::vector<proj::Projection> all_orbits = proj::sub_orbits(t);
    
    std::vector<proj::Projection> orbits_by_axis[3];
    for (const auto& p : all_orbits) {
        orbits_by_axis[p.axis].push_back(p);
    }
    
    for (int axis = 0; axis < 3; ++axis) {
        if (orbits_by_axis[axis].empty()) continue;
        
        std::cout << get_indent(depth) << "Found " << orbits_by_axis[axis].size() 
                  << " non-zero orbits for axis " << (axis == 0 ? "zero" : (axis == 1 ? "one" : "two")) 
                  << ". Trying to prove each has a rank at least " << (int)(target_lb - 1) << ".\n";
        
        std::vector<U8> proj_ranks;
        proj_ranks.reserve(orbits_by_axis[axis].size());
        
        U8 min_proj_rank = 255;
        int orbit_idx = 1;
        for (const auto& p : orbits_by_axis[axis]) {
            U8 r = get_ub(p.tensor, 100000);
            proj_ranks.push_back(r);
            
            std::cout << get_indent(depth) << "ORBIT " << orbit_idx++ << "/" << orbits_by_axis[axis].size() 
                      << " " << format_kernel(p.kernel, axis, shape[axis]) << ". ";
            
            if (r == conjectured_rank) {
                std::cout << "Found a projection with zero rank drop, setting " 
                          << format_kernel(p.kernel, axis, shape[axis]) 
                          << " and trying to prove a lower bound of " << (int)target_lb << "...\n";
                if (topdown_lb(p.tensor, r, target_lb, depth + 1)) {
                    return true;
                }
            } else {
                std::cout << "Upper bound is " << (int)r << ".\n";
            }
            
            if (r < min_proj_rank) {
                min_proj_rank = r;
            }
        }
        
        if (min_proj_rank >= target_lb - 1) {
            bool all_proved = true;
            for (std::size_t i = 0; i < orbits_by_axis[axis].size(); ++i) {
                std::cout << get_indent(depth) << "Trying to prove lower bound of " << (int)(target_lb - 1) << " on orbit " << (i+1) << "...\n";
                if (!topdown_lb(orbits_by_axis[axis][i].tensor, proj_ranks[i], target_lb - 1, depth + 1)) {
                    std::cout << get_indent(depth) << "Failed to prove orbit " << (i+1) << ".\n";
                    all_proved = false;
                    break;
                }
            }
            if (all_proved) {
                std::cout << get_indent(depth) << "Proved target lower bound via sub-orbits on axis " << axis << ".\n";
                return true;
            }
        }
    }
    
    // 3. Forced Products
    auto forced_opt = forced::find(t);
    if (forced_opt.has_value()) {
        auto res = forced_opt.value();
        const Tensor transformed = proj::apply(t, res.action);
        
        for (U8 coord = 0; coord < res.count; ++coord) {
            Matrix M = forced::slice(transformed, res.shape, 0, coord);
            U16 left = 0, right = 0;
            if (forced::factor_rank_one(M, res.shape[1], left, right)) {
                // 1. Project on axis 1 using 'left'
                Tensor proj1 = proj::project_tensor(transformed, res.shape, 1, left);
                if (topdown_lb(proj1, get_ub(proj1, 100000), target_lb - 1, depth + 1)) {
                    return true;
                }
                
                // 2. Project on axis 2 using 'right'
                Tensor proj2 = proj::project_tensor(transformed, res.shape, 2, right);
                if (topdown_lb(proj2, get_ub(proj2, 100000), target_lb - 1, depth + 1)) {
                    return true;
                }
                
                // 3. Branch over all c
                bool all_branches_proved = true;
                std::size_t num_elements = std::size_t{1} << res.shape[0];
                for (U16 c = 1; c < num_elements; ++c) {
                    if ((c & (1 << coord)) == 0) continue;
                    
                    Term term = { c, left, right };
                    Tensor T_minus_term = transformed;
                    proj::add_term(T_minus_term, term);
                    
                    if (!topdown_lb(T_minus_term, get_ub(T_minus_term, 100000), target_lb - 1, depth + 1)) {
                        all_branches_proved = false;
                        break;
                    }
                }
                if (all_branches_proved) {
                    return true;
                }
            }
        }
    }

    return false;
    }; // end of solve lambda
    
    bool result = solve();
    lb_cache[key] = result;
    return result;
}

} // namespace fgs
