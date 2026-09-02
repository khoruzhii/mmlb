#pragma once

#include "tensor.h"
#include "matrix.h"
#include "types.h"
#include <vector>
#include <string>


inline std::string format_orbit(U16 u, int axis);

struct ForcedProduct {
    int axis;
    int index;
    Term term;

    std::string to_string() const {
        std::string axis_name = (axis == 0) ? "a" : (axis == 1) ? "b" : "c";
        return "FP(axis=" + axis_name + ", index=" + std::to_string(index) + ", term=(" +
               format_orbit(term[0], 0) + ", " +
               format_orbit(term[1], 1) + ", " +
               format_orbit(term[2], 2) + "))";
    }
};

// Only in standard basis for now, potential for improvement in future...
inline std::pair<std::vector<ForcedProduct>, std::vector<ForcedProduct>> find_forced_products(const Tensor& T) {
    std::vector<ForcedProduct> fps1;
    std::vector<ForcedProduct> fps2;
    std::array<std::vector<U16>, 3> axis_factors;
    
    Tensor T_transposed;

    for(int axis = 0; axis < 3; axis++) {
        if (axis == 0) T_transposed = T;
        if (axis == 1) T_transposed = T.transpose_AB();
        else if (axis == 2) T_transposed = T.transpose_BC().transpose_AB();

        for (size_t i = 0; i < T_transposed.shape[0]; i++) {
            U16 v = 0;
            U16 w = 0;
            for (size_t j = 0; j < T_transposed.shape[1]; j++) {
                U16 row = T_transposed.get_row(i, j);
                if (row != 0) {
                    if (v == 0) {
                        v = row;
                        w = (1 << j);
                    } else {
                        if (row != v) {
                            v = 0;
                            break;
                        }
                        w |= (1 << j);
                    }
                }
            }
            if (v != 0) {
                switch (axis) {
                    case 0: 
                        fps1.push_back({0, (int)i, { (U16)(1 << i), w, v }}); 
                        break;
                    case 1: 
                        fps1.push_back({1, (int)i, { w, (U16)(1 << i), v }}); 
                        break;
                    case 2: 
                        fps1.push_back({2, (int)i, { w, v, (U16)(1 << i) }});
                        break;
                }
            }
        }

        for (size_t i = 0; i < T_transposed.shape[0]; i++) {
            for (size_t j = i+1; j < T_transposed.shape[0]; j++) {
                U16 v = 0;
                U16 w = 0;
                for (size_t k = 0; k < T_transposed.shape[1]; k++) {
                    U16 row = T_transposed.get_row(i, k) ^ T_transposed.get_row(j, k);
                    if (row != 0) {
                        if (v == 0) {
                            v = row;
                            w = (1 << k);
                        } else {
                            if (row != v) {
                                v = 0;
                                break;
                            }
                            w |= (1 << k);
                        }
                    }
                }
                if (v != 0) {
                    switch (axis) {
                        case 0: 
                            fps2.push_back({0, (int)i, { (U16)((1 << i) | (1 << j)), w, v }}); 
                            break;
                        case 1: 
                            fps2.push_back({1, (int)i, { w, (U16)((1 << i) | (1 << j)), v }});
                            break;
                        case 2: 
                            fps2.push_back({2, (int)i, { w, v, (U16)((1 << i) | (1 << j)) }});
                            break;
                    }
                }
            }
        }
    }

    return {fps1, fps2};
}
