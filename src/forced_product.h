#pragma once

#include "tensor.h"
#include "matrix.h"
#include "types.h"
#include <vector>

struct ForcedProduct {
    int axis;
    int index;
    Term term;
};

// Only in standard basis for now, potential for improvement in future...
inline std::vector<ForcedProduct> find_forced_products(const Tensor& T) {
    std::vector<ForcedProduct> fps;
    
    for (size_t i = 0; i < T.shape[0]; i++) {
        Matrix M(T.shape[1], T.shape[2]);
        for (size_t j = 0; j < T.shape[1]; j++) {
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (T.get_bit(i, j, k)) M.set(j, k, true);
            }
        }
        Matrix M_copy = M; // so we don't mutate it when calling rank()
        if (M_copy.rank() == 1) { // then we have a forced product
            U16 w = 0;
            int first_nonzero_col = -1;
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (M_copy.get(0, k)) {
                    w |= (1 << k);
                    if (first_nonzero_col == -1) first_nonzero_col = k;
                }
            }
            U16 v = 0;
            for (size_t j = 0; j < T.shape[1]; j++) {
                if (M.get(j, first_nonzero_col)) {
                    v |= (1 << j);
                }
            }
            Term t = { (U16)(1 << i), v, w };
            fps.push_back({0, (int)i, t});
        }
    }
    
    for (size_t j = 0; j < T.shape[1]; j++) {
        Matrix M(T.shape[0], T.shape[2]);
        for (size_t i = 0; i < T.shape[0]; i++) {
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (T.get_bit(i, j, k)) M.set(i, k, true);
            }
        }
        Matrix M_copy = M;
        if (M_copy.rank() == 1) {
            U16 w = 0;
            int first_nonzero_col = -1;
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (M_copy.get(0, k)) {
                    w |= (1 << k);
                    if (first_nonzero_col == -1) first_nonzero_col = k;
                }
            }
            U16 u = 0;
            for (size_t i = 0; i < T.shape[0]; i++) {
                if (M.get(i, first_nonzero_col)) {
                    u |= (1 << i);
                }
            }
            Term t = { u, (U16)(1 << j), w };
            fps.push_back({1, (int)j, t});
        }
    }
    
    for (size_t k = 0; k < T.shape[2]; k++) {
        Matrix M(T.shape[0], T.shape[1]);
        for (size_t i = 0; i < T.shape[0]; i++) {
            for (size_t j = 0; j < T.shape[1]; j++) {
                if (T.get_bit(i, j, k)) M.set(i, j, true);
            }
        }
        Matrix M_copy = M;
        if (M_copy.rank() == 1) {
            U16 v = 0;
            int first_nonzero_col = -1;
            for (size_t j = 0; j < T.shape[1]; j++) {
                if (M_copy.get(0, j)) {
                    v |= (1 << j);
                    if (first_nonzero_col == -1) first_nonzero_col = j;
                }
            }
            U16 u = 0;
            for (size_t i = 0; i < T.shape[0]; i++) {
                if (M.get(i, first_nonzero_col)) {
                    u |= (1 << i);
                }
            }
            Term t = { u, v, (U16)(1 << k) };
            fps.push_back({2, (int)k, t});
        }
    }
    
    return fps;
}
