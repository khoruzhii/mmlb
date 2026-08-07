#pragma once

#include "tensor.h"
#include "matrix.h"
#include <algorithm>

inline bool flattening_bound(const Tensor& T, std::size_t target_lb) {
    std::size_t s0 = T.shape[0], s1 = T.shape[1], s2 = T.shape[2];

    if (s0 >= target_lb) {
        Matrix m_A_BC(T.raw(), s0, s1 * s2, 4);
        if (m_A_BC.rank() >= target_lb) {
            return true;
        }
    }

    if (s1 >= target_lb) {
        Matrix m_B_AC(s1, s0 * s2);
        for (std::size_t i = 0; i < s0; i++) {
            for (std::size_t j = 0; j < s1; j++) {
                U16 row = T.get_row(i, j);
                if (row == 0) continue;
                std::size_t col_base = i * s2;
                for (std::size_t k = 0; k < s2; k++) {
                    if ((row >> k) & 1) {
                        m_B_AC.set(j, col_base + k, true);
                    }
                }
            }
        }
        if (m_B_AC.rank() >= target_lb) {
            return true;
        }
    }

    if (s2 >= target_lb) {
        Matrix m_C_AB(s2, s0 * s1);
        for (std::size_t i = 0; i < s0; i++) {
            for (std::size_t j = 0; j < s1; j++) {
                U16 row = T.get_row(i, j);
                if (row == 0) continue;
                std::size_t col_base = i * s1 + j;
                for (std::size_t k = 0; k < s2; k++) {
                    if ((row >> k) & 1) {
                        m_C_AB.set(k, col_base, true);
                    }
                }
            }
        }
        if (m_C_AB.rank() >= target_lb) {
            return true;
        }
    }

    return false;
}
