#pragma once

#include "tensor.h"
#include "matrix.h"
#include <algorithm>

inline Tensor permute_axes(const Tensor& T, int p0, int p1, int p2) {
    if (p0 == 0 && p1 == 1 && p2 == 2) return T;
    if (p0 == 0 && p1 == 2 && p2 == 1) return T.transpose_BC();
    if (p0 == 1 && p1 == 0 && p2 == 2) return T.transpose_AB();
    if (p0 == 1 && p1 == 2 && p2 == 0) return T.transpose_AB().transpose_BC();
    if (p0 == 2 && p1 == 0 && p2 == 1) return T.transpose_BC().transpose_AB();
    return T.transpose_AB().transpose_BC().transpose_AB();
}

inline Matrix koszul_flatten(const Tensor& T, int wedge_axis, int input_axis) {
    int output_axis = 3 - wedge_axis - input_axis;
    Tensor Tp = permute_axes(T, wedge_axis, input_axis, output_axis);

    std::size_t sw = Tp.shape[0], si = Tp.shape[1], so = Tp.shape[2];
    std::size_t npairs = sw * (sw - 1) / 2;
    Matrix M(npairs * si, sw * so);

    std::size_t pair_row_base = 0;
    for (std::size_t i = 0; i < sw; i++) {
        for (std::size_t j = i + 1; j < sw; j++) {
            for (std::size_t l = 0; l < si; l++) {
                std::size_t row = pair_row_base + l;
                U16 row_i = Tp.get_row(i, l); 
                U16 row_j = Tp.get_row(j, l);
                M.set_word(row, i * so, row_j, so);
                M.set_word(row, j * so, row_i, so);
            }
            pair_row_base += si;
        }
    }
    return M;
}

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

    if (s0 > 1 && koszul_flatten(T, 0, 1).rank()/(s0 -1) >= target_lb) {
        std::cout << "Koszul flattening used: (0,1)" << std::endl;
        return true;
    }

    if (s0 > 1 && koszul_flatten(T, 0, 2).rank()/(s0 -1) >= target_lb) {
        std::cout << "Koszul flattening used: (0,2)" << std::endl;
        return true;
    }

    if (s1 > 1 && koszul_flatten(T, 1, 0).rank()/(s1 -1) >= target_lb) {
        std::cout << "Koszul flattening used: (1,0)" << std::endl;
        return true;
    }

    if (s1 > 1 && koszul_flatten(T, 1, 2).rank()/(s1 -1) >= target_lb) {
        std::cout << "Koszul flattening used: (1,2)" << std::endl;
        return true;
    }

    if (s2 > 1 && koszul_flatten(T, 2, 0).rank()/(s2 -1) >= target_lb) {
        std::cout << "Koszul flattening used: (2,0)" << std::endl;
        return true;
    }

    if (s2 > 1 && koszul_flatten(T, 2, 1).rank()/(s2 -1) >= target_lb) {
        std::cout << "Koszul flattening used: (2,1)" << std::endl;
        return true;
    }
    return false;
}
