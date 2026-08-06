#pragma once

#include "tensor.h"
#include "matrix.h"
#include <algorithm>

inline std::size_t flatten_rank(const Tensor& T) {
    std::size_t rank1 = 0, rank2 = 0, rank3 = 0;
    
    // Mode 1: A x (B*C)
    Matrix m1(T.shape[0], T.shape[1] * T.shape[2]);
    for(std::size_t i=0; i<T.shape[0]; ++i) {
        for(std::size_t j=0; j<T.shape[1]; ++j) {
            for(std::size_t k=0; k<T.shape[2]; ++k) {
                if(T.get_bit(i, j, k)) m1.set(i, j * T.shape[2] + k, true);
            }
        }
    }
    rank1 = m1.rank();

    // Mode 2: B x (A*C)
    Matrix m2(T.shape[1], T.shape[0] * T.shape[2]);
    for(std::size_t i=0; i<T.shape[0]; ++i) {
        for(std::size_t j=0; j<T.shape[1]; ++j) {
            for(std::size_t k=0; k<T.shape[2]; ++k) {
                if(T.get_bit(i, j, k)) m2.set(j, k * T.shape[0] + i, true);
            }
        }
    }
    rank2 = m2.rank();

    // Mode 3: C x (A*B)
    Matrix m3(T.shape[2], T.shape[0] * T.shape[1]);
    for(std::size_t i=0; i<T.shape[0]; ++i) {
        for(std::size_t j=0; j<T.shape[1]; ++j) {
            for(std::size_t k=0; k<T.shape[2]; ++k) {
                if(T.get_bit(i, j, k)) m3.set(k, i * T.shape[1] + j, true);
            }
        }
    }
    rank3 = m3.rank();

    return std::max(rank1, std::max(rank2, rank3));
}
