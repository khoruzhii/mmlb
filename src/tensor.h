#pragma once

#include "types.h"

class Tensor {
  private:
    std::array<U64,64> data;
  public:
    bool get_bit(size_t i, size_t j, size_t k) const {
        return (data[4*i + j/4] >> (k + 16*(j%4))) & U64(1);
    }

    bool operator[](size_t i, size_t j, size_t k) const {
        return get_bit(i, j, k);
    }

    U16 operator[](size_t i, size_t j) const {
        return (data[4*i + j/4] >> (16*(j%4))) & U64(0xFFFF);
    }

