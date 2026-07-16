#pragma once

#include "types.h"

#include <bit>
#include <vector>

namespace fgs::bco {

struct Transvection {
    U8 axis;
    U8 source;
    U8 target;
};

inline U16 nnz(const Tensor &tensor);
inline U16 optimize(Tensor &tensor);
inline U16 optimize(Tensor &tensor,
                    std::vector<Transvection> &transvections);

inline U16 nnz(const Tensor &tensor) {
    U16 weight = 0;
    for (const U64 word : tensor) {
        weight = static_cast<U16>(weight + std::popcount(word));
    }
    return weight;
}

constexpr int kDimension = 9;

inline int tensor_bit(int axis, int coordinate, int first, int second) {
    if (axis == 0) {
        return (coordinate * kDimension + first) * kDimension + second;
    }
    if (axis == 1) {
        return (first * kDimension + coordinate) * kDimension + second;
    }
    return (first * kDimension + second) * kDimension + coordinate;
}

inline bool get_bit(const Tensor &tensor, int bit) {
    return (tensor[bit / 64] & (U64{1} << (bit % 64))) != 0;
}

inline int delta(const Tensor &tensor, int axis, int source, int target) {
    int result = 0;
    for (int first = 0; first < kDimension; ++first) {
        for (int second = 0; second < kDimension; ++second) {
            const int source_bit = tensor_bit(axis, source, first, second);
            if (get_bit(tensor, source_bit)) {
                const int target_bit = tensor_bit(axis, target, first, second);
                result += get_bit(tensor, target_bit) ? -1 : 1;
            }
        }
    }
    return result;
}

inline void apply(Tensor &tensor, int axis, int source, int target) {
    for (int first = 0; first < kDimension; ++first) {
        for (int second = 0; second < kDimension; ++second) {
            const int source_bit = tensor_bit(axis, source, first, second);
            if (get_bit(tensor, source_bit)) {
                const int target_bit = tensor_bit(axis, target, first, second);
                tensor[target_bit / 64] ^= U64{1} << (target_bit % 64);
            }
        }
    }
}

inline U16 optimize_impl(Tensor &tensor,
                         std::vector<Transvection> *transvections) {
    U16 steps = 0;
    while (true) {
        int best_delta = 0;
        int best_axis = 0;
        int best_source = 0;
        int best_target = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int source = 0; source < kDimension; ++source) {
                for (int target = 0; target < kDimension; ++target) {
                    if (source == target) {
                        continue;
                    }
                    const int candidate = delta(tensor, axis, source, target);
                    if (candidate < best_delta) {
                        best_delta = candidate;
                        best_axis = axis;
                        best_source = source;
                        best_target = target;
                    }
                }
            }
        }
        if (best_delta == 0) {
            return steps;
        }
        apply(tensor, best_axis, best_source, best_target);
        if (transvections != nullptr) {
            transvections->push_back({static_cast<U8>(best_axis),
                                      static_cast<U8>(best_source),
                                      static_cast<U8>(best_target)});
        }
        ++steps;
    }
}

// Apply the best nnz-reducing coordinate transvection until a strict local
// minimum is reached. Returns the number of applied transvections.
inline U16 optimize(Tensor &tensor) {
    return optimize_impl(tensor, nullptr);
}

// Also record the coordinate changes so a decomposition of the optimized
// tensor can be mapped back to the original tensor.
inline U16 optimize(Tensor &tensor,
                    std::vector<Transvection> &transvections) {
    transvections.clear();
    return optimize_impl(tensor, &transvections);
}

} // namespace fgs::bco
