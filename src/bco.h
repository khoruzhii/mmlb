#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace fgs::bco {

using Tensor = std::array<std::uint64_t, 12>;

struct Transvection {
    std::uint8_t axis;
    std::uint8_t source;
    std::uint8_t target;
};

inline std::uint16_t nnz(const Tensor &tensor);
inline std::uint16_t optimize(Tensor &tensor);
inline std::uint16_t optimize(Tensor &tensor,
                              std::vector<Transvection> &transvections);

inline std::uint16_t nnz(const Tensor &tensor) {
    std::uint16_t weight = 0;
    for (const std::uint64_t word : tensor) {
        weight = static_cast<std::uint16_t>(weight + std::popcount(word));
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
    return (tensor[bit / 64] & (std::uint64_t{1} << (bit % 64))) != 0;
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
                tensor[target_bit / 64] ^= std::uint64_t{1}
                                           << (target_bit % 64);
            }
        }
    }
}

inline std::uint16_t optimize_impl(Tensor &tensor,
                                   std::vector<Transvection> *transvections) {
    std::uint16_t steps = 0;
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
            transvections->push_back({static_cast<std::uint8_t>(best_axis),
                                      static_cast<std::uint8_t>(best_source),
                                      static_cast<std::uint8_t>(best_target)});
        }
        ++steps;
    }
}

// Apply the best nnz-reducing coordinate transvection until a strict local
// minimum is reached. Returns the number of applied transvections.
inline std::uint16_t optimize(Tensor &tensor) {
    return optimize_impl(tensor, nullptr);
}

// Also record the coordinate changes so a decomposition of the optimized
// tensor can be mapped back to the original tensor.
inline std::uint16_t optimize(Tensor &tensor,
                              std::vector<Transvection> &transvections) {
    transvections.clear();
    return optimize_impl(tensor, &transvections);
}

} // namespace fgs::bco
