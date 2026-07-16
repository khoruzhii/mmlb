#pragma once

#include "support.h"

#include <array>
#include <bit>
#include <stdexcept>
#include <vector>

namespace proj {

// One canonical quotient for an axis and a one-dimensional kernel. Shape is
// the target-space shape, even if the projected tensor has smaller support.
struct Projection {
    Tensor tensor;
    Shape shape;
    U8 axis;
    U16 kernel;
};

// Enumerate axes in order and, within each axis, all nonzero kernel vectors in
// increasing order. Over F_2 every nonzero vector represents a distinct line.
// The input must already have prefix mode supports ordered by nondecreasing
// dimension, as produced by proj::normalize.
inline std::vector<Projection> all(const Tensor &tensor);

inline U16 prefix_mask(U8 dimension) {
    return dimension == 0
               ? U16{0}
               : static_cast<U16>((U16{1} << dimension) - 1);
}

inline void validate_normalized(const Tensor &tensor, const Shape &shape) {
    if (multilinear_rank(tensor) != shape) {
        throw std::invalid_argument(
            "projection input must have prefix mode supports");
    }
    if (shape[0] > shape[1] || shape[1] > shape[2]) {
        throw std::invalid_argument(
            "projection input dimensions must be nondecreasing");
    }
}

inline U16 project_factor(U16 value, U16 kernel, U8 dimension) {
    if (dimension == 0 || dimension > kDimension || kernel == 0 ||
        (kernel & ~prefix_mask(dimension)) != 0) {
        throw std::invalid_argument("invalid elementary projection kernel");
    }
    if ((value & ~prefix_mask(dimension)) != 0) {
        throw std::invalid_argument(
            "projected vector exceeds its input dimension");
    }

    const int pivot = std::countr_zero(kernel);
    const bool pivot_value = (value & (U16{1} << pivot)) != 0;
    U16 result = 0;
    int output = 0;
    for (int input = 0; input < dimension; ++input) {
        if (input == pivot) {
            continue;
        }
        bool bit = (value & (U16{1} << input)) != 0;
        bit ^= pivot_value && (kernel & (U16{1} << input)) != 0;
        if (bit) {
            result |= U16{1} << output;
        }
        ++output;
    }
    return result;
}

inline Tensor project_tensor(const Tensor &tensor, const Shape &shape, U8 axis,
                             U16 kernel) {
    if (axis >= 3) {
        throw std::invalid_argument("projection axis must be in [0, 2]");
    }
    const U8 dimension = shape[axis];
    if (dimension == 0 || kernel == 0 ||
        (kernel & ~prefix_mask(dimension)) != 0) {
        throw std::invalid_argument("invalid elementary projection kernel");
    }

    const int pivot = std::countr_zero(kernel);
    const U8 first_axis = static_cast<U8>((axis + 1) % 3);
    const U8 second_axis = static_cast<U8>((axis + 2) % 3);
    Tensor result{};
    std::array<int, 3> input{};
    std::array<int, 3> output{};

    int projected_coordinate = 0;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        if (coordinate == pivot) {
            continue;
        }
        input[axis] = coordinate;
        output[axis] = projected_coordinate++;
        for (int first = 0; first < shape[first_axis]; ++first) {
            input[first_axis] = first;
            output[first_axis] = first;
            for (int second = 0; second < shape[second_axis]; ++second) {
                input[second_axis] = second;
                output[second_axis] = second;
                bool value =
                    tensor_bit(tensor, input[0], input[1], input[2]);
                if ((kernel & (U16{1} << coordinate)) != 0) {
                    input[axis] = pivot;
                    value ^=
                        tensor_bit(tensor, input[0], input[1], input[2]);
                    input[axis] = coordinate;
                }
                if (value) {
                    toggle_tensor_bit(result, output[0], output[1],
                                      output[2]);
                }
            }
        }
    }
    return result;
}

inline std::vector<Projection> all(const Tensor &tensor) {
    validate_tensor(tensor);
    const Shape shape = natural_shape(tensor);
    validate_normalized(tensor, shape);

    std::size_t count = 0;
    for (const U8 dimension : shape) {
        count += prefix_mask(dimension);
    }

    std::vector<Projection> result;
    result.reserve(count);
    for (U8 axis = 0; axis < 3; ++axis) {
        for (U16 kernel = 1; kernel <= prefix_mask(shape[axis]); ++kernel) {
            Shape projected_shape = shape;
            --projected_shape[axis];
            result.push_back({project_tensor(tensor, shape, axis, kernel),
                              projected_shape, axis, kernel});
        }
    }
    return result;
}

} // namespace proj
