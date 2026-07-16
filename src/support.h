#pragma once

#include "types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <vector>

namespace proj {

// An element of GL(9,2)^3 semidirect S3. The three linear maps act first.
// permutation[output_axis] then gives the corresponding input axis.
struct Action {
    std::array<Matrix, 3> gl;
    std::array<U8, 3> permutation;

    bool operator==(const Action &) const = default;
};

inline Action identity();
inline Action inverse(const Action &action);
inline Term apply(const Term &term, const Action &action);
inline Tensor apply(const Tensor &tensor, const Action &action);

// Coordinate bounding-box dimensions in the current basis. The zero tensor
// has shape {0,0,0}.
inline Shape natural_shape(const Tensor &tensor);

// The basis-independent dimensions of the three mode supports.
inline Shape multilinear_rank(const Tensor &tensor);

// Move every mode support to the first coordinates and order the axes so that
// natural_shape(tensor)[0] <= natural_shape(tensor)[1] <=
// natural_shape(tensor)[2]. The returned action maps the input tensor to the
// normalized tensor and can be inverted to transport decompositions back.
inline Action normalize(Tensor &tensor);

constexpr int kDimension = 9;
constexpr int kTensorBits = kDimension * kDimension * kDimension;
constexpr U16 kVectorMask = (U16{1} << kDimension) - 1;
constexpr U64 kLastWordMask = (U64{1} << (kTensorBits % 64)) - 1;
inline constexpr std::array<std::array<U8, 2>, 3> kOtherAxes = {
    {{{1, 2}}, {{0, 2}}, {{0, 1}}}};

inline void validate_tensor(const Tensor &tensor) {
    if ((tensor.back() & ~kLastWordMask) != 0) {
        throw std::invalid_argument("tensor has nonzero padding bits");
    }
}

inline bool tensor_bit(const Tensor &tensor, int i, int j, int k) {
    const int bit = (kDimension * i + j) * kDimension + k;
    return (tensor[bit / 64] & (U64{1} << (bit % 64))) != 0;
}

inline void toggle_tensor_bit(Tensor &tensor, int i, int j, int k) {
    const int bit = (kDimension * i + j) * kDimension + k;
    tensor[bit / 64] ^= U64{1} << (bit % 64);
}

inline bool tensor_fits_shape(const Tensor &tensor, const Shape &shape) {
    for (const U8 dimension : shape) {
        if (dimension > kDimension) {
            throw std::invalid_argument("tensor shape exceeds dimension 9");
        }
    }
    if (shape == Shape{kDimension, kDimension, kDimension}) {
        return true;
    }
    Tensor allowed{};
    const U64 fiber_mask =
        shape[2] == 0 ? U64{0} : (U64{1} << shape[2]) - 1;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            const int bit = (kDimension * i + j) * kDimension;
            const int word = bit / 64;
            const int offset = bit % 64;
            allowed[word] |= fiber_mask << offset;
            if (offset + shape[2] > 64) {
                allowed[word + 1] |= fiber_mask >> (64 - offset);
            }
        }
    }
    for (int word = 0; word < 12; ++word) {
        if ((tensor[word] & ~allowed[word]) != 0) {
            return false;
        }
    }
    return true;
}

template <U8 axis>
inline std::array<Matrix, kDimension>
tensor_slices_on_axis(const Tensor &tensor) {
    static_assert(axis < 3);
    std::array<Matrix, kDimension> result{};
    for (int word = 0; word < 12; ++word) {
        U64 bits = tensor[word];
        while (bits != 0) {
            const int offset = std::countr_zero(bits);
            const int linear = 64 * word + offset;
            if (linear >= kTensorBits) {
                break;
            }
            const int i = linear / 81;
            const int j = linear / 9 % 9;
            const int k = linear % 9;
            if constexpr (axis == 0) {
                result[i][j] |= U16{1} << k;
            } else if constexpr (axis == 1) {
                result[j][i] |= U16{1} << k;
            } else {
                result[k][i] |= U16{1} << j;
            }
            bits &= bits - 1;
        }
    }
    return result;
}

inline std::array<Matrix, kDimension>
tensor_slices(const Tensor &tensor, U8 axis) {
    if (axis == 0) {
        return tensor_slices_on_axis<0>(tensor);
    }
    if (axis == 1) {
        return tensor_slices_on_axis<1>(tensor);
    }
    if (axis != 2) {
        throw std::invalid_argument("tensor slice axis must be in [0, 2]");
    }
    return tensor_slices_on_axis<2>(tensor);
}

inline bool extend_basis(std::array<U16, kDimension> &pivots, U16 vector) {
    for (int bit = kDimension - 1; bit >= 0 && vector != 0; --bit) {
        if ((vector & (U16{1} << bit)) == 0) {
            continue;
        }
        if (pivots[bit] != 0) {
            vector ^= pivots[bit];
        } else {
            pivots[bit] = vector;
            return true;
        }
    }
    return false;
}

inline U8 vector_rank(const Matrix &vectors) {
    std::array<U16, kDimension> pivots{};
    U8 rank = 0;
    for (const U16 vector : vectors) {
        rank += extend_basis(pivots, vector);
    }
    return rank;
}

inline U8 matrix_rank(const Matrix &rows, U8 row_count,
                      U8 column_count) {
    std::array<U16, kDimension> pivots{};
    const U16 mask = column_count == 0
                         ? U16{0}
                         : static_cast<U16>((U16{1} << column_count) - 1);
    U8 rank = 0;
    for (int row = 0; row < row_count; ++row) {
        U16 vector = rows[row] & mask;
        while (vector != 0) {
            const int pivot = std::bit_width(vector) - 1;
            if (pivots[pivot] != 0) {
                vector ^= pivots[pivot];
            } else {
                pivots[pivot] = vector;
                ++rank;
                break;
            }
        }
    }
    return rank;
}

inline bool matrix_full_rank(const Matrix &rows, U8 dimension) {
    std::array<U16, kDimension> pivots{};
    const U16 mask = dimension == 0
                         ? U16{0}
                         : static_cast<U16>((U16{1} << dimension) - 1);
    for (int row = 0; row < dimension; ++row) {
        U16 vector = rows[row] & mask;
        while (vector != 0) {
            const int pivot = std::bit_width(vector) - 1;
            if (pivots[pivot] != 0) {
                vector ^= pivots[pivot];
            } else {
                pivots[pivot] = vector;
                break;
            }
        }
        if (vector == 0) {
            return false;
        }
    }
    return true;
}

inline Matrix identity_matrix() {
    Matrix result{};
    for (int i = 0; i < kDimension; ++i) {
        result[i] = U16{1} << i;
    }
    return result;
}

inline U16 apply_matrix(const Matrix &matrix, U16 vector) {
    U16 result = 0;
    for (; vector != 0; vector &= static_cast<U16>(vector - 1)) {
        result ^= matrix[std::countr_zero(vector)];
    }
    return result;
}

inline Matrix inverse_matrix(const Matrix &matrix) {
    std::array<U32, kDimension> rows{};
    for (int row = 0; row < kDimension; ++row) {
        for (int column = 0; column < kDimension; ++column) {
            if ((matrix[column] & (U16{1} << row)) != 0) {
                rows[row] |= U32{1} << column;
            }
        }
        rows[row] |= U32{1} << (kDimension + row);
    }

    for (int column = 0; column < kDimension; ++column) {
        int pivot = column;
        while (pivot < kDimension &&
               (rows[pivot] & (U32{1} << column)) == 0) {
            ++pivot;
        }
        if (pivot == kDimension) {
            throw std::invalid_argument("support action matrix is singular");
        }
        std::swap(rows[column], rows[pivot]);
        for (int row = 0; row < kDimension; ++row) {
            if (row != column &&
                (rows[row] & (U32{1} << column)) != 0) {
                rows[row] ^= rows[column];
            }
        }
    }

    Matrix result{};
    for (int row = 0; row < kDimension; ++row) {
        const U16 inverse_row =
            static_cast<U16>(rows[row] >> kDimension) & kVectorMask;
        for (int column = 0; column < kDimension; ++column) {
            if ((inverse_row & (U16{1} << column)) != 0) {
                result[column] |= U16{1} << row;
            }
        }
    }
    return result;
}

inline void validate_action(const Action &action) {
    std::array<bool, 3> seen{};
    for (const U8 axis : action.permutation) {
        if (axis >= 3 || seen[axis]) {
            throw std::invalid_argument("invalid support axis permutation");
        }
        seen[axis] = true;
    }
    for (const Matrix &matrix : action.gl) {
        if (vector_rank(matrix) != kDimension) {
            throw std::invalid_argument("support action matrix is singular");
        }
    }
}

inline Action identity() {
    Action result{};
    result.gl.fill(identity_matrix());
    result.permutation = {0, 1, 2};
    return result;
}

inline Action inverse(const Action &action) {
    validate_action(action);
    Action result = identity();
    for (int output_axis = 0; output_axis < 3; ++output_axis) {
        const U8 input_axis = action.permutation[output_axis];
        result.permutation[input_axis] = static_cast<U8>(output_axis);
        result.gl[output_axis] = inverse_matrix(action.gl[input_axis]);
    }
    return result;
}

inline Term apply_term_unchecked(const Term &term, const Action &action) {
    Term result{};
    for (int output_axis = 0; output_axis < 3; ++output_axis) {
        const U8 input_axis = action.permutation[output_axis];
        result[output_axis] =
            apply_matrix(action.gl[input_axis], term[input_axis]);
    }
    return result;
}

inline Term apply(const Term &term, const Action &action) {
    validate_action(action);
    for (const U16 factor : term) {
        if ((factor & ~kVectorMask) != 0) {
            throw std::invalid_argument("term factor exceeds dimension 9");
        }
    }
    return apply_term_unchecked(term, action);
}

inline void add_term(Tensor &tensor, const Term &term) {
    for (U16 first = term[0]; first != 0;
         first &= static_cast<U16>(first - 1)) {
        const int i = std::countr_zero(first);
        for (U16 second = term[1]; second != 0;
             second &= static_cast<U16>(second - 1)) {
            const int j = std::countr_zero(second);
            for (U16 third = term[2]; third != 0;
                 third &= static_cast<U16>(third - 1)) {
                toggle_tensor_bit(tensor, i, j, std::countr_zero(third));
            }
        }
    }
}

inline Tensor apply(const Tensor &tensor, const Action &action) {
    validate_tensor(tensor);
    validate_action(action);
    Tensor result{};
    for (int i = 0; i < kDimension; ++i) {
        for (int j = 0; j < kDimension; ++j) {
            for (int k = 0; k < kDimension; ++k) {
                if (!tensor_bit(tensor, i, j, k)) {
                    continue;
                }
                const Term term = {static_cast<U16>(U16{1} << i),
                                   static_cast<U16>(U16{1} << j),
                                   static_cast<U16>(U16{1} << k)};
                add_term(result, apply_term_unchecked(term, action));
            }
        }
    }
    return result;
}

inline Shape natural_shape(const Tensor &tensor) {
    validate_tensor(tensor);
    Shape result{};
    for (int i = 0; i < kDimension; ++i) {
        for (int j = 0; j < kDimension; ++j) {
            for (int k = 0; k < kDimension; ++k) {
                if (tensor_bit(tensor, i, j, k)) {
                    result[0] = std::max(result[0], static_cast<U8>(i + 1));
                    result[1] = std::max(result[1], static_cast<U8>(j + 1));
                    result[2] = std::max(result[2], static_cast<U8>(k + 1));
                }
            }
        }
    }
    return result;
}

inline U16 fiber(const Tensor &tensor, U8 axis, int first, int second) {
    constexpr std::array<std::array<U8, 2>, 3> other_axes = {
        {{{1, 2}}, {{0, 2}}, {{0, 1}}}};
    std::array<int, 3> index{};
    index[other_axes[axis][0]] = first;
    index[other_axes[axis][1]] = second;
    U16 result = 0;
    for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
        index[axis] = coordinate;
        if (tensor_bit(tensor, index[0], index[1], index[2])) {
            result |= U16{1} << coordinate;
        }
    }
    return result;
}

inline std::vector<U16> mode_basis(const Tensor &tensor, U8 axis) {
    std::array<U16, kDimension> pivots{};
    std::vector<U16> result;
    result.reserve(kDimension);
    for (int first = 0; first < kDimension; ++first) {
        for (int second = 0; second < kDimension; ++second) {
            const U16 candidate = fiber(tensor, axis, first, second);
            if (extend_basis(pivots, candidate)) {
                result.push_back(candidate);
            }
        }
    }
    return result;
}

inline Matrix support_map(const Tensor &tensor, U8 axis, U8 &dimension) {
    const std::vector<U16> support = mode_basis(tensor, axis);
    dimension = static_cast<U8>(support.size());
    const U16 prefix = dimension == 0
                           ? U16{0}
                           : static_cast<U16>((U16{1} << dimension) - 1);
    if (std::all_of(support.begin(), support.end(), [prefix](U16 vector) {
            return (vector & ~prefix) == 0;
        })) {
        return identity_matrix();
    }

    std::array<U16, kDimension> pivots{};
    std::vector<U16> basis;
    basis.reserve(kDimension);
    for (const U16 vector : support) {
        if (!extend_basis(pivots, vector)) {
            throw std::logic_error("mode support basis is dependent");
        }
        basis.push_back(vector);
    }
    for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
        const U16 vector = U16{1} << coordinate;
        if (extend_basis(pivots, vector)) {
            basis.push_back(vector);
        }
    }
    if (basis.size() != kDimension) {
        throw std::logic_error("failed to extend a mode support basis");
    }

    Matrix basis_matrix{};
    std::copy(basis.begin(), basis.end(), basis_matrix.begin());
    return inverse_matrix(basis_matrix);
}

inline Shape multilinear_rank(const Tensor &tensor) {
    validate_tensor(tensor);
    Shape result{};
    for (U8 axis = 0; axis < 3; ++axis) {
        result[axis] = static_cast<U8>(mode_basis(tensor, axis).size());
    }
    return result;
}

inline Action normalize(Tensor &tensor) {
    validate_tensor(tensor);
    Action result = identity();
    Shape dimensions{};
    for (U8 axis = 0; axis < 3; ++axis) {
        result.gl[axis] = support_map(tensor, axis, dimensions[axis]);
    }

    result.permutation = {0, 1, 2};
    std::stable_sort(result.permutation.begin(), result.permutation.end(),
                     [&dimensions](U8 left, U8 right) {
                         return dimensions[left] < dimensions[right];
                     });
    tensor = ::proj::apply(tensor, result);

    Shape expected{};
    for (int axis = 0; axis < 3; ++axis) {
        expected[axis] = dimensions[result.permutation[axis]];
    }
    if (natural_shape(tensor) != expected ||
        multilinear_rank(tensor) != expected) {
        throw std::logic_error("support normalization failed");
    }
    return result;
}

} // namespace proj
