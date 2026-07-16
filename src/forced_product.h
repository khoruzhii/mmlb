#pragma once

#include "support.h"

#include <array>
#include <bit>
#include <optional>
#include <stdexcept>
#include <vector>

namespace forced {

// The action sends the winning input direction to output axis 0. Its first
// count slices are linearly independent rank-one matrices. Shape is the tensor
// shape after the axis permutation.
struct Result {
    proj::Action action;
    Shape shape;
    U8 count;
};

// The input must first be normalized by proj::normalize. Return no value iff
// no nonzero slice combination has matrix rank one on any of the three axes.
inline std::optional<Result> find(const Tensor &tensor);

struct Product {
    U16 slices;
    U16 left;
    U16 right;
};

constexpr std::array<std::array<U8, 2>, 3> kOtherAxes = {
    {{{1, 2}}, {{0, 2}}, {{0, 1}}}};

inline Matrix slice(const Tensor &tensor, const Shape &shape, U8 axis,
                    U8 coordinate) {
    const U8 row_axis = kOtherAxes[axis][0];
    const U8 column_axis = kOtherAxes[axis][1];
    Matrix result{};
    std::array<int, 3> index{};
    index[axis] = coordinate;
    for (int row = 0; row < shape[row_axis]; ++row) {
        index[row_axis] = row;
        for (int column = 0; column < shape[column_axis]; ++column) {
            index[column_axis] = column;
            if (proj::tensor_bit(tensor, index[0], index[1], index[2])) {
                result[row] |= U16{1} << column;
            }
        }
    }
    return result;
}

inline bool factor_rank_one(const Matrix &matrix, U8 rows, U16 &left,
                            U16 &right) {
    left = 0;
    right = 0;
    for (int row = 0; row < rows; ++row) {
        if (matrix[row] == 0) {
            continue;
        }
        if (right == 0) {
            right = matrix[row];
        } else if (matrix[row] != right) {
            return false;
        }
        left |= U16{1} << row;
    }
    return right != 0;
}

// A basis of all rank-one elements in this slice space. Greedy Gaussian
// elimination is exact: every possible coordinate slice is a covector mask,
// and a GL change can include precisely an independent family of such masks.
inline std::vector<Product> products(const Tensor &tensor, const Shape &shape,
                                     U8 axis) {
    const U8 row_axis = kOtherAxes[axis][0];
    std::array<Matrix, 9> slices{};
    for (U8 coordinate = 0; coordinate < shape[axis]; ++coordinate) {
        slices[coordinate] = slice(tensor, shape, axis, coordinate);
    }

    std::array<Matrix, 512> sums{};
    std::array<U16, 9> basis{};
    std::vector<Product> result;
    result.reserve(shape[axis]);
    for (U16 subset = 1; subset < (U16{1} << shape[axis]); ++subset) {
        const U16 previous = subset & static_cast<U16>(subset - 1);
        const int coordinate = std::countr_zero(subset);
        for (int row = 0; row < shape[row_axis]; ++row) {
            sums[subset][row] =
                sums[previous][row] ^ slices[coordinate][row];
        }

        U16 left = 0;
        U16 right = 0;
        if (factor_rank_one(sums[subset], shape[row_axis], left, right) &&
            proj::extend_basis(basis, subset)) {
            result.push_back({subset, left, right});
        }
    }
    return result;
}

inline Matrix rows_to_matrix(const std::array<U16, 9> &rows) {
    Matrix result{};
    for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
            if ((rows[row] & (U16{1} << column)) != 0) {
                result[column] |= U16{1} << row;
            }
        }
    }
    return result;
}

inline Matrix slice_map(const std::vector<Product> &values, U8 dimension) {
    std::array<U16, 9> rows{};
    std::array<U16, 9> basis{};
    U8 count = 0;
    for (const Product &product : values) {
        if (!proj::extend_basis(basis, product.slices)) {
            throw std::logic_error("dependent forced-product slice masks");
        }
        rows[count++] = product.slices;
    }
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        const U16 value = U16{1} << coordinate;
        if (proj::extend_basis(basis, value)) {
            rows[count++] = value;
        }
    }
    for (int coordinate = dimension; coordinate < 9; ++coordinate) {
        rows[count++] = U16{1} << coordinate;
    }
    if (count != 9) {
        throw std::logic_error("failed to complete forced-product slice basis");
    }
    return rows_to_matrix(rows);
}

inline bool is_rank_one(const Tensor &tensor, const Shape &shape, U8 axis,
                        U8 coordinate) {
    const Matrix matrix = slice(tensor, shape, axis, coordinate);
    U16 left = 0;
    U16 right = 0;
    return factor_rank_one(matrix, shape[kOtherAxes[axis][0]], left, right);
}

inline std::optional<Result> find(const Tensor &tensor) {
    proj::validate_tensor(tensor);
    const Shape shape = proj::natural_shape(tensor);
    if (proj::multilinear_rank(tensor) != shape || shape[0] > shape[1] ||
        shape[1] > shape[2]) {
        throw std::invalid_argument(
            "forced-product input must be support-normalized");
    }

    U8 winning_axis = 0;
    std::vector<Product> winning_products;
    for (U8 axis = 0; axis < 3; ++axis) {
        std::vector<Product> candidates = products(tensor, shape, axis);
        if (candidates.size() > winning_products.size()) {
            winning_axis = axis;
            winning_products = std::move(candidates);
        }
    }
    if (winning_products.empty()) {
        return std::nullopt;
    }

    const U8 row_axis = kOtherAxes[winning_axis][0];
    const U8 column_axis = kOtherAxes[winning_axis][1];
    proj::Action action = proj::identity();
    action.gl[winning_axis] =
        slice_map(winning_products, shape[winning_axis]);
    action.permutation = {winning_axis, row_axis, column_axis};
    const Shape output_shape = {
        shape[winning_axis], shape[row_axis], shape[column_axis]};

    const Tensor transformed = proj::apply(tensor, action);
    U8 count = 0;
    for (U8 coordinate = 0; coordinate < output_shape[0]; ++coordinate) {
        count += is_rank_one(transformed, output_shape, 0, coordinate);
    }
    if (count != winning_products.size()) {
        throw std::logic_error("forced-product count mismatch");
    }
    return Result{action, output_shape, count};
}

} // namespace forced
