#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "scheme.h" 

namespace fgs {

struct Matrix {
    std::size_t num_rows;
    std::size_t num_cols;
    std::size_t words_per_row;
    
    std::vector<std::vector<U64>> rows; // note that the rows in flattenings can end up too long, so we may need several words per row

    Matrix(std::size_t r, std::size_t c) : num_rows(r), num_cols(c), words_per_row((c + 63) / 64) {
        rows.assign(r, std::vector<U64>(words_per_row, 0));
    }

    // Set the cell in row r, col c to be bit
    void set(std::size_t r, std::size_t c, bool bit) {
        rows[r][c / 64] = (U64{bit} << (c % 64));
    }

    // Get the cell in row r, col c
    bool get(std::size_t r, std::size_t c) const {
        return (rows[r][c / 64] >> (c % 64)) & 1;
    }

    // Destructively computes the rank of the matrix
    std::size_t rank() {
        std::size_t out = 0;
        // Go through the columns, reducing each one
        for (std::size_t c = 0; c < num_cols && out < num_rows && out < num_cols; c++) {
            // Find a row that has a 1 in column c
            std::size_t pivot = out;
            while (pivot < num_rows && !get(pivot, c)) pivot++;

            // If no pivot is found, just move on
            if (pivot == num_rows) continue;

            // Swap the pivot row into the right place
            if (pivot != out) std::swap(rows[out], rows[pivot]);

            // Fix this column in all subsequent rows
            for (std::size_t r = out + 1; r < num_rows; ++r) {
                if (get(r, c)) {
                    for (std::size_t w = 0; w < words_per_row; ++w) {
                        rows[r][w] ^= rows[out][w];
                    }
                }
            }
            out++;
        }
        
        return out;
    }
};

enum class FlatteningType { AB_C, AC_B, BC_A }; // This should make it easier to write each of the three flattenings in a single function

inline Matrix basic_flatten(const Scheme& scheme, const Scheme::Shape& shape, FlatteningType type) {
    int component0 = 0, component1 = 1, component2 = 2;
    if (type == FlatteningType::AC_B) {
        component1 = 2;
        component2 = 1;
    } else if (type == FlatteningType::BC_A) {
        component0 = 2;
        component2 = 0;
    }

    std::size_t dim0 = shape[component0];
    std::size_t dim1 = shape[component1];
    std::size_t dim2 = shape[component2];

    Matrix mat(dim0*dim1,dim2);

    const auto& terms = scheme.terms();

    for (U16 l : scheme.live) {
        const auto& term = terms[l];
        for (std::size_t i=0; i<dim0; i++) {
            if (!((term[component0] >> i) & 1)) continue;
            for (std::size_t j=0; j<dim1; j++) {
                if (!((term[component1] >> j) & 1)) continue;
                std::size_t row = i * dim1 + j;
                mat.rows[row][0] ^= static_cast<U64>(term[component2]); // note that we don't need multiple column ints for this flattening.
            }
        }
    }
    return mat;
}

inline std::size_t flat_rank(const Scheme& scheme, const Scheme::Shape& shape) {
    Matrix M1 = basic_flatten(scheme,shape,FlatteningType::AB_C);
    Matrix M2 = basic_flatten(scheme,shape,FlatteningType::AC_B);
    Matrix M3 = basic_flatten(scheme,shape,FlatteningType::BC_A);
    return std::max({M1.rank(), M2.rank(), M3.rank()});
}

} // namespace fgs