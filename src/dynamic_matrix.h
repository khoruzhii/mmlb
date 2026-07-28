#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace proj {

class DynamicMatrix {
public:
    DynamicMatrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
        words_per_row_ = (cols + 63) / 64;
        data_.resize(rows * words_per_row_, 0);
    }

    void set(std::size_t r, std::size_t c, bool val) {
        if (r >= rows_ || c >= cols_) return;
        std::size_t word_idx = r * words_per_row_ + (c / 64);
        std::size_t bit_idx = c % 64;
        if (val) {
            data_[word_idx] |= (uint64_t(1) << bit_idx);
        } else {
            data_[word_idx] &= ~(uint64_t(1) << bit_idx);
        }
    }

    bool get(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) return false;
        std::size_t word_idx = r * words_per_row_ + (c / 64);
        std::size_t bit_idx = c % 64;
        return (data_[word_idx] & (uint64_t(1) << bit_idx)) != 0;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Returns a basis for the right kernel (nullspace): {v | Mv = 0}
    std::vector<std::vector<bool>> right_kernel() const {
        DynamicMatrix M = *this; // Copy to mutate

        // Bring to RREF
        std::size_t pivot_row = 0;
        std::vector<std::size_t> pivot_cols;
        std::vector<bool> is_pivot_col(cols_, false);

        for (std::size_t c = 0; c < cols_ && pivot_row < rows_; ++c) {
            // Find pivot
            std::size_t r = pivot_row;
            while (r < rows_ && !M.get(r, c)) {
                ++r;
            }

            if (r == rows_) {
                continue; // No pivot in this column
            }

            // Swap rows
            if (r != pivot_row) {
                for (std::size_t w = 0; w < words_per_row_; ++w) {
                    std::swap(M.data_[pivot_row * words_per_row_ + w], M.data_[r * words_per_row_ + w]);
                }
            }

            // Eliminate other rows
            for (std::size_t r_other = 0; r_other < rows_; ++r_other) {
                if (r_other != pivot_row && M.get(r_other, c)) {
                    for (std::size_t w = 0; w < words_per_row_; ++w) {
                        M.data_[r_other * words_per_row_ + w] ^= M.data_[pivot_row * words_per_row_ + w];
                    }
                }
            }

            pivot_cols.push_back(c);
            is_pivot_col[c] = true;
            ++pivot_row;
        }

        // Construct nullspace basis
        std::vector<std::vector<bool>> basis;
        for (std::size_t c = 0; c < cols_; ++c) {
            if (!is_pivot_col[c]) {
                // Free variable, create a basis vector
                std::vector<bool> vec(cols_, false);
                vec[c] = true;
                
                // Set dependent variables
                for (std::size_t p = 0; p < pivot_cols.size(); ++p) {
                    std::size_t pc = pivot_cols[p];
                    if (M.get(p, c)) {
                        vec[pc] = true;
                    }
                }
                basis.push_back(vec);
            }
        }

        return basis;
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t words_per_row_;
    std::vector<uint64_t> data_;
};

} // namespace proj
