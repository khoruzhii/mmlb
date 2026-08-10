#pragma once

#include <vector>
#include "types.h"

class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t words_per_row;
    std::vector<U64> data_;

    std::vector<std::size_t> pivot_cols;
    std::vector<bool> is_pivot_col;
public:
    Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
        words_per_row = (cols + 63) / 64;
        data_.resize(rows_ * words_per_row);
    }

    Matrix(const std::array<U64, 64>& data, std::size_t rows, std::size_t cols, std::size_t stride = 0) : rows_(rows), cols_(cols) {
        words_per_row = (cols + 63) / 64;
        if (stride == 0) {
            data_.assign(data.begin(), data.begin() + rows_ * words_per_row);
        } else {
            data_.resize(rows_ * words_per_row);
            for (std::size_t i = 0; i < rows_; i++) {
                for (std::size_t w = 0; w < words_per_row; w++) {
                    data_[i * words_per_row + w] = data[i * stride + w];
                }
            }
        }
    }

    void set(std::size_t row, std::size_t col, bool val) {
        std::size_t word_index = col / 64;
        std::size_t bit_index = col % 64;
        if (val) {
            data_[row * words_per_row + word_index] |= (1ULL << bit_index);
        } else {
            data_[row * words_per_row + word_index] &= ~(1ULL << bit_index);
        }
    }

    bool get(std::size_t row, std::size_t col) const {
        std::size_t word_index = col / 64;
        std::size_t bit_index = col % 64;
        return (data_[row * words_per_row + word_index] >> bit_index) & 1;
    }

    // swap two rows
    void swap(std::size_t row1, std::size_t row2) {
        std::size_t word1_start = row1 * words_per_row;
        std::size_t word2_start = row2 * words_per_row;
        for (std::size_t i = 0; i < words_per_row; i++) {
            std::swap(data_[word1_start + i], data_[word2_start + i]);
        }
    }

    // set a word of bits starting at col_start in row
    void set_word(std::size_t row, std::size_t col_start, U64 bits, unsigned count) {
        if (count < 64) 
            bits &= (U64(1) << count) - 1;
        std::size_t word_index = col_start / 64;
        std::size_t bit_offset = col_start % 64;
        std::size_t base = row * words_per_row;
        data_[base + word_index] ^= (bits << bit_offset);
        if (bit_offset + count > 64) {
            data_[base + word_index + 1] ^= (bits >> (64 - bit_offset));
        }
    }

    // add row2 to row1
    void add(std::size_t row1, std::size_t row2) {
        std::size_t word1_start = row1 * words_per_row;
        std::size_t word2_start = row2 * words_per_row;
        for (std::size_t i = 0; i < words_per_row; i++) {
            data_[word1_start + i] ^= data_[word2_start + i];
        }
    }

    // row echelon form
    void rref() {
        pivot_cols.clear();
        std::size_t pivot_row = 0;
        is_pivot_col.assign(cols_, false);
        for (std::size_t col = 0; col < cols_ && pivot_row < rows_; col++) {
            std::size_t current_row = pivot_row;
            while (current_row < rows_ && !get(current_row, col)) {
                current_row++;
            }
            if (current_row < rows_) {
                swap(pivot_row, current_row);
                is_pivot_col[col] = true;
                pivot_cols.push_back(col);
                for (std::size_t row = 0; row < rows_; row++) {
                    if (row != pivot_row && get(row, col)) {
                        add(row, pivot_row);
                    }
                }
                pivot_row++;
            }
        }
    }

    std::vector<std::vector<U64>> right_nullspace_basis() {
        rref();
        std::vector<std::vector<U64>> basis;
        for (std::size_t col = 0; col < cols_; col++) {
            if (!is_pivot_col[col]) {
                std::vector<U64> vec(words_per_row, 0);
                vec[col / 64] |= (1ULL << (col % 64));
                for (std::size_t p = 0; p < pivot_cols.size(); p++) {
                    if (get(p,col)) {
                        vec[pivot_cols[p] / 64] |= (1ULL << (pivot_cols[p] % 64));
                    }
                }
                basis.push_back(vec);
            }
        }
        return basis;
    }

    std::size_t rank() {
        rref();
        return pivot_cols.size();
    }
};
