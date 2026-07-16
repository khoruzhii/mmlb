#pragma once

#include "support.h"

#include <array>
#include <bit>
#include <optional>
#include <stdexcept>

namespace lb {

// A three-dimensional space of dual functionals on one tensor mode. If all
// seven nonzero contractions are invertible 9 x 9 matrices, the certificate
// proves that the tensor has rank at least 17.
struct FanoCertificate {
    U8 axis{};
    std::array<U16, 3> basis{};

    bool operator==(const FanoCertificate &) const = default;
};

// Verify a supplied certificate directly. The tensor need not be normalized.
inline bool fano_verify(const Tensor &tensor,
                        const FanoCertificate &certificate);

// Exhaustively find a certificate on one axis or on the first successful
// axis. Supplying a known prefix shape avoids recomputing it. No normalization
// or ordering of the dimensions is required.
inline std::optional<FanoCertificate>
fano_find_axis(const Tensor &tensor, const Shape &shape, U8 axis);
inline std::optional<FanoCertificate> fano_find(const Tensor &tensor,
                                                const Shape &shape);
inline std::optional<FanoCertificate> fano_find(const Tensor &tensor);

inline constexpr std::array<U64, 6> kFanoShuffleMasks = {
    0x5555555555555555ULL, 0x3333333333333333ULL,
    0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
    0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL};
inline constexpr U16 kFanoMask = (U16{1} << 9) - 1;

constexpr std::array<std::array<U64, 8>, 512>
make_fano_linear_patterns() {
    std::array<std::array<U64, 8>, 512> result{};
    for (int coefficient = 0; coefficient < 512; ++coefficient) {
        for (int value = 0; value < 512; ++value) {
            if ((std::popcount(
                     static_cast<unsigned>(coefficient & value)) &
                 1) != 0) {
                result[coefficient][value / 64] |=
                    U64{1} << (value % 64);
            }
        }
    }
    return result;
}

inline constexpr std::array<std::array<U64, 8>, 512> kFanoLinearPatterns =
    make_fano_linear_patterns();

inline void validate_fano_shape(const Tensor &tensor, const Shape &shape) {
    proj::validate_tensor(tensor);
    if (!proj::tensor_fits_shape(tensor, shape)) {
        throw std::invalid_argument("tensor exceeds its Fano shape");
    }
}

inline Matrix fano_contraction(const std::array<Matrix, 9> &slices,
                               U16 functional) {
    Matrix result{};
    while (functional != 0) {
        const int coordinate = std::countr_zero(functional);
        for (int row = 0; row < 9; ++row) {
            result[row] ^= slices[coordinate][row];
        }
        functional &= static_cast<U16>(functional - 1);
    }
    return result;
}

inline Matrix fano_xor(const Matrix &left, const Matrix &right) {
    Matrix result{};
    for (int row = 0; row < 9; ++row) {
        result[row] = left[row] ^ right[row];
    }
    return result;
}

inline bool fano_invertible(const Matrix &matrix) {
    return proj::matrix_full_rank(matrix, 9);
}

inline std::array<U64, 8>
fano_invertibility_profile(const std::array<Matrix, 9> &slices,
                           U8 dimension = 9) {
    std::array<Matrix, 9> coefficients{};
    for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
            U16 coefficient = 0;
            for (int slice = 0; slice < 9; ++slice) {
                coefficient |= static_cast<U16>(
                    ((slices[slice][row] >> column) & U16{1}) << slice);
            }
            coefficients[row][column] = coefficient;
        }
    }

    const U16 limit = U16{1} << dimension;
    const int word_count = (limit + 63) / 64;
    using Bits = std::array<U64, 8>;
    // One bit is one functional. Keeping the eight 64-functional batches in
    // the innermost dimension lets the compiler vectorize the same Gaussian
    // elimination across all 512 contractions.
    alignas(64) std::array<std::array<Bits, 9>, 9> rows;
    for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
            const U16 coefficient = coefficients[row][column];
            for (int word = 0; word < word_count; ++word) {
                rows[row][column][word] =
                    kFanoLinearPatterns[coefficient][word];
            }
        }
    }

    Bits valid{};
    for (int word = 0; word < word_count; ++word) {
        valid[word] = ~U64{0};
    }
    for (int pivot = 0; pivot < 9; ++pivot) {
        alignas(64) std::array<Bits, 9> selected;
        for (int word = 0; word < word_count; ++word) {
            U64 remaining = valid[word];
            for (int row = pivot; row < 9; ++row) {
                selected[row][word] =
                    remaining & rows[row][pivot][word];
                remaining ^= selected[row][word];
            }
            valid[word] ^= remaining;
        }

        for (int row = pivot + 1; row < 9; ++row) {
            for (int column = pivot; column < 9; ++column) {
                for (int word = 0; word < word_count; ++word) {
                    const U64 difference =
                        (rows[pivot][column][word] ^
                         rows[row][column][word]) &
                        selected[row][word];
                    rows[pivot][column][word] ^= difference;
                    rows[row][column][word] ^= difference;
                }
            }
        }
        for (int row = pivot + 1; row < 9; ++row) {
            alignas(64) Bits eliminate_mask{};
            for (int word = 0; word < word_count; ++word) {
                eliminate_mask[word] =
                    rows[row][pivot][word] & valid[word];
            }
            for (int column = pivot + 1; column < 9; ++column) {
                for (int word = 0; word < word_count; ++word) {
                    rows[row][column][word] ^=
                        rows[pivot][column][word] &
                        eliminate_mask[word];
                }
            }
        }
    }

    std::array<U64, 8> result = valid;
    if (limit < 64) {
        result[0] &= (U64{1} << limit) - 1;
    }
    return result;
}

inline U64 fano_xor_permute(U64 value, U16 translation) {
    for (int bit = 0; bit < 6; ++bit) {
        if ((translation & (U16{1} << bit)) == 0) {
            continue;
        }
        const int shift = 1 << bit;
        const U64 mask = kFanoShuffleMasks[bit];
        value = ((value & mask) << shift) | ((value >> shift) & mask);
    }
    return value;
}

inline std::optional<FanoCertificate>
fano_find_axis_unchecked(const Tensor &tensor, const Shape &shape, U8 axis) {
    const U8 row_axis = proj::kOtherAxes[axis][0];
    const U8 column_axis = proj::kOtherAxes[axis][1];
    if (shape[axis] < 3 || shape[row_axis] < 9 || shape[column_axis] < 9) {
        return std::nullopt;
    }

    const std::array<Matrix, 9> slices =
        proj::tensor_slices(tensor, axis);
    const std::array<U64, 8> invertible_bits =
        fano_invertibility_profile(slices, shape[axis]);

    int value_count = 0;
    for (const U64 word : invertible_bits) {
        value_count += std::popcount(word);
    }
    if (value_count < 7) {
        return std::nullopt;
    }

    std::array<std::array<U64, 8>, 512> compatible;
    std::array<bool, 512> built{};
    // compatible[t] = {x : x and x + t are both invertible contractions}.
    const auto compatibility = [&](U16 translation) -> const auto & {
        if (!built[translation]) {
            const U16 word_translation = translation / 64;
            const U16 bit_translation = translation % 64;
            for (int word = 0; word < 8; ++word) {
                const U64 translated = fano_xor_permute(
                    invertible_bits[word ^ word_translation],
                    bit_translation);
                compatible[translation][word] =
                    invertible_bits[word] & translated;
            }
            built[translation] = true;
        }
        return compatible[translation];
    };

    for (int first_word = 0; first_word < 8; ++first_word) {
        U64 firsts = invertible_bits[first_word];
        while (firsts != 0) {
            const U16 first = static_cast<U16>(
                64 * first_word + std::countr_zero(firsts));
            const auto &first_candidates = compatibility(first);
            const U16 start = static_cast<U16>(first + 1);
            for (int second_word = start / 64; second_word < 8;
                 ++second_word) {
                U64 seconds = first_candidates[second_word];
                if (second_word == start / 64) {
                    seconds &= ~U64{0} << (start % 64);
                }
                while (seconds != 0) {
                    const U16 second = static_cast<U16>(
                        64 * second_word + std::countr_zero(seconds));
                    const U16 third = first ^ second;
                    if (second < third) {
                        const auto &second_candidates = compatibility(second);
                        const auto &third_candidates = compatibility(third);
                        for (int word = 0; word < 8; ++word) {
                            const U64 candidates = first_candidates[word] &
                                                   second_candidates[word] &
                                                   third_candidates[word];
                            if (candidates != 0) {
                                const U16 fourth = static_cast<U16>(
                                    64 * word +
                                    std::countr_zero(candidates));
                                return FanoCertificate{
                                    axis, {first, second, fourth}};
                            }
                        }
                    }
                    seconds &= seconds - 1;
                }
            }
            firsts &= firsts - 1;
        }
    }
    return std::nullopt;
}

inline bool fano_verify(const Tensor &tensor,
                        const FanoCertificate &certificate) {
    proj::validate_tensor(tensor);
    if (certificate.axis >= 3) {
        throw std::invalid_argument("Fano axis must be in [0, 2]");
    }
    const U16 used_coordinates = certificate.basis[0] |
                                 certificate.basis[1] |
                                 certificate.basis[2];
    if ((used_coordinates & ~kFanoMask) != 0) {
        throw std::invalid_argument("Fano functional exceeds dimension 9");
    }
    const std::array<Matrix, 9> slices =
        proj::tensor_slices(tensor, certificate.axis);
    const Matrix first = fano_contraction(slices, certificate.basis[0]);
    const Matrix second = fano_contraction(slices, certificate.basis[1]);
    const Matrix third = fano_contraction(slices, certificate.basis[2]);
    const Matrix first_second = fano_xor(first, second);
    const Matrix first_third = fano_xor(first, third);
    const Matrix second_third = fano_xor(second, third);
    const Matrix all = fano_xor(first_second, third);
    return fano_invertible(first) && fano_invertible(second) &&
           fano_invertible(first_second) && fano_invertible(third) &&
           fano_invertible(first_third) &&
           fano_invertible(second_third) && fano_invertible(all);
}

inline std::optional<FanoCertificate>
fano_find_axis(const Tensor &tensor, const Shape &shape, U8 axis) {
    validate_fano_shape(tensor, shape);
    if (axis >= 3) {
        throw std::invalid_argument("Fano axis must be in [0, 2]");
    }
    return fano_find_axis_unchecked(tensor, shape, axis);
}

inline std::optional<FanoCertificate> fano_find(const Tensor &tensor,
                                                const Shape &shape) {
    validate_fano_shape(tensor, shape);
    for (U8 axis = 0; axis < 3; ++axis) {
        if (const std::optional<FanoCertificate> result =
                fano_find_axis_unchecked(tensor, shape, axis)) {
            return result;
        }
    }
    return std::nullopt;
}

inline std::optional<FanoCertificate> fano_find(const Tensor &tensor) {
    const Shape shape = proj::natural_shape(tensor);
    for (U8 axis = 0; axis < 3; ++axis) {
        if (const std::optional<FanoCertificate> result =
                fano_find_axis_unchecked(tensor, shape, axis)) {
            return result;
        }
    }
    return std::nullopt;
}

} // namespace lb
