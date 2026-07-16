#pragma once

#include "support.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace lb {

// For one selected mode, ranks[a] is the F_2 matrix rank of the contraction
// by the nonzero functional a. Entry zero is always zero.
struct SliceProfile {
    U8 dimension{};
    std::array<U8, 512> ranks{};
};

// Integer dual certificate for the one-mode covering LP. If p[a] is the
// nonnegative weight at a, put
//
//   D = max_(x != 0) sum_(a dot x = 1) p[a].
//
// Then p / D is dual feasible and proves
//
//   rank(T) >= ceil(sum_a p[a] rank(T(a)) / D).
//
// The fixed array also accepts dense certificates produced by an optional LP
// solver. The finder below normally returns sparse 0/1 certificates.
struct SliceCertificate {
    U8 axis{};
    std::array<U32, 512> weights{};

    bool operator==(const SliceCertificate &) const = default;
};

struct SliceResult {
    U8 lower_bound{};
    SliceCertificate certificate{};

    bool operator==(const SliceResult &) const = default;
};

inline constexpr U32 kSliceTrials = 4096;

inline SliceProfile slice_profile(const Tensor &tensor, const Shape &shape,
                                  U8 axis);
inline SliceProfile slice_profile(const Tensor &tensor, U8 axis);

// Recompute and verify a supplied certificate exactly. This never trusts a
// stored objective or denominator.
inline U8 slice_verify(const Tensor &tensor, const Shape &shape,
                       const SliceCertificate &certificate);
inline U8 slice_verify(const Tensor &tensor,
                       const SliceCertificate &certificate);

// Search uniform subspace certificates. Dimensions at most seven are searched
// exhaustively. For larger modes the cheap dimensions are exhaustive and the
// remaining dimensions use a deterministic rank-biased search.
inline SliceResult slice_find_axis(const Tensor &tensor, const Shape &shape,
                                   U8 axis, U32 trials = kSliceTrials);
inline SliceResult slice_find(const Tensor &tensor, const Shape &shape,
                              U32 trials = kSliceTrials);
inline SliceResult slice_find(const Tensor &tensor,
                              U32 trials = kSliceTrials);

// A general inexpensive tensor-rank lower bound: ordinary mode flattenings
// together with the best one-mode covering certificate found on any axis.
inline U8 slice(const Tensor &tensor, const Shape &shape,
                U32 trials = kSliceTrials);
inline U8 slice(const Tensor &tensor, U32 trials = kSliceTrials);

inline void validate_slice_shape(const Tensor &tensor, const Shape &shape) {
    proj::validate_tensor(tensor);
    if (!proj::tensor_fits_shape(tensor, shape)) {
        throw std::invalid_argument("tensor exceeds its slice shape");
    }
}

inline SliceProfile slice_profile_unchecked(const Tensor &tensor,
                                            const Shape &shape, U8 axis) {
    const U8 row_axis = proj::kOtherAxes[axis][0];
    const U8 column_axis = proj::kOtherAxes[axis][1];
    const std::array<Matrix, 9> matrices =
        proj::tensor_slices(tensor, axis);
    SliceProfile result{shape[axis], {}};
    Matrix contraction{};
    U16 previous = 0;
    const U16 limit = U16{1} << shape[axis];
    for (U16 step = 1; step < limit; ++step) {
        const U16 functional = step ^ (step >> 1);
        const U16 changed = functional ^ previous;
        const int coordinate = std::countr_zero(changed);
        for (int row = 0; row < shape[row_axis]; ++row) {
            contraction[row] ^= matrices[coordinate][row];
        }
        result.ranks[functional] = proj::matrix_rank(
            contraction, shape[row_axis], shape[column_axis]);
        previous = functional;
    }
    return result;
}

inline SliceProfile slice_profile(const Tensor &tensor, const Shape &shape,
                                  U8 axis) {
    validate_slice_shape(tensor, shape);
    if (axis >= 3) {
        throw std::invalid_argument("slice axis must be in [0, 2]");
    }
    return slice_profile_unchecked(tensor, shape, axis);
}

inline SliceProfile slice_profile(const Tensor &tensor, U8 axis) {
    const Shape shape = proj::natural_shape(tensor);
    if (axis >= 3) {
        throw std::invalid_argument("slice axis must be in [0, 2]");
    }
    return slice_profile_unchecked(tensor, shape, axis);
}

inline U8 slice_verify_profile(const SliceProfile &profile,
                               const SliceCertificate &certificate) {
    const U16 limit = U16{1} << profile.dimension;
    if (certificate.weights[0] != 0) {
        throw std::invalid_argument("slice weight at zero must vanish");
    }
    for (int functional = limit; functional < 512; ++functional) {
        if (certificate.weights[functional] != 0) {
            throw std::invalid_argument("slice weight exceeds mode dimension");
        }
    }

    std::array<std::int64_t, 512> transform{};
    U64 total = 0;
    U64 numerator = 0;
    for (int functional = 1; functional < limit; ++functional) {
        const U32 weight = certificate.weights[functional];
        transform[functional] = weight;
        total += weight;
        numerator += U64{weight} * profile.ranks[functional];
    }
    if (total == 0) {
        return 0;
    }

    for (int length = 1; length < limit; length *= 2) {
        for (int start = 0; start < limit; start += 2 * length) {
            for (int offset = 0; offset < length; ++offset) {
                const std::int64_t left = transform[start + offset];
                const std::int64_t right =
                    transform[start + length + offset];
                transform[start + offset] = left + right;
                transform[start + length + offset] = left - right;
            }
        }
    }

    U64 denominator = 0;
    for (int vector = 1; vector < limit; ++vector) {
        const std::int64_t difference =
            static_cast<std::int64_t>(total) - transform[vector];
        if (difference < 0 || (difference & 1) != 0) {
            throw std::logic_error("invalid slice Walsh transform");
        }
        denominator = std::max(
            denominator, static_cast<U64>(difference / 2));
    }
    if (denominator == 0) {
        throw std::logic_error("nonzero slice certificate has zero load");
    }
    const U64 bound = (numerator + denominator - 1) / denominator;
    if (bound > std::numeric_limits<U8>::max()) {
        throw std::overflow_error("slice lower bound exceeds U8");
    }
    return static_cast<U8>(bound);
}

inline SliceProfile
slice_weighted_profile_unchecked(const Tensor &tensor, const Shape &shape,
                                 const SliceCertificate &certificate) {
    const U16 limit = U16{1} << shape[certificate.axis];
    int nonzero = 0;
    for (int functional = 1; functional < limit; ++functional) {
        nonzero += certificate.weights[functional] != 0;
    }
    if (nonzero == 0) {
        return SliceProfile{shape[certificate.axis], {}};
    }
    if (nonzero > 384) {
        return slice_profile_unchecked(tensor, shape, certificate.axis);
    }

    const U8 row_axis = proj::kOtherAxes[certificate.axis][0];
    const U8 column_axis = proj::kOtherAxes[certificate.axis][1];
    const std::array<Matrix, 9> matrices =
        proj::tensor_slices(tensor, certificate.axis);
    SliceProfile profile{shape[certificate.axis], {}};
    for (int functional = 1; functional < limit; ++functional) {
        if (certificate.weights[functional] == 0) {
            continue;
        }
        Matrix contraction{};
        U16 selected = static_cast<U16>(functional);
        while (selected != 0) {
            const int coordinate = std::countr_zero(selected);
            for (int row = 0; row < shape[row_axis]; ++row) {
                contraction[row] ^= matrices[coordinate][row];
            }
            selected &= selected - 1;
        }
        profile.ranks[functional] = proj::matrix_rank(
            contraction, shape[row_axis], shape[column_axis]);
    }
    return profile;
}

inline U8 slice_verify(const Tensor &tensor, const Shape &shape,
                       const SliceCertificate &certificate) {
    if (certificate.axis >= 3) {
        throw std::invalid_argument("slice axis must be in [0, 2]");
    }
    validate_slice_shape(tensor, shape);
    return slice_verify_profile(
        slice_weighted_profile_unchecked(tensor, shape, certificate),
        certificate);
}

inline U8 slice_verify(const Tensor &tensor,
                       const SliceCertificate &certificate) {
    const Shape shape = proj::natural_shape(tensor);
    if (certificate.axis >= 3) {
        throw std::invalid_argument("slice axis must be in [0, 2]");
    }
    return slice_verify_profile(
        slice_weighted_profile_unchecked(tensor, shape, certificate),
        certificate);
}

struct SliceSearch {
    U64 numerator{};
    U32 denominator{1};
    SliceCertificate certificate{};
};

inline bool slice_better(U64 numerator, U32 denominator,
                         const SliceSearch &best) {
    return numerator * best.denominator > best.numerator * denominator;
}

inline U8 slice_search_bound(const SliceSearch &search) {
    return static_cast<U8>(
        (search.numerator + search.denominator - 1) /
        search.denominator);
}

inline U8 slice_subspace_upper_bound(const SliceProfile &profile,
                                     U8 dimension) {
    int remaining = (1 << dimension) - 1;
    U64 numerator = 0;
    const int limit = 1 << profile.dimension;
    for (int rank = 9; rank >= 0 && remaining != 0; --rank) {
        int count = 0;
        for (int functional = 1; functional < limit; ++functional) {
            count += profile.ranks[functional] == rank;
        }
        const int used = std::min(remaining, count);
        numerator += U64{static_cast<U32>(used)} * rank;
        remaining -= used;
    }
    const U32 denominator = U32{1} << (dimension - 1);
    return static_cast<U8>((numerator + denominator - 1) / denominator);
}

template <U8 dimension>
inline U64 slice_subspace_score_fixed(
    const SliceProfile &profile, const std::array<U16, 9> &basis) {
    U64 score = 0;
    U16 value = 0;
    U16 previous = 0;
    for (int step = 1; step < (1 << dimension); ++step) {
        const U16 gray = static_cast<U16>(step ^ (step >> 1));
        const int changed = std::countr_zero(
            static_cast<U16>(gray ^ previous));
        value ^= basis[changed];
        score += profile.ranks[value];
        previous = gray;
    }
    return score;
}

inline U64 slice_subspace_score(const SliceProfile &profile,
                                const std::array<U16, 9> &basis,
                                U8 dimension) {
    switch (dimension) {
    case 1:
        return slice_subspace_score_fixed<1>(profile, basis);
    case 2:
        return slice_subspace_score_fixed<2>(profile, basis);
    case 3:
        return slice_subspace_score_fixed<3>(profile, basis);
    case 4:
        return slice_subspace_score_fixed<4>(profile, basis);
    case 5:
        return slice_subspace_score_fixed<5>(profile, basis);
    case 6:
        return slice_subspace_score_fixed<6>(profile, basis);
    case 7:
        return slice_subspace_score_fixed<7>(profile, basis);
    case 8:
        return slice_subspace_score_fixed<8>(profile, basis);
    case 9:
        return slice_subspace_score_fixed<9>(profile, basis);
    default:
        return 0;
    }
}

inline void slice_store_subspace(SliceSearch &best, U8 axis,
                                 const std::array<U16, 9> &basis,
                                 U8 dimension, U64 score) {
    const U32 denominator = U32{1} << (dimension - 1);
    if (!slice_better(score, denominator, best)) {
        return;
    }
    best.numerator = score;
    best.denominator = denominator;
    best.certificate = {};
    best.certificate.axis = axis;
    U16 value = 0;
    U16 previous = 0;
    for (int step = 1; step < (1 << dimension); ++step) {
        const U16 gray = static_cast<U16>(step ^ (step >> 1));
        value ^= basis[std::countr_zero(
            static_cast<U16>(gray ^ previous))];
        best.certificate.weights[value] = 1;
        previous = gray;
    }
}

template <class Callback>
inline void slice_rref_subspaces(U8 ambient, U8 dimension,
                                 Callback callback) {
    std::array<U8, 9> pivots{};
    const auto choose = [&](auto &&self, U8 row, U8 first) -> void {
        if (row != dimension) {
            for (U8 pivot = first;
                 pivot <= static_cast<U8>(ambient - dimension + row);
                 ++pivot) {
                pivots[row] = pivot;
                self(self, static_cast<U8>(row + 1),
                     static_cast<U8>(pivot + 1));
            }
            return;
        }

        std::array<U8, 81> free_positions{};
        U8 free_count = 0;
        U16 pivot_mask = 0;
        for (int index = 0; index < dimension; ++index) {
            pivot_mask |= U16{1} << pivots[index];
        }
        for (U8 index = 0; index < dimension; ++index) {
            for (U8 column = static_cast<U8>(pivots[index] + 1);
                 column < ambient; ++column) {
                if ((pivot_mask & (U16{1} << column)) == 0) {
                    free_positions[free_count++] =
                        static_cast<U8>(9 * index + column);
                }
            }
        }

        const U32 configurations = U32{1} << free_count;
        for (U32 configuration = 0; configuration < configurations;
             ++configuration) {
            std::array<U16, 9> basis{};
            for (int index = 0; index < dimension; ++index) {
                basis[index] = U16{1} << pivots[index];
            }
            U32 bits = configuration;
            while (bits != 0) {
                const int free = std::countr_zero(bits);
                const U8 position = free_positions[free];
                basis[position / 9] |= U16{1} << (position % 9);
                bits &= bits - 1;
            }
            callback(basis);
        }
    };
    choose(choose, 0, 0);
}

inline void slice_exhaust_dimension(const SliceProfile &profile, U8 axis,
                                    U8 dimension, SliceSearch &best) {
    slice_rref_subspaces(
        profile.dimension, dimension,
        [&](const std::array<U16, 9> &basis) {
            const U64 score =
                slice_subspace_score(profile, basis, dimension);
            slice_store_subspace(best, axis, basis, dimension, score);
        });
}

inline U8 slice_span_rank(const std::array<U16, 511> &vectors,
                          int count) {
    std::array<U16, 9> pivots{};
    U8 rank = 0;
    for (int index = 0; index < count; ++index) {
        U16 value = vectors[index];
        while (value != 0) {
            const int pivot = std::bit_width(value) - 1;
            if (pivots[pivot] != 0) {
                value ^= pivots[pivot];
            } else {
                pivots[pivot] = value;
                ++rank;
                break;
            }
        }
    }
    return rank;
}

inline U64 slice_random(U64 &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

inline std::array<U16, 511>
slice_rank_order(const SliceProfile &profile) {
    std::array<U16, 511> ordered{};
    int position = 0;
    const int count = (1 << profile.dimension) - 1;
    for (int rank = 9; rank >= 0; --rank) {
        for (int vector = 1; vector <= count; ++vector) {
            if (profile.ranks[vector] == rank) {
                ordered[position++] = static_cast<U16>(vector);
            }
        }
    }
    return ordered;
}

inline void slice_random_dimension(const SliceProfile &profile, U8 axis,
                                   U8 dimension, U32 trials,
                                   const std::array<U16, 511> &ordered,
                                   SliceSearch &best, U64 &state) {
    const U8 upper_bound =
        slice_subspace_upper_bound(profile, dimension);
    if (upper_bound <= slice_search_bound(best)) {
        return;
    }
    const int count = (1 << profile.dimension) - 1;
    int elite = 1;
    while (elite < count &&
           profile.ranks[ordered[elite]] ==
               profile.ranks[ordered[elite - 1]]) {
        ++elite;
    }
    while (elite < count && slice_span_rank(ordered, elite) < dimension) {
        const U8 next_rank = profile.ranks[ordered[elite]];
        while (elite < count &&
               profile.ranks[ordered[elite]] == next_rank) {
            ++elite;
        }
    }
    for (U32 trial = 0; trial < trials; ++trial) {
        // Interleave elite and unrestricted samples so increasing the trial
        // budget extends the same deterministic search prefix.
        const int pool = trial % 4 != 3 ? elite : count;
        std::array<U16, 9> basis{};
        std::array<U16, 9> pivots{};
        U8 basis_size = 0;
        for (int attempt = 0; basis_size < dimension && attempt < 64;
             ++attempt) {
            const U16 original = ordered[slice_random(state) % pool];
            U16 value = original;
            while (value != 0) {
                const int pivot = std::bit_width(value) - 1;
                if (pivots[pivot] != 0) {
                    value ^= pivots[pivot];
                } else {
                    pivots[pivot] = value;
                    basis[basis_size++] = original;
                    break;
                }
            }
        }
        if (basis_size != dimension) {
            continue;
        }
        const U64 score =
            slice_subspace_score(profile, basis, dimension);
        slice_store_subspace(best, axis, basis, dimension, score);
        if (slice_search_bound(best) == upper_bound) {
            return;
        }
    }
}

constexpr std::array<U16, 168> make_slice_elliptic_sets() {
    std::array<U16, 168> result{};
    int index = 0;
    for (U32 set = 0; set < (U32{1} << 15); ++set) {
        if (std::popcount(set) != 5) {
            continue;
        }
        int sum = 0;
        U32 values = set;
        while (values != 0) {
            sum ^= std::countr_zero(values) + 1;
            values &= values - 1;
        }
        if (sum == 0) {
            result[index++] = static_cast<U16>(set);
        }
    }
    return result;
}

inline constexpr std::array<U16, 168> kSliceEllipticSets =
    make_slice_elliptic_sets();
static_assert(kSliceEllipticSets.back() != 0);

inline void slice_store_set(const SliceProfile &profile, U8 axis, U16 set,
                            U32 denominator, SliceSearch &best) {
    U64 numerator = 0;
    U16 values = set;
    while (values != 0) {
        const int functional = std::countr_zero(values) + 1;
        numerator += profile.ranks[functional];
        values &= values - 1;
    }
    if (!slice_better(numerator, denominator, best)) {
        return;
    }
    best.numerator = numerator;
    best.denominator = denominator;
    best.certificate = {};
    best.certificate.axis = axis;
    values = set;
    while (values != 0) {
        const int functional = std::countr_zero(values) + 1;
        best.certificate.weights[functional] = 1;
        values &= values - 1;
    }
}

inline void slice_dimension_four_vertices(const SliceProfile &profile,
                                           U8 axis, SliceSearch &best) {
    constexpr U16 all = (U16{1} << 15) - 1;
    for (const U16 set : kSliceEllipticSets) {
        slice_store_set(profile, axis, set, 4, best);
        slice_store_set(profile, axis, static_cast<U16>(all ^ set), 6,
                        best);
    }
}

inline SliceResult slice_find_axis_unchecked(const Tensor &tensor,
                                             const Shape &shape, U8 axis,
                                             U32 trials) {
    const SliceProfile profile =
        slice_profile_unchecked(tensor, shape, axis);
    SliceSearch best{};
    best.certificate.axis = axis;
    if (profile.dimension == 0) {
        return {};
    }

    std::array<U16, 9> full_basis{};
    for (int index = 0; index < profile.dimension; ++index) {
        full_basis[index] = U16{1} << index;
    }
    slice_store_subspace(
        best, axis, full_basis, profile.dimension,
        slice_subspace_score(profile, full_basis, profile.dimension));

    if (profile.dimension <= 7) {
        for (U8 dimension = 1; dimension <= profile.dimension;
             ++dimension) {
            if (dimension == profile.dimension ||
                slice_subspace_upper_bound(profile, dimension) <=
                    slice_search_bound(best)) {
                continue;
            }
            slice_exhaust_dimension(profile, axis, dimension, best);
        }
    } else {
        if (slice_subspace_upper_bound(profile, 1) >
            slice_search_bound(best)) {
            slice_exhaust_dimension(profile, axis, 1, best);
        }
        if (slice_subspace_upper_bound(profile, 2) >
            slice_search_bound(best)) {
            slice_exhaust_dimension(profile, axis, 2, best);
        }
        if (profile.dimension == 8 &&
            slice_subspace_upper_bound(profile, 3) >
                slice_search_bound(best)) {
            slice_exhaust_dimension(profile, axis, 3, best);
        }

        if (trials != 0) {
            U64 seed = 0x9e3779b97f4a7c15ULL ^
                       (U64{axis} << 56) ^
                       (U64{profile.dimension} << 48);
            for (int functional = 1;
                 functional < (1 << profile.dimension); ++functional) {
                seed ^= static_cast<U64>(profile.ranks[functional] + 1) *
                        (0xbf58476d1ce4e5b9ULL + functional);
                seed = (seed << 17) | (seed >> 47);
            }
            const std::array<U16, 511> ordered =
                slice_rank_order(profile);
            for (U8 dimension = 3;
                 dimension < profile.dimension && dimension <= 5;
                 ++dimension) {
                U64 state = seed ^
                            (0x94d049bb133111ebULL * (dimension + 1));
                slice_random_dimension(profile, axis, dimension, trials,
                                       ordered, best, state);
            }
        }
    }

    // For four-dimensional modes the remaining LP vertices are the 168
    // elliptic five-sets and their complements. Together with subspaces this
    // is an exact finder for the full one-mode covering LP in dimension four.
    if (profile.dimension == 4) {
        slice_dimension_four_vertices(profile, axis, best);
    }

    const U64 bound =
        (best.numerator + best.denominator - 1) / best.denominator;
    return {static_cast<U8>(bound), best.certificate};
}

inline SliceResult slice_find_axis(const Tensor &tensor, const Shape &shape,
                                   U8 axis, U32 trials) {
    validate_slice_shape(tensor, shape);
    if (axis >= 3) {
        throw std::invalid_argument("slice axis must be in [0, 2]");
    }
    return slice_find_axis_unchecked(tensor, shape, axis, trials);
}

inline SliceResult slice_find_unchecked(const Tensor &tensor,
                                        const Shape &shape, U32 trials) {
    SliceResult best{};
    for (U8 axis = 0; axis < 3; ++axis) {
        const SliceResult candidate =
            slice_find_axis_unchecked(tensor, shape, axis, trials);
        if (candidate.lower_bound > best.lower_bound) {
            best = candidate;
        }
    }
    return best;
}

inline SliceResult slice_find(const Tensor &tensor, const Shape &shape,
                              U32 trials) {
    validate_slice_shape(tensor, shape);
    return slice_find_unchecked(tensor, shape, trials);
}

inline SliceResult slice_find(const Tensor &tensor, U32 trials) {
    const Shape shape = proj::natural_shape(tensor);
    return slice_find_unchecked(tensor, shape, trials);
}

inline U8 slice(const Tensor &tensor, const Shape &shape, U32 trials) {
    validate_slice_shape(tensor, shape);
    const SliceResult covering = slice_find_unchecked(tensor, shape, trials);
    const Shape support = proj::multilinear_rank(tensor);
    return std::max({covering.lower_bound, support[0], support[1],
                     support[2]});
}

inline U8 slice(const Tensor &tensor, U32 trials) {
    const Shape shape = proj::natural_shape(tensor);
    const SliceResult covering = slice_find_unchecked(tensor, shape, trials);
    const Shape support = proj::multilinear_rank(tensor);
    return std::max({covering.lower_bound, support[0], support[1],
                     support[2]});
}

} // namespace lb
