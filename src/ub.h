#pragma once

#include "bco.h"
#include "reduce.h"
#include "scheme.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fgs {

// 2-reduction followed by flips with plus every 2k; default is 1M flips.
inline U16 ub(const Tensor &tensor, U32 flips = 1'000'000);

// The same upper bound together with a decomposition of the input tensor.
inline std::vector<Term> ubd(const Tensor &tensor, U32 flips = 1'000'000);

inline std::vector<Term> ubd(const Tensor &input, U32 flips_limit) {
    constexpr U32 plus_interval = 2'000;
    constexpr U32 seed = 42;
    constexpr int last_bits = 9 * 9 * 9 % 64;
    constexpr U64 last_mask = (U64{1} << last_bits) - 1;
    if ((input.back() & ~last_mask) != 0) {
        throw std::invalid_argument("bits outside a 9x9x9 tensor must be zero");
    }

    std::vector<Term> terms;
    std::size_t weight = 0;
    for (const U64 word : input) {
        weight += std::popcount(word);
    }
    terms.reserve(weight);
    for (int bit = 0; bit < 9 * 9 * 9; ++bit) {
        if ((input[bit / 64] & (U64{1} << (bit % 64))) == 0) {
            continue;
        }
        const int i = bit / 81;
        const int j = (bit / 9) % 9;
        const int k = bit % 9;
        terms.push_back({static_cast<U16>(1U << i), static_cast<U16>(1U << j),
                         static_cast<U16>(1U << k)});
    }

    reduce2(terms);
    Scheme scheme({9, 9, 9}, std::move(terms), seed);
    U16 best_rank = scheme.rank();
    std::vector<Term> best_terms = scheme.terms();
    const auto update_best = [&] {
        if (scheme.rank() < best_rank) {
            best_rank = scheme.rank();
            best_terms = scheme.terms();
        }
    };

    U32 flips = 0;
    while (flips < flips_limit) {
        if (!scheme.flip()) {
            if (!scheme.plus()) {
                break;
            }
            update_best();
            continue;
        }
        ++flips;
        update_best();
        if (flips % plus_interval == 0 && scheme.plus()) {
            update_best();
        }
    }

    std::erase_if(best_terms,
                  [](const Term &term) { return term[0] == 0; });
    return best_terms;
}

inline U16 ub(const Tensor &tensor, U32 flips) {
    return static_cast<U16>(ubd(tensor, flips).size());
}

} // namespace fgs
