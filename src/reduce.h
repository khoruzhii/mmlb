#pragma once

#include "types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fgs {

inline U16 reduce2(std::vector<Term> &terms);

constexpr std::size_t kMaxTerms = 1000;

using PairVector = std::array<U64, 2>;
using Coefficients = std::array<U64, (kMaxTerms + 63) / 64>;
constexpr int kPairBits = 81;

inline bool is_zero(const Term &term) {
    return term[0] == 0 || term[1] == 0 || term[2] == 0;
}

inline void compact(std::vector<Term> &terms) {
    std::erase_if(terms, [](const Term &term) { return is_zero(term); });
}

inline PairVector outer_product(U16 first, U16 second) {
    PairVector result{};
    for (; first != 0; first &= static_cast<U16>(first - 1)) {
        const int i = std::countr_zero(first);
        for (U16 value = second; value != 0;
             value &= static_cast<U16>(value - 1)) {
            const int j = std::countr_zero(value);
            const int bit = 9 * i + j;
            result[bit / 64] |= U64{1} << (bit % 64);
        }
    }
    return result;
}

inline bool is_zero(const PairVector &value) {
    return value[0] == 0 && value[1] == 0;
}

inline int leading_bit(const PairVector &value) {
    if (value[1] != 0) {
        return 64 + static_cast<int>(std::bit_width(value[1])) - 1;
    }
    return static_cast<int>(std::bit_width(value[0])) - 1;
}

inline void xor_with(PairVector &target, const PairVector &source) {
    target[0] ^= source[0];
    target[1] ^= source[1];
}

inline void xor_with(Coefficients &target, const Coefficients &source) {
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] ^= source[i];
    }
}

inline bool coefficient(const Coefficients &values, std::size_t index) {
    return (values[index / 64] & (U64{1} << (index % 64))) != 0;
}

inline void apply_all(std::vector<Term> &terms, int first_component,
                      int second_component, int remaining_component) {
    std::array<PairVector, kPairBits> basis{};
    std::array<Coefficients, kPairBits> combinations{};
    std::array<bool, kMaxTerms> independent{};

    for (std::size_t p = 0; p < terms.size(); ++p) {
        PairVector value = outer_product(terms[p][first_component],
                                         terms[p][second_component]);
        Coefficients relation{};
        relation[p / 64] |= U64{1} << (p % 64);

        while (!is_zero(value)) {
            const int pivot = leading_bit(value);
            if (is_zero(basis[pivot])) {
                basis[pivot] = value;
                combinations[pivot] = relation;
                independent[p] = true;
                break;
            }
            xor_with(value, basis[pivot]);
            xor_with(relation, combinations[pivot]);
        }
        if (!is_zero(value)) {
            continue;
        }

        const U16 factor = terms[p][remaining_component];
        for (std::size_t t = 0; t < p; ++t) {
            if (coefficient(relation, t)) {
                terms[t][remaining_component] ^= factor;
            }
        }
    }

    std::vector<Term> reduced;
    reduced.reserve(terms.size());
    for (std::size_t t = 0; t < terms.size(); ++t) {
        if (independent[t] && terms[t][remaining_component] != 0) {
            reduced.push_back(terms[t]);
        }
    }
    terms = std::move(reduced);
}

// Batch-reduce in the order UV|W, UW|V, VW|U and repeat full passes until no
// direction decreases the rank. Returns the final 2-irreducible rank.
inline U16 reduce2(std::vector<Term> &terms) {
    if (terms.size() > kMaxTerms) {
        throw std::invalid_argument("2-reduction supports at most 1000 terms");
    }
    compact(terms);
    while (true) {
        const std::size_t before = terms.size();
        apply_all(terms, 0, 1, 2);
        apply_all(terms, 0, 2, 1);
        apply_all(terms, 1, 2, 0);
        if (terms.size() == before) {
            return static_cast<U16>(terms.size());
        }
    }
}

} // namespace fgs
