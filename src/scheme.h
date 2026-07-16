#pragma once

#include "types.h"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fgs {

// A CP scheme over F_2. Each term is (u, v, w), with every factor stored as a
// bit vector of dimension at most 9. A zero factor denotes a dead term slot.
struct Scheme {
    static constexpr int kMaxDimension = 9;
    static constexpr std::size_t kMaxTerms = 1000;
    static constexpr std::size_t kValueCount = 1U << kMaxDimension;

    explicit Scheme(Shape shape, std::vector<Term> terms, U32 seed = 42)
        : data(std::move(terms)), rng(seed) {
        for (const U8 dimension : shape) {
            if (dimension == 0 || dimension > kMaxDimension) {
                throw std::invalid_argument(
                    "scheme dimensions must be in [1, 9]");
            }
        }
        flippable_slots.fill(kNone);
        if (data.size() > kMaxTerms) {
            throw std::invalid_argument(
                "a scheme can contain at most 1000 terms");
        }

        data.reserve(30);
        position_slots.reserve(30);
        position_slots.resize(data.size());
        live_slots.reserve(30);
        live_slots.resize(data.size());
        free_slots.reserve(30);
        live.reserve(30);

        for (U16 index = 0; index < data.size(); ++index) {
            validate_term(data[index], shape);
            add_live_index(index);
        }
    }

    // Apply a random flip. The shared component value is selected uniformly
    // among all component/value pairs occurring in at least two live terms.
    bool flip() {
        int shared_component = 0;
        U16 first = kNone;
        U16 second = kNone;
        if (!sample_flip_pair(shared_component, first, second)) {
            return false;
        }

        const int next = (shared_component + 1) % 3;
        const int prev = (shared_component + 2) % 3;

        const U16 new_first = data[first][next] ^ data[second][next];
        const U16 new_second = data[first][prev] ^ data[second][prev];
        set_component(first, next, new_first);
        set_component(second, prev, new_second);
        return true;
    }

    // Apply a random 2-to-3 plus transition.
    bool plus() {
        U16 first = kNone;
        U16 second = kNone;
        if (!sample_live_pair(first, second)) {
            return false;
        }

        const int orientation = rng() % 3U;
        const int next = (orientation + 1) % 3;
        const int prev = (orientation + 2) % 3;
        const Term a = data[first];
        const Term b = data[second];
        Term x = a;
        Term y = b;
        Term z{};

        x[orientation] ^= b[orientation];
        y[prev] ^= a[prev];
        z[orientation] = b[orientation];
        z[next] = b[next] ^ a[next];
        z[prev] = a[prev];

        canonicalize_zero(x);
        canonicalize_zero(y);
        canonicalize_zero(z);
        if (live.size() == kMaxTerms && !is_zero(x) && !is_zero(y) &&
            !is_zero(z)) {
            return false;
        }

        replace_term(first, x);
        replace_term(second, y);
        if (!is_zero(z)) {
            add_term(z);
        }
        return true;
    }

    U16 rank() const { return static_cast<U16>(live.size()); }
    const std::vector<Term> &terms() const { return data; }

    static constexpr U16 kNone = std::numeric_limits<U16>::max();
    static constexpr std::size_t kBucketCount = 3 * kValueCount;

    std::vector<Term> data;
    std::vector<std::array<U16, 3>> position_slots;
    std::vector<U16> live;
    std::vector<U16> live_slots;
    std::vector<U16> free_slots;
    std::array<std::vector<U16>, kBucketCount> positions;
    std::vector<U16> flippable;
    std::array<U16, kBucketCount> flippable_slots{};
    std::mt19937 rng;

    static bool is_zero(const Term &term) { return term[0] == 0; }

    static U16 bucket(std::size_t comp, std::size_t value) {
        return static_cast<U16>(comp * kValueCount + value);
    }

    static void canonicalize_zero(Term &term) {
        if (term[0] == 0 || term[1] == 0 || term[2] == 0) {
            term = {};
        }
    }

    static void validate_term(const Term &term, const Shape &shape) {
        for (std::size_t comp = 0; comp < 3; ++comp) {
            const U32 limit = U32{1} << shape[comp];
            if (term[comp] == 0 || term[comp] >= limit) {
                throw std::invalid_argument("invalid scheme factor");
            }
        }
    }

    void add_term(const Term &term) {
        U16 index = kNone;
        if (free_slots.empty()) {
            index = static_cast<U16>(data.size());
            data.push_back(term);
            position_slots.push_back({});
            live_slots.push_back(kNone);
        } else {
            index = free_slots.back();
            free_slots.pop_back();
            data[index] = term;
        }
        add_live_index(index);
    }

    void add_live_index(U16 index) {
        live_slots[index] = static_cast<U16>(live.size());
        live.push_back(index);
        for (int comp = 0; comp < 3; ++comp) {
            add_position(index, comp, data[index][comp]);
        }
    }

    void remove_live_index(U16 index) {
        const U16 slot = live_slots[index];
        const U16 moved = live.back();
        live[slot] = moved;
        live_slots[moved] = slot;
        live.pop_back();
        live_slots[index] = kNone;
    }

    void add_position(U16 index, int comp, U16 value) {
        const U16 key = bucket(comp, value);
        auto &list = positions[key];
        position_slots[index][comp] = static_cast<U16>(list.size());
        if (list.size() == 1) {
            flippable_slots[key] = static_cast<U16>(flippable.size());
            flippable.push_back(key);
        }
        list.push_back(index);
    }

    void remove_position(U16 index, int comp, U16 value) {
        const U16 key = bucket(comp, value);
        auto &list = positions[key];
        const U16 slot = position_slots[index][comp];
        const U16 moved = list.back();
        list[slot] = moved;
        position_slots[moved][comp] = slot;
        if (list.size() == 2) {
            remove_flippable(key);
        }
        list.pop_back();
    }

    void remove_flippable(U16 key) {
        const U16 slot = flippable_slots[key];
        const U16 moved = flippable.back();
        flippable[slot] = moved;
        flippable_slots[moved] = slot;
        flippable.pop_back();
        flippable_slots[key] = kNone;
    }

    void erase_term(U16 index) {
        const Term old = data[index];
        for (int comp = 0; comp < 3; ++comp) {
            remove_position(index, comp, old[comp]);
        }
        remove_live_index(index);
        data[index] = {};
        free_slots.push_back(index);
    }

    void set_component(U16 index, int comp, U16 value) {
        const U16 old = data[index][comp];
        if (old == value) {
            return;
        }
        if (value == 0) {
            erase_term(index);
            return;
        }
        remove_position(index, comp, old);
        data[index][comp] = value;
        add_position(index, comp, value);
    }

    void replace_term(U16 index, const Term &term) {
        if (data[index] == term) {
            return;
        }
        if (is_zero(term)) {
            erase_term(index);
            return;
        }

        const Term old = data[index];
        for (int comp = 0; comp < 3; ++comp) {
            remove_position(index, comp, old[comp]);
        }
        data[index] = term;
        for (int comp = 0; comp < 3; ++comp) {
            add_position(index, comp, term[comp]);
        }
    }

    bool sample_flip_pair(int &component, U16 &first, U16 &second) {
        if (flippable.empty()) {
            return false;
        }

        const U32 sample = rng();
        const U16 key = flippable[sample % flippable.size()];
        component = key / kValueCount;
        const auto &list = positions[key];
        if (list.size() == 2) {
            const bool reverse = (sample & 0x10000U) != 0;
            first = list[reverse ? 1 : 0];
            second = list[reverse ? 0 : 1];
            return true;
        }

        const std::size_t first_slot = (sample >> 16U) % list.size();
        std::size_t second_slot = rng() % (list.size() - 1);
        if (second_slot >= first_slot) {
            ++second_slot;
        }
        first = list[first_slot];
        second = list[second_slot];
        return true;
    }

    bool sample_live_pair(U16 &first, U16 &second) {
        if (live.size() < 2) {
            return false;
        }
        const U32 sample = rng();
        const std::size_t first_slot = sample % live.size();
        std::size_t second_slot = (sample >> 16U) % (live.size() - 1);
        if (second_slot >= first_slot) {
            ++second_slot;
        }
        first = live[first_slot];
        second = live[second_slot];
        return true;
    }
};

} // namespace fgs
