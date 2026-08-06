#pragma once

#include "../third_party/updated_code/tensor.hpp"
#include <random>

inline int ub(const Tensor& T) {
    // first make the Scheme

    int maxrank = 0;
    for (size_t i = 0; i < T.shape[0]; i++) {
        for (size_t j = 0; j < T.shape[1]; j++) {
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (T.get_bit(i,j,k)) {
                    maxrank++;
                }
            }
        }
    }
    Scheme scheme;
    scheme.maxrank = maxrank + 10;
    scheme.data = new factor[3 * scheme.maxrank];
    scheme.flips = new PairSet[3];
    scheme.rank = maxrank;

    int current_row = 0;
    for (size_t i = 0; i < T.shape[0]; i++) {
        for (size_t j = 0; j < T.shape[1]; j++) {
            for (size_t k = 0; k < T.shape[2]; k++) {
                if (T.get_bit(i,j,k)) {
                    scheme.data[3*current_row] = (1ULL << i);
                    scheme.data[3*current_row+1] = (1ULL << j);
                    scheme.data[3*current_row+2] = (1ULL << k);
                    current_row++;
                }
            }
        }
    }
    scheme.init();
    if (scheme.rank <= 1) return scheme.rank;

    // Now set up the rng and fgs
    static std::mt19937 gen(42); // static to avoid entropy drain
    std::uniform_int_distribution<> coinflip(0, 1);
    std::uniform_int_distribution<> d3(0, 2);
    int min_rank = scheme.rank;
    int flips_since_plus = 0;
    int pathlength = 50000;

    for (int step = 0; step < pathlength; step++) {
        int numflips = scheme.flips[0].size() + scheme.flips[1].size() + scheme.flips[2].size();
        if (flips_since_plus >= 5000 || numflips == 0) {
            scheme.randomsplit(gen, coinflip, d3, 10);
            flips_since_plus = 0;
        }
        bool reduced = scheme.randomflip(gen, coinflip, true);
        flips_since_plus++;
        if (reduced && scheme.rank < min_rank) {
            min_rank = scheme.rank;
            flips_since_plus = 0;
        }
    }
    return min_rank;
}