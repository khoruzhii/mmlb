#include "tensor.h"
#include "types.h"
#include <bit>

Tensor apply_substitution(const Tensor& T_orig, U16 u, int axis) {
    Tensor T = T_orig;
    int idx = std::countr_zero(u);
    U16 rest = u ^ (1 << idx);

    if (axis == 0) {
        for (int b = 0; b < 16; b++) {
            if ((rest >> b) & 1) {
                for (int w = 0; w < 4; w++) {
                    T.mut_raw()[4*b+w] ^= T.raw()[4*idx + w];
                }
            }
        }
        for (int w = 0; w < 4; w++) {
            T.mut_raw()[4 * idx + w] = 0;
        }
    } else {
        if (axis == 1) {
            T = T.transpose_BC();
        }
        // now we only need to worry about axis == 2
        U64 mask = 0x0001000100010001ULL << idx;
        for (int w = 0; w < 64; w++) {
            U64 maskandT = (T.raw()[w] & mask) >> idx;
            T.mut_raw()[w] ^= (maskandT * rest);
            T.mut_raw()[w] &= ~mask;
        }

        if (axis == 1) {
            T = T.transpose_BC();
        }
    }
    T.nf();
    return T;
}