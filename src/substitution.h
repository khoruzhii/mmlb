#include "tensor.h"
#include "types.h"
#include <bit>

Tensor apply_substitution(const Tensor& T_orig, U16 u, int axis) {
    Tensor T = T_orig;
    int idx = std::countr_zero(u);
    U16 rest = u ^ (1 << idx);

    if (axis == 1) {
        T = T.transpose_AB();
    } else if (axis == 2) {
        T = T.transpose_BC().transpose_AB();
    }
    for (int b = 0; b < 16; b++) {
        if ((rest >> b) & 1) {
            for (int w = 0; w < 4; w++) {
                T.mut_raw()[4*b+w] ^= T.raw()[4*idx + w];
            }
        }
    }
    int last = T.shape[0] - 1;
    if (idx != last) {
        for (int w = 0; w < 4; w++) {
            T.mut_raw()[4 * idx + w] = T.raw()[4 * last + w];
        }
    }
    for (int w = 0; w < 4; w++) {
        T.mut_raw()[4 * last + w] = 0;
    }
    T.shape[0]--;
    return T.nf();
}

// So that we can play around with the elements of the tensor in the places we thought they were in before the substitution...
inline Tensor apply_substitution_no_nf(const Tensor& T_orig, U16 u, int axis) {
    Tensor T = T_orig;
    int idx = std::countr_zero(u);
    U16 rest = u ^ (1 << idx);

    if (axis == 1) {
        T = T.transpose_AB();
    } else if (axis == 2) {
        T = T.transpose_BC().transpose_AB();
    }
    for (int b = 0; b < 16; b++) {
        if ((rest >> b) & 1) {
            for (int w = 0; w < 4; w++) {
                T.mut_raw()[4*b+w] ^= T.raw()[4*idx + w];
            }
        }
    }
    int last = T.shape[0] - 1;
    if (idx != last) {
        for (int w = 0; w < 4; w++) {
            T.mut_raw()[4 * idx + w] = T.raw()[4 * last + w];
        }
    }
    for (int w = 0; w < 4; w++) {
        T.mut_raw()[4 * last + w] = 0;
    }
    T.shape[0]--;
    
    // Transpose back so dimensions are in their original axes, just with reduced size
    if (axis == 1) {
        T = T.transpose_AB();
    } else if (axis == 2) {
        T = T.transpose_AB().transpose_BC();
    }
    return T;
}