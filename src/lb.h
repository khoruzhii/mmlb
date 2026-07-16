#pragma once

#include "lb_fano.h"
#include "lb_slice.h"

namespace lb {

// General inexpensive tensor-rank lower bound over F_2. The tensor need not
// be support-normalized or have ordered dimensions.
inline U8 lower_bound(const Tensor &tensor, const Shape &shape,
                      U32 slice_trials = kSliceTrials) {
    U8 result = slice(tensor, shape, slice_trials);
    if (result < 17 && fano_find(tensor, shape).has_value()) {
        result = 17;
    }
    return result;
}

inline U8 lower_bound(const Tensor &tensor,
                      U32 slice_trials = kSliceTrials) {
    return lower_bound(tensor, proj::natural_shape(tensor), slice_trials);
}

} // namespace lb
