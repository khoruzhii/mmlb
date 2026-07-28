#pragma once

#include "support.h"
#include "dynamic_matrix.h"

#include <array>
#include <bit>
#include <stdexcept>
#include <vector>

namespace proj {

// One canonical quotient for an axis and a one-dimensional kernel. Shape is
// the target-space shape, even if the projected tensor has smaller support.
struct Projection {
    Tensor tensor;
    Shape shape;
    U8 axis;
    U16 kernel;
};

// Enumerate axes in order and, within each axis, all nonzero kernel vectors in
// increasing order. Over F_2 every nonzero vector represents a distinct line.
// The input must already have prefix mode supports ordered by nondecreasing
// dimension, as produced by proj::normalize.
inline std::vector<Projection> all(const Tensor &tensor);

inline U16 prefix_mask(U8 dimension) {
    return dimension == 0
               ? U16{0}
               : static_cast<U16>((U16{1} << dimension) - 1);
}

inline void validate_normalized(const Tensor &tensor, const Shape &shape) {
    if (multilinear_rank(tensor) != shape) {
        throw std::invalid_argument(
            "projection input must have prefix mode supports");
    }
    if (shape[0] > shape[1] || shape[1] > shape[2]) {
        throw std::invalid_argument(
            "projection input dimensions must be nondecreasing");
    }
}

inline U16 project_factor(U16 value, U16 kernel, U8 dimension) {
    if (dimension == 0 || dimension > kDimension || kernel == 0 ||
        (kernel & ~prefix_mask(dimension)) != 0) {
        throw std::invalid_argument("invalid elementary projection kernel");
    }
    if ((value & ~prefix_mask(dimension)) != 0) {
        throw std::invalid_argument(
            "projected vector exceeds its input dimension");
    }

    const int pivot = std::countr_zero(kernel);
    const bool pivot_value = (value & (U16{1} << pivot)) != 0;
    U16 result = 0;
    int output = 0;
    for (int input = 0; input < dimension; ++input) {
        if (input == pivot) {
            continue;
        }
        bool bit = (value & (U16{1} << input)) != 0;
        bit ^= pivot_value && (kernel & (U16{1} << input)) != 0;
        if (bit) {
            result |= U16{1} << output;
        }
        ++output;
    }
    return result;
}

inline Tensor project_tensor(const Tensor &tensor, const Shape &shape, U8 axis,
                             U16 kernel) {
    if (axis >= 3) {
        throw std::invalid_argument("projection axis must be in [0, 2]");
    }
    const U8 dimension = shape[axis];
    if (dimension == 0 || kernel == 0 ||
        (kernel & ~prefix_mask(dimension)) != 0) {
        throw std::invalid_argument("invalid elementary projection kernel");
    }

    const int pivot = std::countr_zero(kernel);
    const U8 first_axis = static_cast<U8>((axis + 1) % 3);
    const U8 second_axis = static_cast<U8>((axis + 2) % 3);
    Tensor result{};
    std::array<int, 3> input{};
    std::array<int, 3> output{};

    int projected_coordinate = 0;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        if (coordinate == pivot) {
            continue;
        }
        input[axis] = coordinate;
        output[axis] = projected_coordinate++;
        for (int first = 0; first < shape[first_axis]; ++first) {
            input[first_axis] = first;
            output[first_axis] = first;
            for (int second = 0; second < shape[second_axis]; ++second) {
                input[second_axis] = second;
                output[second_axis] = second;
                bool value =
                    tensor_bit(tensor, input[0], input[1], input[2]);
                if ((kernel & (U16{1} << coordinate)) != 0) {
                    input[axis] = pivot;
                    value ^=
                        tensor_bit(tensor, input[0], input[1], input[2]);
                    input[axis] = coordinate;
                }
                if (value) {
                    toggle_tensor_bit(result, output[0], output[1],
                                      output[2]);
                }
            }
        }
    }
    return result;
}

inline std::vector<Projection> all(const Tensor &tensor) {
    validate_tensor(tensor);
    const Shape shape = natural_shape(tensor);
    validate_normalized(tensor, shape);

    std::size_t count = 0;
    for (const U8 dimension : shape) {
        count += prefix_mask(dimension);
    }

    std::vector<Projection> result;
    result.reserve(count);
    for (U8 axis = 0; axis < 3; ++axis) {
        for (U16 kernel = 1; kernel <= prefix_mask(shape[axis]); ++kernel) {
            Shape projected_shape = shape;
            --projected_shape[axis];
            result.push_back({project_tensor(tensor, shape, axis, kernel),
                              projected_shape, axis, kernel});
        }
    }
    return result;
}

inline std::vector<Projection> sub_orbits(const Tensor &tensor) {
    /*
    Mathematical Explanation:
    This function groups projection kernels into sub-orbits under the action of the tensor's symmetry group.
    Finding the exact symmetry group of a tensor is computationally expensive. Instead, we compute the
    "Derivation Algebra" (the Lie algebra of the symmetry group) which is defined by a linear system:
      T(X_A a, b, c) + T(a, X_B b, c) + T(a, b, X_C c) = 0
    Solving this system gives a basis of derivations (triples of matrices X_A, X_B, X_C).
    
    If a derivation X is nilpotent of degree 2 (i.e. X^2 = 0), we can exponentiate it to get a unipotent
    automorphism: exp(X) = I + X + X^2/2 + ... = I + X (since X^2 = 0 and we are over F_2).
    Because X is a derivation, I + X is an exact symmetry of the tensor.
    
    We collect these unipotent generators (I+X) for each axis. Then, we use Breadth-First Search (BFS) 
    to compute orbits on the set of possible projection kernels (all nonzero vectors in F_2^d) under 
    the action of these generators. Finally, we only return one representative kernel from each sub-orbit.
    
    This does not guarantee that the returned projections are completely distinct up to all symmetries 
    (since we only use unipotent generators derived from the Lie algebra), but it drastically reduces 
    redundancy in practice.
    */
    validate_tensor(tensor);
    const Shape shape = natural_shape(tensor);
    validate_normalized(tensor, shape);

    // 1. Build the 729 x 243 linear system for the derivation algebra.
    auto var_idx = [](int mat_id, int r, int c) {
        return mat_id * 81 + r * kDimension + c;
    };

    DynamicMatrix M(kTensorBits, 243);
    for (int x = 0; x < kDimension; ++x) {
        for (int y = 0; y < kDimension; ++y) {
            for (int z = 0; z < kDimension; ++z) {
                int row = (x * kDimension + y) * kDimension + z;
                for (int m = 0; m < kDimension; ++m) {
                    if (tensor_bit(tensor, m, y, z)) {
                        M.set(row, var_idx(0, m, x), !M.get(row, var_idx(0, m, x)));
                    }
                    if (tensor_bit(tensor, x, m, z)) {
                        M.set(row, var_idx(1, m, y), !M.get(row, var_idx(1, m, y)));
                    }
                    if (tensor_bit(tensor, x, y, m)) {
                        M.set(row, var_idx(2, m, z), !M.get(row, var_idx(2, m, z)));
                    }
                }
            }
        }
    }

    // 2. Solve for the nullspace
    std::vector<std::vector<bool>> basis = M.right_kernel();

    // 3. Extract unipotent group generators for each space
    std::array<std::vector<Matrix>, 3> generators;
    for (const auto& vec : basis) {
        for (int axis = 0; axis < 3; ++axis) {
            Matrix X{};
            bool is_zero = true;
            for (int r = 0; r < kDimension; ++r) {
                for (int c = 0; c < kDimension; ++c) {
                    if (vec[var_idx(axis, r, c)]) {
                        X[c] |= (U16{1} << r);
                        is_zero = false;
                    }
                }
            }

            if (!is_zero) {
                bool nil2 = true;
                for (int r = 0; r < kDimension && nil2; ++r) {
                    for (int c = 0; c < kDimension && nil2; ++c) {
                        bool val = false;
                        for (int k = 0; k < kDimension; ++k) {
                            if ((X[k] & (U16{1} << r)) && (X[c] & (U16{1} << k))) {
                                val ^= true;
                            }
                        }
                        if (val) nil2 = false;
                    }
                }

                if (nil2) {
                    Matrix I_plus_X = X;
                    for (int i = 0; i < kDimension; ++i) {
                        I_plus_X[i] ^= (U16{1} << i);
                    }
                    generators[axis].push_back(I_plus_X);
                }
            }
        }
    }

    // 4. Compute sub-orbits on the possible elements
    std::vector<Projection> result;
    for (U8 axis = 0; axis < 3; ++axis) {
        std::size_t num_elements = std::size_t{1} << shape[axis];
        std::vector<bool> visited(num_elements, false);

        for (U16 start_kernel = 1; start_kernel < num_elements; ++start_kernel) {
            if (!visited[start_kernel]) {
                U16 representative = start_kernel;
                
                std::vector<U16> queue;
                queue.push_back(start_kernel);
                visited[start_kernel] = true;

                std::size_t head = 0;
                while (head < queue.size()) {
                    U16 curr = queue[head++];
                    if (curr < representative) {
                        representative = curr;
                    }

                    for (const auto& g : generators[axis]) {
                        U16 next_v = apply_matrix(g, curr);
                        next_v &= prefix_mask(shape[axis]);
                        
                        if (next_v > 0 && next_v < num_elements && !visited[next_v]) {
                            visited[next_v] = true;
                            queue.push_back(next_v);
                        }
                    }
                }

                Shape projected_shape = shape;
                --projected_shape[axis];
                result.push_back({project_tensor(tensor, shape, axis, representative),
                                  projected_shape, axis, representative});
            }
        }
    }

    return result;
}

} // namespace proj
