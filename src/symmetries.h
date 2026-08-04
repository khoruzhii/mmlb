#include "matrix.h"
#include "tensor.h"
#include <vector>
#include <bit>

namespace symmetries {
// Computing the full group of symmetries is too hard. Instead we use the following approach:
// Suppose that T lies in A * B * C and that U acts only on A, V acts on B and W acts on C. 
// (U,V,W) form a symmetry if U(V(W(T))) = T.
// Let X = I+U, Y = I+V, Z = I+W. Then (U,V,W) form a symmetry if (X+I)(Y+I)(Z+I)T=T.
// Expanding this and rearranging gives XYZT + (XY+YZ+XZ)T + (X+Y+Z)T = 0.
// Solving this properly is too hard, but we can easily look only for solutions that have (X+Y+Z)T = 0.
// This is a linear system of which we want the nullspace, so we find a basis for the right kernel of some matrix.
// Next we check if the (X,Y,Z) found correspond to (U,V,W) that are actually symmetries.


Matrix annihilator_matrix(Tensor& T) {
    size_t sa = T.shape[0];
    size_t sb = T.shape[1];
    size_t sc = T.shape[2];
    Matrix M(sa*sb*sc, sa*sa+sb*sb+sc*sc); // 
    size_t row_idx = 0;
    for (size_t i = 0; i < sa; i++) {
        for (size_t j = 0; j < sb; j++) {
            for (size_t k = 0; k < sc; k++) {
                for (size_t X = 0; X < sa; X++) {
                    if (T.get_bit(X,j,k)) M.set(row_idx, i*sa + X, true); 
                }
                for (size_t Y = 0; Y < sb; Y++) {
                    if (T.get_bit(i,Y,k)) M.set(row_idx, sa*sa + j*sb + Y, true); 
                }
                for (size_t Z = 0; Z < sc; Z++) {
                    if (T.get_bit(i,j,Z)) M.set(row_idx, sa*sa+sb*sb + k*sc + Z, true); 
                }
                row_idx++;
            }
        }
    }
    return M;
}

std::vector<std::vector<U64>> get_candidate_symmetries(Tensor& T) {
    size_t sa = T.shape[0];
    size_t sb = T.shape[1];
    size_t sc = T.shape[2];
    Matrix M = annihilator_matrix(T);
    auto nullspace = M.right_nullspace_basis(); // this is a basis for (X,Y,Z) such that (X+Y+Z)T = 0.
    std::vector<std::vector<U64>> symmetries;
    for (auto& short_vec : nullspace) {
        // first put the vector into the maximum size vector we could have (for 16*16*16 tensors)

        // vec will be the (U,V,W) and not the (X,Y,Z)
        std::vector<U64> vec(12,0);
        for (size_t i = 0; i < 256; i += 17) {
            vec[i/64] ^= (1ULL << (i%64));
            vec[4 + i/64] ^= (1ULL << (i%64));
            vec[8 + i/64] ^= (1ULL << (i%64));
        }

        auto get_short = [&](size_t idx) { return (short_vec[idx/64] >> (idx%64)) & 1; };
        auto flip_vec = [&](size_t idx) { vec[idx/64] ^= (1ULL << (idx%64)); };

        size_t offset = 0;
        for (size_t i = 0; i < sa; i++) {
            for (size_t X = 0; X < sa; X++) {
                if (get_short(offset + i*sa + X) == 1) flip_vec(i * 16 + X);
            }
        }
        offset = sa * sa;
        for (size_t j = 0; j < sb; j++) {
            for (size_t Y = 0; Y < sb; Y++) {
                if (get_short(offset + j * sb + Y) == 1) flip_vec(256 + j * 16 + Y);
            }
        }
        offset += sb * sb;
        for (size_t k = 0; k < sc; k++) {
            for (size_t Z = 0; Z < sc; Z++) {
                if (get_short(offset + k * sc + Z) == 1) flip_vec(512 + k * 16 + Z);
            }
        }

        symmetries.push_back(vec);
        
    }
    return symmetries;
}

bool is_symmetry(Tensor& T, std::vector<U64>& vec) {
    // confirms if a candidate symmetry really works
    std::vector<U64> U(4);
    std::vector<U64> V(4);
    std::vector<U64> W(4);
    for (size_t i = 0; i < 4; i++) {
        U[i] = vec[i];
        V[i] = vec[4 + i];
        W[i] = vec[8 + i];
    }

    Tensor T1;
    T1.shape = T.shape;
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 16; k++) {
                bool T_val = false;
                for (size_t a = 0; a < 16; a++) {
                    T_val ^= (T.get_bit(a,j,k) && ((U[(i*16 + a) / 64] >> ((i*16 + a) % 64)) & 1));
                }
                T1.set_bit(i,j,k,T_val);
            }
        }
    }
    Tensor T2;
    T2.shape = T.shape;
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 16; k++) {
                bool T_val = false;
                for (size_t b = 0; b < 16; b++) {
                    T_val ^= (T1.get_bit(i,b,k) && ((V[(j*16 + b) / 64] >> ((j*16 + b) % 64)) & 1));
                }
                T2.set_bit(i,j,k,T_val);
            }
        }
    }
    Tensor T3;
    T3.shape = T.shape;
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 16; k++) {
                bool T_val = false;
                for (size_t c = 0; c < 16; c++) {
                    T_val ^= (T2.get_bit(i,j,c) && ((W[(k*16 + c) / 64] >> ((k*16 + c) % 64)) & 1));
                }
                T3.set_bit(i,j,k,T_val);
            }
        }
    }
    
    return T == T3; // We need to have a == operator on tensors!
}

std::vector<std::vector<U64>> symmetry_generators(Tensor& T) {
    std::vector<std::vector<U64>> candidates = get_candidate_symmetries(T);
    std::vector<std::vector<U64>> symmetries;
    
    for (auto& candidate : candidates) {
        if (is_symmetry(T, candidate)) {
            symmetries.push_back(candidate);
        }
    }
    return symmetries;
}

U16 apply_symmetry(const std::vector<U64>& sym, U16 vec, size_t dim, int axis) {
    // recall that sym is the three 16*16 matrices (U,V,W)
    U16 out = 0;
    for (size_t i=0; i < dim; i++) {
        U16 row = (sym[axis * 4 + i / 4] >> (16 * (i% 4))) & 0xFFFF;
        out ^= (std::popcount((unsigned int)(row & vec)) & 1) << i;
    }
    return out;
}

std::vector<std::vector<U16>> get_all_orbits(const std::vector<std::vector<U64>>& symmetries, size_t dim, int axis) {
    std::vector<bool> visited(1 << dim, false);
    std::vector<std::vector<U16>> orbits;
    for (size_t v0 = 0; v0 < (1 << dim); v0++) {
        if (visited[v0]) continue;
        std::vector<U16> orbit;
        orbit.push_back(v0);
        visited[v0] = true;
        size_t head = 0;
        while (head < orbit.size()) {
            U16 curr = orbit[head];
            head++;
            for (const auto& sym : symmetries) {
                U16 next = apply_symmetry(sym, curr, dim, axis);
                if (!visited[next]) {
                    visited[next] = true;
                    orbit.push_back(next);
                }
            }
        }
        orbits.push_back(orbit);
    }
    return orbits;
}

} // namespace symmetries