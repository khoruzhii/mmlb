#pragma once

#include "types.h"

#include <array>
#include <immintrin.h>

class Tensor {
  private:
    std::array<U64,64> data;

  public:
    Shape shape;

    Tensor() : shape{} {}
    explicit Tensor(const std::array<U64,64>& d, Shape s = {}) : data(d), shape(s) {}

    bool get_bit(size_t i, size_t j, size_t k) const {
      return (data[4*i + j/4] >> (k + 16*(j%4))) & U64(1);
    }

    U16 get_row(size_t i, size_t j) const {
      return (data[4*i + j/4] >> (16*(j%4))) & U64(0xFFFF);
    }

#if __cplusplus >= 202302L
    bool operator[](size_t i, size_t j, size_t k) const {
      return get_bit(i, j, k);
    }

    U16 operator[](size_t i, size_t j) const {
      return (data[4*i + j/4] >> (16*(j%4))) & U64(0xFFFF);
    }
#endif
    
    void set_bit(size_t i, size_t j, size_t k, bool value) {
      if (value) {
          data[4*i + j/4] |= (U64(1) << (k + 16*(j%4)));
      } else {
        data[4*i + j/4] &= ~(U64(1) << (k + 16*(j%4)));
      }
    }

    const std::array<U64,64>& raw() const { return data; }

    inline Tensor transpose_BC() const {
      std::array<U64,64> out{};
      for (size_t i = 0; i < 16; i++) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data[i*4]));

        // ── Phase 1: Deinterleave bytes ─────────────────────────────
        // Within each 128-bit lane (= 8 rows), move the low bytes
        // (cols 0-7) to the low qword and high bytes (cols 8-15)
        // to the high qword.
        //
        // Input lane:  [r0_lo, r0_hi, r1_lo, r1_hi, ..., r3_lo, r3_hi]
        //               byte0  byte1  byte2  byte3  ...  byte6  byte7
        // Output lane: [r0_lo, r1_lo, ..., r3_lo,  r0_hi, r1_hi, ..., r3_hi]
        //              ──── block A/C ────  ──── block B/D ────
        const __m256i deinterleave = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0,2,4,6,8,10,12,14,  1,3,5,7,9,11,13,15));
        x = _mm256_shuffle_epi8(x, deinterleave);
        // x = [A, B, C, D]  (four 8×8 bit-matrices, one per 64-bit lane)

        // ── Phase 2: Transpose each 8×8 block (3 delta-swaps) ──────
        //
        // delta_swap(x, δ, mask):
        //   t = (x ^ (x >> δ)) & mask;   x ^= t ^ (t << δ);
        //
        // Each swap exchanges one pair of row/column index bits.
        __m256i t;

        // δ = 7:  swap 1×1 blocks
        const __m256i m7 = _mm256_set1_epi64x(0x00AA00AA00AA00AAULL);
        t = _mm256_and_si256(
                _mm256_xor_si256(x, _mm256_srli_epi64(x, 7)), m7);
        x = _mm256_xor_si256(x,
                _mm256_xor_si256(t, _mm256_slli_epi64(t, 7)));

        // δ = 14: swap 2×2 blocks
        const __m256i m14 = _mm256_set1_epi64x(0x0000CCCC0000CCCCULL);
        t = _mm256_and_si256(
                _mm256_xor_si256(x, _mm256_srli_epi64(x, 14)), m14);
        x = _mm256_xor_si256(x,
                _mm256_xor_si256(t, _mm256_slli_epi64(t, 14)));

        // δ = 28: swap 4×4 blocks
        const __m256i m28 = _mm256_set1_epi64x(0x00000000F0F0F0F0ULL);
        t = _mm256_and_si256(
                _mm256_xor_si256(x, _mm256_srli_epi64(x, 28)), m28);
        x = _mm256_xor_si256(x,
                _mm256_xor_si256(t, _mm256_slli_epi64(t, 28)));

        // x = [A^T, B^T, C^T, D^T]

        // ── Phase 3: Reinterleave into 16-bit rows ─────────────────
        //
        // Output row j (j < 8)  = byte j of A^T  |  (byte j of C^T) << 8
        // Output row j (j >= 8) = byte j of B^T  |  (byte j of D^T) << 8
        //
        // punpcklbw interleaves the low 8 bytes of two registers,
        // punpckhbw interleaves the high 8 bytes.
        __m128i lo = _mm256_castsi256_si128(x);         // [A^T, B^T]
        __m128i hi = _mm256_extracti128_si256(x, 1);    // [C^T, D^T]

        __m128i top = _mm_unpacklo_epi8(lo, hi);        // rows 0–7  of M^T
        __m128i bot = _mm_unpackhi_epi8(lo, hi);        // rows 8–15 of M^T

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i*4 + 0]), top);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i*4 + 2]), bot);
      }
      return Tensor(out, {shape[0], shape[2], shape[1]});
    }
};
