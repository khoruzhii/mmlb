#pragma once

#include "types.h"

#include <array>
#include <immintrin.h>

class Tensor {
  private:
    std::array<U64,64> data;

  public:
    Shape shape;

    Tensor() : data{}, shape{} {}
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
    std::array<U64,64>& mut_raw() { return data; }

    bool operator==(const Tensor& other) const {
        if (shape != other.shape) return false;
        for (size_t i = 0; i < 64; i++) {
            if (data[i] != other.data[i]) return false;
        }
        return true;
    }

    bool operator<(const Tensor& other) const {
        if (shape[0] != other.shape[0]) return shape[0] < other.shape[0];
        if (shape[1] != other.shape[1]) return shape[1] < other.shape[1];
        if (shape[2] != other.shape[2]) return shape[2] < other.shape[2];
        return data < other.data;
    }

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


  inline Tensor transpose_AB() const {
    std::array<U64,64> out{};
    __m256i r[16], t[16], u[16];

    // ── Load: one YMM per row of the 16×16 word matrix ─────────
    for (int i = 0; i < 16; i++)
      r[i] = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(&data[4 * i]));

    // ── Stage 1: 16-bit interleave ─────────────────────────────
    // Pairs rows (0,1), (2,3), ..., (14,15).
    // lo: merges elements at even positions within each 128-bit lane
    // hi: merges elements at odd positions
    for (int i = 0; i < 16; i += 2) {
      t[i + 0] = _mm256_unpacklo_epi16(r[i], r[i + 1]);
      t[i + 1] = _mm256_unpackhi_epi16(r[i], r[i + 1]);
    }

    // ── Stage 2: 32-bit interleave ─────────────────────────────
    // Merges groups of 2 rows into groups of 4.
    // Pairs (t[0],t[2]), (t[1],t[3]) within each block of 4.
    for (int i = 0; i < 16; i += 4) {
      u[i + 0] = _mm256_unpacklo_epi32(t[i + 0], t[i + 2]);
      u[i + 1] = _mm256_unpackhi_epi32(t[i + 0], t[i + 2]);
      u[i + 2] = _mm256_unpacklo_epi32(t[i + 1], t[i + 3]);
      u[i + 3] = _mm256_unpackhi_epi32(t[i + 1], t[i + 3]);
    }

    // ── Stage 3: 64-bit interleave ─────────────────────────────
    // Merges groups of 4 rows into groups of 8.
    // After this, each lane holds one complete column for 8 rows.
    for (int i = 0; i < 16; i += 8) {
      t[i + 0] = _mm256_unpacklo_epi64(u[i + 0], u[i + 4]);
      t[i + 1] = _mm256_unpackhi_epi64(u[i + 0], u[i + 4]);
      t[i + 2] = _mm256_unpacklo_epi64(u[i + 1], u[i + 5]);
      t[i + 3] = _mm256_unpackhi_epi64(u[i + 1], u[i + 5]);
      t[i + 4] = _mm256_unpacklo_epi64(u[i + 2], u[i + 6]);
      t[i + 5] = _mm256_unpackhi_epi64(u[i + 2], u[i + 6]);
      t[i + 6] = _mm256_unpacklo_epi64(u[i + 3], u[i + 7]);
      t[i + 7] = _mm256_unpackhi_epi64(u[i + 3], u[i + 7]);
    }

    // ── Stage 4: 128-bit cross-lane permute ────────────────────
    // Merges the two 8-row halves into complete 16-element columns.
    for (int i = 0; i < 8; i++) {
      r[i]     = _mm256_permute2x128_si256(t[i], t[i + 8], 0x20);
      r[i + 8] = _mm256_permute2x128_si256(t[i], t[i + 8], 0x31);
    }

    // ── Store ──────────────────────────────────────────────────
    for (int i = 0; i < 16; i++)
      _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(&out[4 * i]), r[i]);

    return Tensor(out, {shape[1], shape[0], shape[2]});
  }


  
  Tensor nf() const {
    Tensor t;
    int comp = (shape[0] > shape[1])<<2 | (shape[1] > shape[2])<<1 | (shape[0] > shape[2]);
    switch (comp) {
      case 0: t = *this; break;
      case 2: t = transpose_BC(); break;
      case 3: t = transpose_BC().transpose_AB(); break;
      case 4: t = transpose_AB(); break;
      case 5: t = transpose_AB().transpose_BC(); break;
      case 7: t = transpose_AB().transpose_BC().transpose_AB(); break;
    }

    return t;
  }
};
