#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "types.h"


inline constexpr U64 kEntries = (1ull << 27);

inline U32 encode_bit_index(int i, int j, int k) {
    return static_cast<uint32_t>(i * 9 + j * 3 + k);
}

inline bool load_table(const std::string& path, std::vector<U8>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::perror("fopen");
        return false;
    }

    out.assign(kEntries, 0);
    size_t read = std::fread(out.data(), 1, static_cast<size_t>(kEntries), f);

    std::fclose(f);

    if (read != static_cast<size_t>(kEntries)) {
        std::fprintf(stderr, "Short read: got %zu, expected %llu\n",
                     read, static_cast<unsigned long long>(kEntries));
        out.clear();
        return false;
    }
    return true;
}

inline U8 rank_table_lookup(const Tensor &t, const std::vector<U8>& table) {
    if (std::max({t.shape[0],t.shape[1],t.shape[2]}) > 3)
        return 0;
    U32 lookup = 0;
    for (auto i = 0; i < 3; i++) {
      for (auto j = 0; j < 3; j++) {
        lookup |= U64((t.get_row(i,j)&U16(7)))<<(i * 9 + j * 3);
      }
    }
    return table[lookup];
}
