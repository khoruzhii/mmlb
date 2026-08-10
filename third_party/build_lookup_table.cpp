#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static inline uint32_t encode_index(int i, int j, int k) {
    return static_cast<uint32_t>(i * 9 + j * 3 + k);
}

static std::vector<uint32_t> build_rank1_tensors() {
    std::vector<uint32_t> rank1;
    rank1.reserve(7 * 7 * 7);

    for (int a = 1; a < 8; a++) {
        for (int b = 1; b < 8; b++) {
            for (int c = 1; c < 8; c++) {
                uint32_t t = 0u;
                for (int i = 0; i < 3; i++) {
                    if (((a >> i) & 1) == 0) continue;
                    for (int j = 0; j < 3; j++) {
                        if (((b >> j) & 1) == 0) continue;
                        for (int k = 0; k < 3; k++) {
                            if (((c >> k) & 1) == 0) continue;
                            t |= (1u << encode_index(i, j, k));
                        }
                    }
                }
                rank1.push_back(t);
            }
        }
    }
    return rank1;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "rank_table_3x3x3_f2.raw";

    const uint64_t N = (1ull << 27); // 128M entries
    std::vector<uint8_t> rank_table;
    try {
        rank_table.assign(N, 0xFF); // 255 means "unreached"
    } catch (...) {
        std::fprintf(stderr, "Failed to allocate %zu bytes for rank_table\n", size_t(N));
        return 1;
    }


    auto rank1 = build_rank1_tensors();
    std::printf("Distinct rank-1 tensors: %zu\n", rank1.size()); // expect 343


    std::vector<uint32_t> current, next;
    current.reserve(1 << 20);
    next.reserve(1 << 20);

    rank_table[0] = 0; 
    current.push_back(0);

    int r = 0;
    uint64_t visited = 1;

    while (!current.empty()) {
        std::printf("Rank %d: %zu tensors (visited: %llu / %llu)\n",
                    r, current.size(),
                    static_cast<unsigned long long>(visited),
                    static_cast<unsigned long long>(N));

        next.clear();
        next.reserve(current.size() * 2);

        for (uint32_t T : current) {
            for (uint32_t v : rank1) {
                uint32_t S = T ^ v;
                if (rank_table[S] == 0xFF) {
                    rank_table[S] = static_cast<uint8_t>(r + 1);
                    next.push_back(S);
                }
            }
        }
        visited += next.size();
        current.swap(next);
        r++;
    }

    int max_rank = 0;
    std::array<uint64_t, 32> hist{};
    for (uint64_t i = 0; i < N; i++) {
        uint8_t rr = rank_table[i];
        if (rr == 0xFF) {
            std::fprintf(stderr, "WARNING: Unreached index: %llu\n",
                         static_cast<unsigned long long>(i));
            continue;
        }
        hist[rr]++;
        if (rr > max_rank) max_rank = rr;
    }

    std::printf("Max rank: %d\n", max_rank);
    for (int i = 0; i <= max_rank; i++) {
        std::printf("  Rank %d: %llu\n", i,
                    static_cast<unsigned long long>(hist[i]));
    }


    std::FILE* f = std::fopen(out_path, "wb");
    if (!f) {
        std::perror("fopen");
        return 1;
    }
    size_t wrote = std::fwrite(rank_table.data(), 1, size_t(N), f);
    if (wrote != size_t(N)) {
        std::perror("fwrite data");
        std::fclose(f);
        return 1;
    }
    std::fclose(f);
    std::printf("Wrote %s (%llu bytes)\n",
                out_path, static_cast<unsigned long long>(N));
    return 0;
}