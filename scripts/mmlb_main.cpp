#include <iostream>
#include <string>
#include <cstdlib>
#include "../src/tensor.h"
#include "../src/forced_product.h"
#include "../src/ub.h"
#include "../src/flatten.h"
#include "../src/topdown_lb.h"

int oldrank = 0;
std::string filename = "";
int correctness_check = 0;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <n> <m> <p> <target_lb>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);
    int p = std::atoi(argv[3]);
    int target_lb = std::atoi(argv[4]);

    if (n <= 0 || m <= 0 || p <= 0 || target_lb <= 0) {
        std::cerr << "All arguments must be strictly positive integers." << std::endl;
        return 1;
    }

    U8 sa = n * m;
    U8 sb = m * p;
    U8 sc = n * p;

    if (sa > 16 || sb > 16 || sc > 16) {
        std::cerr << "Tensor dimensions " << sa << "x" << sb << "x" << sc 
                  << " exceed maximum supported shape (16x16x16)." << std::endl;
        return 1;
    }
    
    load_table("third_party/rank_table_3x3x3_f2.raw", rank_table);
    clear_cache();
    Shape shape = {sa, sb, sc};
    Tensor T(std::array<U64, 64>{}, shape);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                int a_idx = i * m + k;
                int b_idx = k * p + j;
                int c_idx = i * p + j;
                T.set_bit(a_idx, b_idx, c_idx, true);
            }
        }
    }

    std::cout << "=== Running test for <" << n << "," << m << "," << p << "> (target >= " << target_lb << ") ===\n";
    
    int conj_rank = ub(T);
    std::cout << "Initial upper bound (conj_rank): " << conj_rank << "\n";
    
    bool lb = topdown_lb(T, target_lb, conj_rank);
    
    std::cout << "Rank >= " << target_lb << ": " << (lb ? "True" : "False") << std::endl;

    return 0;
}
