#include "topdown_lb.h"
#include "support.h"
#include <iostream>
#include <chrono>

using namespace fgs;
using namespace proj;

Tensor construct_matmul_tensor(int a, int b, int c) {
    Tensor tensor{};
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < b; ++j) {
            for (int k = 0; k < c; ++k) {
                int x = b * i + j;
                int y = c * j + k;
                int z = a * k + i;
                toggle_tensor_bit(tensor, x, y, z);
            }
        }
    }
    return tensor;
}

void benchmark(int a, int b, int c, U8 target_lb) {
    std::cout << "=================================================\n";
    std::cout << "Benchmarking <" << a << "," << b << "," << c << "> for target lower bound " << (int)target_lb << "\n";
    std::cout << "=================================================\n";
    Tensor t = construct_matmul_tensor(a, b, c);
    
    auto start = std::chrono::high_resolution_clock::now();
    bool result = topdown_lb(t, ub(t, 100000), target_lb);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "\nRESULT: <" << a << "," << b << "," << c << "> Rank >= " << (int)target_lb 
              << " proven: " << (result ? "YES" : "NO") 
              << " (" << duration.count() << " ms)\n\n";
}

int main() {
    benchmark(2, 2, 2, 7);
    benchmark(2, 2, 3, 11);
    benchmark(3, 3, 3, 15);
    return 0;
}
