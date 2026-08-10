# clang + LTO, generic x86-64-v3 (AVX2) target.
CXX = clang++
CXXFLAGS = -I. -O3 -mavx2 -std=c++20 -flto
LDFLAGS = -flto
