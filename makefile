CXX = g++
CXXFLAGS = -I. -O3 -mavx2 -std=c++20
SRC = scripts/mmlb_main.cpp third_party/updated_code/tensor.cpp third_party/updated_code/pairSet.cpp third_party/updated_code/mm.cpp
DEPS = third_party/updated_code/tensor.hpp third_party/updated_code/pairSet.hpp third_party/updated_code/mm.hpp src/tensor.h src/flatten.h src/forced_product.h src/ub.h src/topdown_lb.h
OUT = mmlb

all: $(OUT)

$(OUT): $(DEPS)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC)

clean:
	rm -f $(OUT)
