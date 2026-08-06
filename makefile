CXX = g++
CXXFLAGS = -I. -O3 -mavx2 -std=c++20
SRC = scripts/mmlb_main.cpp third_party/updated_code/tensor.cpp third_party/updated_code/pairSet.cpp third_party/updated_code/mm.cpp
OUT = mmlb

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC)

clean:
	rm -f $(OUT)
