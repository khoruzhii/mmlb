# Select a build configuration from config/*.mk, e.g.:
#   make CONFIG=default
#   make CONFIG=clang-flto
CONFIG ?= default
CONFIG_FILE = config/$(CONFIG).mk

ifeq ($(wildcard $(CONFIG_FILE)),)
$(error Unknown CONFIG '$(CONFIG)': no such file '$(CONFIG_FILE)')
endif

include $(CONFIG_FILE)

SRC = scripts/mmlb_main.cpp third_party/updated_code/tensor.cpp third_party/updated_code/pairSet.cpp third_party/updated_code/mm.cpp
DEPS = third_party/updated_code/tensor.hpp third_party/updated_code/pairSet.hpp third_party/updated_code/mm.hpp src/tensor.h src/flatten.h src/forced_product.h src/ub.h src/rank_table.h src/topdown_lb.h
OUT = mmlb

all: $(OUT)

$(OUT): $(DEPS)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT)
