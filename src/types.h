#pragma once

#include <array>
#include <cstdint>

using U8 = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

using Tensor = std::array<U64, 12>;
using Matrix = std::array<U16, 9>;
using Term = std::array<U16, 3>;
using Shape = std::array<U8, 3>;
