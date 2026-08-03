#pragma once

#include <array>
#include <cstdint>

using U8 = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

using Matrix = std::array<U16, 16>;
using Term = std::array<U16, 3>;
using Shape = std::array<U8, 3>;
