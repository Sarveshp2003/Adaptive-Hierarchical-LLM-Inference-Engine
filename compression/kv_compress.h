#pragma once

#include <vector>
#include <cstdint>

// Simple float32 <-> float16 conversion utilities for proto KV compression
std::vector<uint16_t> float32_to_float16(const std::vector<float> &in);
std::vector<float> float16_to_float32(const std::vector<uint16_t> &in);
