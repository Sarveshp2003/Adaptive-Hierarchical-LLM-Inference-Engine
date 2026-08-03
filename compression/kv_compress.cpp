#include "kv_compress.h"
#include <cstdint>
#include <cstring>

// Reference conversion adapted for prototyping. Not fully IEEE-perfect but sufficient for testing.
static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mant = x & 0x007FFFFF;
    int32_t exp = ((x >> 23) & 0xFF) - 127;

    if (exp > 15) {
        // overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7C00);
    } else if (exp > -15) {
        // normalized
        exp += 15;
        uint16_t half = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
        return half;
    } else {
        // underflow -> zero
        return static_cast<uint16_t>(sign);
    }
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = h & 0x03FFu;
    uint32_t x;
    if (exp == 0) {
        // zero / subnormal -> treat as zero for proto
        x = sign;
    } else if (exp == 0x1F) {
        // Inf/NaN
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        // normalized
        exp = exp - 15 + 127;
        x = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

std::vector<uint16_t> float32_to_float16(const std::vector<float> &in) {
    std::vector<uint16_t> out;
    out.reserve(in.size());
    for (float v : in) out.push_back(float_to_half(v));
    return out;
}

std::vector<float> float16_to_float32(const std::vector<uint16_t> &in) {
    std::vector<float> out;
    out.reserve(in.size());
    for (uint16_t h : in) out.push_back(half_to_float(h));
    return out;
}
