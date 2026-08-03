#include "KVCompressionEngine.h"
#include "Logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>

KVCompressionEngine::KVCompressionEngine()
    : strategy(CompressionType::FP16),
      enableAdaptive(false),
      compressionThreshold(100.0f) {
    LOG_INFO("KVCompressionEngine initialized");
}

KVCompressionEngine::~KVCompressionEngine() {
}

void KVCompressionEngine::setCompressionType(CompressionType type) {
    strategy = type;
    const char* typeStr;
    switch (type) {
        case CompressionType::NONE: typeStr = "NONE"; break;
        case CompressionType::FP16: typeStr = "FP16"; break;
        case CompressionType::INT8: typeStr = "INT8"; break;
        case CompressionType::NF4: typeStr = "NF4"; break;
        default: typeStr = "UNKNOWN"; break;
    }
    LOG_INFO("KV compression strategy set to: " + std::string(typeStr));
}

void KVCompressionEngine::enableAdaptiveCompression(float tokenAgeThreshold) {
    enableAdaptive = true;
    compressionThreshold = tokenAgeThreshold;
    LOG_INFO("Adaptive compression enabled, threshold=" + std::to_string(tokenAgeThreshold));
}

void KVCompressionEngine::compressToFP16(const float* input, uint16_t* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        float val = input[i];
        
        // Simplified FP32→FP16 conversion
        // In production, use hardware instruction or library like half.hpp
        uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
        uint16_t sign = (bits >> 31) & 0x1;
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

        uint16_t fp16;
        if (exponent == 0xFF) {
            // Infinity or NaN
            fp16 = (sign << 15) | 0x7C00;
        } else if (exponent == 0) {
            // Zero
            fp16 = (sign << 15);
        } else {
            // Normal number: convert exponent and mantissa
            uint16_t fp16_exp = ((int32_t)exponent - 127 + 15) & 0x1F;
            uint16_t fp16_mantissa = (mantissa >> 13) & 0x3FF;
            fp16 = (sign << 15) | (fp16_exp << 10) | fp16_mantissa;
        }
        
        output[i] = fp16;
    }
}

void KVCompressionEngine::decompressFP16(const uint16_t* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint16_t fp16 = input[i];
        
        uint16_t sign = (fp16 >> 15) & 0x1;
        uint16_t exponent = (fp16 >> 10) & 0x1F;
        uint16_t mantissa = fp16 & 0x3FF;

        uint32_t fp32_sign = sign << 31;
        uint32_t fp32_exponent = ((exponent + 127 - 15) & 0xFF) << 23;
        uint32_t fp32_mantissa = mantissa << 13;

        uint32_t bits = fp32_sign | fp32_exponent | fp32_mantissa;
        float val = *reinterpret_cast<float*>(&bits);
        output[i] = val;
    }
}

void KVCompressionEngine::compressToINT8(const float* input, int8_t* output,
                                        size_t count, QuantizationStats& stats) {
    // Find min and max
    float minVal = input[0];
    float maxVal = input[0];
    
    for (size_t i = 1; i < count; i++) {
        minVal = std::min(minVal, input[i]);
        maxVal = std::max(maxVal, input[i]);
    }

    // Compute scale
    stats.minVal = minVal;
    stats.maxVal = maxVal;
    stats.scale = (maxVal - minVal) / 255.0f;
    stats.zeroPoint = minVal;
    stats.type = CompressionType::INT8;

    if (stats.scale == 0.0f) stats.scale = 1.0f;

    // Quantize
    for (size_t i = 0; i < count; i++) {
        float normalized = (input[i] - minVal) / stats.scale;
        int32_t q = (int32_t)std::round(normalized);
                q = std::clamp(q, 0, 255);
                output[i] = static_cast<int8_t>(q - 128);
    }

    LOG_DEBUG("INT8 compression: range=[" + std::to_string(minVal) + ", " 
              + std::to_string(maxVal) + "], scale=" + std::to_string(stats.scale));
}

void KVCompressionEngine::decompressINT8(const int8_t* input, float* output,
                                        size_t count, const QuantizationStats& stats) {
    for (size_t i = 0; i < count; i++) {
        int32_t quantized = input[i] + 128;
        float val = stats.zeroPoint + quantized * stats.scale;
        output[i] = val;
    }
}

void KVCompressionEngine::compressToNF4(const float* input, uint8_t* output,
                                       size_t count, QuantizationStats& stats) {
    // NF4: Normalized 4-bit Float
    // Maps 16 values evenly across [-1, 1] range
    // Then scales to input range

    float minVal = input[0];
    float maxVal = input[0];
    
    for (size_t i = 1; i < count; i++) {
        minVal = std::min(minVal, input[i]);
        maxVal = std::max(maxVal, input[i]);
    }

    stats.minVal = minVal;
    stats.maxVal = maxVal;
    stats.scale = (maxVal - minVal) / 15.0f;
    stats.zeroPoint = minVal;
    stats.type = CompressionType::NF4;

    if (stats.scale == 0.0f) stats.scale = 1.0f;

    // Quantize to 4-bit (2 values per byte)
    for (size_t i = 0; i < count; i += 2) {
        float normalized1 = (input[i] - minVal) / stats.scale;
        int q1 = (int)std::round(normalized1);
                q1 = std::clamp(q1, 0, 15);
                uint8_t quantized1 = static_cast<uint8_t>(q1);

                uint8_t quantized2 = 0;
                if (i + 1 < count) {
                    float normalized2 = (input[i + 1] - minVal) / stats.scale;
                    int q2 = (int)std::round(normalized2);
                    q2 = std::clamp(q2, 0, 15);
                    quantized2 = static_cast<uint8_t>(q2);
                }

        output[i / 2] = (quantized1 << 4) | quantized2;
    }

    LOG_DEBUG("NF4 compression: range=[" + std::to_string(minVal) + ", " 
              + std::to_string(maxVal) + "]");
}

void KVCompressionEngine::decompressNF4(const uint8_t* input, float* output,
                                       size_t count, const QuantizationStats& stats) {
    for (size_t i = 0; i < count; i += 2) {
        uint8_t packed = input[i / 2];
        uint8_t quantized1 = (packed >> 4) & 0xF;
        uint8_t quantized2 = packed & 0xF;

        float val1 = stats.zeroPoint + quantized1 * stats.scale;
        output[i] = val1;

        if (i + 1 < count) {
            float val2 = stats.zeroPoint + quantized2 * stats.scale;
            output[i + 1] = val2;
        }
    }
}

float KVCompressionEngine::compress(const float* input, void* output, size_t count,
                                   QuantizationStats& stats) {
    stats.type = strategy;

    switch (strategy) {
        case CompressionType::NONE:
            std::memcpy(output, input, count * sizeof(float));
            return 1.0f;

        case CompressionType::FP16:
            compressToFP16(input, (uint16_t*)output, count);
            return 0.5f;

        case CompressionType::INT8:
            compressToINT8(input, (int8_t*)output, count, stats);
            return 0.25f;

        case CompressionType::NF4:
            compressToNF4(input, (uint8_t*)output, count, stats);
            return 0.125f;

        default:
            return 1.0f;
    }
}

void KVCompressionEngine::decompress(const void* input, float* output, size_t count,
                                    const QuantizationStats& stats) {
    switch (stats.type) {
        case CompressionType::NONE:
            std::memcpy(output, input, count * sizeof(float));
            break;

        case CompressionType::FP16:
            decompressFP16((const uint16_t*)input, output, count);
            break;

        case CompressionType::INT8:
            decompressINT8((const int8_t*)input, output, count, stats);
            break;

        case CompressionType::NF4:
            decompressNF4((const uint8_t*)input, output, count, stats);
            break;

        default:
            break;
    }
}

size_t KVCompressionEngine::getCompressedSize(CompressionType type, size_t elementCount) const {
    switch (type) {
        case CompressionType::NONE:
            return elementCount * sizeof(float);
        case CompressionType::FP16:
            return elementCount * sizeof(uint16_t);
        case CompressionType::INT8:
            return elementCount * sizeof(int8_t);
        case CompressionType::NF4:
            return (elementCount + 1) / 2;  // 2 values per byte
        default:
            return elementCount * sizeof(float);
    }
}

float KVCompressionEngine::getCompressionRatio(CompressionType type) const {
    switch (type) {
        case CompressionType::NONE:
            return 1.0f;
        case CompressionType::FP16:
            return 0.5f;
        case CompressionType::INT8:
            return 0.25f;
        case CompressionType::NF4:
            return 0.125f;
        default:
            return 1.0f;
    }
}

size_t KVCompressionEngine::estimateSavings(size_t uncompressedBytes, CompressionType type) const {
    size_t compressedSize = uncompressedBytes * getCompressionRatio(type);
    return uncompressedBytes - compressedSize;
}
