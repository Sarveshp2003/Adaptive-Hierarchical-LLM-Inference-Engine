#ifndef KV_COMPRESSION_ENGINE_H
#define KV_COMPRESSION_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

/**
 * KV cache compression engine.
 * 
 * Reduces KV cache memory by quantizing to lower precision:
 * - FP32 → FP16 (50% compression)
 * - FP32 → INT8 (75% compression)
 * - FP32 → NF4 (87.5% compression)
 */

enum class CompressionType {
    NONE = 0,      // No compression (FP32)
    FP16 = 1,      // 16-bit float (50% reduction)
    INT8 = 2,      // 8-bit integer (75% reduction)
    NF4 = 3,       // 4-bit normalized float (87.5% reduction)
};

/**
 * Quantization statistics for inverse transformation
 */
struct QuantizationStats {
    float minVal;
    float maxVal;
    float scale;
    float zeroPoint;
    CompressionType type;
};

class KVCompressionEngine {
private:
    CompressionType strategy;
    bool enableAdaptive;
    float compressionThreshold;  // Token age threshold for compression

    /**
     * FP32 → FP16 compression
     */
    void compressToFP16(const float* input, uint16_t* output, size_t count);

    /**
     * FP32 → INT8 compression with quantization
     */
    void compressToINT8(const float* input, int8_t* output, 
                       size_t count, QuantizationStats& stats);

    /**
     * FP32 → NF4 compression (normalized 4-bit)
     */
    void compressToNF4(const float* input, uint8_t* output,
                      size_t count, QuantizationStats& stats);

    /**
     * Decompress FP16 → FP32
     */
    void decompressFP16(const uint16_t* input, float* output, size_t count);

    /**
     * Decompress INT8 → FP32
     */
    void decompressINT8(const int8_t* input, float* output,
                       size_t count, const QuantizationStats& stats);

    /**
     * Decompress NF4 → FP32
     */
    void decompressNF4(const uint8_t* input, float* output,
                      size_t count, const QuantizationStats& stats);

public:
    KVCompressionEngine();
    ~KVCompressionEngine();

    /**
     * Set compression strategy
     */
    void setCompressionType(CompressionType type);

    /**
     * Enable adaptive compression (choose based on age)
     */
    void enableAdaptiveCompression(float tokenAgeThreshold);

    /**
     * Compress KV cache data
     * 
     * @param input Input KV data (FP32)
     * @param output Compressed output
     * @param count Number of elements
     * @param stats Statistics for decompression
     * @return Compression ratio (output_size / input_size)
     */
    float compress(const float* input, void* output, size_t count,
                  QuantizationStats& stats);

    /**
     * Decompress KV cache data
     * 
     * @param input Compressed data
     * @param output Output KV data (FP32)
     * @param count Number of elements
     * @param stats Quantization statistics
     */
    void decompress(const void* input, float* output, size_t count,
                   const QuantizationStats& stats);

    /**
     * Get compressed size for given type and count
     */
    size_t getCompressedSize(CompressionType type, size_t elementCount) const;

    /**
     * Get uncompressed size
     */
    size_t getUncompressedSize(size_t elementCount) const {
        return elementCount * sizeof(float);
    }

    /**
     * Calculate compression ratio
     */
    float getCompressionRatio(CompressionType type) const;

    /**
     * Get current strategy
     */
    CompressionType getStrategy() const { return strategy; }

    /**
     * Estimate memory savings
     */
    size_t estimateSavings(size_t uncompressedBytes, CompressionType type) const;
};

#endif
