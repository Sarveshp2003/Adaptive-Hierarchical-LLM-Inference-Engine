#ifndef GPU_CALLBACK_H
#define GPU_CALLBACK_H

#include <cstdint>
#include <string>
#include <functional>

/**
 * Callback interface for GPU operations to report status back to Java.
 * 
 * This enables Java to:
 * - Monitor GPU progress
 * - Implement cancellation
 * - Track memory usage
 * - Handle errors
 */

enum class GPUCallbackType {
    TENSOR_TRANSFER_START = 0,
    TENSOR_TRANSFER_COMPLETE = 1,
    LAYER_COMPUTE_START = 2,
    LAYER_COMPUTE_COMPLETE = 3,
    MEMORY_ALLOCATED = 4,
    MEMORY_FREED = 5,
    KV_CACHE_UPDATE = 6,
    GPU_ERROR = 7,
    ATTENTION_COMPUTED = 8,
};

struct GPUCallbackEvent {
    GPUCallbackType type;
    uint64_t timestamp;
    
    // Event-specific data
    struct {
        std::string tensorName;
        size_t bytesTransferred;
        int layerId;
        double computeTimeMs;
        size_t memoryAllocated;
        void* memoryPtr;
        int errorCode;
        std::string errorMessage;
    } data;
};

/**
 * Abstract callback interface
 */
class IGPUCallback {
public:
    virtual ~IGPUCallback() = default;

    /**
     * Called when GPU event occurs
     */
    virtual void onGPUEvent(const GPUCallbackEvent& event) = 0;

    /**
     * Check if operation should be cancelled
     */
    virtual bool shouldCancel() = 0;
};

/**
 * Default callback implementation (logging)
 */
class DefaultGPUCallback : public IGPUCallback {
public:
    void onGPUEvent(const GPUCallbackEvent& event) override;
    bool shouldCancel() override { return false; }
};

#endif
