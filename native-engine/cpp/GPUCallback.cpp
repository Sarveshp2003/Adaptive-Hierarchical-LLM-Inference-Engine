#include "GPUCallback.h"
#include "Logger.h"

void DefaultGPUCallback::onGPUEvent(const GPUCallbackEvent& event) {
    switch (event.type) {
        case GPUCallbackType::TENSOR_TRANSFER_START:
            LOG_INFO("GPU Transfer START: " + event.data.tensorName);
            break;

        case GPUCallbackType::TENSOR_TRANSFER_COMPLETE:
            LOG_INFO("GPU Transfer COMPLETE: " + event.data.tensorName 
                    + " bytes=" + std::to_string(event.data.bytesTransferred));
            break;

        case GPUCallbackType::LAYER_COMPUTE_START:
            LOG_INFO("Layer COMPUTE START: layer=" + std::to_string(event.data.layerId));
            break;

        case GPUCallbackType::LAYER_COMPUTE_COMPLETE:
            LOG_INFO("Layer COMPUTE COMPLETE: layer=" + std::to_string(event.data.layerId)
                    + " time=" + std::to_string(event.data.computeTimeMs) + "ms");
            break;

        case GPUCallbackType::MEMORY_ALLOCATED:
            LOG_INFO("GPU MEMORY ALLOCATED: " 
                    + std::to_string(event.data.memoryAllocated / (1024*1024)) + "MB");
            break;

        case GPUCallbackType::MEMORY_FREED:
            LOG_INFO("GPU MEMORY FREED: " 
                    + std::to_string(event.data.memoryAllocated / (1024*1024)) + "MB");
            break;

        case GPUCallbackType::KV_CACHE_UPDATE:
            LOG_INFO("KV Cache UPDATE");
            break;

        case GPUCallbackType::GPU_ERROR:
            LOG_ERROR("GPU ERROR: " + event.data.errorMessage 
                     + " code=" + std::to_string(event.data.errorCode));
            break;

        case GPUCallbackType::ATTENTION_COMPUTED:
            LOG_INFO("Attention COMPUTED");
            break;

        default:
            LOG_WARN("Unknown GPU event type");
            break;
    }
}
