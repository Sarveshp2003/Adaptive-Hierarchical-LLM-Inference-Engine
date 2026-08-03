#include "NativeEngine.h"

#include <iostream>
#include <stdexcept>
#include <chrono>

#include <cuda_runtime.h>

#include "CUDAStream.h"
#include "RuntimeMemory.h"
#include "GGUFLoader.h"
#include "GPUCallback.h"
#include "KVCacheGPU.h"
#include "Logger.h"



NativeEngine::NativeEngine()

:
initialized(false),
modelLoader(std::make_shared<MMapWeightLoader>())

{

}




NativeEngine::~NativeEngine()

{

    shutdown();

}






void NativeEngine::initialize(

    int gpuDevice,

    size_t gpuMemoryBytes

)

{

    std::cout
        << "Initializing GPU "
        << gpuDevice
        << std::endl;



    cudaError_t error =
        cudaSetDevice(gpuDevice);



    if(error != cudaSuccess)

    {

        throw std::runtime_error(

            "CUDA device initialization failed"

        );

    }

    // Install a top-level unhandled exception filter to write minidumps on native crashes
    GPUMemoryPool::installUnhandledExceptionFilter();



    RuntimeMemory::initializeGPU(
        gpuMemoryBytes
    );



    CUDAStream::initialize();

    this->gpuDevice = gpuDevice;
    this->gpuCallback = std::make_unique<DefaultGPUCallback>();
    this->kvCacheGPU = std::make_unique<KVCacheGPU>(
        gpuMemoryBytes / 2,  // Half GPU memory for KV cache
        128 * 1024           // 128KB pages
    );

    LOG_INFO("Native Engine initialized with GGUF, GPU callbacks, and KV cache");

        // Start background cleanup thread for pending pinned buffers
    cleanupRunning = true;
    cleanupThread = std::thread([this]() {
        int counter = 0;
        while(cleanupRunning) {
            pollCleanupPendingCopies();
            // Periodically snapshot allocator state for diagnostics (increased frequency ~100ms)
            if(++counter % 10 == 0) {
                try {
                    char ts[64]; auto now = std::chrono::system_clock::now(); std::time_t t = std::chrono::system_clock::to_time_t(now); tm localtm; localtime_s(&localtm, &t); sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", localtm.tm_year+1900, localtm.tm_mon+1, localtm.tm_mday, localtm.tm_hour, localtm.tm_min, localtm.tm_sec);
                    std::string statePath = "out/allocator_state_periodic_" + std::to_string((long long)counter) + "_" + ts + ".json";
                    RuntimeMemory::serializePoolState(statePath);
                } catch(...) { LOG_WARN_STREAM("Periodic serializePoolState failed"); }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    initialized = true;


    std::cout
        << "Native Engine initialized"
        << std::endl;


}








void NativeEngine::addLayer(

    int hiddenSize

)

{

    if(!initialized)

    {

        throw std::runtime_error(

            "Engine not initialized"

        );

    }



    layers.push_back(

        std::make_unique<TransformerLayer>(

            hiddenSize

        )

    );



    std::cout
        << "Added Transformer Layer "
        << layers.size() - 1
        << std::endl;


}









void NativeEngine::executeLayer(

    int index,

    Tensor& input,

    Tensor& output

)

{

    if(!initialized)

    {

        throw std::runtime_error(

            "Engine not initialized"

        );

    }



    if(index < 0 ||
       index >= static_cast<int>(layers.size()))

    {

        throw std::runtime_error(

            "Invalid transformer layer index"

        );

    }



    std::cout
        << "Executing Transformer Layer "
        << index
        << std::endl;



    layers[index]->forward(

        input,

        output

    );

    // After executing the layer, snapshot allocator state for diagnostics
    try {
        char ts[64]; auto now = std::chrono::system_clock::now(); std::time_t t = std::chrono::system_clock::to_time_t(now); tm localtm; localtime_s(&localtm, &t); sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", localtm.tm_year+1900, localtm.tm_mon+1, localtm.tm_mday, localtm.tm_hour, localtm.tm_min, localtm.tm_sec);
        std::string statePath = "out/allocator_state_layer_" + std::to_string(index) + "_" + ts + ".json";
        RuntimeMemory::serializePoolState(statePath);
    } catch(...) {
        LOG_WARN_STREAM("serializePoolState failed for layer " << index);
    }

}










bool NativeEngine::loadModel(const std::string& path)
{
   if(!modelLoader)
   {
       modelLoader = std::make_shared<MMapWeightLoader>();
   }

   return modelLoader->open(path);
}

bool NativeEngine::loadLayer(int layerId)
{
   if(!modelLoader)
   {
       return false;
   }

   return modelLoader->loadLayer(layerId);
}

void NativeEngine::releaseLayer(int layerId)
{
   if(modelLoader)
   {
       modelLoader->releaseLayer(layerId);
   }
}

void NativeEngine::shutdown()
  
{
  
    if(!initialized)
  
        return;
  
 
  
 
    // Stop background cleanup thread and attempt to cleanup any pending pinned buffers
    cleanupRunning = false;
    if(cleanupThread.joinable()) {
        cleanupThread.join();
    }
    // Drain remaining pending entries once more
    pollCleanupPendingCopies();

    // Ensure all CUDA work is finished before shutdown
    CUDAStream::synchronize();

    // After sync, free any remaining pending pinned buffers and destroy events
    {
        std::lock_guard<std::mutex> lock(pendingMutex);
        for(auto &pc : pendingCopies) {
            if(pc.event) cudaEventDestroy(pc.event);
            if(pc.pinnedPtr) RuntimeMemory::releasePinnedHost(pc.pinnedPtr);
        }
        pendingCopies.clear();
    }

    layers.clear();
    
    if (kvCacheGPU) {
       kvCacheGPU->clear();
       kvCacheGPU.reset();
    }
    
    ggufLoader.reset();
    gpuCallback.reset();


    CUDAStream::shutdown();



    RuntimeMemory::shutdown();



    initialized = false;
 
 
 
 
    std::cout
       << "Native Engine shutdown"
       << std::endl;
 
 

}

// ============ GGUF Model Loading ============

bool NativeEngine::loadModelGGUF(const std::string& path) {
    if (!initialized) {
       LOG_ERROR("Engine not initialized");
       return false;
    }

    ggufLoader = std::make_unique<GGUFLoader>();
    
    if (ggufLoader->loadMetadata(path)) {
       // Extract model configuration
       numLayers = ggufLoader->getNumLayers();
       hiddenDim = ggufLoader->getHiddenDim();
       
       LOG_INFO("GGUF Model loaded: layers=" + std::to_string(numLayers) + 
               " hiddenDim=" + std::to_string(hiddenDim));
       
       // Report callback
       GPUCallbackEvent event;
       event.type = GPUCallbackType::TENSOR_TRANSFER_COMPLETE;
       event.data.tensorName = "Model metadata";
       event.data.bytesTransferred = ggufLoader->getTotalModelSize();
       reportCallback(event);

       return true;
    }

    // clear ggufLoader instance since we'll try the fallback reader
    ggufLoader.reset();

    // Fallback: try using the repo gguf reader which handles more GGUF variants
    try {
        loader::ModelArchive arch = loader::GGUFReader::load(path);
        archCache = std::make_unique<loader::ModelArchive>(std::move(arch));
        // derive simple metadata by parsing tensor names (look for 'blk.<N>' patterns)
        numLayers = 0; hiddenDim = 0;
        for(auto &lr : archCache->layers) {
            // attempt to find 'blk.' pattern used by Llama-style models
            std::string name = lr.name;
            size_t pos = name.find("blk.");
            if(pos != std::string::npos) {
                size_t start = pos + 4;
                size_t end = start;
                while(end < name.size() && isdigit((unsigned char)name[end])) ++end;
                if(end > start) {
                    try {
                        int lid = std::stoi(name.substr(start, end-start));
                        if(lid + 1 > numLayers) numLayers = lid + 1;
                        lr.layer_id = lid; // update record
                    } catch(...) {}
                }
            }
            if(!lr.shape.empty()) hiddenDim = std::max(hiddenDim, lr.shape.back());
        }
        if(numLayers == 0) {
            // fallback: if layer ids weren't found, try the generic layer_id values set by reader
            for(const auto &lr : archCache->layers) {
                if(lr.layer_id + 1 > numLayers) numLayers = lr.layer_id + 1;
            }
        }
        LOG_INFO("GGUF Model loaded via fallback reader: layers=" + std::to_string(numLayers) + " hiddenDim=" + std::to_string(hiddenDim));
        return true;
    } catch(const std::exception &e) {
        LOG_ERROR(std::string("Failed to load GGUF metadata: ") + e.what());
        ggufLoader.reset();
        archCache.reset();
        return false;
    }
}

bool NativeEngine::loadLayerFromGGUF(int layerId) {
    if (!ggufLoader && !archCache) {
       LOG_ERROR("GGUF loader not initialized and no fallback archive available");
       return false;
    }

    if (layerId < 0 || layerId >= numLayers) {
       LOG_ERROR("Invalid layer ID: " + std::to_string(layerId));
       return false;
    }

    // Construct tensor names for this layer (llama.attention, llama.feed_forward, etc)
    std::string layerPrefix = "blk." + std::to_string(layerId) + ".";
    
    std::vector<std::string> requiredTensors = {
       layerPrefix + "attn_q.weight",
       layerPrefix + "attn_k.weight",
       layerPrefix + "attn_v.weight",
       layerPrefix + "attn_output.weight",
       layerPrefix + "ffn_gate.weight",
       layerPrefix + "ffn_up.weight",
       layerPrefix + "ffn_down.weight",
    };

    // Load each tensor
    if (ggufLoader) {
        for (const auto& tensorName : requiredTensors) {
           if (!ggufLoader->hasTensor(tensorName)) {
               LOG_WARN("Tensor not found: " + tensorName);
               continue;
           }

           const GGUFTensor* tensor = ggufLoader->getTensor(tensorName);
           if (!tensor) continue;

           size_t bytes = tensor->size;
           if(bytes == 0) {
               LOG_WARN_STREAM("Tensor has zero size: " << tensorName);
               continue;
           }

           // Allocate pinned host buffer for streaming
           LOG_INFO_STREAM("Preparing to stream tensor: " << tensorName << " bytes=" << bytes << " loadedTensors=" << loadedTensors.size());
           void* pinned = RuntimeMemory::allocatePinnedHost(bytes);
           if(!pinned) {
               LOG_ERROR_STREAM("Failed to allocate pinned host memory for tensor: " << tensorName);
               return false;
           }
           LOG_INFO_STREAM("Pinned host buffer allocated for " << tensorName << " at " << pinned);

           // Stream tensor data from GGUF into pinned host buffer
           if(!ggufLoader->streamTensorData(tensorName, pinned, bytes)) {
               LOG_ERROR_STREAM("Failed to stream tensor data: " << tensorName);
               RuntimeMemory::releasePinnedHost(pinned);
               return false;
           }

           // Allocate GPU memory and async copy from pinned host -> device
           void* devPtr = RuntimeMemory::allocateGPU(bytes);
           if(!devPtr) {
               LOG_WARN_STREAM("GPU allocation failed for tensor: " << tensorName << " — attempting eviction and retry");
               // Try evicting KV cache pages first, then free previously loaded tensors and retry allocation a few times.
               bool allocated = false;
               for(int attempt=0; attempt<4 && !allocated; ++attempt) {
                   // Evict an LRU KV page if possible
                   if(kvCacheGPU && kvCacheGPU->evictLRUPage()) {
                       LOG_INFO_STREAM("Evicted one KV page to free GPU memory (attempt " << attempt << ")");
                   }
                   // Free one previously loaded tensor (attempt to free largest ones first by name heuristic)
                   // Attempt to free an entire oldest layer's tensors to make room.
                   int oldestLayer = INT_MAX;
                   for(const auto &kv : loadedTensors) {
                       const std::string &n = kv.first;
                       size_t p = n.find("blk.");
                       if(p != std::string::npos) {
                           size_t s = p + 4;
                           size_t e = s;
                           while(e < n.size() && isdigit((unsigned char)n[e])) ++e;
                           if(e > s) {
                               try {
                                   int lid = std::stoi(n.substr(s, e-s));
                                   if(lid < oldestLayer) oldestLayer = lid;
                               } catch(...) {}
                           }
                       }
                   }
                   if(oldestLayer != INT_MAX) {
                       LOG_INFO_STREAM("Releasing all tensors from oldest layer " << oldestLayer << " to free GPU memory");
                       // free matching entries
                       for(auto it = loadedTensors.begin(); it != loadedTensors.end(); ) {
                           if(it->first.find(std::string("blk.") + std::to_string(oldestLayer) + ".") != std::string::npos) {
                               RuntimeMemory::releaseGPU(it->second.ptr);
                               it = loadedTensors.erase(it);
                           } else ++it;
                       }
                       // Ask model loader to release layer backing if available
                       if(modelLoader) {
                           try { modelLoader->releaseLayer(oldestLayer); } catch(...) {}
                       }
                   } else {
                       // Fallback: free a single tensor if we couldn't detect layer ids
                       for(auto it = loadedTensors.begin(); it != loadedTensors.end(); ) {
                           if(it->first == tensorName) { ++it; continue; }
                           LOG_INFO_STREAM("Releasing previously loaded tensor to free GPU: " << it->first << " ptr=" << it->second.ptr);
                           RuntimeMemory::releaseGPU(it->second.ptr);
                           it = loadedTensors.erase(it);
                           break;
                       }
                   }
                   // small pause to allow allocator merging (no sync)
                   devPtr = RuntimeMemory::allocateGPU(bytes);
                   if(devPtr) allocated = true;
               }
               if(!devPtr) {
                   LOG_ERROR_STREAM("GPU allocation failed after eviction attempts for tensor: " << tensorName);
                   RuntimeMemory::releasePinnedHost(pinned);
                   return false;
               }
           }

           LOG_INFO_STREAM("cudaMemcpyAsync H2D for " << tensorName << " dst=" << devPtr << " bytes=" << bytes);
           cudaError_t err = cudaMemcpyAsync(devPtr, pinned, bytes, cudaMemcpyHostToDevice, CUDAStream::get());
           if(err != cudaSuccess) {
               LOG_ERROR_STREAM("cudaMemcpyAsync failed for tensor " << tensorName << ": " << cudaGetErrorString(err));
               RuntimeMemory::releaseGPU(devPtr);
               RuntimeMemory::releasePinnedHost(pinned);
               return false;
           } else {
               LOG_INFO_STREAM("cudaMemcpyAsync queued for " << tensorName << " dst=" << devPtr);
           }

           // Record pointer so it can be freed on engine shutdown or layer release
           loadedTensors[tensorName] = { devPtr, true };

           // Report transfer (bytes moved)
           GPUCallbackEvent event;
           event.type = GPUCallbackType::TENSOR_TRANSFER_COMPLETE;
           event.data.tensorName = tensorName;
           event.data.bytesTransferred = bytes;
           reportCallback(event);

           // Create an event to know when the async copy is complete, so pinned host buffer can be freed without synchronizing the whole stream.
           cudaEvent_t ev = nullptr;
           cudaError_t evErr = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
           if(evErr != cudaSuccess) {
               LOG_WARN_STREAM("cudaEventCreateWithFlags failed: " << cudaGetErrorString(evErr) << " — falling back to stream synchronize");
               cudaError_t syncErr = cudaStreamSynchronize(CUDAStream::get());
               if(syncErr != cudaSuccess) LOG_WARN_STREAM("cudaStreamSynchronize failed: " << cudaGetErrorString(syncErr));
               RuntimeMemory::releasePinnedHost(pinned);
           } else {
               // Record event on the stream; when the event is complete the copy finished.
               cudaError_t recErr = cudaEventRecord(ev, CUDAStream::get());
               if(recErr != cudaSuccess) {
                   LOG_WARN_STREAM("cudaEventRecord failed: " << cudaGetErrorString(recErr) << " — freeing pinned buffer now via sync");
                   cudaEventDestroy(ev);
                   cudaError_t syncErr = cudaStreamSynchronize(CUDAStream::get());
                   if(syncErr != cudaSuccess) LOG_WARN_STREAM("cudaStreamSynchronize failed: " << cudaGetErrorString(syncErr));
                   RuntimeMemory::releasePinnedHost(pinned);
               } else {
                   // Push to pending list for later cleanup
                   std::lock_guard<std::mutex> lock(pendingMutex);
                   pendingCopies.push_back({ev, pinned});
               }
           }
        }

        return true;
    }

    // Fallback: if we have an archCache produced by loader::GGUFReader, stream tensors from that archive
    if (archCache) {
        for (const auto &tensorName : requiredTensors) {
            // find record in archCache
            const loader::LayerRecord* found = nullptr;
            for (const auto &lr : archCache->layers) {
                if (lr.name == tensorName) { found = &lr; break; }
            }
            if (!found) {
                LOG_WARN("Tensor not found in archive: " + tensorName);
                continue;
            }

            size_t bytes = found->weights.size() * sizeof(float);
            if (bytes == 0) { LOG_WARN_STREAM("Tensor has zero size: " << tensorName); continue; }

            // Allocate pinned host buffer and copy floats into it
            void* pinned = RuntimeMemory::allocatePinnedHost(bytes);
            if (!pinned) { LOG_ERROR_STREAM("Failed to allocate pinned host memory for tensor: " << tensorName); return false; }
            std::memcpy(pinned, found->weights.data(), bytes);

            // Allocate GPU memory and async copy
            void* devPtr = RuntimeMemory::allocateGPU(bytes);
            if (!devPtr) {
                LOG_WARN_STREAM("GPU allocation failed for tensor: " << tensorName << " — attempting eviction and retry");
                bool allocated = false;
                for(int attempt=0; attempt<4 && !allocated; ++attempt) {
                    if(kvCacheGPU && kvCacheGPU->evictLRUPage()) {
                        LOG_INFO_STREAM("Evicted one KV page to free GPU memory (attempt " << attempt << ")");
                    }
                    // Attempt to free an entire oldest layer's tensors to make room.
                    int oldestLayer = INT_MAX;
                    for(const auto &kv : loadedTensors) {
                        const std::string &n = kv.first;
                        size_t p = n.find("blk.");
                        if(p != std::string::npos) {
                            size_t s = p + 4;
                            size_t e = s;
                            while(e < n.size() && isdigit((unsigned char)n[e])) ++e;
                            if(e > s) {
                                try {
                                    int lid = std::stoi(n.substr(s, e-s));
                                    if(lid < oldestLayer) oldestLayer = lid;
                                } catch(...) {}
                            }
                        }
                    }
                    if(oldestLayer != INT_MAX) {
                        LOG_INFO_STREAM("Releasing all tensors from oldest layer " << oldestLayer << " to free GPU memory");
                        // free matching entries
                        for(auto it = loadedTensors.begin(); it != loadedTensors.end(); ) {
                            if(it->first.find(std::string("blk.") + std::to_string(oldestLayer) + ".") != std::string::npos) {
                                RuntimeMemory::releaseGPU(it->second.ptr);
                                it = loadedTensors.erase(it);
                            } else ++it;
                        }
                        // Ask model loader to release layer backing if available
                        if(modelLoader) {
                            try { modelLoader->releaseLayer(oldestLayer); } catch(...) {}
                        }
                    } else {
                        // Fallback: free a single tensor if we couldn't detect layer ids
                        for(auto it = loadedTensors.begin(); it != loadedTensors.end(); ) {
                            if(it->first == tensorName) { ++it; continue; }
                            LOG_INFO_STREAM("Releasing previously loaded tensor to free GPU: " << it->first << " ptr=" << it->second.ptr);
                            RuntimeMemory::releaseGPU(it->second.ptr);
                            it = loadedTensors.erase(it);
                            break;
                        }
                    }
                    devPtr = RuntimeMemory::allocateGPU(bytes);
                    if(devPtr) allocated = true;
                }
                if(!devPtr) { LOG_ERROR_STREAM("GPU allocation failed after eviction attempts for tensor: " << tensorName); RuntimeMemory::releasePinnedHost(pinned); return false; }
            }
            // Alignment-aware copy: some GPUs/drivers return "misaligned address" for async copies.
            uintptr_t devAddr = reinterpret_cast<uintptr_t>(devPtr);
            uintptr_t hostAddr = reinterpret_cast<uintptr_t>(pinned);
            bool misaligned = ((devAddr % 256) != 0) || ((hostAddr % 256) != 0);
            cudaError_t err;
            if(misaligned) {
                LOG_WARN_STREAM("Misaligned pointer detected for tensor " << tensorName << " — using synchronous cudaMemcpy fallback");
                err = cudaMemcpy(devPtr, pinned, bytes, cudaMemcpyHostToDevice);
                if(err != cudaSuccess) {
                    LOG_ERROR_STREAM("cudaMemcpy (sync) failed for tensor " << tensorName << ": " << cudaGetErrorString(err)); RuntimeMemory::releaseGPU(devPtr); RuntimeMemory::releasePinnedHost(pinned); return false; }
            } else {
                err = cudaMemcpyAsync(devPtr, pinned, bytes, cudaMemcpyHostToDevice, CUDAStream::get());
                if (err != cudaSuccess) { LOG_ERROR_STREAM("cudaMemcpyAsync failed for tensor " << tensorName << ": " << cudaGetErrorString(err)); RuntimeMemory::releaseGPU(devPtr); RuntimeMemory::releasePinnedHost(pinned); return false; }
            }

            loadedTensors[tensorName] = { devPtr, true };

            GPUCallbackEvent event;
            event.type = GPUCallbackType::TENSOR_TRANSFER_COMPLETE;
            event.data.tensorName = tensorName;
            event.data.bytesTransferred = bytes;
            reportCallback(event);

            cudaEvent_t ev = nullptr;
            cudaError_t evErr = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (evErr != cudaSuccess) {
                LOG_WARN_STREAM("cudaEventCreateWithFlags failed: " << cudaGetErrorString(evErr) << " — falling back to stream synchronize");
                cudaError_t syncErr = cudaStreamSynchronize(CUDAStream::get());
                if (syncErr != cudaSuccess) LOG_WARN_STREAM("cudaStreamSynchronize failed: " << cudaGetErrorString(syncErr));
                RuntimeMemory::releasePinnedHost(pinned);
            } else {
                // Record event on the stream; when the event is complete the copy finished.
                cudaError_t recErr = cudaEventRecord(ev, CUDAStream::get());
                if (recErr != cudaSuccess) {
                    LOG_WARN_STREAM("cudaEventRecord failed: " << cudaGetErrorString(recErr) << " — freeing pinned buffer now via sync");
                    cudaEventDestroy(ev);
                    cudaError_t syncErr = cudaStreamSynchronize(CUDAStream::get());
                    if (syncErr != cudaSuccess) LOG_WARN_STREAM("cudaStreamSynchronize failed: " << cudaGetErrorString(syncErr));
                    RuntimeMemory::releasePinnedHost(pinned);
                } else {
                    LOG_INFO_STREAM("Push pending copy entry: tensor=" << tensorName << " pinned=" << pinned << " event=" << ev);
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    pendingCopies.push_back({ev, pinned});
                    // Inform GPU pool that this device pointer is referenced by an async operation
                    RuntimeMemory::attachEventToGPU(devPtr, ev);
                }
            }
        }
        return true;
    }

    LOG_ERROR("No GGUF source available to stream tensors");
    return false;
}

// Metadata getters (GGUF)
int NativeEngine::getNumLayers() const {
    if (ggufLoader) return ggufLoader->getNumLayers();
    return numLayers;
}

int NativeEngine::getHiddenDim() const {
    if (ggufLoader) return ggufLoader->getHiddenDim();
    return hiddenDim;
}

int NativeEngine::getNumHeads() const {
    if (ggufLoader) return ggufLoader->getNumHeads();
    return 0;
}

int NativeEngine::getVocabSize() const {
    if (ggufLoader) return ggufLoader->getVocabSize();
    return 0;
}

uint64_t NativeEngine::getTotalModelSize() const {
    if (ggufLoader) return ggufLoader->getTotalModelSize();
    return 0;
}

std::string NativeEngine::getModelName() const {
    if (ggufLoader) return ggufLoader->getMetadata("model_name");
    return std::string();
}

// ============ GPU Callbacks ============

void NativeEngine::pollCleanupPendingCopies() {
    std::lock_guard<std::mutex> lock(pendingMutex);
    auto it = pendingCopies.begin();
    while(it != pendingCopies.end()) {
        cudaError_t q = cudaEventQuery(it->event);
        if(q == cudaSuccess) {
            // Copy finished — free pinned buffer and destroy event
            LOG_INFO_STREAM("Pending copy completed: event=" << it->event << " ptr=" << it->pinnedPtr);
            RuntimeMemory::releasePinnedHost(it->pinnedPtr);
            cudaEventDestroy(it->event);
            it = pendingCopies.erase(it);
        } else if(q == cudaErrorNotReady) {
            // not ready; keep
            ++it;
        } else {
            LOG_WARN_STREAM("cudaEventQuery returned error: " << cudaGetErrorString(q) << " — cleaning up entry");
            RuntimeMemory::releasePinnedHost(it->pinnedPtr);
            cudaEventDestroy(it->event);
            it = pendingCopies.erase(it);
        }
    }
}

void NativeEngine::reportCallback(const GPUCallbackEvent& event) {
    if (gpuCallback) {
       gpuCallback->onGPUEvent(event);
    }
}

void NativeEngine::setGPUCallback(std::unique_ptr<IGPUCallback> callback) {
    gpuCallback = std::move(callback);
    LOG_INFO("GPU callback registered");
}

// ============ KV Cache Management ============

void* NativeEngine::transferKVToGPU(const void* hostPtr, size_t size, int pageId) {
    if (!kvCacheGPU) {
       LOG_ERROR("KV cache GPU not initialized");
       return nullptr;
    }

    GPUCallbackEvent event;
    event.type = GPUCallbackType::KV_CACHE_UPDATE;
    reportCallback(event);

    return kvCacheGPU->transferToGPU(hostPtr, size, pageId);
}

bool NativeEngine::transferKVFromGPU(void* gpuPtr, void* hostPtr, size_t size) {
    if (!kvCacheGPU) {
       LOG_ERROR("KV cache GPU not initialized");
       return false;
    }

    return kvCacheGPU->transferToHost(gpuPtr, hostPtr, size);
}

void NativeEngine::markKVPageUsed(int pageId, uint64_t token) {
    if (kvCacheGPU) {
       kvCacheGPU->markPageUsed(pageId, token);
    }
}

// ============ Tensor Transfer ============

void* NativeEngine::transferTensorToGPU(const Tensor& tensor) {
    LOG_INFO_STREAM("Transferring tensor to GPU");
    try {
        // Tensor API expects non-const for allocation/upload; cast away const safely
        Tensor& t = const_cast<Tensor&>(tensor);

        if(!t.cpu()) {
            LOG_ERROR_STREAM("transferTensorToGPU: tensor has no CPU data to upload");
            return nullptr;
        }

        // Allocate GPU backing and copy host -> device
        t.allocateGPU();
        size_t bytes = t.bytes();
        if(bytes == 0) return nullptr;

        cudaError_t err = cudaMemcpy(t.gpu(), t.cpu(), bytes, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
            LOG_ERROR_STREAM("cudaMemcpy H2D failed: " << cudaGetErrorString(err));
            return nullptr;
        }

#ifdef CUDA_DEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        LOG_INFO_STREAM("transferTensorToGPU: upload complete, bytes=" << bytes << " ptr=" << t.gpu());
        return static_cast<void*>(t.gpu());
    }
    catch(const std::exception& ex) {
        LOG_ERROR_STREAM("transferTensorToGPU exception: " << ex.what());
        return nullptr;
    }
}

bool NativeEngine::transferTensorFromGPU(void* gpuPtr, Tensor& tensor) {
    LOG_INFO_STREAM("Transferring tensor from GPU");
    try {
        // Ensure CPU buffer exists to receive data
        tensor.allocateCPU();
        size_t bytes = tensor.bytes();
        if(bytes == 0) return true;

        cudaError_t err = cudaMemcpy(tensor.cpu(), gpuPtr, bytes, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
            LOG_ERROR_STREAM("cudaMemcpy D2H failed: " << cudaGetErrorString(err));
            return false;
        }

#ifdef CUDA_DEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        LOG_INFO_STREAM("transferTensorFromGPU: download complete, bytes=" << bytes << " dst=" << tensor.cpu());
        return true;
    }
    catch(const std::exception& ex) {
        LOG_ERROR_STREAM("transferTensorFromGPU exception: " << ex.what());
        return false;
    }
}

void NativeEngine::removeLastLayer() {
    if(!layers.empty()) {
        layers.pop_back();
        LOG_INFO("Removed last runtime transformer layer");
    } else {
        LOG_WARN("removeLastLayer called but no runtime layers exist");
    }
}
