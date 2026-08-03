#ifndef NATIVE_ENGINE_H
#define NATIVE_ENGINE_H


#include <vector>
#include <memory>
#include <string>


#include "TransformerLayer.h"
#include "Tensor.h"
#include "MMapWeightLoader.h"
#include "GGUFLoader.h"
#include "..\..\loader\gguf_reader.h"
#include "GPUCallback.h"
#include "KVCacheGPU.h"
#include <map>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <atomic>


class NativeEngine
{

private:

    bool initialized;


    std::vector<
        std::unique_ptr<TransformerLayer>
    > layers;

    std::shared_ptr<MMapWeightLoader> modelLoader;
    std::unique_ptr<GGUFLoader> ggufLoader;
    // Fallback archive cache when GGUFLoader cannot parse certain GGUF v3 variants
    std::unique_ptr<loader::ModelArchive> archCache;
    std::unique_ptr<IGPUCallback> gpuCallback;
    std::unique_ptr<KVCacheGPU> kvCacheGPU;

    int gpuDevice;
    int numLayers;
    int hiddenDim;

    struct LoadedTensorEntry { void* ptr; bool fromPool; };
    // Track loaded tensors (tensor name -> GPU pointer + ownership)
    std::map<std::string, LoadedTensorEntry> loadedTensors;

    struct PendingCopy {
        cudaEvent_t event;
        void* pinnedPtr;
    };

    std::vector<PendingCopy> pendingCopies;
    std::mutex pendingMutex;

    void pollCleanupPendingCopies();

    // Background cleanup thread for pinned buffers
    std::thread cleanupThread;
    std::atomic_bool cleanupRunning{false};

    void reportCallback(const GPUCallbackEvent& event);

public:


    NativeEngine();


    ~NativeEngine();



    void initialize(
        int gpuDevice,
        size_t gpuMemoryBytes = 1024ULL * 1024ULL * 1024ULL  // 1GB default
    );


    void addLayer(
        int hiddenSize
    );


    void executeLayer(

        int index,

        Tensor& input,

        Tensor& output

    );


    void shutdown();

    // GGUF Model Loading
    bool loadModelGGUF(const std::string& path);
    bool loadLayerFromGGUF(int layerId);
    
    // Legacy support
    bool loadModel(const std::string& path);
    bool loadLayer(int layerId);
    void releaseLayer(int layerId);
    // Remove the most-recently added runtime transformer layer (frees its GPU resources)
    void removeLastLayer();

    // GPU Callbacks
    void setGPUCallback(std::unique_ptr<IGPUCallback> callback);

    // KV Cache Management
    void* transferKVToGPU(const void* hostPtr, size_t size, int pageId);
    bool transferKVFromGPU(void* gpuPtr, void* hostPtr, size_t size);
    void markKVPageUsed(int pageId, uint64_t token);

    // Tensor transfer
    void* transferTensorToGPU(const Tensor& tensor);
    bool transferTensorFromGPU(void* gpuPtr, Tensor& tensor);

    // Metadata getters (GGUF)
    int getNumLayers() const;
    int getHiddenDim() const;
    int getNumHeads() const;
    int getVocabSize() const;
    uint64_t getTotalModelSize() const;
    std::string getModelName() const;

};



#endif