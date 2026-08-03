#include <jni.h>

#include <iostream>
#include <map>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <cuda_runtime.h>
#include "RuntimeMemory.h"

#include "NativeEngine.h"



static NativeEngine engine;



extern "C"
{


JNIEXPORT void JNICALL
Java_com_adaptivellm_nativeengine_NativeEngine_initialize(

        JNIEnv *env,

        jobject obj,

        jint gpuDevice

)
{

    // Query GPU memory and allocate intelligently
    size_t gpu_free = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    
    // Allocate 85% of available memory, but cap at 8GB
    size_t to_allocate = std::min(
        (gpu_total * 85) / 100,
        8ULL * 1024ULL * 1024ULL * 1024ULL
    );
    
    // Minimum 1GB, maximum available
    if (to_allocate < 1024ULL * 1024ULL * 1024ULL) {
        to_allocate = 1024ULL * 1024ULL * 1024ULL;
    }
    
    std::cout << "GPU Memory: Total=" << (gpu_total / (1024.0*1024*1024)) << "GB, "
              << "Free=" << (gpu_free / (1024.0*1024*1024)) << "GB, "
              << "Allocating=" << (to_allocate / (1024.0*1024*1024)) << "GB" << std::endl;
    
    engine.initialize(
        static_cast<int>(gpuDevice),
        to_allocate
    );

}

// Simple allocation tracking for JNI stubs
struct AllocationInfo {
    void* ptr;
    size_t size;
    bool runtimeAlloc;
};
static std::unordered_map<uint64_t, AllocationInfo> g_allocations;
static uint64_t g_next_handle = 1;

// ---------------- GPUMemoryService JNI stubs ----------------

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_initializeGPU(JNIEnv* env, jobject obj) {
    std::cout << "JNI: GPUMemoryService_initializeGPU\n";
    try {
        // Query GPU and allocate intelligently
        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);
        
        // Allocate 85% of available memory, but cap at 8GB
        size_t to_allocate = std::min(
            (gpu_total * 85) / 100,
            8ULL * 1024ULL * 1024ULL * 1024ULL
        );
        
        // Minimum 1GB, maximum available
        if (to_allocate < 1024ULL * 1024ULL * 1024ULL) {
            to_allocate = 1024ULL * 1024ULL * 1024ULL;
        }
        
        std::cout << "Initializing GPU memory: " << (to_allocate / (1024.0*1024*1024)) << "GB\n";
        RuntimeMemory::initializeGPU(to_allocate);
    } catch(const std::exception& e) {
        std::cerr << "GPU init failed: " << e.what() << "\n";
    } catch(...) {}
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_shutdown(JNIEnv* env, jobject obj) {
    try { RuntimeMemory::shutdown(); } catch(...) {}
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_nativeAllocateGPU(JNIEnv* env, jobject obj, jlong bytes) {
    void* ptr = RuntimeMemory::allocateGPU(static_cast<size_t>(bytes));
    if (!ptr) {
        return static_cast<jlong>(0);
    }
    uint64_t handle = g_next_handle++;
    g_allocations[handle] = { ptr, static_cast<size_t>(bytes), true };
    std::cout << "JNI allocate GPU: handle=" << handle << " size=" << bytes << std::endl;
    return static_cast<jlong>(handle);
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_nativeFreeGPU(JNIEnv* env, jobject obj, jlong handle) {
    uint64_t h = static_cast<uint64_t>(handle);
    auto it = g_allocations.find(h);
    if (it == g_allocations.end()) return;
    void* ptr = it->second.ptr;
    if (it->second.runtimeAlloc) {
        RuntimeMemory::releaseGPU(ptr);
    } else {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree failed in nativeFreeGPU: " << cudaGetErrorString(err) << std::endl;
        }
    }
    g_allocations.erase(it);
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_nativeGetSize(JNIEnv* env, jobject obj, jlong handle) {
    uint64_t h = static_cast<uint64_t>(handle);
    auto it = g_allocations.find(h);
    if (it != g_allocations.end()) return static_cast<jlong>(it->second.size);
    return 0;
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_nativeMemcpyHostToDevice(JNIEnv* env, jobject obj, jlong hostPtr, jlong gpuHandle, jlong bytes) {
    uint64_t h = static_cast<uint64_t>(gpuHandle);
    auto it = g_allocations.find(h);
    if (it == g_allocations.end()) return;
    void* devPtr = it->second.ptr;
    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(hostPtr));
    cudaError_t err = cudaMemcpy(devPtr, src, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy HostToDevice failed: " << cudaGetErrorString(err) << std::endl;
    }
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_GPUMemoryService_nativeMemcpyDeviceToHost(JNIEnv* env, jobject obj, jlong gpuHandle, jlong hostPtr, jlong bytes) {
    uint64_t h = static_cast<uint64_t>(gpuHandle);
    auto it = g_allocations.find(h);
    if (it == g_allocations.end()) return;
    void* devPtr = it->second.ptr;
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(hostPtr));
    cudaError_t err = cudaMemcpy(dst, devPtr, static_cast<size_t>(bytes), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy DeviceToHost failed: " << cudaGetErrorString(err) << std::endl;
    }
}

// ---------------- KVCacheGPUManager JNI stubs ----------------

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_initializeKVCache(JNIEnv* env, jobject obj) {
    // No-op (engine initialization handled elsewhere)
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_shutdownKVCache(JNIEnv* env, jobject obj) {
    // No-op
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_nativeTransferToGPU(JNIEnv* env, jobject obj, jlong hostPtr, jlong size, jint pageId) {
    const void* hptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(hostPtr));
    void* gpuPtr = engine.transferKVToGPU(hptr, static_cast<size_t>(size), static_cast<int>(pageId));
    if (!gpuPtr) return static_cast<jlong>(0);
    uint64_t handle = g_next_handle++;
    g_allocations[handle] = { gpuPtr, static_cast<size_t>(size), false };
    std::cout << "JNI KV transfer to GPU: page=" << pageId << " handle=" << handle << " size=" << size << std::endl;
    return static_cast<jlong>(handle);
}

JNIEXPORT jboolean JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_nativeTransferFromGPU(JNIEnv* env, jobject obj, jlong gpuHandle, jlong hostPtr, jlong size) {
    uint64_t h = static_cast<uint64_t>(gpuHandle);
    auto it = g_allocations.find(h);
    if (it == g_allocations.end()) {
        return JNI_FALSE;
    }
    void* gpuPtr = it->second.ptr;
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(hostPtr));
    bool ok = engine.transferKVFromGPU(gpuPtr, dst, static_cast<size_t>(size));
    if (!ok) return JNI_FALSE;
    // Free GPU memory associated with this page
    cudaError_t err = cudaFree(gpuPtr);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree failed in nativeTransferFromGPU: " << cudaGetErrorString(err) << std::endl;
    }
    g_allocations.erase(it);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_nativeMarkPageUsed(JNIEnv* env, jobject obj, jint pageId, jlong tokenPosition) {
    try {
        engine.markKVPageUsed(static_cast<int>(pageId), static_cast<uint64_t>(tokenPosition));
    } catch(...) {}
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_nativeGetUsedMemory(JNIEnv* env, jobject obj) {
    size_t total = 0;
    for (auto &p : g_allocations) total += p.second.size;
    return static_cast<jlong>(total);
}

JNIEXPORT jboolean JNICALL Java_com_adaptivellm_runtime_services_KVCacheGPUManager_nativeEvictLRU(JNIEnv* env, jobject obj) {
    if (g_allocations.empty()) return JNI_FALSE;
    auto it = g_allocations.begin();
    void* ptr = it->second.ptr;
    if (it->second.runtimeAlloc) {
        RuntimeMemory::releaseGPU(ptr);
    } else {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree failed in nativeEvictLRU: " << cudaGetErrorString(err) << std::endl;
        }
    }
    g_allocations.erase(it);
    return JNI_TRUE;
}

// ---------------- NativeEngine simple adapters ----------------

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_startRuntime(JNIEnv* env, jobject obj) {
    std::cout << "JNI: NativeEngine_startRuntime (shim)" << std::endl;
}

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_stopRuntime(JNIEnv* env, jobject obj) {
    try { engine.shutdown(); } catch(...) {}
}

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_requestLayer(JNIEnv* env, jobject obj, jint layerId) {
    try { engine.loadLayer(static_cast<int>(layerId)); } catch(...) {}
}

// ---------------- GPU Profiler & Model Validator stubs ----------------

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_GPUProfiler_nativeGetGPUMemoryUsed(JNIEnv* env, jobject obj) { return Java_com_adaptivellm_runtime_services_KVCacheGPUManager_nativeGetUsedMemory(env,obj); }
JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_GPUProfiler_nativeGetGPUMemoryTotal(JNIEnv* env, jobject obj) { return 1024LL * 1024LL * 1024LL; }
JNIEXPORT jfloat JNICALL Java_com_adaptivellm_runtime_GPUProfiler_nativeGetGPUTemperature(JNIEnv* env, jobject obj) { return 55.0f; }
JNIEXPORT jfloat JNICALL Java_com_adaptivellm_runtime_GPUProfiler_nativeGetGPUPowerDraw(JNIEnv* env, jobject obj) { return 120.0f; }
JNIEXPORT jdouble JNICALL Java_com_adaptivellm_runtime_GPUProfiler_nativeGetMemoryBandwidth(JNIEnv* env, jobject obj) { return 250.0; }

JNIEXPORT jboolean JNICALL Java_com_adaptivellm_runtime_ModelValidator_nativeValidateModel(JNIEnv* env, jobject obj, jstring path) { return JNI_TRUE; }
JNIEXPORT jboolean JNICALL Java_com_adaptivellm_runtime_ModelValidator_nativeCheckGPU(JNIEnv* env, jobject obj) { return JNI_TRUE; }
JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_ModelValidator_nativeGetGPUMemory(JNIEnv* env, jobject obj) { return 4LL * 1024LL * 1024LL * 1024LL; }
JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_ModelValidator_nativeGetComputeCapability(JNIEnv* env, jobject obj) { return 86; }

// ---------------- GGUFModelLoader adapter ----------------

JNIEXPORT jboolean JNICALL Java_com_adaptivellm_runtime_GGUFModelLoader_nativeLoadModelGGUF(JNIEnv* env, jobject obj, jstring path) {
    const char* p = env->GetStringUTFChars(path, nullptr);
    bool ok = engine.loadModelGGUF(std::string(p));
    env->ReleaseStringUTFChars(path, p);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_adaptivellm_runtime_GGUFModelLoader_nativeLoadLayerFromGGUF(JNIEnv* env, jobject obj, jint layerId) {
    bool ok = engine.loadLayer(static_cast<int>(layerId));
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL Java_com_adaptivellm_runtime_GGUFModelLoader_nativeGetMetadata(JNIEnv* env, jobject obj) {
    // Build ModelMetadata(numLayers, hiddenDim, numHeads, vocabSize, totalSize, modelName)
    jint numLayers = static_cast<jint>(engine.getNumLayers());
    jint hiddenDim = static_cast<jint>(engine.getHiddenDim());
    jint numHeads = static_cast<jint>(engine.getNumHeads());
    jint vocabSize = static_cast<jint>(engine.getVocabSize());
    jlong totalSize = static_cast<jlong>(engine.getTotalModelSize());
    std::string name = engine.getModelName();
    jstring jname = env->NewStringUTF(name.c_str());

    jclass metaClass = env->FindClass("com/adaptivellm/runtime/GGUFModelLoader$ModelMetadata");
    if(!metaClass) return nullptr;

    jmethodID ctor = env->GetMethodID(metaClass, "<init>", "(IIIIJLjava/lang/String;)V");
    if(!ctor) return nullptr;

    jobject metaObj = env->NewObject(metaClass, ctor, numLayers, hiddenDim, numHeads, vocabSize, totalSize, jname);
    return metaObj;
}







JNIEXPORT void JNICALL
Java_com_adaptivellm_nativeengine_NativeEngine_executeLayer(

        JNIEnv *env,

        jobject obj,

        jint layerId,
        jfloatArray inputData,
        jint batch,
        jint seqLen,
        jfloatArray outputData

)
{
    std::cout << "JNI execute layer request: " << layerId << std::endl;

    if(inputData == nullptr || outputData == nullptr)
    {
        std::cerr << "JNI executeLayer: input or output array is null" << std::endl;
        return;
    }

    jsize inLen = env->GetArrayLength(inputData);
    jsize outLen = env->GetArrayLength(outputData);

    // Create tensors with provided shape
    try
    {
        Tensor input({ static_cast<int>(batch), static_cast<int>(seqLen) }, DataType::FP32);
        Tensor output({ static_cast<int>(batch), static_cast<int>(seqLen) }, DataType::FP32);

        input.allocateCPU(); input.allocateGPU();
        output.allocateCPU(); output.allocateGPU();

        if(inLen > 0)
        {
            env->GetFloatArrayRegion(inputData, 0, inLen, input.cpu());
            input.upload();
        }

        // Execute layer
        engine.executeLayer(static_cast<int>(layerId), input, output);

        output.download();

        if(outLen > 0)
        {
            // Copy back up to min(outLen, output.elements())
            jsize copyCount = std::min<jsize>(outLen, static_cast<jsize>(output.elements()));
            env->SetFloatArrayRegion(outputData, 0, copyCount, output.cpu());
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "JNI executeLayer exception: " << ex.what() << std::endl;
    }
}





JNIEXPORT void JNICALL
Java_com_adaptivellm_nativeengine_NativeEngine_shutdown(

        JNIEnv *env,

        jobject obj

)
{

    engine.shutdown();

}


JNIEXPORT void JNICALL
Java_com_adaptivellm_nativeengine_NativeEngine_addLayer(
    JNIEnv* env,
    jobject obj,
    jint hiddenSize
)
{
    try
    {
        engine.addLayer(static_cast<int>(hiddenSize));
    }
    catch(const std::exception& ex)
    {
        std::cerr << "JNI addLayer exception: " << ex.what() << std::endl;
    }
}


}