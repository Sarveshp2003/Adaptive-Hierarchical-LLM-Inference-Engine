#include <jni.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>

extern "C" {
    // Declarations for adaptive engine C exports
    int adaptive_engine_tokenize(const char* text, int* output_tokens, int max_tokens);
    int adaptive_engine_detokenize(const int* tokens, int token_count, char* output_text, int max_len);
    int adaptive_engine_get_vocab_size();
    int adaptive_engine_get_eos_token();
    int adaptive_engine_infer(const int* input_tokens, int token_count, float* logits_out, int max_logits);
    double adaptive_engine_compute_perplexity(const int* tokens, int token_count);
    typedef struct NativeEngineApi {
        void (*start)();
        void (*stop)();
        long (*prefetchLayer)(int layerId);
        long (*evictLayer)(int layerId);
        long (*keepLayer)(int layerId);
        long (*moveKvToRam)(long kvPageId);
        long (*moveKvToGpu)(long kvPageId);
        long (*compressKv)(long kvPageId);
        long (*offloadKv)(long kvPageId);
        int  (*getCurrentLayer)();
        long (*getGpuMemory)();
        int  (*getKvPages)();
        int  (*getCachedLayers)();
    } NativeEngineApi;
    NativeEngineApi* adaptive_engine_get_api();
}

// JNI bridge for com.adaptivellm.runtime.NativeInferenceEngine
extern "C" {

static void configure_gpu_env(jboolean enableGpu) {
    const char* value = enableGpu ? "1" : "0";
#ifdef _WIN32
    _putenv_s("ADAPTIVELLM_ENABLE_GPU", value);
#else
    setenv("ADAPTIVELLM_ENABLE_GPU", value, 1);
#endif
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeInitialize(JNIEnv* env, jobject obj) {
    configure_gpu_env(true);
    NativeEngineApi* api = adaptive_engine_get_api();
    if (api && api->start) api->start();
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeInitialize__Z(JNIEnv* env, jobject obj, jboolean enableGpu) {
    configure_gpu_env(enableGpu);
    NativeEngineApi* api = adaptive_engine_get_api();
    if (api && api->start) api->start();
}

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeShutdown(JNIEnv* env, jobject obj) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (api && api->stop) api->stop();
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeGetVocabSize(JNIEnv* env, jobject obj) {
    int v = adaptive_engine_get_vocab_size();
    return (jint)v;
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeGetEosToken(JNIEnv* env, jobject obj) {
    int v = adaptive_engine_get_eos_token();
    return (jint)v;
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeTokenize(JNIEnv* env, jobject obj, jstring text, jintArray outputTokens, jint maxTokens) {
    if (text == NULL) return (jint)-1;
    const char* s = env->GetStringUTFChars(text, NULL);
    if (s == NULL) return (jint)-1;
    std::vector<int> buf(maxTokens);
    int count = adaptive_engine_tokenize(s, buf.data(), maxTokens);
    env->ReleaseStringUTFChars(text, s);
    // Propagate negative return codes from the native tokenizer (e.g., -N indicates N required tokens)
    if (count < 0) return (jint)count;
    if (outputTokens != NULL) {
        env->SetIntArrayRegion(outputTokens, 0, count, reinterpret_cast<jint*>(buf.data()));
    }
    return (jint)count;
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeDetokenize(JNIEnv* env, jobject obj, jintArray tokens, jint tokenCount, jbyteArray outputText, jint maxLen) {
    if (tokenCount <= 0 || tokens == NULL || outputText == NULL) return (jint)-1;
    std::vector<int> tvec(tokenCount);
    env->GetIntArrayRegion(tokens, 0, tokenCount, reinterpret_cast<jint*>(tvec.data()));
    std::vector<char> out(maxLen);
    int bytes = adaptive_engine_detokenize(tvec.data(), tokenCount, out.data(), maxLen);
    if (bytes < 0) return (jint)-1;
    env->SetByteArrayRegion(outputText, 0, bytes, reinterpret_cast<jbyte*>(out.data()));
    return (jint)bytes;
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeInfer(JNIEnv* env, jobject obj, jintArray inputTokens, jint tokenCount, jfloatArray logitsOut, jint maxLogits) {
    if (tokenCount < 0 || (tokenCount > 0 && inputTokens == NULL)) return (jint)-1;
    std::vector<int> invec(tokenCount);
    if (tokenCount > 0) env->GetIntArrayRegion(inputTokens, 0, tokenCount, reinterpret_cast<jint*>(invec.data()));
    std::vector<float> logits(maxLogits);
    int nextToken = adaptive_engine_infer(invec.data(), tokenCount, logits.data(), maxLogits);
    if (nextToken < 0) return (jint)-1;
    if (logitsOut != NULL) {
        int copyCount = std::min(maxLogits, (jint)maxLogits);
        env->SetFloatArrayRegion(logitsOut, 0, copyCount, reinterpret_cast<jfloat*>(logits.data()));
    }
    return (jint)nextToken;
}

JNIEXPORT jdouble JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeComputePerplexity(JNIEnv* env, jobject obj, jintArray tokens, jint tokenCount) {
    if (tokenCount <= 0 || tokens == NULL) return (jdouble)-1.0;
    std::vector<int> tvec(tokenCount);
    env->GetIntArrayRegion(tokens, 0, tokenCount, reinterpret_cast<jint*>(tvec.data()));
    double p = adaptive_engine_compute_perplexity(tvec.data(), tokenCount);
    return (jdouble)p;
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativePrefetchLayer(JNIEnv* env, jobject obj, jint layerId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->prefetchLayer) return -1;
    return static_cast<jlong>(api->prefetchLayer(static_cast<int>(layerId)));
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeEvictLayer(JNIEnv* env, jobject obj, jint layerId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->evictLayer) return -1;
    return static_cast<jlong>(api->evictLayer(static_cast<int>(layerId)));
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeKeepLayer(JNIEnv* env, jobject obj, jint layerId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->keepLayer) return -1;
    return static_cast<jlong>(api->keepLayer(static_cast<int>(layerId)));
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeMoveKvToRam(JNIEnv* env, jobject obj, jlong kvPageId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->moveKvToRam) return -1;
    return static_cast<jlong>(api->moveKvToRam(static_cast<long>(kvPageId)));
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeMoveKvToGpu(JNIEnv* env, jobject obj, jlong kvPageId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->moveKvToGpu) return -1;
    return static_cast<jlong>(api->moveKvToGpu(static_cast<long>(kvPageId)));
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeCompressKv(JNIEnv* env, jobject obj, jlong kvPageId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->compressKv) return -1;
    return static_cast<jlong>(api->compressKv(static_cast<long>(kvPageId)));
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeOffloadKv(JNIEnv* env, jobject obj, jlong kvPageId) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->offloadKv) return -1;
    return static_cast<jlong>(api->offloadKv(static_cast<long>(kvPageId)));
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeGetCurrentLayer(JNIEnv* env, jobject obj) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->getCurrentLayer) return -1;
    return static_cast<jint>(api->getCurrentLayer());
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeGetGpuMemory(JNIEnv* env, jobject obj) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->getGpuMemory) return -1;
    return static_cast<jlong>(api->getGpuMemory());
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeGetKvPages(JNIEnv* env, jobject obj) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->getKvPages) return -1;
    return static_cast<jint>(api->getKvPages());
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeGetCachedLayers(JNIEnv* env, jobject obj) {
    NativeEngineApi* api = adaptive_engine_get_api();
    if (!api || !api->getCachedLayers) return -1;
    return static_cast<jint>(api->getCachedLayers());
}

} // extern C
