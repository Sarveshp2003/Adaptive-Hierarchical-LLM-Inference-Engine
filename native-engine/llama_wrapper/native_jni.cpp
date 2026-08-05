#include <jni.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>

extern "C" {
    // Declarations for adaptive engine C exports
    int adaptive_engine_tokenize(const char* text, int* output_tokens, int max_tokens);
    int adaptive_engine_detokenize(const int* tokens, int token_count, char* output_text, int max_len);
    int adaptive_engine_get_vocab_size();
    int adaptive_engine_infer(const int* input_tokens, int token_count, float* logits_out, int max_logits);
    double adaptive_engine_compute_perplexity(const int* tokens, int token_count);
    typedef struct NativeEngineApi {
        void (*start)();
        void (*stop)();
    } NativeEngineApi;
    NativeEngineApi* adaptive_engine_get_api();
}

// JNI bridge for com.adaptivellm.runtime.NativeInferenceEngine
extern "C" {

JNIEXPORT void JNICALL Java_com_adaptivellm_runtime_NativeInferenceEngine_nativeInitialize(JNIEnv* env, jobject obj) {
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

} // extern C
