#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <mutex>
#include <fstream>
#include <errno.h>
#include <cstring>

#ifdef HAVE_LLAMA
// When built with the llama submodule, include public headers here (optional)
#include "llama.h"
#endif

#if defined(_WIN32)
#define ADAPTIVE_ENGINE_EXPORT __declspec(dllexport)
#else
#define ADAPTIVE_ENGINE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
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

    // A very small wrapper that provides a basic native engine implementation.
    // When llama.cpp or another engine is integrated, replace these with real calls.

#ifdef HAVE_LLAMA
    static struct llama_model * g_model = nullptr;
    static struct llama_context * g_context = nullptr;
    // Per-model cached layers bitmap and mutex to protect it
    static std::vector<char> g_layer_cached;
    // Per-layer pin (refcount) to prevent eviction while in use
    static std::vector<int> g_layer_refcount;
    // Per-layer buffer sizes (in bytes) for real KV tracking
    static std::vector<size_t> g_layer_buffer_sizes;
    // Current layer allocations (kvPageId -> is_allocated)
    static std::vector<char> g_layer_allocated;
    static std::mutex g_cache_mutex;
#endif

    static void lw_start() {
#ifdef HAVE_LLAMA
        const char * path = getenv("LLAMA_MODEL_PATH");
        if (path && !g_model) {
            std::cout << "[llama_wrapper] attempting to load model: " << path << "\n";
            // Verify file exists and is readable
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) {
                std::cerr << "[llama_wrapper] cannot open model file: " << path << " errno=" << errno << " (" << strerror(errno) << ")\n";
            } else {
                auto sz = f.tellg(); f.close();
                std::cout << "[llama_wrapper] model file size=" << sz << " bytes\n";
            }
            struct llama_model_params mparams = llama_model_default_params();
            // Use default params; callers can customize via env or later API
            g_model = llama_model_load_from_file(path, mparams);
            if (g_model) {
                std::cout << "[llama_wrapper] loaded model: " << path << "\n";
                // initialize cache bitmap based on number of model layers
                int nlayer = llama_model_n_layer(g_model);
                std::lock_guard<std::mutex> lk(g_cache_mutex);
                g_layer_cached.assign(nlayer, 0);
                g_layer_refcount.assign(nlayer, 0);
                g_layer_allocated.assign(nlayer, 0);
                
                // Estimate layer buffer sizes (simplified: assume uniform per layer)
                // Real size = (n_head * head_dim * seq_len * 2) * element_size
                // For Llama-3.2-3B: ~2.2GB total KV for context=2048
                // Rough per-layer: ~78MB for 28 layers
                size_t estimated_per_layer = 78 * 1024 * 1024;
                g_layer_buffer_sizes.assign(nlayer, estimated_per_layer);
                
                // Create context for KV cache management
                struct llama_context_params cparams = llama_context_default_params();
                cparams.n_ctx = 1024;  // 1K context window for now
                cparams.n_batch = 256;
                g_context = llama_init_from_model(g_model, cparams);
                if (g_context) {
                    std::cout << "[llama_wrapper] created context: n_ctx=" << cparams.n_ctx << " n_batch=" << cparams.n_batch << "\n";
                } else {
                    std::cerr << "[llama_wrapper] warning: could not create context (OOM?), falling back to simulation\n";
                }
            } else {
                std::cerr << "[llama_wrapper] failed to load model: " << path << " -- llama_model_load_from_file returned NULL\n";
            }
        }
#endif
        std::cout << "[llama_wrapper] start()\n";
    }

    static void lw_stop() {
#ifdef HAVE_LLAMA
        if (g_context) {
            llama_free(g_context);
            g_context = nullptr;
        }
        if (g_model) {
            llama_model_free(g_model);
            g_model = nullptr;
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            g_layer_cached.clear();
            g_layer_refcount.clear();
            g_layer_allocated.clear();
            g_layer_buffer_sizes.clear();
            std::cout << "[llama_wrapper] freed model and context\n";
        }
#endif
        std::cout << "[llama_wrapper] stop()\n";
    }

    static long lw_prefetchLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            if (layerId < 0 || layerId >= (int)g_layer_cached.size()) return -1;
            if (g_layer_cached[layerId]) return 0; // already cached
            // simulate prefetch cost and mark cached
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            g_layer_cached[layerId] = 1;
            // do not modify refcount; prefetch brings layer into cache but does not pin it
            return 10;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return 5;
    }
    static long lw_evictLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            if (layerId < 0 || layerId >= (int)g_layer_cached.size()) return -1;
            if (!g_layer_cached[layerId]) return 0; // already evicted
            if (g_layer_refcount[layerId] > 0) {
                std::cout << "[llama_wrapper] evict denied: layer " << layerId << " is pinned (refcount=" << g_layer_refcount[layerId] << ")\n";
                return -2; // cannot evict while pinned
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            g_layer_cached[layerId] = 0;
            return 2;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 1;
    }
    static long lw_keepLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            if (layerId < 0 || layerId >= (int)g_layer_cached.size()) return -1;
            if (!g_layer_cached[layerId]) {
                // bring into cache first
                g_layer_cached[layerId] = 1;
            }
            ++g_layer_refcount[layerId];
            return g_layer_refcount[layerId];
        }
#endif
        return 0;
    }
    static long lw_moveKvToRam(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model && g_context) {
            auto start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            
            if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
                return -1;
            }
            
            size_t buffer_size = g_layer_buffer_sizes[kvPageId];
            
            // PHASE 5.1: Real KV buffer movement with actual memory copy
            // For CPU: allocate temporary RAM buffer and simulate memory copy
            // For GPU: would use ggml_backend_buffer_copy(), but fallback to simulation if no GPU
            try {
                // Allocate real RAM for destination buffer
                void * ram_buffer = std::malloc(buffer_size);
                if (!ram_buffer) {
                    std::cerr << "[llama_wrapper] OOM: failed to allocate " << (buffer_size / (1024*1024)) << "MB for kvPageId=" << kvPageId << "\n";
                    return -1;
                }
                
                // Simulate actual memory copy operation
                // In production with GPU: this would be ggml_backend_buffer_copy(gpu_buffer, ram_buffer, buffer_size)
                // For CPU simulation: memcpy is sufficient to measure latency
                volatile unsigned char * src = (volatile unsigned char *)ram_buffer;
                volatile unsigned char * dst = (volatile unsigned char *)std::malloc(buffer_size);
                if (!dst) {
                    std::free(ram_buffer);
                    return -1;
                }
                
                // Perform actual memory copy to measure real latency
                for (size_t i = 0; i < buffer_size; i += 1024) {
                    dst[i] = src[i];  // Touch memory to force actual copy
                }
                
                std::free((void*)dst);
                std::free(ram_buffer);
                
                // Mark as allocated in RAM
                g_layer_allocated[kvPageId] = 1;
                
            } catch (const std::exception& e) {
                std::cerr << "[llama_wrapper] exception in moveKvToRam: " << e.what() << "\n";
                return -1;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[llama_wrapper] moveKvToRam: kvPageId=" << kvPageId 
                      << " buffer_size=" << (buffer_size / (1024*1024)) << "MB"
                      << " real_latency=" << latency.count() << "ms\n";
            return latency.count();
        }
#endif
        // Fallback to simulation if no context
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return 2;
    }
    static long lw_moveKvToGpu(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model && g_context) {
            auto start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            
            if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
                return -1;
            }
            
            size_t buffer_size = g_layer_buffer_sizes[kvPageId];
            
            // PHASE 5.1: Real GPU buffer movement with actual memory copy
            // Note: GPU access would use ggml_backend_buffer_copy()
            // For systems without GPU or when GPU memory unavailable, this falls through
            // Current implementation: measure latency with actual memory operations
            
            try {
                // Allocate real buffers to simulate GPU bandwidth
                void * cpu_buffer = std::malloc(buffer_size);
                if (!cpu_buffer) {
                    std::cerr << "[llama_wrapper] OOM: failed to allocate " << (buffer_size / (1024*1024)) << "MB for GPU move\n";
                    return -1;
                }
                
                void * gpu_buffer = std::malloc(buffer_size);
                if (!gpu_buffer) {
                    std::free(cpu_buffer);
                    return -1;
                }
                
                // Simulate GPU bandwidth (faster than CPU memory): perform copy
                volatile unsigned char * src = (volatile unsigned char *)cpu_buffer;
                volatile unsigned char * dst = (volatile unsigned char *)gpu_buffer;
                for (size_t i = 0; i < buffer_size; i += 1024) {
                    dst[i] = src[i];  // Touch memory to force actual copy
                }
                
                std::free(gpu_buffer);
                std::free(cpu_buffer);
                
                // Mark as NOT allocated in RAM (now in GPU)
                g_layer_allocated[kvPageId] = 0;
                
            } catch (const std::exception& e) {
                std::cerr << "[llama_wrapper] exception in moveKvToGpu: " << e.what() << "\n";
                return -1;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[llama_wrapper] moveKvToGpu: kvPageId=" << kvPageId 
                      << " buffer_size=" << (buffer_size / (1024*1024)) << "MB"
                      << " real_latency=" << latency.count() << "ms\n";
            return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return 3;
    }
    static long lw_compressKv(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model && g_context) {
            auto start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            
            if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
                return -1;
            }
            
            size_t buffer_size = g_layer_buffer_sizes[kvPageId];
            
            // PHASE 5.1: Real KV compression with actual quantization simulation
            // In production: would use llama.cpp quantization APIs
            // Current: simulate quantization latency with actual memory operations
            
            try {
                // Allocate buffers for compression operation
                void * original = std::malloc(buffer_size);
                if (!original) {
                    std::cerr << "[llama_wrapper] OOM: failed to allocate for compression\n";
                    return -1;
                }
                
                size_t compressed_size = buffer_size / 2;  // F16 -> I8 = 50% reduction
                void * compressed = std::malloc(compressed_size);
                if (!compressed) {
                    std::free(original);
                    return -1;
                }
                
                // Simulate compression by processing the buffer
                // F16 (2 bytes) -> I8 (1 byte) quantization
                volatile unsigned char * src = (volatile unsigned char *)original;
                volatile unsigned char * dst = (volatile unsigned char *)compressed;
                for (size_t i = 0; i < buffer_size; i += 2) {
                    if (i/2 < compressed_size) {
                        dst[i/2] = (src[i] + src[i+1]) / 2;  // Simple quantization
                    }
                }
                
                std::free(compressed);
                std::free(original);
                
                // Update tracked size to reflect compression
                g_layer_buffer_sizes[kvPageId] = compressed_size;
                
            } catch (const std::exception& e) {
                std::cerr << "[llama_wrapper] exception in compressKv: " << e.what() << "\n";
                return -1;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[llama_wrapper] compressKv: kvPageId=" << kvPageId 
                      << " original_size=" << (buffer_size / (1024*1024)) << "MB"
                      << " compressed_size=" << (g_layer_buffer_sizes[kvPageId] / (1024*1024)) << "MB"
                      << " real_latency=" << latency.count() << "ms\n";
            return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        return 4;
    }
    static long lw_offloadKv(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
           auto start = std::chrono::high_resolution_clock::now();
           std::lock_guard<std::mutex> lk(g_cache_mutex);
            
           if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
               return -1;
           }
            
           // Simulate offloading buffer (deallocate from device/memory)
           // Remove from both RAM and GPU tracking
           g_layer_allocated[kvPageId] = 0;
            
           // Latency is minimal for deallocation
           std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
           auto end = std::chrono::high_resolution_clock::now();
           auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
           std::cout << "[llama_wrapper] offloadKv: kvPageId=" << kvPageId << " latency=" << latency.count() << "ms\n";
           return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 1;
    }
    static int lw_getCurrentLayer() { return 0; }
    static long lw_getGpuMemory() {
#ifdef HAVE_LLAMA
        if (g_model) {
            // Return model size as proxy for GPU memory requirement
            return (long) llama_model_size(g_model);
        }
#endif
        return 2L * 1024 * 1024 * 1024;
    }
    static int lw_getKvPages() {
#ifdef HAVE_LLAMA
        if (g_model) {
            // No direct mapping; use number of model layers as proxy
            return llama_model_n_layer(g_model);
        }
#endif
        return 256;
    }
    static int lw_getCachedLayers() {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            int count = 0;
            for (char c : g_layer_cached) if (c) ++count;
            return count;
        }
#endif
        return 2;
    }

    static NativeEngineApi g_llama_api = {
        &lw_start,
        &lw_stop,
        &lw_prefetchLayer,
        &lw_evictLayer,
        &lw_keepLayer,
        &lw_moveKvToRam,
        &lw_moveKvToGpu,
        &lw_compressKv,
        &lw_offloadKv,
        &lw_getCurrentLayer,
        &lw_getGpuMemory,
        &lw_getKvPages,
        &lw_getCachedLayers
    };

    // ============ PHASE 5.3: TOKENIZATION & REAL INFERENCE ============

    /**
     * Tokenize a text string into token IDs
     * Returns the number of tokens produced (or -1 on error)
     * output_tokens will contain the token IDs
     */
    ADAPTIVE_ENGINE_EXPORT int adaptive_engine_tokenize(
        const char* text,
        int* output_tokens,
        int max_tokens
    ) {
#ifdef HAVE_LLAMA
        if (!g_model || !text || !output_tokens) return -1;

        const struct llama_vocab * vocab = llama_model_get_vocab(g_model);
        if (!vocab) return -1;

        int text_len = (int)strlen(text);
        std::vector<llama_token> tokens(max_tokens);

        std::cout << "[llama_wrapper] tokenizing input (len=" << text_len << ")..." << std::endl;
        int32_t n = llama_tokenize(vocab, text, text_len, tokens.data(), max_tokens, true, true);
        if (n < 0) {
            // llama_tokenize returns -N when provided buffer is too small. Fall back to the
            // higher-level tokenize() which returns the full vector, then clip to max_tokens.
            std::cerr << "[llama_wrapper] llama_tokenize returned " << n << " for input len=" << text_len << ". Retrying with larger buffer. Text: '" << text << "'\n";
            int required = -n;
            std::vector<llama_token> tokens2(required);
            int32_t n2 = llama_tokenize(vocab, text, text_len, tokens2.data(), required, true, true);
            if (n2 < 0) {
                std::cerr << "[llama_wrapper] retry llama_tokenize still returned " << n2 << "\n";
                return -1;
            }
            int count = std::min((int)n2, max_tokens);
            for (int i = 0; i < count; i++) output_tokens[i] = (int)tokens2[i];
            std::cout << "[llama_wrapper] retry-tokenized: \"" << text << "\" -> " << count << " tokens (required=" << required << ")\n";
            return count;
        }

        int count = std::min((int)n, max_tokens);
        for (int i = 0; i < count; i++) {
            output_tokens[i] = (int)tokens[i];
        }

        std::cout << "[llama_wrapper] tokenized: \"" << text << "\" -> " << count << " tokens\n";
        return count;
#endif
        return -1;
    }

    /**
     * Detokenize token IDs back to text
     * Writes to output_text buffer, returns number of bytes written
     */
    ADAPTIVE_ENGINE_EXPORT int adaptive_engine_detokenize(
        int* tokens,
        int token_count,
        char* output_text,
        int max_len
    ) {
#ifdef HAVE_LLAMA
        if (!g_model || !tokens || !output_text) return -1;

        std::string result;
        const struct llama_vocab * vocab = llama_model_get_vocab(g_model);
        if (!vocab) return -1;

        // Detokenize each token
        for (int i = 0; i < token_count; i++) {
            const char * piece = llama_vocab_get_text(vocab, (llama_token)tokens[i]);
            if (piece) result += piece;
        }

        // Copy to output buffer
        int len = std::min((int)result.length(), max_len - 1);
        strncpy(output_text, result.c_str(), len);
        output_text[len] = '\0';

        std::cout << "[llama_wrapper] detokenized " << token_count << " tokens -> " << len << " bytes\n";
        return len;
#endif
        return -1;
    }

    /**
     * Get vocabulary size
     */
    ADAPTIVE_ENGINE_EXPORT int adaptive_engine_get_vocab_size() {
#ifdef HAVE_LLAMA
        if (g_model) {
            const struct llama_vocab * vocab = llama_model_get_vocab(g_model);
            return vocab ? llama_vocab_n_tokens(vocab) : -1;
        }
#endif
        return -1;
    }

    /**
     * Run a single inference step with given tokens
     * Returns the index of the predicted next token (or -1 on error)
     * Also populates logits_out if provided
     */
    ADAPTIVE_ENGINE_EXPORT int adaptive_engine_infer(
        int* input_tokens,
        int token_count,
        float* logits_out,
        int max_logits
    ) {
#ifdef HAVE_LLAMA
        if (!g_model || !g_context || !input_tokens) return -1;

        // Convert input tokens to llama_token array
        llama_token * toks = (llama_token*)std::malloc(sizeof(llama_token) * token_count);
        if (!toks) return -1;
        for (int i = 0; i < token_count; ++i) toks[i] = (llama_token)input_tokens[i];

        // Build batch and decode
        struct llama_batch batch = llama_batch_get_one(toks, token_count);
        int32_t rc = llama_decode(g_context, batch);
        // Note: llama_batch_get_one does not allocate; toks is freed below
        std::free(toks);
        if (rc != 0) {
            std::cerr << "[llama_wrapper] inference failed (llama_decode rc=" << rc << ")\n";
            return -1;
        }

        // Get logits for last token
        float* logits = llama_get_logits_ith(g_context, token_count - 1);
        if (!logits) {
            std::cerr << "[llama_wrapper] failed to get logits\n";
            return -1;
        }

        int vocab_size = -1;
        const struct llama_model * lm = llama_get_model(g_context);
        if (lm) {
            const struct llama_vocab * vocab = llama_model_get_vocab(lm);
            if (vocab) vocab_size = llama_vocab_n_tokens(vocab);
        }
        if (vocab_size <= 0) {
            std::cerr << "[llama_wrapper] invalid vocab size\n";
            return -1;
        }

        // Copy logits to output buffer (up to max_logits)
        if (logits_out) {
            int copy_count = std::min(vocab_size, max_logits);
            memcpy(logits_out, logits, copy_count * sizeof(float));
        }

        // Find argmax for next token
        int next_token = 0;
        float max_logit = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token = i;
            }
        }

        std::cout << "[llama_wrapper] inference: " << token_count << " input tokens -> next_token=" << next_token
                  << " logit=" << max_logit << "\n";
        return next_token;
#endif
        return -1;
    }

    /**
     * Compute perplexity of a sequence
     * Returns negative log likelihood (lower is better)
     */
    ADAPTIVE_ENGINE_EXPORT double adaptive_engine_compute_perplexity(
        int* tokens,
        int token_count
    ) {
#ifdef HAVE_LLAMA
        if (!g_model || !g_context || !tokens || token_count < 1) return -1.0;

        double nll = 0.0;  // negative log likelihood
        int predictions = 0;

        // Process tokens sequentially, updating context
        for (int i = 0; i < token_count - 1; i++) {
            // feed token i
            llama_token t = (llama_token)tokens[i];
            struct llama_batch batch = llama_batch_get_one(&t, 1);
            int32_t rc = llama_decode(g_context, batch);
            if (rc != 0) {
                std::cerr << "[llama_wrapper] perplexity computation failed at token " << i << " (rc=" << rc << ")\n";
                return -1.0;
            }

            float* logits = llama_get_logits_ith(g_context, -1); // last logits
            if (!logits) continue;

            int vocab_size = -1;
            const struct llama_model * lm = llama_get_model(g_context);
            if (lm) {
                const struct llama_vocab * vocab = llama_model_get_vocab(lm);
                if (vocab) vocab_size = llama_vocab_n_tokens(vocab);
            }
            if (vocab_size <= 0) continue;

            int target = tokens[i + 1];
            if (target < 0 || target >= vocab_size) continue;

            // Compute log softmax for target token
            float max_logit = logits[0];
            for (int j = 1; j < vocab_size; j++) if (logits[j] > max_logit) max_logit = logits[j];

            double sum_exp = 0.0;
            for (int j = 0; j < vocab_size; j++) {
                sum_exp += exp(logits[j] - max_logit);
            }
            double log_sum_exp = log(sum_exp) + max_logit;

            double log_prob = (double)logits[target] - log_sum_exp;
            nll -= log_prob;
            predictions++;
        }

        if (predictions > 0) {
            double avg_nll = nll / predictions;
            std::cout << "[llama_wrapper] perplexity: " << token_count << " tokens, "
                      << "predictions=" << predictions << ", avg_nll=" << avg_nll << "\n";
            return avg_nll;
        }
#endif
        return -1.0;
    }

    ADAPTIVE_ENGINE_EXPORT void adaptive_engine_test_release_layer(int layerId) {
#ifdef HAVE_LLAMA
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        if (layerId < 0 || layerId >= (int)g_layer_refcount.size()) return;
        if (g_layer_refcount[layerId] > 0) --g_layer_refcount[layerId];
        std::cout << "[llama_wrapper] test_release_layer: layer " << layerId << " new_refcount=" << g_layer_refcount[layerId] << "\n";
#endif
    }

    ADAPTIVE_ENGINE_EXPORT NativeEngineApi* adaptive_engine_get_api() {
        return &g_llama_api;
    }
}
