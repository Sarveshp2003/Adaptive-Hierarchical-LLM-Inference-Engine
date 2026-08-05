package com.adaptivellm.runtime;

/**
 * JNI wrapper for native inference engine with tokenization and real model inference.
 * 
 * Phase 5.3: Real Model Inference Integration
 * 
 * Provides:
 * - Tokenization (string -> token IDs)
 * - Detokenization (token IDs -> string)
 * - Real model inference (forward pass)
 * - Perplexity computation (convergence metric)
 */
public final class NativeInferenceEngine {
    
    private volatile boolean initialized = false;
    private volatile int vocabSize = -1;
    private volatile int eosToken = -1;

    /**
     * Initialize the native inference engine.
     * Must be called before any inference operations.
     */
    public void initialize() {
        if (!initialized) {
            nativeInitialize();
            this.vocabSize = nativeGetVocabSize();
            this.eosToken = nativeGetEosToken();
            this.initialized = true;
            System.out.println("[NativeInferenceEngine] Initialized with vocab_size=" + vocabSize + " eos_token=" + eosToken);
        }
    }

    /**
     * Tokenize a text string into token IDs.
     * 
     * @param text Input text to tokenize
     * @return Array of token IDs
     * @throws RuntimeException if tokenization fails
     */
    public int[] tokenize(String text) throws RuntimeException {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }

        if (text == null || text.isEmpty()) {
            return new int[0];
        }

        // Estimate tokens and retry with exponential backoff if buffer is too small.
        int words = Math.max(1, text.split("\\s+").length);
        int maxTokens = Math.min(1024, Math.max(10, words * 2)); // start conservatively
        final int MAX_CAP = 65536;
        int attempts = 0;

        while (attempts < 6) {
            int[] tokens = new int[maxTokens];
            int tokenCount = nativeTokenize(text, tokens, maxTokens);

            if (tokenCount < 0) {
                // Native returned negative: -N means N tokens required. Try to resize exactly if possible.
                int required = -tokenCount;
                if (required > 0 && required <= MAX_CAP && required > maxTokens) {
                    maxTokens = Math.min(required + 8, MAX_CAP); // add small slack
                    attempts++;
                    continue;
                }

                // Otherwise fall back to exponential growth
                if (maxTokens < MAX_CAP) {
                    maxTokens = Math.min(maxTokens * 2, MAX_CAP);
                    attempts++;
                    continue;
                }

                throw new RuntimeException(ErrorCode.JNI_ERROR, "Tokenization failed for: " + text);
            }

            // If tokenCount equals or exceeds buffer, assume truncation and retry with larger buffer
            if (tokenCount >= maxTokens && maxTokens < MAX_CAP) {
                maxTokens = Math.min(maxTokens * 2, MAX_CAP);
                attempts++;
                continue;
            }

            // Success: trim to actual size
            if (tokenCount < maxTokens) {
                int[] result = new int[tokenCount];
                System.arraycopy(tokens, 0, result, 0, tokenCount);
                return result;
            }

            // tokenCount == maxTokens but at cap — return as-is
            return tokens;
        }

        throw new RuntimeException(ErrorCode.JNI_ERROR, "Tokenization failed (exceeded retries) for: " + text);
    }

    /**
     * Detokenize an array of token IDs back to text.
     * 
     * @param tokens Array of token IDs
     * @return Reconstructed text
     * @throws RuntimeException if detokenization fails
     */
    public String detokenize(int[] tokens) throws RuntimeException {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        
        if (tokens == null || tokens.length == 0) {
            return "";
        }
        
        // Allocate buffer for text (estimate: ~4 bytes per token avg)
        byte[] outputBuffer = new byte[tokens.length * 8 + 128];
        
        int bytesWritten = nativeDetokenize(tokens, tokens.length, outputBuffer, outputBuffer.length);
        
        if (bytesWritten < 0) {
            throw new RuntimeException(ErrorCode.JNI_ERROR, "Detokenization failed");
        }
        
        return new String(outputBuffer, 0, bytesWritten, java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * Run a single inference step.
     * Given input tokens, predict the next token and get logits.
     * 
     * @param inputTokens Token IDs to process
     * @return Prediction result with next token and logits
     * @throws RuntimeException if inference fails
     */
    public InferencePrediction infer(int[] inputTokens) throws RuntimeException {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        
        if (inputTokens == null || inputTokens.length == 0) {
            throw new IllegalArgumentException("Input tokens cannot be null or empty");
        }
        
        if (vocabSize <= 0) {
            throw new IllegalStateException("Invalid vocabulary size");
        }
        
        // Allocate logits buffer
        float[] logits = new float[vocabSize];
        
        // Run inference
        int nextToken = nativeInfer(inputTokens, inputTokens.length, logits, vocabSize);
        
        if (nextToken < 0) {
            throw new RuntimeException(ErrorCode.JNI_ERROR, "Inference failed");
        }
        
        return new InferencePrediction(nextToken, logits);
    }

    /**
     * Compute perplexity of a token sequence.
     * Lower perplexity indicates better predictions.
     * 
     * @param tokens Token sequence
     * @return Average negative log likelihood (perplexity metric)
     * @throws RuntimeException if computation fails
     */
    public double computePerplexity(int[] tokens) throws RuntimeException {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        
        if (tokens == null || tokens.length < 2) {
            throw new IllegalArgumentException("Need at least 2 tokens for perplexity");
        }
        
        double perplexity = nativeComputePerplexity(tokens, tokens.length);
        
        if (perplexity < 0) {
            throw new RuntimeException(ErrorCode.JNI_ERROR, "Perplexity computation failed");
        }
        
        return perplexity;
    }

    /**
     * Get vocabulary size.
     */
    public int getVocabSize() {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return vocabSize;
    }

    /**
     * Check if engine is initialized.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get EOS token id as reported by native engine. May be -1 if unavailable.
     */
    public int getEosToken() {
        if (!initialized) throw new IllegalStateException("Engine not initialized");
        return eosToken;
    }

    /**
     * Cleanup and release resources.
     */
    public void shutdown() {
        if (initialized) {
            nativeShutdown();
            initialized = false;
            vocabSize = -1;
        }
    }

    /**
     * Result of a single inference step.
     */
    public static class InferencePrediction {
        public final int nextToken;
        public final float[] logits;
        
        public InferencePrediction(int nextToken, float[] logits) {
            this.nextToken = nextToken;
            this.logits = logits.clone();
        }
        
        /**
         * Get top-k predictions.
         */
        public int[] getTopTokens(int k) {
            if (k <= 0) throw new IllegalArgumentException("k must be positive");
            
            // Find indices of top k logits
            int[] indices = new int[Math.min(k, logits.length)];
            float[] values = new float[indices.length];
            
            // Initialize with first k values
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
                values[i] = logits[i];
            }
            
            // Simple selection sort for top k
            for (int i = 0; i < indices.length; i++) {
                int maxIdx = i;
                for (int j = i + 1; j < logits.length; j++) {
                    if (logits[j] > values[maxIdx]) {
                        maxIdx = j;
                    }
                }
                if (maxIdx != i) {
                    int tmpIdx = indices[i];
                    float tmpVal = values[i];
                    indices[i] = indices[maxIdx];
                    values[i] = values[maxIdx];
                    // Note: indices[maxIdx] doesn't change, we just care about top k
                }
            }
            
            return indices;
        }
        
        /**
         * Get confidence of top prediction.
         * Uses softmax approximation (just comparing logits).
         */
        public double getConfidence() {
            if (logits.length == 0) return 0.0;
            
            float max = logits[0];
            for (float v : logits) {
                if (v > max) max = v;
            }
            
            // Gap between top prediction and second best
            float topLogit = logits[nextToken];
            float gap = topLogit - (max == topLogit ? getSecondMax() : max);
            
            // Normalize to [0, 1]
            return Math.min(1.0, Math.max(0.0, 0.5 + gap / 10.0));
        }
        
        private float getSecondMax() {
            float max1 = Float.NEGATIVE_INFINITY;
            float max2 = Float.NEGATIVE_INFINITY;
            for (float v : logits) {
                if (v > max1) {
                    max2 = max1;
                    max1 = v;
                } else if (v > max2) {
                    max2 = v;
                }
            }
            return max2;
        }
        
        @Override
        public String toString() {
            return String.format(
                "InferencePrediction{nextToken=%d, confidence=%.2f%%}",
                nextToken, getConfidence() * 100
            );
        }
    }

    // ============ Native JNI Calls ============

    private native void nativeInitialize();
    private native void nativeShutdown();
    
    private native int nativeTokenize(String text, int[] outputTokens, int maxTokens);
    private native int nativeDetokenize(int[] tokens, int tokenCount, byte[] outputText, int maxLen);
    
    private native int nativeGetVocabSize();
    private native int nativeGetEosToken();
    private native int nativeInfer(int[] inputTokens, int tokenCount, float[] logitsOut, int maxLogits);
    private native double nativeComputePerplexity(int[] tokens, int tokenCount);

    static {
        try {
            System.loadLibrary("adaptive_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Warning: Failed to load adaptive_engine native library: " + e.getMessage());
        }
    }
}
