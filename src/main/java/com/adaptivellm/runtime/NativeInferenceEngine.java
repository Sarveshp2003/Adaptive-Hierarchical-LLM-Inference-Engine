package com.adaptivellm.runtime;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.regex.Pattern;

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

    private static final String NATIVE_LIB_NAME = "adaptive_engine";
    private static volatile boolean nativeLibraryLoaded = false;
    
    private volatile boolean initialized = false;
    private volatile int vocabSize = -1;
    private volatile int eosToken = -1;

    static {
        loadNativeLibrary();
    }

    /**
     * Try loading common dependency DLLs from the same directory.
     */
    private static void tryLoadDependencies(Path dllDir) {
        String[] dependencies = { "llama.dll", "adaptive_engine_llama.dll" };
        for (String dep : dependencies) {
            Path depPath = dllDir.resolve(dep);
            if (Files.isRegularFile(depPath)) {
                try {
                    System.load(depPath.toAbsolutePath().toString());
                    System.out.println("[NativeInferenceEngine] debug: loaded dependency " + depPath.toAbsolutePath());
                } catch (UnsatisfiedLinkError e) {
                    System.out.println("[NativeInferenceEngine] debug: could not load dependency " + depPath.toAbsolutePath() + ": " + e.getMessage());
                }
            }
        }
    }

    private static void loadNativeLibrary() {
        if (nativeLibraryLoaded) {
            return;
        }

        try {
            System.loadLibrary(NATIVE_LIB_NAME);
            nativeLibraryLoaded = true;
            return;
        } catch (UnsatisfiedLinkError ignored) {
            // Fall back to a few common build output locations.
        }

        String libraryName = System.mapLibraryName(NATIVE_LIB_NAME);
        Path userDir = Paths.get(System.getProperty("user.dir", "")).toAbsolutePath().normalize();
        // Try Release directory first (most likely to have all dependencies)
        Path[] candidateRoots = new Path[] {
                userDir.resolve("native-engine").resolve("llama_wrapper").resolve("build").resolve("lib").resolve("Release"),
                userDir.resolve("native-engine").resolve("llama_wrapper").resolve("build").resolve("bin").resolve("Release"),
                userDir.resolve("native-engine").resolve("llama_wrapper").resolve("build").resolve("lib"),
                userDir.resolve("native-engine").resolve("llama_wrapper").resolve("build").resolve("bin"),
                userDir.resolve("native-engine").resolve("llama_wrapper").resolve("build"),
                userDir,
                userDir.resolve("lib"),
                userDir.resolve("target").resolve("classes")
        };

        String javaLibraryPath = System.getProperty("java.library.path", "");
        System.out.println("[NativeInferenceEngine] debug: user.dir=" + System.getProperty("user.dir") + " java.library.path=" + javaLibraryPath + " ADAPTIVELLM_NATIVE_LIB=" + System.getenv("ADAPTIVELLM_NATIVE_LIB"));
        if (!javaLibraryPath.isEmpty()) {
            for (String entry : javaLibraryPath.split(Pattern.quote(File.pathSeparator))) {
                if (entry == null || entry.isBlank()) {
                    continue;
                }
                candidateRoots = Arrays.copyOf(candidateRoots, candidateRoots.length + 1);
                candidateRoots[candidateRoots.length - 1] = Paths.get(entry).toAbsolutePath().normalize();
            }
        }

        for (Path root : candidateRoots) {
            if (root == null) {
                continue;
            }
            Path candidate = root.resolve(libraryName);
            System.out.println("[NativeInferenceEngine] debug: checking candidate=" + candidate.toAbsolutePath());
            if (Files.isRegularFile(candidate)) {
                System.out.println("[NativeInferenceEngine] debug: found file at " + candidate.toAbsolutePath());
                try {
                    // Try loading dependencies first (llama.dll, etc.)
                    tryLoadDependencies(root);
                    System.load(candidate.toAbsolutePath().toString());
                    nativeLibraryLoaded = true;
                    System.out.println("[NativeInferenceEngine] debug: System.load succeeded for " + candidate.toAbsolutePath());
                    return;
                } catch (UnsatisfiedLinkError ignored2) {
                    System.err.println("[NativeInferenceEngine] debug: System.load failed for " + candidate.toAbsolutePath() + " -> " + ignored2.getMessage());
                    // Try the next candidate.
                }
            }
        }

        // Final fallback: allow explicit absolute path via ADAPTIVELLM_NATIVE_LIB env var
        String explicit = System.getenv("ADAPTIVELLM_NATIVE_LIB");
        if (explicit != null && !explicit.isBlank()) {
            try {
                Path p = Paths.get(explicit).toAbsolutePath().normalize();
                if (Files.isRegularFile(p)) {
                    try {
                        System.load(p.toString());
                        nativeLibraryLoaded = true;
                        return;
                    } catch (UnsatisfiedLinkError ignored3) {
                        // fallthrough to error
                    }
                }
            } catch (Exception ex) {
                // ignore and fall through to error below
            }
        }

        throw new UnsatisfiedLinkError("Unable to locate native library '" + libraryName + "' in java.library.path or common build directories");
    }

    /**
     * Initialize the native inference engine.
     * Must be called before any inference operations.
     */
    public void initialize() {
        initialize(true);
    }

    public void initialize(boolean enableGpu) {
        if (!initialized) {
            nativeInitialize(enableGpu);
            this.vocabSize = nativeGetVocabSize();
            this.eosToken = nativeGetEosToken();
            this.initialized = true;
            System.out.println("[NativeInferenceEngine] Initialized with vocab_size=" + vocabSize + " eos_token=" + eosToken + " gpu_enabled=" + enableGpu);
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
        final int MAX_CAP = 262144; // allow much larger temporary buffers for pathological cases
        int attempts = 0;

        while (attempts < 12) {
            int[] tokens = new int[maxTokens];
            int tokenCount = nativeTokenize(text, tokens, maxTokens);

            if (tokenCount < 0) {
                // Native returned negative: -N means N tokens required. Try to resize exactly if possible.
                int required = -tokenCount;
                System.out.println("[NativeInferenceEngine] tokenize: native returned negative (" + tokenCount + ") => required=" + required + " (buffer=" + maxTokens + ")");
                // Defensive: if native reports non-positive or a required size <= current buffer, treat as error.
                if (required <= 0 || required <= maxTokens) {
                    throw new RuntimeException(ErrorCode.JNI_ERROR, "Tokenization failed (native returned " + tokenCount + ") for: " + text);
                }
                if (required <= MAX_CAP) {
                    maxTokens = Math.min(required + 16, MAX_CAP); // add slack
                    attempts++;
                    continue;
                }

                // Otherwise fall back to exponential growth
                if (maxTokens < MAX_CAP) {
                    maxTokens = Math.min(maxTokens * 2, MAX_CAP);
                    attempts++;
                    continue;
                }

                throw new RuntimeException(ErrorCode.JNI_ERROR, "Tokenization failed (native required=" + required + ") for: " + text);
            }

            // If tokenCount equals or exceeds buffer, assume truncation and retry with larger buffer
            if (tokenCount >= maxTokens && maxTokens < MAX_CAP) {
                System.out.println("[NativeInferenceEngine] tokenize: tokenCount >= buffer (" + tokenCount + "), growing buffer from " + maxTokens);
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
        
        // Default behavior: return native-picked token (greedy) plus logits
        InferencePrediction basePred = new InferencePrediction(nextToken, logits);

        return basePred;
    }

    /**
     * Run a single inference step and sample next token using decoding parameters.
     * If temperature == 0, behaves deterministically (argmax).
     */
    public InferencePrediction inferWithSampling(int[] inputTokens, double temperature, int topK, double topP, int[] recentTokens, double repetitionPenalty) throws RuntimeException {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        if (inputTokens == null || inputTokens.length == 0) {
            throw new IllegalArgumentException("Input tokens cannot be null or empty");
        }
        if (vocabSize <= 0) {
            throw new IllegalStateException("Invalid vocabulary size");
        }

        // Get logits from native inference
        float[] logits = new float[vocabSize];
        int nativeNext = nativeInfer(inputTokens, inputTokens.length, logits, vocabSize);
        if (nativeNext < 0) {
            throw new RuntimeException(ErrorCode.JNI_ERROR, "Inference failed");
        }

        int sampled = sampleFromLogits(logits, temperature, topK, topP, recentTokens, repetitionPenalty);
        return new InferencePrediction(sampled, logits);
    }

    /**
     * Sample a token id from logits using temperature, top-k, top-p (nucleus), and repetition penalty.
     * This is implemented in Java so JNI/native rebuild is not required.
     */
    public static int sampleFromLogits(float[] logits, double temperature, int topK, double topP, int[] recentTokens, double repetitionPenalty) {
        if (logits == null || logits.length == 0) return -1;
        int vocab = logits.length;

        // If temperature == 0 → deterministic argmax
        if (temperature <= 0.0) {
            int argmax = 0;
            float max = logits[0];
            for (int i = 1; i < vocab; ++i) {
                if (logits[i] > max) { max = logits[i]; argmax = i; }
            }
            return argmax;
        }

        // Copy logits to a double array for stable processing
        double[] scores = new double[vocab];
        for (int i = 0; i < vocab; ++i) scores[i] = logits[i];

        // Apply repetition penalty: for each recent token, penalize its score
        if (recentTokens != null && recentTokens.length > 0 && repetitionPenalty > 1.0) {
            for (int rt : recentTokens) {
                if (rt >= 0 && rt < vocab) {
                    // Typical penalty: divide score for repeating token (logit-space)
                    scores[rt] = scores[rt] / repetitionPenalty;
                }
            }
        }

        // Temperature scaling (divide logits by temperature)
        double invTemp = 1.0 / temperature;
        for (int i = 0; i < vocab; ++i) scores[i] = scores[i] * invTemp;

        // Apply top-k: keep only topK scores, others -> -Inf
        int k = Math.max(1, Math.min(topK, vocab));
        if (k < vocab) {
            // find threshold via partial selection
            double[] copy = Arrays.copyOf(scores, vocab);
            java.util.Arrays.sort(copy); // ascending
            double kth = copy[vocab - k];
            for (int i = 0; i < vocab; ++i) {
                if (scores[i] < kth) scores[i] = Double.NEGATIVE_INFINITY;
            }
        }

        // Convert scores to probabilities using softmax, but support top-p by zeroing small probs
        double maxScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; ++i) if (scores[i] > maxScore) maxScore = scores[i];
        double sumExp = 0.0;
        double[] expScores = new double[vocab];
        for (int i = 0; i < vocab; ++i) {
            if (scores[i] == Double.NEGATIVE_INFINITY) { expScores[i] = 0.0; continue; }
            double v = Math.exp(scores[i] - maxScore);
            expScores[i] = v;
            sumExp += v;
        }
        if (sumExp <= 0.0) {
            // fallback to argmax
            int argmax = 0; double best = scores[0];
            for (int i = 1; i < vocab; ++i) { if (scores[i] > best) { best = scores[i]; argmax = i; } }
            return argmax;
        }

        // Normalize to probabilities
        double[] probs = new double[vocab];
        for (int i = 0; i < vocab; ++i) probs[i] = expScores[i] / sumExp;

        // Apply top-p nucleus sampling: keep smallest set with cumulative prob >= topP
        if (topP > 0.0 && topP < 1.0) {
            // create array of indices sorted by prob desc
            Integer[] idx = new Integer[vocab];
            for (int i = 0; i < vocab; ++i) idx[i] = i;
            java.util.Arrays.sort(idx, (a, b) -> Double.compare(probs[b], probs[a]));
            double cum = 0.0;
            boolean[] keep = new boolean[vocab];
            for (int j = 0; j < vocab; ++j) {
                int i = idx[j];
                cum += probs[i];
                keep[i] = true;
                if (cum >= topP) break;
            }
            // zero out the rest
            double newSum = 0.0;
            for (int i = 0; i < vocab; ++i) {
                if (!keep[i]) probs[i] = 0.0; else newSum += probs[i];
            }
            if (newSum <= 0.0) {
                // fallback argmax
                int argmax = 0; double best = probs[0];
                for (int i = 1; i < vocab; ++i) { if (probs[i] > best) { best = probs[i]; argmax = i; } }
                return argmax;
            }
            // renormalize
            for (int i = 0; i < vocab; ++i) probs[i] /= newSum;
        }

        // Sample from probs
        double r = java.util.concurrent.ThreadLocalRandom.current().nextDouble();
        double acc = 0.0;
        for (int i = 0; i < vocab; ++i) {
            acc += probs[i];
            if (r <= acc) return i;
        }
        // numerical fallback
        for (int i = vocab - 1; i >= 0; --i) if (probs[i] > 0.0) return i;
        // last resort
        int argmax = 0; double best = probs[0];
        for (int i = 1; i < vocab; ++i) { if (probs[i] > best) { best = probs[i]; argmax = i; } }
        return argmax;
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
     * Retrieve the tokenizer.chat_template string from the native model metadata (if available).
     */
    public String getChatTemplate() {
        if (!initialized) throw new IllegalStateException("Engine not initialized");
        try {
            String tmpl = nativeGetChatTemplate();
            if (tmpl != null && !tmpl.isBlank()) return tmpl;
        } catch (UnsatisfiedLinkError | NoSuchMethodError e) {
            // native getter not available in this build; fall through to env fallback
        }
        String env = System.getenv("ADAPTIVELLM_CHAT_TEMPLATE");
        return env == null ? "" : env;
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

    public long prefetchLayer(int layerId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativePrefetchLayer(layerId);
    }

    public long evictLayer(int layerId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeEvictLayer(layerId);
    }

    public long keepLayer(int layerId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeKeepLayer(layerId);
    }

    public long moveKvToRam(long kvPageId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeMoveKvToRam(kvPageId);
    }

    public long moveKvToGpu(long kvPageId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeMoveKvToGpu(kvPageId);
    }

    public long compressKv(long kvPageId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeCompressKv(kvPageId);
    }

    public long offloadKv(long kvPageId) {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeOffloadKv(kvPageId);
    }

    public int getCurrentLayer() {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeGetCurrentLayer();
    }

    public long getGpuMemory() {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeGetGpuMemory();
    }

    public int getKvPages() {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeGetKvPages();
    }

    public int getCachedLayers() {
        if (!initialized) {
            throw new IllegalStateException("Engine not initialized");
        }
        return nativeGetCachedLayers();
    }

    private native void nativeInitialize();
    private native void nativeInitialize(boolean enableGpu);
    private native void nativeShutdown();
    
    private native int nativeTokenize(String text, int[] outputTokens, int maxTokens);
    private native int nativeDetokenize(int[] tokens, int tokenCount, byte[] outputText, int maxLen);
    
    private native int nativeGetVocabSize();
    private native int nativeGetEosToken();
    private native int nativeInfer(int[] inputTokens, int tokenCount, float[] logitsOut, int maxLogits);
    private native double nativeComputePerplexity(int[] tokens, int tokenCount);
    private native long nativePrefetchLayer(int layerId);
    
    // Optional native getter for model chat template (may be unavailable if native not rebuilt)
    private native String nativeGetChatTemplate();
    private native long nativeEvictLayer(int layerId);
    private native long nativeKeepLayer(int layerId);
    private native long nativeMoveKvToRam(long kvPageId);
    private native long nativeMoveKvToGpu(long kvPageId);
    private native long nativeCompressKv(long kvPageId);
    private native long nativeOffloadKv(long kvPageId);
    private native int nativeGetCurrentLayer();
    private native long nativeGetGpuMemory();
    private native int nativeGetKvPages();
    private native int nativeGetCachedLayers();
}
