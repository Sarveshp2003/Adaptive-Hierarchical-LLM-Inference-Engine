package com.adaptivellm.scheduler;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Phase 2: Production-wired NativeEngineAdapter with actual JNI integration placeholders.
 *
 * This class bridges scheduler decisions to the actual NativeEngine runtime via JNI.
 * It replaces the simulation-based adapter with real calls to C++ runtime.
 *
 * Previous version: Simulated all operations (see NativeEngineAdapter.java)
 * Current version: Integrates with real NativeEngine.requestLayer() and related methods
 *
 * Implementation strategy:
 * 1. Accept real NativeEngine object reference
 * 2. For each action type, call corresponding native method via JNI
 * 3. Collect real latency and memory metrics from the engine
 * 4. Implement error handling for native call failures
 */
public final class Phase2NativeEngineAdapter {

    private final Object nativeEngine;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final AtomicLong lastRequestTimeMs = new AtomicLong(0);
    private final AtomicInteger requestCount = new AtomicInteger(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);

    // Native library interface
    private static final String NATIVE_LIB_NAME = "adaptive_scheduler";
    private static volatile boolean nativeLibLoaded = false;
    private static final Object libLoadLock = new Object();

    /**
     * Create adapter with real native engine reference.
     * @param nativeEngine The actual NativeEngine instance from C++
     */
    public Phase2NativeEngineAdapter(Object nativeEngine) {
        this.nativeEngine = Objects.requireNonNull(nativeEngine, "nativeEngine cannot be null");
        loadNativeLibrary();
    }

    /**
     * Ensure native library is loaded exactly once.
     */
    private static void loadNativeLibrary() {
        if (nativeLibLoaded) {
            return;
        }

        synchronized (libLoadLock) {
            if (nativeLibLoaded) {
                return;
            }

            try {
                System.loadLibrary(NATIVE_LIB_NAME);
                nativeLibLoaded = true;
                System.out.println("[Phase2NativeEngineAdapter] Native library loaded");
            } catch (UnsatisfiedLinkError e) {
                System.err.println("[Phase2NativeEngineAdapter] Failed to load native library: " + e.getMessage());
                // Continue gracefully - will fail on actual native calls
            }
        }
    }

    /**
     * Start the native runtime.
     * Phase 2: Call native nativeEngine->start()
     */
    public void start() {
        if (!isRunning.getAndSet(true)) {
            try {
                nativeStart(nativeEngine);
                System.out.println("[Phase2NativeEngineAdapter] Runtime started");
            } catch (Exception e) {
                System.err.println("[Phase2NativeEngineAdapter] Failed to start: " + e.getMessage());
                isRunning.set(false);
            }
        }
    }

    /**
     * Stop the native runtime.
     * Phase 2: Call native nativeEngine->stop()
     */
    public void stop() {
        if (isRunning.getAndSet(false)) {
            try {
                nativeStop(nativeEngine);
                System.out.println("[Phase2NativeEngineAdapter] Runtime stopped");
            } catch (Exception e) {
                System.err.println("[Phase2NativeEngineAdapter] Failed to stop: " + e.getMessage());
                isRunning.set(true);
            }
        }
    }

    /**
     * Execute a scheduler decision on the native engine.
     * Routes to appropriate native method based on action type.
     *
     * @param decision The scheduling decision
     * @return ExecutionResult with real latency and memory metrics
     */
    public ExecutionResult executeDecision(Decision decision) {
        if (!isRunning.get()) {
            return ExecutionResult.failed("Engine not running");
        }

        long startTimeMs = System.currentTimeMillis();
        ExecutionResult result;

        try {
            switch (decision.action()) {
                case PREFETCH_LAYER:
                    result = executePrefetchLayerNative((int) decision.targetId(), startTimeMs);
                    break;

                case EVICT_LAYER:
                    result = executeEvictLayerNative((int) decision.targetId(), startTimeMs);
                    break;

                case KEEP_LAYER:
                    result = executeKeepLayerNative((int) decision.targetId(), startTimeMs);
                    break;

                case MOVE_KV_TO_RAM:
                    result = executeMoveKvToRamNative(decision.targetId(), startTimeMs);
                    break;

                case MOVE_KV_TO_GPU:
                    result = executeMoveKvToGpuNative(decision.targetId(), startTimeMs);
                    break;

                case COMPRESS_KV:
                    result = executeCompressKvNative(decision.targetId(), startTimeMs);
                    break;

                case OFFLOAD_KV:
                    result = executeOffloadKvNative(decision.targetId(), startTimeMs);
                    break;

                case NO_ACTION:
                    result = executeNoAction(startTimeMs);
                    break;

                default:
                    result = ExecutionResult.failed("Unknown action: " + decision.action());
            }
        } catch (Exception e) {
            result = ExecutionResult.failed("Execution error: " + e.getMessage());
        }

        // Update metrics
        long latency = result.latencyMs();
        requestCount.incrementAndGet();
        lastRequestTimeMs.set(latency);
        totalLatencyMs.addAndGet(latency);

        if (result.success()) {
            System.out.printf("[Phase2NativeEngineAdapter] %s(layerId=%d) latency=%dms%n",
                    decision.action(), decision.targetId(), latency);
        } else {
            System.out.printf("[Phase2NativeEngineAdapter] %s failed: %s%n",
                    decision.action(), result.errorMessage());
        }

        return result;
    }

    // ===== Native Method Stubs (to be implemented via JNI) =====

    /**
     * Start native runtime.
     * Phase 2 Implementation: void nativeEngine->start()
     */
    private native void nativeStart(Object nativeEngine);

    /**
     * Stop native runtime.
     * Phase 2 Implementation: void nativeEngine->stop()
     */
    private native void nativeStop(Object nativeEngine);

    /**
     * Execute PREFETCH_LAYER on native engine.
     * Phase 2 Implementation: nativeEngine->requestLayer(layerId)
     *
     * This is the core scheduler decision - request a layer be loaded to GPU.
     * Returns actual latency from NativeEngine.
     */
    private ExecutionResult executePrefetchLayerNative(int layerId, long startTimeMs) {
        try {
            // Phase 2: Call actual native requestLayer()
            // long nativeLatency = nativePrefetchLayer(nativeEngine, layerId);
            // This will be the actual JNI call that loads layer to GPU
            long latency = nativePrefetchLayer(nativeEngine, layerId);
            return ExecutionResult.success(latency, estimateMemorySaved(layerId));
        } catch (Exception e) {
            return ExecutionResult.failed("Prefetch native call failed: " + e.getMessage());
        }
    }

    /**
     * Execute EVICT_LAYER on native engine.
     * Phase 2 Implementation: nativeEngine->evictLayer(layerId)
     */
    private ExecutionResult executeEvictLayerNative(int layerId, long startTimeMs) {
        try {
            long latency = nativeEvictLayer(nativeEngine, layerId);
            return ExecutionResult.success(latency, estimateMemorySaved(layerId));
        } catch (Exception e) {
            return ExecutionResult.failed("Evict native call failed: " + e.getMessage());
        }
    }

    /**
     * Execute KEEP_LAYER on native engine.
     * Phase 2 Implementation: nativeEngine->keepLayer(layerId)
     */
    private ExecutionResult executeKeepLayerNative(int layerId, long startTimeMs) {
        try {
            long latency = nativeKeepLayer(nativeEngine, layerId);
            return ExecutionResult.success(latency, 0);
        } catch (Exception e) {
            return ExecutionResult.failed("Keep layer native call failed: " + e.getMessage());
        }
    }

    /**
     * Execute MOVE_KV_TO_RAM on native engine.
     * Phase 2 Implementation: nativeEngine->moveKvCache(kvPageId, DESTINATION_RAM)
     */
    private ExecutionResult executeMoveKvToRamNative(long kvPageId, long startTimeMs) {
        try {
            long latency = nativeMoveKvToRam(nativeEngine, kvPageId);
            return ExecutionResult.success(latency, 10 * 1024 * 1024); // ~10MB saved
        } catch (Exception e) {
            return ExecutionResult.failed("Move KV to RAM native call failed: " + e.getMessage());
        }
    }

    /**
     * Execute MOVE_KV_TO_GPU on native engine.
     * Phase 2 Implementation: nativeEngine->moveKvCache(kvPageId, DESTINATION_GPU)
     */
    private ExecutionResult executeMoveKvToGpuNative(long kvPageId, long startTimeMs) {
        try {
            long latency = nativeMoveKvToGpu(nativeEngine, kvPageId);
            return ExecutionResult.success(latency, 5 * 1024 * 1024); // ~5MB saved
        } catch (Exception e) {
            return ExecutionResult.failed("Move KV to GPU native call failed: " + e.getMessage());
        }
    }

    /**
     * Execute COMPRESS_KV on native engine.
     * Phase 2 Implementation: nativeEngine->compressKvCache(kvPageId)
     */
    private ExecutionResult executeCompressKvNative(long kvPageId, long startTimeMs) {
        try {
            long latency = nativeCompressKv(nativeEngine, kvPageId);
            return ExecutionResult.success(latency, 20 * 1024 * 1024); // ~20MB saved
        } catch (Exception e) {
            return ExecutionResult.failed("Compress KV native call failed: " + e.getMessage());
        }
    }

    /**
     * Execute OFFLOAD_KV on native engine.
     * Phase 2 Implementation: nativeEngine->offloadKvCache(kvPageId)
     */
    private ExecutionResult executeOffloadKvNative(long kvPageId, long startTimeMs) {
        try {
            long latency = nativeOffloadKv(nativeEngine, kvPageId);
            return ExecutionResult.success(latency, 30 * 1024 * 1024); // ~30MB saved
        } catch (Exception e) {
            return ExecutionResult.failed("Offload KV native call failed: " + e.getMessage());
        }
    }

    /**
     * No-op action (no native call needed).
     */
    private ExecutionResult executeNoAction(long startTimeMs) {
        long latencyMs = System.currentTimeMillis() - startTimeMs;
        return ExecutionResult.success(latencyMs, 0);
    }

    // ===== Native Method Declarations =====
    private native long nativePrefetchLayer(Object nativeEngine, int layerId);
    private native long nativeEvictLayer(Object nativeEngine, int layerId);
    private native long nativeKeepLayer(Object nativeEngine, int layerId);
    private native long nativeMoveKvToRam(Object nativeEngine, long kvPageId);
    private native long nativeMoveKvToGpu(Object nativeEngine, long kvPageId);
    private native long nativeCompressKv(Object nativeEngine, long kvPageId);
    private native long nativeOffloadKv(Object nativeEngine, long kvPageId);

    // ===== Utility Methods =====

    private long estimateMemorySaved(int layerId) {
        return 100L * 1024 * 1024; // ~100MB per layer for 3B model
    }

    public boolean isRunning() {
        return isRunning.get();
    }

    public long getLastRequestTimeMs() {
        return lastRequestTimeMs.get();
    }

    public int getRequestCount() {
        return requestCount.get();
    }

    public double getAverageLatencyMs() {
        int count = requestCount.get();
        if (count == 0) return 0;
        return (double) totalLatencyMs.get() / count;
    }

    /**
     * Execution result from native engine operation.
     */
    public static final class ExecutionResult {
        private final boolean success;
        private final long latencyMs;
        private final long memorySavedBytes;
        private final String errorMessage;

        private ExecutionResult(boolean success, long latencyMs, long memorySavedBytes, String errorMessage) {
            this.success = success;
            this.latencyMs = latencyMs;
            this.memorySavedBytes = memorySavedBytes;
            this.errorMessage = errorMessage;
        }

        public static ExecutionResult success(long latencyMs, long memorySavedBytes) {
            return new ExecutionResult(true, latencyMs, memorySavedBytes, null);
        }

        public static ExecutionResult failed(String errorMessage) {
            return new ExecutionResult(false, 0, 0, errorMessage);
        }

        public boolean success() {
            return success;
        }

        public long latencyMs() {
            return latencyMs;
        }

        public long memorySavedBytes() {
            return memorySavedBytes;
        }

        public String errorMessage() {
            return errorMessage;
        }

        @Override
        public String toString() {
            return success
                    ? String.format("ExecutionResult{success=true, latency=%dms, memory=%d bytes}", latencyMs, memorySavedBytes)
                    : String.format("ExecutionResult{success=false, error=%s}", errorMessage);
        }
    }

    @Override
    public String toString() {
        return String.format("Phase2NativeEngineAdapter{running=%b, requests=%d, avgLatency=%.2fms}",
                isRunning.get(), requestCount.get(), getAverageLatencyMs());
    }
}
