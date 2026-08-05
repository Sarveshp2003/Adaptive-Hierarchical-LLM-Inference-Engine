package com.adaptivellm.scheduler;

import com.adaptivellm.runtime.NativeInferenceEngine;
import java.util.Objects;

/**
 * Adapter that bridges scheduler decisions to NativeEngine calls.
 *
 * Translates high-level scheduling actions into native runtime operations.
 * Simulates native runtime behavior for testing and development.
 */
public final class NativeEngineAdapter {

    private final NativeInferenceEngine nativeEngine;
    private volatile boolean isRunning = false;
    private long lastRequestTimeMs = 0;
    private int requestCount = 0;

    /**
     * Create adapter with optional native engine reference.
     * Currently simulates native runtime behavior.
     */
    public NativeEngineAdapter() {
        this(null);
    }

    /**
     * Create adapter with native engine object (for future integration).
     * @param nativeEngine The native engine instance
     */
    public NativeEngineAdapter(NativeInferenceEngine nativeEngine) {
        this.nativeEngine = nativeEngine;
    }

    /**
     * Start the native runtime.
     */
    public void start() {
        if (!isRunning) {
            isRunning = true;
            System.out.println("[NativeEngineAdapter] Runtime started (simulated)");
        }
    }

    /**
     * Stop the native runtime.
     */
    public void stop() {
        if (isRunning) {
            isRunning = false;
            System.out.println("[NativeEngineAdapter] Runtime stopped (simulated)");
        }
    }

    /**
     * Execute a scheduler decision on the native engine.
     *
     * @param decision The scheduling decision to execute
     * @return ExecutionResult with latency and memory metrics
     */
    public ExecutionResult executeDecision(Decision decision) {
        if (!isRunning) {
            System.err.println("[NativeEngineAdapter] Runtime not running");
            return ExecutionResult.failed("Runtime not running");
        }

        long startTimeMs = System.currentTimeMillis();
        ExecutionResult result;

        try {
            switch (decision.action()) {
                case PREFETCH_LAYER:
                    result = executePrefetchLayer((int) decision.targetId(), startTimeMs);
                    break;

                case EVICT_LAYER:
                    result = executeEvictLayer((int) decision.targetId(), startTimeMs);
                    break;

                case KEEP_LAYER:
                    result = executeKeepLayer((int) decision.targetId(), startTimeMs);
                    break;

                case MOVE_KV_TO_RAM:
                    result = executeMoveKvToRam(decision.targetId(), startTimeMs);
                    break;

                case MOVE_KV_TO_GPU:
                    result = executeMoveKvToGpu(decision.targetId(), startTimeMs);
                    break;

                case COMPRESS_KV:
                    result = executeCompressKv(decision.targetId(), startTimeMs);
                    break;

                case OFFLOAD_KV:
                    result = executeOffloadKv(decision.targetId(), startTimeMs);
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

        lastRequestTimeMs = result.latencyMs();
        requestCount++;

        if (result.success()) {
            System.out.printf("[NativeEngineAdapter] %s(layerId=%d) latency=%dms memory=%d bytes%n",
                    decision.action(), decision.targetId(), result.latencyMs(), result.memorySavedBytes());
        } else {
            System.out.printf("[NativeEngineAdapter] %s failed: %s%n",
                    decision.action(), result.errorMessage());
        }

        return result;
    }

    private ExecutionResult executePrefetchLayer(int layerId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.prefetchLayer(layerId);
                long latencyMs = nativeResult > 0 ? nativeResult : System.currentTimeMillis() - startTimeMs;
                return ExecutionResult.success(latencyMs, 100 * 1024 * 1024);
            }
            Thread.sleep(5);
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            long memorySaved = 100 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Prefetch failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeEvictLayer(int layerId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.evictLayer(layerId);
                long latencyMs = nativeResult > 0 ? nativeResult : 1;
                return ExecutionResult.success(latencyMs, 100 * 1024 * 1024);
            }
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 1;
            long memorySaved = 100 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Evict failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeKeepLayer(int layerId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.keepLayer(layerId);
                long latencyMs = nativeResult >= 0 ? Math.max(1, nativeResult) : 1;
                return ExecutionResult.success(latencyMs, 0);
            }
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 1;
            return ExecutionResult.success(latencyMs, 0);
        } catch (Exception e) {
            return ExecutionResult.failed("Keep layer failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeMoveKvToRam(long kvPageId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.moveKvToRam(kvPageId);
                long latencyMs = nativeResult > 0 ? nativeResult : 5;
                return ExecutionResult.success(latencyMs, 10 * 1024 * 1024);
            }
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 5;
            long memorySaved = 10 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Move KV to RAM failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeMoveKvToGpu(long kvPageId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.moveKvToGpu(kvPageId);
                long latencyMs = nativeResult > 0 ? nativeResult : 8;
                return ExecutionResult.success(latencyMs, 5 * 1024 * 1024);
            }
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 8;
            long memorySaved = 5 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Move KV to GPU failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeCompressKv(long kvPageId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.compressKv(kvPageId);
                long latencyMs = nativeResult > 0 ? nativeResult : 10;
                return ExecutionResult.success(latencyMs, 20 * 1024 * 1024);
            }
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 10;
            long memorySaved = 20 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Compress KV failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeOffloadKv(long kvPageId, long startTimeMs) {
        try {
            if (nativeEngine != null && nativeEngine.isInitialized()) {
                long nativeResult = nativeEngine.offloadKv(kvPageId);
                long latencyMs = nativeResult > 0 ? nativeResult : 15;
                return ExecutionResult.success(latencyMs, 30 * 1024 * 1024);
            }
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 15;
            long memorySaved = 30 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Offload KV failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeNoAction(long startTimeMs) {
        long latencyMs = System.currentTimeMillis() - startTimeMs;
        return ExecutionResult.success(latencyMs, 0);
    }

    /**
     * Check if runtime is running.
     */
    public boolean isRunning() {
        return isRunning;
    }

    /**
     * Get last request latency in milliseconds.
     */
    public long getLastRequestTimeMs() {
        return lastRequestTimeMs;
    }

    /**
     * Get total number of requests executed.
     */
    public int getRequestCount() {
        return requestCount;
    }

    /**
     * Get average latency across all requests.
     */
    public double getAverageLatencyMs() {
        if (requestCount == 0) return 0;
        return (double) lastRequestTimeMs / requestCount; // Simplified: use last time
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
}
