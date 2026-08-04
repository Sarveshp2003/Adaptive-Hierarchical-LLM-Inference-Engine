package com.adaptivellm.scheduler;

import java.util.Objects;

/**
 * Adapter that bridges scheduler decisions to NativeEngine calls.
 *
 * Translates high-level scheduling actions into native runtime operations.
 * Simulates native runtime behavior for testing and development.
 */
public final class NativeEngineAdapter {

    private volatile boolean isRunning = false;
    private long lastRequestTimeMs = 0;
    private int requestCount = 0;

    /**
     * Create adapter with optional native engine reference.
     * Currently simulates native runtime behavior.
     */
    public NativeEngineAdapter() {
    }

    /**
     * Create adapter with native engine object (for future integration).
     * @param nativeEngine The native engine instance
     */
    public NativeEngineAdapter(Object nativeEngine) {
        // Accept any object for now - for future integration with real NativeEngine
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
            // Simulate prefetch operation (in production, call nativeEngine.requestLayer(layerId))
            // This would be: nativeEngine.requestLayer(layerId);
            Thread.sleep(5); // Simulate 5ms network latency
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            // Estimate memory saved based on layer size (rough estimate: 100MB per layer for 3B model)
            long memorySaved = 100 * 1024 * 1024;
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Prefetch failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeEvictLayer(int layerId, long startTimeMs) {
        // Eviction is typically faster than prefetch
        try {
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 1; // Minimum 1ms for eviction
            long memorySaved = 100 * 1024 * 1024; // Assume 100MB per layer
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Evict failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeKeepLayer(int layerId, long startTimeMs) {
        // No-op operation
        long latencyMs = System.currentTimeMillis() - startTimeMs;
        if (latencyMs == 0) latencyMs = 1;
        return ExecutionResult.success(latencyMs, 0);
    }

    private ExecutionResult executeMoveKvToRam(long kvPageId, long startTimeMs) {
        // Simulate KV move operation
        try {
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 5; // Typical 5ms for KV move
            // KV cache pages are smaller than layers
            long memorySaved = 10 * 1024 * 1024; // ~10MB per page
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Move KV to RAM failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeMoveKvToGpu(long kvPageId, long startTimeMs) {
        try {
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 8;
            long memorySaved = 5 * 1024 * 1024; // ~5MB GPU overhead
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Move KV to GPU failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeCompressKv(long kvPageId, long startTimeMs) {
        try {
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 10; // Compression takes ~10ms
            long memorySaved = 20 * 1024 * 1024; // ~20% compression = 20MB saved
            return ExecutionResult.success(latencyMs, memorySaved);
        } catch (Exception e) {
            return ExecutionResult.failed("Compress KV failed: " + e.getMessage());
        }
    }

    private ExecutionResult executeOffloadKv(long kvPageId, long startTimeMs) {
        try {
            long latencyMs = System.currentTimeMillis() - startTimeMs;
            if (latencyMs == 0) latencyMs = 15; // Offload to SSD takes ~15ms
            long memorySaved = 30 * 1024 * 1024; // ~30MB freed from main memory
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
