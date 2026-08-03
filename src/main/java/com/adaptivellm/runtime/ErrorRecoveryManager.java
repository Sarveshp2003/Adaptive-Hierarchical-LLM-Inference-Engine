package com.adaptivellm.runtime;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Comprehensive error recovery and monitoring system.
 * 
 * Features:
 * - Automatic retry with exponential backoff
 * - Circuit breaker pattern
 * - State recovery from checkpoints
 * - Performance metrics collection
 * - Health monitoring
 */
public final class ErrorRecoveryManager {

    private final Map<String, RecoveryHandler> handlers;
    private final PerformanceMetrics metrics;
    private final CircuitBreaker circuitBreaker;
    private volatile boolean recovering;

    public ErrorRecoveryManager() {
        this.handlers = new ConcurrentHashMap<>();
        this.metrics = new PerformanceMetrics();
        this.circuitBreaker = new CircuitBreaker(5, 60_000);  // 5 failures, 60s timeout
        this.recovering = false;
    }

    /**
     * Handle error with recovery strategy
     */
    public <T> T handleError(String operation, Throwable error, 
                            RecoveryStrategy<T> strategy) throws Exception {
        
        // Check circuit breaker
        if (circuitBreaker.isOpen(operation)) {
            throw new IllegalStateException("Circuit breaker open for: " + operation);
        }

        // Record error
        metrics.recordError(operation, error);
        System.err.println("Error in " + operation + ": " + error.getMessage());

        // Get or create handler
        RecoveryHandler handler = handlers.computeIfAbsent(operation, 
            k -> new RecoveryHandler(operation, 3, 1000));  // 3 retries, 1s initial delay

        try {
            recovering = true;
            T result = handler.retryWithBackoff(strategy);
            circuitBreaker.recordSuccess(operation);
            recovering = false;
            return result;
        } catch (Exception e) {
            circuitBreaker.recordFailure(operation);
            recovering = false;
            throw e;
        }
    }

    /**
     * Create checkpoint for state recovery
     */
    public void createCheckpoint(String label, CheckpointData data) {
        try {
            long checkpointId = System.currentTimeMillis();
            String path = "./checkpoints/" + label + "_" + checkpointId + ".ckpt";
            
            // Serialize checkpoint
            nativeCreateCheckpoint(path, data);
            
            System.out.println("Checkpoint created: " + label + " at " + path);
            metrics.recordCheckpoint(label);
        } catch (Exception e) {
            System.err.println("Failed to create checkpoint: " + e.getMessage());
        }
    }

    /**
     * Recover from checkpoint
     */
    public CheckpointData recoverFromCheckpoint(String label) throws Exception {
        System.out.println("Recovering from checkpoint: " + label);
        
        CheckpointData data = nativeRecoverCheckpoint(label);
        metrics.recordRecovery(label);
        
        return data;
    }

    /**
     * Get performance metrics
     */
    public PerformanceMetrics getMetrics() {
        return metrics;
    }

    /**
     * Reset all recovery state
     */
    public void reset() {
        handlers.clear();
        circuitBreaker.reset();
        recovering = false;
        System.out.println("Recovery manager reset");
    }

    // ============ Native Methods ============

    private native void nativeCreateCheckpoint(String path, CheckpointData data);
    private native CheckpointData nativeRecoverCheckpoint(String label);

    // ============ Inner Classes ============

    /**
     * Retry handler with exponential backoff
     */
    private static class RecoveryHandler {
        private final String operation;
        private final int maxRetries;
        private final long initialDelayMs;
        private int failureCount;

        public RecoveryHandler(String operation, int maxRetries, long initialDelayMs) {
            this.operation = operation;
            this.maxRetries = maxRetries;
            this.initialDelayMs = initialDelayMs;
            this.failureCount = 0;
        }

        public <T> T retryWithBackoff(RecoveryStrategy<T> strategy) throws Exception {
            for (int attempt = 0; attempt <= maxRetries; attempt++) {
                try {
                    T result = strategy.execute();
                    failureCount = 0;
                    return result;
                } catch (Exception e) {
                    failureCount++;
                    
                    if (attempt == maxRetries) {
                        throw e;
                    }

                    // Exponential backoff
                    long delayMs = initialDelayMs * (1L << attempt);
                    System.out.println("Retry " + (attempt + 1) + "/" + maxRetries + 
                                      " for " + operation + " after " + delayMs + "ms");
                    
                    Thread.sleep(delayMs);
                }
            }

            throw new Exception("All retries failed for: " + operation);
        }
    }

    /**
     * Circuit breaker for cascading failure prevention
     */
    public static class CircuitBreaker {
        private final Map<String, CircuitState> states = new ConcurrentHashMap<>();
        private final int failureThreshold;
        private final long timeoutMs;

        public CircuitBreaker(int failureThreshold, long timeoutMs) {
            this.failureThreshold = failureThreshold;
            this.timeoutMs = timeoutMs;
        }

        public boolean isOpen(String operation) {
            CircuitState state = states.get(operation);
            if (state == null) return false;

            if (state.isOpen) {
                // Check if timeout expired
                if (System.currentTimeMillis() - state.openedAt > timeoutMs) {
                    state.isOpen = false;
                    state.failureCount = 0;
                    return false;
                }
                return true;
            }

            return false;
        }

        public void recordFailure(String operation) {
            CircuitState state = states.computeIfAbsent(operation, 
                k -> new CircuitState());

            state.failureCount++;
            if (state.failureCount >= failureThreshold) {
                state.isOpen = true;
                state.openedAt = System.currentTimeMillis();
                System.err.println("Circuit breaker OPEN for: " + operation);
            }
        }

        public void recordSuccess(String operation) {
            CircuitState state = states.get(operation);
            if (state != null) {
                state.failureCount = 0;
            }
        }

        public void reset() {
            states.clear();
        }

        private static class CircuitState {
            boolean isOpen = false;
            int failureCount = 0;
            long openedAt = 0;
        }
    }

    /**
     * Performance metrics collector
     */
    public static class PerformanceMetrics {
        private final Map<String, ErrorStats> errorStats = new ConcurrentHashMap<>();
        private final AtomicLong totalErrors = new AtomicLong(0);
        private final AtomicLong totalRecoveries = new AtomicLong(0);
        private final List<Long> checkpointTimes = new CopyOnWriteArrayList<>();

        public void recordError(String operation, Throwable error) {
            totalErrors.incrementAndGet();
            errorStats.computeIfAbsent(operation, k -> new ErrorStats())
                     .recordError(error);
        }

        public void recordCheckpoint(String label) {
            checkpointTimes.add(System.currentTimeMillis());
        }

        public void recordRecovery(String label) {
            totalRecoveries.incrementAndGet();
        }

        public String getSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("ErrorRecovery Metrics:\n");
            sb.append("  Total Errors: ").append(totalErrors).append("\n");
            sb.append("  Total Recoveries: ").append(totalRecoveries).append("\n");
            sb.append("  Checkpoint Count: ").append(checkpointTimes.size()).append("\n");
            
            for (Map.Entry<String, ErrorStats> entry : errorStats.entrySet()) {
                sb.append("  ").append(entry.getKey()).append(": ")
                  .append(entry.getValue()).append("\n");
            }

            return sb.toString();
        }

        private static class ErrorStats {
            int count = 0;
            String lastError = "";
            long lastErrorTime = 0;

            void recordError(Throwable error) {
                count++;
                lastError = error.getMessage();
                lastErrorTime = System.currentTimeMillis();
            }

            @Override
            public String toString() {
                return count + " errors, last: " + lastError;
            }
        }
    }

    /**
     * Recovery strategy interface
     */
    @FunctionalInterface
    public interface RecoveryStrategy<T> {
        T execute() throws Exception;
    }

    /**
     * Checkpoint data
     */
    public static class CheckpointData {
        public Map<String, Object> state = new HashMap<>();
        public long timestamp = System.currentTimeMillis();
        public String label;

        public CheckpointData(String label) {
            this.label = label;
        }

        public void putState(String key, Object value) {
            state.put(key, value);
        }

        public Object getState(String key) {
            return state.get(key);
        }
    }

    static {
        try {
            System.loadLibrary("adaptive_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native library: " + e.getMessage());
        }
    }
}
