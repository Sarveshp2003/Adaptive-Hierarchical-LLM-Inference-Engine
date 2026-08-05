package com.adaptivellm.scheduler;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Runtime controller that manages the scheduler decision loop.
 *
 * Responsibilities:
 * 1. Periodically evaluate current memory state
 * 2. Get scheduling decisions from AdaptiveScheduler
 * 3. Execute decisions via NativeEngineAdapter
 * 4. Report execution results back to scheduler (closes feedback loop)
 * 5. Trigger online and batch retraining
 */
public final class SchedulerRuntimeController {

    private final AdaptiveScheduler scheduler;
    private final NativeEngineAdapter nativeAdapter;
    private final MemoryStateProvider stateProvider;
    private final PerformanceMetrics metrics;
    private final ModelPersistence modelPersistence;
    private final String modelName;

    private volatile boolean running = false;
    private Thread controlThread = null;
    private final AtomicLong decisionsExecuted = new AtomicLong(0);
    private final AtomicLong successfulDecisions = new AtomicLong(0);

    // Configuration
    private long decisionIntervalMs = 100; // Evaluate every 100ms by default
    private int batchRetrainThreshold = 50; // Retrain after every 50 decisions
    private double latencyImprovementThreshold = 10.0; // ms

    /**
     * Create controller with scheduler and native engine.
     */
    public SchedulerRuntimeController(
            AdaptiveScheduler scheduler,
            NativeEngineAdapter nativeAdapter,
            MemoryStateProvider stateProvider
    ) {
        this(scheduler, nativeAdapter, stateProvider, null, null);
    }

    public SchedulerRuntimeController(
            AdaptiveScheduler scheduler,
            NativeEngineAdapter nativeAdapter,
            MemoryStateProvider stateProvider,
            ModelPersistence modelPersistence,
            String modelName
    ) {
        this.scheduler = Objects.requireNonNull(scheduler);
        this.nativeAdapter = Objects.requireNonNull(nativeAdapter);
        this.stateProvider = Objects.requireNonNull(stateProvider);
        this.metrics = new PerformanceMetrics();
        this.modelPersistence = modelPersistence;
        this.modelName = modelName;
    }

    /**
     * Start the control loop (runs in separate thread).
     */
    public void start() {
        if (running) {
            System.err.println("[SchedulerController] Already running");
            return;
        }

        running = true;
        nativeAdapter.start();
        loadPersistedModelIfAvailable();

        controlThread = new Thread(() -> {
            System.out.println("[SchedulerController] Control loop started");
            long lastRetrainDecisions = 0;

            while (running) {
                try {
                    // 1. Get current memory state
                    MemoryState state = stateProvider.getCurrentState();

                    // 2. Get scheduling decision
                    ScheduledDecision scheduled = scheduler.evaluate(state);
                    Decision decision = scheduled.decision();

                    // 3. Execute decision if action required
                    if (decision.isActionRequired()) {
                        NativeEngineAdapter.ExecutionResult result =
                                nativeAdapter.executeDecision(decision);

                        // 4. Calculate latency improvement (simplified)
                        double latencyImprovement = 0;
                        long memorySaved = 0;

                        if (result.success()) {
                            successfulDecisions.incrementAndGet();
                            latencyImprovement = estimateLatencyImprovement(decision, result);
                            memorySaved = result.memorySavedBytes();
                        }

                        // 5. Report result back to scheduler (closes feedback loop!)
                        scheduler.reportResult(scheduled, latencyImprovement, memorySaved);

                        // Track metrics
                        metrics.recordDecision(decision, result, latencyImprovement);
                    }

                    decisionsExecuted.incrementAndGet();

                    // 6. Trigger batch retraining if threshold exceeded
                    if (decisionsExecuted.get() - lastRetrainDecisions >= batchRetrainThreshold) {
                        scheduler.retrainOnCollectedResults(10, true);
                        lastRetrainDecisions = decisionsExecuted.get();
                        System.out.printf("[SchedulerController] Batch retrain triggered at %d decisions%n",
                                decisionsExecuted.get());
                    }

                    // Sleep before next decision
                    Thread.sleep(decisionIntervalMs);

                } catch (InterruptedException e) {
                    if (running) {
                        System.err.println("[SchedulerController] Interrupted: " + e.getMessage());
                    }
                    break;
                } catch (Exception e) {
                    System.err.println("[SchedulerController] Error in control loop: " + e.getMessage());
                    e.printStackTrace();
                }
            }

            System.out.println("[SchedulerController] Control loop stopped");
        }, "SchedulerControlThread");

        controlThread.setDaemon(false);
        controlThread.start();
    }

    /**
     * Load a persisted model into the scheduler if configured.
     */
    public void loadPersistedModelIfAvailable() {
        if (modelPersistence == null) {
            return;
        }

        try {
            if (modelName != null && !modelName.isBlank()) {
                scheduler.reloadModelFromDisk(modelName, modelPersistence);
            } else {
                scheduler.reloadLatestModelFromDisk(modelPersistence);
            }
        } catch (Exception e) {
            System.err.println("[SchedulerController] Could not load persisted model: " + e.getMessage());
        }
    }

    /**
     * Stop the control loop.
     */
    public void stop() {
        if (!running) {
            System.err.println("[SchedulerController] Not running");
            return;
        }

        running = false;

        if (controlThread != null) {
            try {
                controlThread.join(5000); // Wait up to 5 seconds
            } catch (InterruptedException e) {
                System.err.println("[SchedulerController] Interrupted while stopping: " + e.getMessage());
                Thread.currentThread().interrupt();
            }
        }

        nativeAdapter.stop();
        System.out.println("[SchedulerController] Stopped");
    }

    /**
     * Estimate latency improvement based on decision and execution result.
     *
     * Simplified heuristic: prefetch saves 50ms, evict saves 30ms, KV operations save 10-20ms.
     */
    private double estimateLatencyImprovement(Decision decision, NativeEngineAdapter.ExecutionResult result) {
        if (!result.success()) {
            return -10.0; // Negative impact if execution failed
        }

        switch (decision.action()) {
            case PREFETCH_LAYER:
                // Prefetching next layer typically saves 40-60ms in inference
                return 50.0;

            case EVICT_LAYER:
                // Evicting unused layers saves 20-40ms
                return 30.0;

            case COMPRESS_KV:
                // KV compression saves 10-20ms per inference
                return 15.0;

            case MOVE_KV_TO_RAM:
            case MOVE_KV_TO_GPU:
            case OFFLOAD_KV:
                // KV operations save 5-15ms
                return 10.0;

            case KEEP_LAYER:
            case NO_ACTION:
                // No improvement from no-op
                return 0.0;

            default:
                return 0.0;
        }
    }

    /**
     * Set decision interval (how often to evaluate).
     */
    public void setDecisionIntervalMs(long intervalMs) {
        if (intervalMs < 1) {
            throw new IllegalArgumentException("Interval must be >= 1ms");
        }
        this.decisionIntervalMs = intervalMs;
        System.out.printf("[SchedulerController] Decision interval set to %dms%n", intervalMs);
    }

    /**
     * Set batch retraining threshold (how many decisions between retraining).
     */
    public void setBatchRetrainThreshold(int threshold) {
        if (threshold < 10) {
            throw new IllegalArgumentException("Threshold must be >= 10");
        }
        this.batchRetrainThreshold = threshold;
        System.out.printf("[SchedulerController] Batch retrain threshold set to %d%n", threshold);
    }

    // --- Metrics and Monitoring ---

    public AdaptiveScheduler getScheduler() {
        return scheduler;
    }

    public boolean isRunning() {
        return running;
    }

    public long getDecisionsExecuted() {
        return decisionsExecuted.get();
    }

    public long getSuccessfulDecisions() {
        return successfulDecisions.get();
    }

    public double getSuccessRate() {
        long total = decisionsExecuted.get();
        if (total == 0) return 0;
        return (double) successfulDecisions.get() / total * 100;
    }

    public PerformanceMetrics getMetrics() {
        return metrics;
    }

    /**
     * Get performance summary.
     */
    public String getPerformanceSummary() {
        return String.format(
                "SchedulerController Performance:\n" +
                "  Running: %b\n" +
                "  Decisions executed: %d\n" +
                "  Successful: %d (%.1f%%)\n" +
                "  Native adapter latency: %.2fms\n" +
                "  Total memory saved: %.1f MB\n" +
                "  Decision interval: %dms\n" +
                "  Next batch retrain at: %d decisions",
                running,
                decisionsExecuted.get(),
                successfulDecisions.get(),
                getSuccessRate(),
                nativeAdapter.getAverageLatencyMs(),
                metrics.getTotalMemorySavedMb(),
                decisionIntervalMs,
                batchRetrainThreshold - (decisionsExecuted.get() % batchRetrainThreshold)
        );
    }

    /**
     * Provider for current memory state.
     * Implement to get live state from runtime.
     */
    public interface MemoryStateProvider {
        MemoryState getCurrentState();
    }

    /**
     * Performance metrics tracking for scheduler decisions.
     */
    public static final class PerformanceMetrics {
        private volatile long totalMemorySavedBytes = 0;
        private volatile long totalLatencyImprovementMs = 0;
        private volatile int prefetchCount = 0;
        private volatile int evictCount = 0;
        private volatile int kvCount = 0;
        private volatile int failureCount = 0;

        public void recordDecision(Decision decision, NativeEngineAdapter.ExecutionResult result,
                                 double latencyImprovement) {
            if (result.success()) {
                totalMemorySavedBytes += result.memorySavedBytes();
                totalLatencyImprovementMs += (long) latencyImprovement;

                switch (decision.action()) {
                    case PREFETCH_LAYER:
                        prefetchCount++;
                        break;
                    case EVICT_LAYER:
                        evictCount++;
                        break;
                    case MOVE_KV_TO_RAM:
                    case MOVE_KV_TO_GPU:
                    case COMPRESS_KV:
                    case OFFLOAD_KV:
                        kvCount++;
                        break;
                }
            } else {
                failureCount++;
            }
        }

        public double getTotalMemorySavedMb() {
            return totalMemorySavedBytes / (1024.0 * 1024.0);
        }

        public long getTotalLatencyImprovementMs() {
            return totalLatencyImprovementMs;
        }

        public int getPrefetchCount() {
            return prefetchCount;
        }

        public int getEvictCount() {
            return evictCount;
        }

        public int getKvCount() {
            return kvCount;
        }

        public int getFailureCount() {
            return failureCount;
        }

        @Override
        public String toString() {
            return String.format(
                    "PerformanceMetrics{prefetch=%d, evict=%d, kv=%d, failures=%d, memory=%.1f MB, latency=+%d ms}",
                    prefetchCount, evictCount, kvCount, failureCount,
                    getTotalMemorySavedMb(), totalLatencyImprovementMs
            );
        }
    }
}
