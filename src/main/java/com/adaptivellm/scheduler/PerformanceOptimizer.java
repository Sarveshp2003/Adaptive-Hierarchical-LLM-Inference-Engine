package com.adaptivellm.scheduler;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Auto-tuning system for adaptive scheduler.
 * 
 * Analyzes execution metrics and suggests optimal parameter settings for:
 * - KV cache compression strategy
 * - Layer eviction policy
 * - Prefetch depth
 * - Checkpoint frequency
 * - Batch sizes
 */
public final class PerformanceOptimizer {

    // Optimization profiles
    public enum OptimizationProfile {
        AGGRESSIVE,      // Maximum compression/eviction
        BALANCED,        // Medium settings (default)
        CONSERVATIVE     // Minimal changes (safety first)
    }

    // Recommended parameters
    public static class OptimizationConfig {
        public String compressionType;      // NONE, FP16, INT8, NF4
        public int compressionThreshold;    // Token age to compress
        public String evictionPolicy;       // LRU, ATTENTION, PREDICTIVE
        public int prefetchDepth;           // Layers to prefetch
        public int checkpointFrequency;     // Tokens between checkpoints
        public int batchSize;               // Inference batch size
        public double gpuMemoryTarget;      // Target GPU utilization (0.0-1.0)
        public boolean enableCompression;
        public boolean enablePrefetch;

        @Override
        public String toString() {
            return String.format(
                "OptimizationConfig{compression=%s@%d, eviction=%s, prefetch=%d, " +
                "checkpoint=%d, batch=%d, gpuTarget=%.1f%%, enabled=%b/%b}",
                compressionType, compressionThreshold, evictionPolicy, prefetchDepth,
                checkpointFrequency, batchSize, gpuMemoryTarget * 100,
                enableCompression, enablePrefetch
            );
        }
    }

    private final AdaptiveScheduler scheduler;
    private final TrainingDataCollector collector;
    private Map<String, Double> currentMetrics;

    public PerformanceOptimizer(AdaptiveScheduler scheduler, TrainingDataCollector collector) {
        this.scheduler = scheduler;
        this.collector = collector;
        this.currentMetrics = new HashMap<>();
    }

    /**
     * Analyze execution traces and recommend parameters.
     */
    public OptimizationConfig analyze(OptimizationProfile profile) {
        OptimizationConfig config = new OptimizationConfig();

        // Get action distribution from training data
        Map<SchedulerAction, Integer> actionDist = getActionDistribution();
        int totalActions = actionDist.values().stream().mapToInt(Integer::intValue).sum();

        // Analyze compression needs
        double compressionFrequency = (double) actionDist.getOrDefault(SchedulerAction.COMPRESS_KV, 0) / totalActions;
        double evictionFrequency = (double) actionDist.getOrDefault(SchedulerAction.EVICT_LAYER, 0) / totalActions;
        double prefetchFrequency = (double) actionDist.getOrDefault(SchedulerAction.PREFETCH_LAYER, 0) / totalActions;

        // Set parameters based on profile
        switch (profile) {
            case AGGRESSIVE:
                config.compressionType = "NF4";  // Maximum compression
                config.compressionThreshold = 50;  // Compress early
                config.evictionPolicy = "ATTENTION";  // Smart eviction
                config.prefetchDepth = 3;  // Deep prefetch
                config.checkpointFrequency = 100;  // Frequent checkpoints
                config.batchSize = 8;  // Small batch
                config.gpuMemoryTarget = 0.7;  // Conservative
                break;

            case CONSERVATIVE:
                config.compressionType = "FP16";  // Light compression
                config.compressionThreshold = 200;  // Compress late
                config.evictionPolicy = "LRU";  // Simple eviction
                config.prefetchDepth = 1;  // Single layer ahead
                config.checkpointFrequency = 500;  // Infrequent checkpoints
                config.batchSize = 32;  // Large batch
                config.gpuMemoryTarget = 0.9;  // Aggressive
                break;

            case BALANCED:
            default:
                // Choose based on observed patterns
                if (compressionFrequency > 0.3) {
                    config.compressionType = "INT8";  // Medium compression
                } else {
                    config.compressionType = "FP16";
                }

                if (evictionFrequency > 0.2) {
                    config.evictionPolicy = "ATTENTION";
                } else {
                    config.evictionPolicy = "LRU";
                }

                config.compressionThreshold = 100;
                config.prefetchDepth = 2;
                config.checkpointFrequency = 250;
                config.batchSize = 16;
                config.gpuMemoryTarget = 0.8;
                break;
        }

        config.enableCompression = compressionFrequency > 0.15;
        config.enablePrefetch = prefetchFrequency > 0.15;

        return config;
    }

    /**
     * Recommend configuration based on workload characteristics.
     */
    public OptimizationConfig recommendConfig(double gpuUtilization, double ramUtilization) {
        OptimizationConfig config = new OptimizationConfig();

        // Memory pressure determines aggressiveness
        double memoryPressure = (gpuUtilization * 0.6) + (ramUtilization * 0.4);

        if (memoryPressure > 0.8) {
            // High pressure: aggressive compression
            config.compressionType = "NF4";
            config.compressionThreshold = 30;
            config.gpuMemoryTarget = 0.6;
        } else if (memoryPressure > 0.6) {
            // Medium pressure: balanced
            config.compressionType = "INT8";
            config.compressionThreshold = 100;
            config.gpuMemoryTarget = 0.75;
        } else {
            // Low pressure: minimal intervention
            config.compressionType = "FP16";
            config.compressionThreshold = 200;
            config.gpuMemoryTarget = 0.85;
        }

        // Default values
        config.evictionPolicy = "ATTENTION";
        config.prefetchDepth = 2;
        config.checkpointFrequency = 250;
        config.batchSize = memoryPressure > 0.7 ? 8 : 16;
        config.enableCompression = true;
        config.enablePrefetch = true;

        return config;
    }

    /**
     * Adaptive tuning based on recent performance.
     */
    public OptimizationConfig tuneAdaptively(List<Double> recentLatencies, 
                                             List<Double> recentMemoryUsage) {
        OptimizationConfig config = new OptimizationConfig();

        if (recentLatencies.isEmpty() || recentMemoryUsage.isEmpty()) {
            return getDefaultConfig();
        }

        // Calculate trends
        double avgLatency = recentLatencies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double latencyTrend = calculateTrend(recentLatencies);
        double memoryTrend = calculateTrend(recentMemoryUsage);

        // If latency increasing and memory stable -> reduce compression
        if (latencyTrend > 0.05 && memoryTrend < 0.02) {
            config.compressionType = "FP16";
            config.compressionThreshold = 150;
        }
        // If memory increasing -> increase compression
        else if (memoryTrend > 0.05) {
            config.compressionType = "INT8";
            config.compressionThreshold = 50;
        }
        // Otherwise: balanced
        else {
            config.compressionType = "INT8";
            config.compressionThreshold = 100;
        }

        config.evictionPolicy = "PREDICTIVE";
        config.prefetchDepth = 2;
        config.checkpointFrequency = 250;
        config.batchSize = 16;
        config.gpuMemoryTarget = 0.8;
        config.enableCompression = true;
        config.enablePrefetch = true;

        return config;
    }

    /**
     * Get default configuration.
     */
    public OptimizationConfig getDefaultConfig() {
        return analyze(OptimizationProfile.BALANCED);
    }

    /**
     * Generate optimization report.
     */
    public String getOptimizationReport(OptimizationProfile profile) {
        OptimizationConfig config = analyze(profile);
        Map<SchedulerAction, Integer> actionDist = getActionDistribution();

        StringBuilder sb = new StringBuilder();
        sb.append("╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║              OPTIMIZATION ANALYSIS REPORT                  ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        sb.append("Profile: ").append(profile).append("\n\n");

        sb.append("Recommended Configuration:\n");
        sb.append(String.format("  Compression:         %s (threshold: %d tokens)\n", 
            config.compressionType, config.compressionThreshold));
        sb.append(String.format("  Eviction Policy:     %s\n", config.evictionPolicy));
        sb.append(String.format("  Prefetch Depth:      %d layers\n", config.prefetchDepth));
        sb.append(String.format("  Checkpoint Freq:     %d tokens\n", config.checkpointFrequency));
        sb.append(String.format("  Batch Size:          %d\n", config.batchSize));
        sb.append(String.format("  GPU Memory Target:   %.1f%%\n", config.gpuMemoryTarget * 100));

        sb.append("\nAction Distribution (from ").append(collector.size()).append(" decisions):\n");
        int total = actionDist.values().stream().mapToInt(Integer::intValue).sum();
        for (SchedulerAction action : SchedulerAction.values()) {
            int count = actionDist.getOrDefault(action, 0);
            double percentage = total > 0 ? (double) count / total * 100 : 0;
            String bar = "█".repeat((int)(percentage / 2)) + "░".repeat(50 - (int)(percentage / 2));
            sb.append(String.format("  %-20s [%s] %.1f%% (%d)\n", 
                action, bar, percentage, count));
        }

        return sb.toString();
    }

    /**
     * Calculate trend in metric sequence (0 = stable, 1 = increasing, -1 = decreasing).
     */
    private double calculateTrend(List<Double> values) {
        if (values.size() < 2) return 0;

        double sumDiff = 0;
        for (int i = 1; i < values.size(); i++) {
            sumDiff += values.get(i) - values.get(i - 1);
        }

        return sumDiff / (values.size() - 1);
    }

    /**
     * Get action distribution from collected data.
     */
    private Map<SchedulerAction, Integer> getActionDistribution() {
        Map<SchedulerAction, Integer> dist = new HashMap<>();
        for (SchedulerAction action : SchedulerAction.values()) {
            dist.put(action, 0);
        }

        for (TrainingSample sample : collector.samples()) {
            SchedulerAction action = sample.decision().action();
            dist.put(action, dist.get(action) + 1);
        }

        return dist;
    }

    /**
     * Suggest batch size based on hardware constraints.
     */
    public int suggestBatchSize(long availableGPUMemory, long modelWeightsSize) {
        long availableForKV = availableGPUMemory - modelWeightsSize;
        
        // Estimate memory per sequence
        long memoryPerSeq = 10 * 1024 * 1024;  // ~10MB per sequence
        
        int suggestedBatch = (int) (availableForKV / memoryPerSeq);
        return Math.max(1, Math.min(suggestedBatch, 32));  // Cap at 32
    }

    /**
     * Suggest compression based on available memory.
     */
    public String suggestCompressionType(long availableMemory, long kvCacheSize) {
        double compressionNeeded = (double) kvCacheSize / availableMemory;

        if (compressionNeeded > 8) {
            return "NF4";   // 8x compression needed
        } else if (compressionNeeded > 4) {
            return "INT8";  // 4x compression needed
        } else if (compressionNeeded > 2) {
            return "FP16";  // 2x compression needed
        } else {
            return "NONE";  // No compression needed
        }
    }

    /**
     * Update current metrics for adaptive tuning.
     */
    public void updateMetrics(String metricName, double value) {
        currentMetrics.put(metricName, value);
    }

    /**
     * Get all current metrics.
     */
    public Map<String, Double> getCurrentMetrics() {
        return new HashMap<>(currentMetrics);
    }
}
