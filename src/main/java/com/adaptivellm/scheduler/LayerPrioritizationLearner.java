package com.adaptivellm.scheduler;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Phase 3.4: Dynamic layer prioritization using policy gradient learning.
 * 
 * Learns which layers are most critical for inference and automatically
 * prioritizes their prefetching and caching based on execution feedback.
 * 
 * Architecture:
 * - Tracks per-layer access frequency and latency impact
 * - Maintains layer priority scores updated by gradient feedback
 * - Generates optimal prefetch order based on learned patterns
 * - Integrates with AdaptiveScheduler for decision-making
 */
public final class LayerPrioritizationLearner {

    /**
     * Per-layer statistics for learning.
     */
    public static class LayerMetrics {
        public final int layerId;
        public int accessCount;           // How often this layer was accessed
        public long totalLatency;         // Total latency when this layer was in use
        public long totalMemorySaved;     // Memory saved by optimizing this layer
        public double priorityScore;      // Learned priority (0.0-1.0)
        public int timesEvicted;          // How many times was this layer evicted
        public double convergenceImpact;  // Impact on overall loss convergence
        
        public LayerMetrics(int layerId) {
            this.layerId = layerId;
            this.accessCount = 0;
            this.totalLatency = 0L;
            this.totalMemorySaved = 0L;
            this.priorityScore = 0.5;  // Uniform initial priority
            this.timesEvicted = 0;
            this.convergenceImpact = 0.0;
        }
        
        public double getAverageLatency() {
            return accessCount > 0 ? (double) totalLatency / accessCount : 0.0;
        }
        
        public double getAverageMemorySaved() {
            return accessCount > 0 ? (double) totalMemorySaved / accessCount : 0.0;
        }
        
        public double getEvictionRate() {
            return accessCount > 0 ? (double) timesEvicted / accessCount : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format(
                "LayerMetrics{id=%d, accesses=%d, avgLatency=%.2fms, priority=%.3f, " +
                "evictionRate=%.1f%%, convergenceImpact=%.4f}",
                layerId, accessCount, getAverageLatency(), priorityScore,
                getEvictionRate() * 100, convergenceImpact
            );
        }
    }

    /**
     * Policy gradient for layer prioritization.
     * Updates priority based on execution outcomes.
     */
    private static class PolicyGradient {
        double learningRate = 0.01;
        
        /**
         * Update priority score using policy gradient.
         * If latency improved and memory saved, increase priority.
         * If layer caused slowdown, decrease priority.
         */
        double updatePriority(
            double currentPriority,
            double latencyImprovement,    // Positive = improvement
            long memorySaved,
            int accessCount
        ) {
            // Gradient components
            double latencyGradient = latencyImprovement > 0 ? 0.001 : -0.001;
            double memoryGradient = memorySaved > 0 ? 0.0001 : -0.0001;
            double frequencyMultiplier = Math.sqrt(accessCount);  // Frequent layers influence more
            
            // Combined gradient
            double gradient = (latencyGradient + memoryGradient) * frequencyMultiplier;
            
            // Update priority with learning rate
            double newPriority = currentPriority + (learningRate * gradient);
            
            // Clamp to [0, 1]
            return Math.max(0.0, Math.min(1.0, newPriority));
        }
    }

    private final int numLayers;
    private final Map<Integer, LayerMetrics> layerMetrics;
    private final PolicyGradient policyGradient;
    private final Deque<TrainingSample> recentSamples;
    private final int maxHistorySize = 1000;
    private int totalDecisions = 0;
    private double cumulativeConvergenceImprovement = 0.0;

    public LayerPrioritizationLearner(int numLayers) {
        this.numLayers = numLayers;
        this.layerMetrics = new HashMap<>();
        this.policyGradient = new PolicyGradient();
        this.recentSamples = new LinkedList<>();
        
        // Initialize metrics for all layers
        for (int i = 0; i < numLayers; i++) {
            layerMetrics.put(i, new LayerMetrics(i));
        }
    }

    /**
     * Record execution result and update layer metrics with policy gradient.
     */
    public void recordLayerAccess(
        int layerId,
        long latency,
        long memorySavedBytes,
        double convergenceImprovement,
        boolean wasEvicted
    ) {
        if (layerId < 0 || layerId >= numLayers) {
            return;
        }
        
        LayerMetrics metrics = layerMetrics.get(layerId);
        metrics.accessCount++;
        metrics.totalLatency += latency;
        metrics.totalMemorySaved += memorySavedBytes;
        metrics.convergenceImpact += convergenceImprovement;
        if (wasEvicted) {
            metrics.timesEvicted++;
        }
        
        // Apply policy gradient to update priority
        metrics.priorityScore = policyGradient.updatePriority(
            metrics.priorityScore,
            convergenceImprovement > 0 ? convergenceImprovement : 0,
            memorySavedBytes,
            metrics.accessCount
        );
        
        totalDecisions++;
        cumulativeConvergenceImprovement += convergenceImprovement;
    }

    /**
     * Record execution feedback from a decision.
     */
    public void recordDecisionFeedback(
        ScheduledDecision decision,
        double latencyImprovement,
        long memorySaved,
        double convergenceImprovement
    ) {
        // Update layer metrics based on decision
        Decision.Layer[] actionLayers = decision.decision().layers();
        if (actionLayers != null) {
            for (Decision.Layer layer : actionLayers) {
                recordLayerAccess(
                    layer.layerId(),
                    0,  // Latency tracked at higher level
                    memorySaved / actionLayers.length,
                    convergenceImprovement / actionLayers.length,
                    false
                );
            }
        }
        
        recentSamples.addLast(decision.sample());
        if (recentSamples.size() > maxHistorySize) {
            recentSamples.removeFirst();
        }
    }

    /**
     * Get optimal prefetch order based on learned layer priorities.
     * Returns layer IDs sorted by priority (highest first).
     */
    public List<Integer> getOptimalPrefetchOrder() {
        return layerMetrics.values().stream()
            .sorted((a, b) -> Double.compare(b.priorityScore, a.priorityScore))
            .map(m -> m.layerId)
            .collect(Collectors.toList());
    }

    /**
     * Get layers that should be kept in cache (high priority, high convergence impact).
     */
    public List<Integer> getCriticalLayers(double priorityThreshold) {
        return layerMetrics.values().stream()
            .filter(m -> m.priorityScore >= priorityThreshold &&
                        m.convergenceImpact > 0)
            .sorted((a, b) -> Double.compare(b.priorityScore, a.priorityScore))
            .map(m -> m.layerId)
            .collect(Collectors.toList());
    }

    /**
     * Adaptive prefetch depth: deeper for high-variance workloads.
     */
    public int getAdaptivePrefetchDepth() {
        double avgLatency = layerMetrics.values().stream()
            .mapToDouble(LayerMetrics::getAverageLatency)
            .average()
            .orElse(0.0);
        
        // More variability → deeper prefetch
        if (avgLatency > 50.0) {
            return 4;  // Aggressive prefetch for slow layers
        } else if (avgLatency > 20.0) {
            return 3;
        } else if (avgLatency > 5.0) {
            return 2;
        } else {
            return 1;  // Single layer ahead for fast layers
        }
    }

    /**
     * Get layer with highest convergence impact.
     */
    public int getMostCriticalLayer() {
        return layerMetrics.values().stream()
            .max(Comparator.comparingDouble(m -> m.convergenceImpact))
            .map(m -> m.layerId)
            .orElse(0);
    }

    /**
     * Get all layer metrics for monitoring/debugging.
     */
    public List<LayerMetrics> getAllMetrics() {
        return layerMetrics.values().stream()
            .sorted(Comparator.comparingDouble(m -> -m.priorityScore))
            .collect(Collectors.toList());
    }

    /**
     * Get summary statistics for the learning progress.
     */
    public String getSummary() {
        double avgPriority = layerMetrics.values().stream()
            .mapToDouble(m -> m.priorityScore)
            .average()
            .orElse(0.5);
        
        double priorityVariance = layerMetrics.values().stream()
            .mapToDouble(m -> Math.pow(m.priorityScore - avgPriority, 2))
            .average()
            .orElse(0.0);
        
        List<LayerMetrics> topLayers = getAllMetrics().stream()
            .limit(5)
            .collect(Collectors.toList());
        
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "LayerPrioritizationLearner{decisions=%d, avgPriority=%.3f, " +
            "variance=%.4f, cumulativeImprovement=%.4f}\n",
            totalDecisions, avgPriority, Math.sqrt(priorityVariance),
            cumulativeConvergenceImprovement
        ));
        
        sb.append("Top 5 Critical Layers:\n");
        for (LayerMetrics m : topLayers) {
            sb.append("  ").append(m.toString()).append("\n");
        }
        
        return sb.toString();
    }

    /**
     * Reset all metrics (for new training phase).
     */
    public void reset() {
        layerMetrics.values().forEach(m -> {
            m.accessCount = 0;
            m.totalLatency = 0L;
            m.totalMemorySaved = 0L;
            m.priorityScore = 0.5;
            m.timesEvicted = 0;
            m.convergenceImpact = 0.0;
        });
        recentSamples.clear();
        totalDecisions = 0;
        cumulativeConvergenceImprovement = 0.0;
    }
}
