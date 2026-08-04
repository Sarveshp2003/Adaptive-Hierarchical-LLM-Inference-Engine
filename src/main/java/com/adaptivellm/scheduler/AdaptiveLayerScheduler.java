package com.adaptivellm.scheduler;

import java.util.*;

/**
 * Phase 3.4: Integration layer for dynamic layer prioritization.
 * 
 * Wraps Phase2NativeEngineAdapter to:
 * 1. Apply learned layer priorities to prefetch decisions
 * 2. Track execution outcomes for continuous learning
 * 3. Adapt scheduler behavior based on convergence feedback
 * 4. Manage adaptive prefetch depth based on workload characteristics
 */
public final class AdaptiveLayerScheduler {

    private final Phase2NativeEngineAdapter nativeAdapter;
    private final AdaptiveScheduler scheduler;
    private final LayerPrioritizationLearner learner;
    private final SchedulerRuntimeController runtimeController;
    private final int numLayers;
    private int decisionCount = 0;
    private double lastLoss = Double.MAX_VALUE;
    private int learningPhaseCounter = 0;
    private static final int LEARNING_PHASE_INTERVAL = 50;  // Every 50 decisions, update strategy

    public AdaptiveLayerScheduler(
        Phase2NativeEngineAdapter nativeAdapter,
        AdaptiveScheduler scheduler,
        LayerPrioritizationLearner learner,
        SchedulerRuntimeController runtimeController,
        int numLayers
    ) {
        this.nativeAdapter = Objects.requireNonNull(nativeAdapter);
        this.scheduler = Objects.requireNonNull(scheduler);
        this.learner = Objects.requireNonNull(learner);
        this.runtimeController = Objects.requireNonNull(runtimeController);
        this.numLayers = numLayers;
    }

    /**
     * Execute a decision using the native adapter and record results for learning.
     */
    public ExecutionResult executeDecisionWithLearning(Decision decision, MemoryState state) {
        decisionCount++;
        
        // Determine which layers are being targeted
        int targetLayer = (int) decision.targetId();
        
        // Execute decision on native engine
        ExecutionResult result = nativeAdapter.executeDecision(decision);
        
        if (!result.success()) {
            System.err.println("[AdaptiveLayerScheduler] Decision execution failed: " + result.errorMessage());
            return result;
        }
        
        // Record layer access for learning
        double convergenceImprovement = 0.0;
        if (lastLoss > 0) {
            convergenceImprovement = lastLoss - state.getEstimatedLoss();
            lastLoss = state.getEstimatedLoss();
        }
        
        learner.recordLayerAccess(
            targetLayer,
            result.latencyMs(),
            state.getMemorySavedBytes(),
            convergenceImprovement,
            decision.action() == SchedulerAction.EVICT_LAYER
        );
        
        // Update scheduler with feedback
        ScheduledDecision scheduled = new ScheduledDecision(decision, null);
        scheduler.reportResult(
            scheduled,
            convergenceImprovement,
            state.getMemorySavedBytes()
        );
        
        // Periodic learning phase: update strategy based on accumulated feedback
        if (decisionCount % LEARNING_PHASE_INTERVAL == 0) {
            updateLearningStrategy();
        }
        
        return result;
    }

    /**
     * Generate next decision using learned layer priorities.
     */
    public Decision generatePrioritizedDecision(MemoryState state) {
        // Get base decision from scheduler
        ScheduledDecision scheduledDecision = scheduler.evaluate(state);
        Decision baseDecision = scheduledDecision.decision();
        
        // Check if this is a prefetch decision - if so, prioritize based on learning
        if (baseDecision.action() == SchedulerAction.PREFETCH_LAYER) {
            // Get optimal layer order based on learned priorities
            List<Integer> optimalOrder = learner.getOptimalPrefetchOrder();
            
            if (!optimalOrder.isEmpty()) {
                // Try to prefetch the highest priority layer
                int priorityLayer = optimalOrder.get(0);
                
                // Create new decision with prioritized layer
                Decision prioritizedDecision = new Decision(
                    baseDecision.action(),
                    priorityLayer,
                    baseDecision.layers(),
                    baseDecision.metadata()
                );
                
                return prioritizedDecision;
            }
        }
        
        // Keep critical layers pinned
        List<Integer> criticalLayers = learner.getCriticalLayers(0.7);
        if (!criticalLayers.isEmpty() && state.getAvailableMemoryBytes() > 500_000_000) {
            // If memory available and layers are critical, pin top critical layer
            int criticalLayer = criticalLayers.get(0);
            if (!state.getCachedLayers().contains(criticalLayer)) {
                Decision keepDecision = new Decision(
                    SchedulerAction.KEEP_LAYER,
                    criticalLayer,
                    new Decision.Layer[]{new Decision.Layer(criticalLayer, 0.99)},
                    "Keeping critical layer based on learned priority"
                );
                return keepDecision;
            }
        }
        
        return baseDecision;
    }

    /**
     * Periodic update to learning strategy based on accumulated feedback.
     */
    private void updateLearningStrategy() {
        learningPhaseCounter++;
        
        System.out.println("\n=== Learning Phase " + learningPhaseCounter + " ===");
        System.out.println(learner.getSummary());
        
        // Get current metrics
        List<LayerPrioritizationLearner.LayerMetrics> metrics = learner.getAllMetrics();
        
        // Identify problematic patterns
        List<LayerPrioritizationLearner.LayerMetrics> problematicLayers = metrics.stream()
            .filter(m -> m.getEvictionRate() > 0.5 && m.convergenceImpact < 0)
            .toList();
        
        if (!problematicLayers.isEmpty()) {
            System.out.println("Warning: Problematic layers with high eviction and negative impact:");
            problematicLayers.forEach(m -> System.out.println("  - Layer " + m.layerId + ": " + m));
        }
        
        // Get critical layers for priority pinning
        List<Integer> criticalLayers = learner.getCriticalLayers(0.6);
        if (!criticalLayers.isEmpty()) {
            System.out.println("Critical layers to prioritize: " + criticalLayers.subList(0, Math.min(3, criticalLayers.size())));
        }
        
        // Adjust prefetch depth
        int adaptivePrefetchDepth = learner.getAdaptivePrefetchDepth();
        System.out.println("Adaptive prefetch depth: " + adaptivePrefetchDepth);
        
        // Periodic retraining if enough samples collected
        if (decisionCount > 100) {
            scheduler.retrainOnCollectedResults(50, true);
        }
    }

    /**
     * Get current learning status for monitoring.
     */
    public String getLearningStatus() {
        return String.format(
            "AdaptiveLayerScheduler{decisions=%d, learningPhases=%d, lastLoss=%.6f}\n%s",
            decisionCount, learningPhaseCounter, lastLoss,
            learner.getSummary()
        );
    }

    /**
     * Reset learning state for new training phase.
     */
    public void resetLearning() {
        learner.reset();
        decisionCount = 0;
        lastLoss = Double.MAX_VALUE;
        learningPhaseCounter = 0;
        System.out.println("[AdaptiveLayerScheduler] Learning state reset");
    }

    /**
     * Get metrics for a specific layer.
     */
    public Optional<LayerPrioritizationLearner.LayerMetrics> getLayerMetrics(int layerId) {
        return learner.getAllMetrics().stream()
            .filter(m -> m.layerId == layerId)
            .findFirst();
    }

    /**
     * Recommend optimization strategy based on learned patterns.
     */
    public PerformanceOptimizer.OptimizationProfile recommendOptimizationProfile() {
        List<LayerPrioritizationLearner.LayerMetrics> metrics = learner.getAllMetrics();
        
        // Calculate variance in layer priorities
        double avgPriority = metrics.stream()
            .mapToDouble(m -> m.priorityScore)
            .average()
            .orElse(0.5);
        
        double priorityVariance = metrics.stream()
            .mapToDouble(m -> Math.pow(m.priorityScore - avgPriority, 2))
            .average()
            .orElse(0.0);
        
        double stdDev = Math.sqrt(priorityVariance);
        
        // High variance = diverse workload = more aggressive optimization needed
        if (stdDev > 0.15) {
            return PerformanceOptimizer.OptimizationProfile.AGGRESSIVE;
        } else if (stdDev > 0.08) {
            return PerformanceOptimizer.OptimizationProfile.BALANCED;
        } else {
            return PerformanceOptimizer.OptimizationProfile.CONSERVATIVE;
        }
    }
}
