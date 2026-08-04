package com.adaptivellm.scheduler;

import java.io.IOException;
import java.util.*;

/**
 * Demonstration of improved backpropagation and feedback loop.
 *
 * This demo shows:
 * 1. Full backpropagation through all network layers
 * 2. Online learning from execution results
 * 3. Batch retraining with outcome weighting
 * 4. Feedback-driven model improvement
 *
 * Usage:
 *   java -cp build FeedbackLoopDemo
 */
public final class FeedbackLoopDemo {

    public static void main(String[] args) throws IOException {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║     AI SCHEDULER: FULL BACKPROP + FEEDBACK LOOP DEMO      ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        // 1. Initialize scheduler
        System.out.println("Step 1: Initializing scheduler components...\n");
        FeatureExtractor extractor = new FeatureExtractor();
        NeuralNetworkPredictor predictor = new NeuralNetworkPredictor();
        TrainingDataCollector collector = new TrainingDataCollector();
        ModelPersistence persistence = new ModelPersistence("./scheduler_models");
        MLTrainer trainer = new MLTrainer(predictor, persistence);
        AdaptiveScheduler scheduler = new AdaptiveScheduler(extractor, predictor, collector, trainer);

        System.out.println("✓ Scheduler created");
        System.out.println("✓ Neural network: " + predictor.getNetworkInfo());
        System.out.println("✓ Feature extractor: " + extractor.describe() + "\n");

        // 2. Simulate execution loop with feedback
        System.out.println("Step 2: Running simulation with 20 decision cycles...\n");
        simulateExecutionCycles(scheduler, 20);

        // 3. Retrain with collected outcomes
        System.out.println("\nStep 3: Retraining model with feedback...\n");
        trainer.retrainWithFeedback(collector.samples());

        // 4. Show improvement metrics
        System.out.println("\nStep 4: Model improvement metrics\n");
        System.out.println(trainer.getFeedbackMetrics());

        // 5. Demonstrate online learning
        System.out.println("Step 5: Online learning demonstration...\n");
        demonstrateOnlineLearning(scheduler, extractor);

        // 6. Save trained model
        System.out.println("\nStep 6: Saving trained model...\n");
        String modelName = trainer.saveModel("feedback_trained", collector.samples());
        System.out.println("✓ Model saved as: " + modelName);

        // 7. Compare baseline vs improved
        System.out.println("\nStep 7: Comparing with baseline rule-based scheduler...\n");
        trainer.compareWithBaseline(collector.samples());

        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║            DEMO COMPLETED                                  ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }

    /**
     * Simulate realistic execution cycles with feedback.
     */
    private static void simulateExecutionCycles(AdaptiveScheduler scheduler, int cycles) {
        Random rand = new Random(42);

        for (int cycle = 0; cycle < cycles; cycle++) {
            // 1. Create realistic memory state
            int layer = rand.nextInt(28);  // 28 layers in Llama 3.2 3B
            long token = rand.nextLong(100_000);
            double gpuUsage = 0.4 + rand.nextDouble() * 0.5;
            double ramUsage = 0.3 + rand.nextDouble() * 0.6;
            double latency = rand.nextDouble() * 500;
            int cachedLayers = rand.nextInt(10);
            int kvPages = rand.nextInt(50);

            MemoryState state = new MemoryState(
                    layer, token, gpuUsage, ramUsage, latency, cachedLayers, kvPages
            );

            // 2. Get scheduler decision
            ScheduledDecision scheduled = scheduler.evaluate(state);

            // 3. Simulate execution with realistic outcome
            double latencyImprovement = simulateExecution(scheduled.decision(), state, rand);
            long memorySaved = (long)(rand.nextDouble() * 1_000_000);

            // 4. Report result (triggers online learning)
            scheduler.reportResult(scheduled, latencyImprovement, memorySaved);

            if ((cycle + 1) % 5 == 0) {
                System.out.println(String.format("  Cycle %2d: action=%s, latency_improvement=%.4f ms, memory_saved=%.1f MB",
                        cycle + 1,
                        scheduled.decision().action(),
                        latencyImprovement,
                        memorySaved / (1024.0 * 1024.0)));
            }
        }

        System.out.println("\n✓ Collected " + scheduler.trainingSamples() + " samples with execution outcomes");
    }

    /**
     * Simulate realistic execution outcome based on action taken.
     */
    private static double simulateExecution(Decision decision, MemoryState state, Random rand) {
        // Better actions produce better outcomes
        double baseImprovement = rand.nextDouble() * 10;

        switch (decision.action()) {
            case PREFETCH_LAYER:
                // Prefetch reduces latency significantly
                return baseImprovement + 5.0 + (state.storageLatency() * 0.01);
            case EVICT_LAYER:
                // Evict saves memory, minor latency impact
                return baseImprovement + 1.0;
            case COMPRESS_KV:
                // Compression helps under high memory pressure
                return baseImprovement + (state.ramUsage() * 8.0);
            case MOVE_KV_TO_RAM:
                // Moving to RAM helps if GPU is full
                return baseImprovement + (state.gpuUsage() * 6.0);
            case MOVE_KV_TO_GPU:
                // Moving to GPU helps if have space
                return baseImprovement + ((1.0 - state.gpuUsage()) * 4.0);
            case NO_ACTION:
                return baseImprovement - 2.0;
            default:
                return baseImprovement;
        }
    }

    /**
     * Demonstrate online learning capability.
     */
    private static void demonstrateOnlineLearning(AdaptiveScheduler scheduler, FeatureExtractor extractor) {
        Random rand = new Random(123);

        System.out.println("Simulating 5 quick decisions with immediate feedback:");
        for (int i = 0; i < 5; i++) {
            // Create state
            MemoryState state = new MemoryState(
                    rand.nextInt(28), rand.nextLong(100_000),
                    rand.nextDouble() * 0.9, rand.nextDouble() * 0.9,
                    rand.nextDouble() * 500, rand.nextInt(10), rand.nextInt(50)
            );

            // Get decision
            ScheduledDecision scheduled = scheduler.evaluate(state);

            // Simulate and report
            double latencyImprovement = 5.0 + rand.nextDouble() * 5.0;
            long memorySaved = (long)(rand.nextDouble() * 2_000_000);
            scheduler.reportResult(scheduled, latencyImprovement, memorySaved);
        }

        System.out.println("\n✓ Online learning samples processed");
    }
}
