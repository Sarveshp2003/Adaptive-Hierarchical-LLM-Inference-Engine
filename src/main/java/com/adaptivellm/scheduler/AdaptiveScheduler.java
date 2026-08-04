package com.adaptivellm.scheduler;

import java.util.Objects;
import java.util.List;

/**
 * Main adaptive memory scheduler with feedback loop.
 *
 * Architecture:
 * 1. evaluate() -> generates decision based on runtime state
 * 2. reportResult() -> records execution outcome
 * 3. feedback loop -> retrains model on execution results
 */
public final class AdaptiveScheduler {

    private final FeatureExtractor extractor;
    private final PredictorModel predictor;
    private final TrainingDataCollector collector;
    private MLTrainer trainer;  // Optional: for feedback-driven learning

    public AdaptiveScheduler(
            FeatureExtractor extractor,
            PredictorModel predictor,
            TrainingDataCollector collector
    ) {
        this.extractor = Objects.requireNonNull(extractor);
        this.predictor = Objects.requireNonNull(predictor);
        this.collector = Objects.requireNonNull(collector);
        this.trainer = null;
    }

    /**
     * Create scheduler with trainer for feedback loop.
     */
    public AdaptiveScheduler(
            FeatureExtractor extractor,
            PredictorModel predictor,
            TrainingDataCollector collector,
            MLTrainer trainer
    ) {
        this(extractor, predictor, collector);
        this.trainer = Objects.requireNonNull(trainer);
    }

    /**
     * Evaluates current runtime state and generates scheduling decision.
     */
    public ScheduledDecision evaluate(MemoryState state) {
        double[] features = extractor.extractNormalized(state);
        Decision decision = predictor.predict(features);
        TrainingSample sample = collector.record(state, decision);
        return new ScheduledDecision(decision, sample);
    }

    /**
     * Reports execution result and updates training sample.
     * This is the feedback signal that drives improvement.
     */
    public void reportResult(
            ScheduledDecision scheduled,
            double latencyImprovement,
            long memorySavedBytes
    ) {
        collector.updateResult(
                scheduled.sample(),
                latencyImprovement,
                memorySavedBytes
        );

        // Trigger online learning if trainer is available
        if (trainer != null) {
            trainer.updateWithExecutionResult(scheduled.sample(), 5);
        }
    }

    /**
     * Batch feedback retraining: retrain on all collected execution results.
     * Call this periodically after accumulating enough samples.
     */
    public void retrainOnCollectedResults(int minSamples, boolean weightByOutcome) {
        if (trainer == null) {
            System.err.println("No trainer available for retraining");
            return;
        }

        List<TrainingSample> samples = collector.samples();
        if (samples.size() < minSamples) {
            System.out.println("Insufficient samples for retraining: " + samples.size() + "/" + minSamples);
            return;
        }

        trainer.retrainWithFeedback(samples);
    }

    /**
     * Periodic batch retraining with optional outcome weighting.
     */
    public void periodicRetrain(int minSamples) {
        if (trainer == null) {
            return;
        }
        trainer.periodicRetrain(minSamples, true);
    }

    /**
     * Get feedback metrics for monitoring.
     */
    public String getFeedbackMetrics() {
        if (trainer == null) {
            return "No trainer available";
        }
        return trainer.getFeedbackMetrics();
    }

    /**
     * Training samples count.
     */
    public int trainingSamples() {
        return collector.size();
    }

    /**
     * Set trainer for feedback loop (useful if created without trainer initially).
     */
    public void setTrainer(MLTrainer trainer) {
        this.trainer = Objects.requireNonNull(trainer);
    }

    /**
     * Get the trainer instance.
     */
    public MLTrainer getTrainer() {
        return trainer;
    }
}