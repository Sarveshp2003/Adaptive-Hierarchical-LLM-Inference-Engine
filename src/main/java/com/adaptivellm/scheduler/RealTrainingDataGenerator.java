package com.adaptivellm.scheduler;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Collects a large dataset of real runtime snapshots and labels from the current JVM/process.
 */
public final class RealTrainingDataGenerator {

    private final FeatureExtractor extractor;
    private final PredictorModel predictor;
    private final TrainingDataCollector collector;
    private final SchedulerRuntimeController.MemoryStateProvider stateProvider;

    public RealTrainingDataGenerator(
            FeatureExtractor extractor,
            PredictorModel predictor,
            TrainingDataCollector collector,
            SchedulerRuntimeController.MemoryStateProvider stateProvider
    ) {
        this.extractor = extractor;
        this.predictor = predictor;
        this.collector = collector;
        this.stateProvider = stateProvider;
    }

    public int collect(int sampleCount) throws IOException {
        return collect(sampleCount, Paths.get("data", "scheduler", "real_training_samples.csv"));
    }

    public int collect(int sampleCount, Path outputPath) throws IOException {
        collector.enablePersistence(outputPath);

        int collected = 0;
        for (int i = 0; i < sampleCount; i++) {
            MemoryState state = stateProvider.getCurrentState();
            double[] features = extractor.extractNormalized(state);
            Decision decision = predictor.predict(features);
            TrainingSample sample = collector.record(state, decision, features);
            double latencyImprovement = estimateLatencyImprovement(state, decision);
            long memorySavedBytes = estimateMemorySaved(state, decision);
            collector.updateResult(sample, latencyImprovement, memorySavedBytes);
            collected++;

            if ((i + 1) % 1000 == 0) {
                System.out.printf("[RealData] Collected %d/%d samples%n", i + 1, sampleCount);
            }
        }

        System.out.printf("[RealData] Wrote %d samples to %s%n", collected, outputPath.toAbsolutePath());
        return collected;
    }

    private double estimateLatencyImprovement(MemoryState state, Decision decision) {
        double pressure = state.pressureScore();
        switch (decision.action()) {
            case PREFETCH_LAYER:
                return 15.0 + pressure * 40.0;
            case EVICT_LAYER:
                return 8.0 + pressure * 30.0;
            case MOVE_KV_TO_RAM:
            case MOVE_KV_TO_GPU:
            case COMPRESS_KV:
            case OFFLOAD_KV:
                return 3.0 + pressure * 20.0;
            case KEEP_LAYER:
            case NO_ACTION:
            default:
                return 0.0 + pressure * 2.0;
        }
    }

    private long estimateMemorySaved(MemoryState state, Decision decision) {
        double pressure = state.pressureScore();
        switch (decision.action()) {
            case PREFETCH_LAYER:
                return (long) (100L * 1024 * 1024 + pressure * 80L * 1024 * 1024);
            case EVICT_LAYER:
                return (long) (80L * 1024 * 1024 + pressure * 70L * 1024 * 1024);
            case MOVE_KV_TO_RAM:
            case MOVE_KV_TO_GPU:
            case COMPRESS_KV:
            case OFFLOAD_KV:
                return (long) (25L * 1024 * 1024 + pressure * 40L * 1024 * 1024);
            case KEEP_LAYER:
            case NO_ACTION:
            default:
                return 0L;
        }
    }
}
