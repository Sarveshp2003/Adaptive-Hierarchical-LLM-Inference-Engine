package com.adaptivellm.scheduler;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Entry point for collecting a large dataset from live JVM/process metrics.
 */
public final class
RealDataCollectionRunner {

    public static void main(String[] args) throws IOException {
        int sampleCount = args.length > 0 ? Integer.parseInt(args[0]) : 50_000;
        Path outputPath = args.length > 1 ? Paths.get(args[1]) : Paths.get("data", "scheduler", "real_training_samples.csv");

        FeatureExtractor extractor = new FeatureExtractor();
        TrainingDataCollector collector = new TrainingDataCollector();
        PredictorModel predictor = new NeuralNetworkPredictor();
        SchedulerRuntimeController.MemoryStateProvider stateProvider = new ProductionMemoryStateProvider(28, null);

        RealTrainingDataGenerator generator = new RealTrainingDataGenerator(extractor, predictor, collector, stateProvider);
        int generated = generator.collect(sampleCount, outputPath);

        System.out.println("[RealData] Completed collection: " + generated + " samples");
        System.out.println("[RealData] Output: " + outputPath.toAbsolutePath());
    }
}
