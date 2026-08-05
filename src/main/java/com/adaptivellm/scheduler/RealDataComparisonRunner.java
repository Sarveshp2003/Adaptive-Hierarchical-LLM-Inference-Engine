package com.adaptivellm.scheduler;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads a saved model and compares it with the rule-based baseline on a CSV dataset.
 */
public final class RealDataComparisonRunner {

    public static void main(String[] args) throws Exception {
        Path csvPath = args.length > 0 ? Paths.get(args[0]) : Paths.get("data", "scheduler", "real_training_samples.csv");
        String modelDir = args.length > 2 ? args[2] : "models";

        ModelPersistence persistence = new ModelPersistence(modelDir);
        NeuralNetworkPredictor model;
        if (args.length > 1) {
            model = persistence.loadModel(args[1]);
        } else {
            model = persistence.loadLatestModel();
        }

        MLTrainer trainer = new MLTrainer(model, persistence);

        List<TrainingSample> samples = loadCsv(csvPath);
        if (samples.isEmpty()) {
            System.err.println("No samples found for comparison: " + csvPath);
            return;
        }

        System.out.println("Running baseline comparison on " + samples.size() + " samples...");
        trainer.compareWithBaseline(samples);
    }

    private static List<TrainingSample> loadCsv(Path csvPath) throws IOException {
        List<TrainingSample> samples = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(csvPath)) {
            String line = reader.readLine(); // header
            int lineNumber = 1;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length < 24) {
                    System.err.println("Skipping malformed row " + lineNumber);
                    continue;
                }
                try {
                    long timestamp = Long.parseLong(parts[0]);
                    SchedulerAction action = SchedulerAction.valueOf(parts[1]);
                    long targetId = Long.parseLong(parts[2]);
                    double confidence = Double.parseDouble(parts[3]);

                    int currentLayer = Integer.parseInt(parts[4]);
                    long currentToken = Long.parseLong(parts[5]);
                    double gpuUsage = Double.parseDouble(parts[6]);
                    double ramUsage = Double.parseDouble(parts[7]);
                    double storageLatency = Double.parseDouble(parts[8]);
                    int cachedLayers = Integer.parseInt(parts[9]);
                    int kvPages = Integer.parseInt(parts[10]);
                    double pressureScore = Double.parseDouble(parts[11]);

                    double[] features = new double[12];
                    for (int i = 0; i < 12; i++) {
                        features[i] = Double.parseDouble(parts[12 + i]);
                    }

                    MemoryState state = new MemoryState(
                            currentLayer,
                            currentToken,
                            gpuUsage,
                            ramUsage,
                            storageLatency,
                            cachedLayers,
                            kvPages
                    );

                    Decision decision = new Decision(action, targetId, confidence);
                    TrainingSample sample = new TrainingSample(state, decision, features);
                    samples.add(sample);
                } catch (Exception e) {
                    System.err.println("Skipping row " + lineNumber + " due to parse error: " + e.getMessage());
                }
            }
        }
        return samples;
    }
}
