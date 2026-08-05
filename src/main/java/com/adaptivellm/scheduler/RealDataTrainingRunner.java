package com.adaptivellm.scheduler;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Trains the scheduler neural network from the persisted real-data CSV.
 */
public final class RealDataTrainingRunner {

    public static void main(String[] args) throws Exception {
        Path csvPath = args.length > 0 ? Paths.get(args[0]) : Paths.get("data", "scheduler", "real_training_samples.csv");
        String modelName = args.length > 1 ? args[1] : "real_scheduler_model";
        String modelDir = args.length > 2 ? args[2] : "models";

        List<TrainingSample> samples = loadCsv(csvPath);
        if (samples.isEmpty()) {
            throw new IllegalStateException("No training samples found in " + csvPath.toAbsolutePath());
        }

        NeuralNetworkPredictor model = new NeuralNetworkPredictor();
        model.setTrainingConfig(0.001, 0.0003, 220, 128);
        ModelPersistence persistence = new ModelPersistence(modelDir);
        MLTrainer trainer = new MLTrainer(model, persistence);

        trainer.train(new ArrayList<>(samples));

        System.out.println("\nTraining metrics:");
        for (var entry : trainer.getMetrics().entrySet()) {
            System.out.printf("  %s = %.6f%n", entry.getKey(), entry.getValue());
        }

        String savedName = trainer.saveModel(modelName, samples);
        System.out.println("\nSaved model: " + savedName);
        System.out.println("Validation report:\n" + trainer.getReport());
    }

    private static List<TrainingSample> loadCsv(Path csvPath) throws IOException {
        List<TrainingSample> samples = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(csvPath)) {
            String line = reader.readLine();
            int lineNumber = 1;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length < 24) {
                    throw new IOException("Unexpected CSV row at line " + lineNumber + ": " + line);
                }
                if (parts[0].equalsIgnoreCase("timestamp") || parts[1].equalsIgnoreCase("action")) {
                    continue;
                }

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
            }
        }

        return samples;
    }
}
