package com.adaptivellm.scheduler;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Machine Learning training engine.
 * 
 * Handles:
 * - Training neural network from collected samples
 * - Model evaluation and validation
 * - Hyperparameter tuning
 * - Cross-validation
 * - Performance reporting
 */
public final class MLTrainer {

    private final NeuralNetworkPredictor model;
    private final ModelPersistence persistence;
    private List<TrainingSample> trainingData;
    private List<TrainingSample> validationData;
    private Map<String, Double> performanceMetrics;

    public MLTrainer(NeuralNetworkPredictor model, ModelPersistence persistence) {
        this.model = model;
        this.persistence = persistence;
        this.performanceMetrics = new LinkedHashMap<>();
    }

    /**
     * Train model on collected samples.
     */
    public void train(List<TrainingSample> samples) {
        if (samples.isEmpty()) {
            System.err.println("No training samples provided");
            return;
        }

        // Split into train/validation (80/20)
        Collections.shuffle(samples);
        int splitIdx = (int) (samples.size() * 0.8);
        trainingData = samples.subList(0, splitIdx);
        validationData = samples.subList(splitIdx, samples.size());

        System.out.println("Training split: " + trainingData.size() + " train, " + validationData.size() + " validation");

        // Train the model
        model.train(trainingData);

        // Evaluate
        evaluate();
    }

    /**
     * Evaluate model on validation set.
     */
    private void evaluate() {
        if (validationData == null || validationData.isEmpty()) {
            return;
        }

        FeatureExtractor extractor = new FeatureExtractor();
        int correct = 0;
        double avgConfidence = 0.0;

        for (TrainingSample sample : validationData) {
            double[] features = extractor.extractNormalized(sample.state());
            Decision prediction = model.predict(features);
            Decision expected = sample.decision();

            if (prediction.action() == expected.action()) {
                correct++;
            }
            avgConfidence += prediction.confidence();
        }

        double accuracy = (double) correct / validationData.size() * 100;
        avgConfidence /= validationData.size();

        performanceMetrics.put("accuracy", accuracy);
        performanceMetrics.put("avg_confidence", avgConfidence);

        System.out.println(String.format("Validation: Accuracy=%.2f%%, Confidence=%.4f", accuracy, avgConfidence));
    }

    /**
     * Fine-tune with new samples.
     */
    public void fineTune(List<TrainingSample> newSamples) {
        if (trainingData == null) {
            trainingData = new ArrayList<>();
        }
        trainingData.addAll(newSamples);

        System.out.println("Fine-tuning with " + newSamples.size() + " new samples (total: " + trainingData.size() + ")");
        model.train(trainingData);
        evaluate();
    }

    /**
     * Hyperparameter optimization (simple grid search).
     */
    public void optimizeHyperparameters(List<TrainingSample> samples) {
        double[] learningRates = {0.001, 0.005, 0.01, 0.05};
        double[] regularizations = {0.0001, 0.001, 0.01};
        int[] epochs = {50, 100, 200};

        System.out.println("Starting hyperparameter optimization...");

        double bestAccuracy = 0.0;
        Map<String, Double> bestParams = new HashMap<>();

        for (double lr : learningRates) {
            for (double reg : regularizations) {
                for (int ep : epochs) {
                    // Reset and retrain with new hyperparameters
                    model.setTrainingConfig(lr, reg, ep, 32);
                    train(samples);

                    double accuracy = performanceMetrics.getOrDefault("accuracy", 0.0);
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestParams.put("learning_rate", lr);
                        bestParams.put("regularization", reg);
                        bestParams.put("epochs", (double) ep);
                    }

                    System.out.println(String.format(
                        "  lr=%.4f, reg=%.6f, epochs=%d -> accuracy=%.2f%%",
                        lr, reg, ep, accuracy));
                }
            }
        }

        // Apply best hyperparameters
        if (!bestParams.isEmpty()) {
            double lr = bestParams.getOrDefault("learning_rate", 0.01);
            double reg = bestParams.getOrDefault("regularization", 0.001);
            int ep = bestParams.getOrDefault("epochs", 100.0).intValue();

            model.setTrainingConfig(lr, reg, ep, 32);
            System.out.println("\nOptimal hyperparameters found:");
            System.out.println(String.format("  Learning Rate: %.4f", lr));
            System.out.println(String.format("  Regularization: %.6f", reg));
            System.out.println(String.format("  Epochs: %d", ep));
            System.out.println(String.format("  Best Accuracy: %.2f%%", bestAccuracy));
        }
    }

    /**
     * K-fold cross-validation.
     */
    public void crossValidate(List<TrainingSample> samples, int k) {
        System.out.println("Starting " + k + "-fold cross-validation...");

        List<TrainingSample> shuffled = new ArrayList<>(samples);
        Collections.shuffle(shuffled);

        int foldSize = samples.size() / k;
        double[] foldAccuracies = new double[k];

        for (int fold = 0; fold < k; fold++) {
            int start = fold * foldSize;
            int end = (fold == k - 1) ? samples.size() : (fold + 1) * foldSize;

            List<TrainingSample> testSet = shuffled.subList(start, end);
            List<TrainingSample> trainSet = new ArrayList<>(shuffled);
            trainSet.removeAll(testSet);

            // Train and evaluate
            model.train(trainSet);
            FeatureExtractor extractor = new FeatureExtractor();
            int correct = 0;

            for (TrainingSample sample : testSet) {
                double[] features = extractor.extractNormalized(sample.state());
                Decision prediction = model.predict(features);
                if (prediction.action() == sample.decision().action()) {
                    correct++;
                }
            }

            foldAccuracies[fold] = (double) correct / testSet.size() * 100;
            System.out.println(String.format("  Fold %d: %.2f%%", fold + 1, foldAccuracies[fold]));
        }

        // Average accuracy
        double avgAccuracy = Arrays.stream(foldAccuracies).average().orElse(0.0);
        double stdDev = Math.sqrt(Arrays.stream(foldAccuracies)
            .map(x -> Math.pow(x - avgAccuracy, 2))
            .average().orElse(0.0));

        System.out.println(String.format("Average Accuracy: %.2f%% ± %.2f%%", avgAccuracy, stdDev));
        performanceMetrics.put("cv_accuracy", avgAccuracy);
        performanceMetrics.put("cv_stddev", stdDev);
    }

    /**
     * Save trained model.
     */
    public String saveModel(String modelName, List<TrainingSample> originalSamples) throws IOException {
        double loss = performanceMetrics.getOrDefault("final_loss", 0.0);
        return persistence.saveModel(model, modelName, originalSamples.size(), loss);
    }

    /**
     * Get training report.
     */
    public String getReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║              ML TRAINING REPORT                           ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        sb.append("Model Information:\n");
        sb.append("  ").append(model.getNetworkInfo()).append("\n");
        sb.append("  ").append(model.getStatistics()).append("\n\n");

        sb.append("Performance Metrics:\n");
        for (Map.Entry<String, Double> metric : performanceMetrics.entrySet()) {
            sb.append(String.format("  %-20s: %.6f\n", metric.getKey(), metric.getValue()));
        }

        if (trainingData != null) {
            sb.append(String.format("\nTraining Data: %d samples\n", trainingData.size()));
        }
        if (validationData != null) {
            sb.append(String.format("Validation Data: %d samples\n", validationData.size()));
        }

        return sb.toString();
    }

    /**
     * Get metrics as map.
     */
    public Map<String, Double> getMetrics() {
        return new HashMap<>(performanceMetrics);
    }

    /**
     * Compare with baseline rule-based predictor.
     */
    public void compareWithBaseline(List<TrainingSample> samples) {
        FeatureExtractor extractor = new FeatureExtractor();
        RuleBasedPredictor baseline = new RuleBasedPredictor();

        int nnCorrect = 0;
        int baselineCorrect = 0;

        for (TrainingSample sample : samples) {
            double[] features = extractor.extractNormalized(sample.state());
            SchedulerAction expected = sample.decision().action();

            Decision nnPred = model.predict(features);
            Decision basePred = baseline.predict(features);

            if (nnPred.action() == expected) nnCorrect++;
            if (basePred.action() == expected) baselineCorrect++;
        }

        double nnAccuracy = (double) nnCorrect / samples.size() * 100;
        double baselineAccuracy = (double) baselineCorrect / samples.size() * 100;
        double improvement = nnAccuracy - baselineAccuracy;

        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║           NEURAL NETWORK vs RULE-BASED COMPARISON          ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println(String.format("NN Accuracy:       %.2f%%", nnAccuracy));
        System.out.println(String.format("Baseline Accuracy: %.2f%%", baselineAccuracy));
        System.out.println(String.format("Improvement:       %.2f%% (%s)", improvement,
            improvement > 0 ? "BETTER" : "WORSE"));
    }

    /**
     * Get action distribution in training data.
     */
    public Map<SchedulerAction, Integer> getActionDistribution() {
        Map<SchedulerAction, Integer> distribution = new HashMap<>();
        for (SchedulerAction action : SchedulerAction.values()) {
            distribution.put(action, 0);
        }

        if (trainingData != null) {
            for (TrainingSample sample : trainingData) {
                SchedulerAction action = sample.decision().action();
                distribution.put(action, distribution.get(action) + 1);
            }
        }

        return distribution;
    }

    /**
     * Feedback loop: retrain on execution results with outcome weighting.
     * 
     * Called after scheduler makes decisions and executes them, with metrics
     * on how well those decisions performed (latency improvement, memory saved).
     */
    public void retrainWithFeedback(List<TrainingSample> samplesWithOutcomes) {
        if (samplesWithOutcomes.isEmpty()) {
            System.err.println("No samples with outcomes for retraining");
            return;
        }

        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║         FEEDBACK-DRIVEN RETRAINING                        ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        // Collect outcome statistics
        double avgLatencyImprovement = 0.0;
        double avgMemorySaved = 0.0;
        int samplesWithGoodOutcome = 0;

        for (TrainingSample sample : samplesWithOutcomes) {
            avgLatencyImprovement += sample.latencyImprovement();
            avgMemorySaved += sample.memorySavedBytes();
            if (sample.latencyImprovement() > 0) {
                samplesWithGoodOutcome++;
            }
        }

        avgLatencyImprovement /= samplesWithOutcomes.size();
        avgMemorySaved /= samplesWithOutcomes.size();

        System.out.println("Outcome Statistics:");
        System.out.println(String.format("  Avg Latency Improvement: %.4f ms", avgLatencyImprovement));
        System.out.println(String.format("  Avg Memory Saved: %.2f MB", avgMemorySaved / (1024.0 * 1024.0)));
        System.out.println(String.format("  Good Decisions: %d / %d (%.1f%%)",
            samplesWithGoodOutcome, samplesWithOutcomes.size(),
            (samplesWithGoodOutcome * 100.0) / samplesWithOutcomes.size()));

        // Retrain with outcome weighting
        model.trainWeighted(samplesWithOutcomes, true);

        // Update training data
        if (trainingData == null) {
            trainingData = new ArrayList<>();
        }
        trainingData.addAll(samplesWithOutcomes);

        // Re-evaluate
        evaluate();

        performanceMetrics.put("avg_latency_improvement", avgLatencyImprovement);
        performanceMetrics.put("avg_memory_saved", avgMemorySaved);
        performanceMetrics.put("good_decision_ratio", (double) samplesWithGoodOutcome / samplesWithOutcomes.size());
    }

    /**
     * Online learning: incrementally adapt to a single execution result.
     * 
     * Called immediately after each scheduler decision is executed to provide
     * real-time feedback and improve decision quality.
     */
    public void updateWithExecutionResult(TrainingSample sample, int trainingIterations) {
        System.out.println("\n[Online Learning] Adapting to execution result:");
        System.out.println(String.format("  Action: %s", sample.decision().action()));
        System.out.println(String.format("  Latency Improvement: %.4f ms", sample.latencyImprovement()));
        System.out.println(String.format("  Memory Saved: %.2f MB", sample.memorySavedBytes() / (1024.0 * 1024.0)));

        // Incremental training on this single sample
        model.trainIncremental(sample, trainingIterations);

        // Record for later batch retraining
        if (trainingData == null) {
            trainingData = new ArrayList<>();
        }
        trainingData.add(sample);

        System.out.println(String.format("  Training Data Size: %d samples", trainingData.size()));
    }

    /**
     * Batch periodic retraining: retrain on accumulated samples periodically.
     */
    public void periodicRetrain(int minSamples, boolean weightByOutcome) {
        if (trainingData == null || trainingData.size() < minSamples) {
            System.out.println("Not enough samples for retraining (need " + minSamples + ", have " +
                (trainingData == null ? 0 : trainingData.size()) + ")");
            return;
        }

        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║       PERIODIC BATCH RETRAINING                           ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        if (weightByOutcome) {
            model.trainWeighted(trainingData, true);
        } else {
            model.train(trainingData);
        }

        evaluate();
    }

    /**
     * Get performance improvement metrics from feedback loop.
     */
    public String getFeedbackMetrics() {
        StringBuilder sb = new StringBuilder();
        sb.append("╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║              FEEDBACK LOOP METRICS                        ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        sb.append("Execution Outcomes:\n");
        double latency = performanceMetrics.getOrDefault("avg_latency_improvement", 0.0);
        double memory = performanceMetrics.getOrDefault("avg_memory_saved", 0.0);
        double ratio = performanceMetrics.getOrDefault("good_decision_ratio", 0.0);

        sb.append(String.format("  Avg Latency Improvement: %.4f ms\n", latency));
        sb.append(String.format("  Avg Memory Saved: %.2f MB\n", memory / (1024.0 * 1024.0)));
        sb.append(String.format("  Good Decision Ratio: %.1f%%\n", ratio * 100));

        sb.append("\nModel Learning:\n");
        sb.append(String.format("  Training Samples: %d\n", trainingData == null ? 0 : trainingData.size()));
        sb.append(String.format("  Accuracy: %.2f%%\n", performanceMetrics.getOrDefault("accuracy", 0.0)));
        sb.append(String.format("  Confidence: %.4f\n", performanceMetrics.getOrDefault("avg_confidence", 0.0)));
        sb.append(String.format("  Loss: %.6f\n", model.getLastLoss()));

        return sb.toString();
    }
}
