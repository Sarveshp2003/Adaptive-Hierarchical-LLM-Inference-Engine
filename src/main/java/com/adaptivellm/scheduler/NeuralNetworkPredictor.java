package com.adaptivellm.scheduler;

import java.io.*;
import java.util.*;

/**
 * Neural network-based predictor for adaptive memory scheduling.
 * 
 * Architecture:
 * - Input layer: 8 features (from FeatureExtractor)
 * - Hidden layer 1: 32 neurons with ReLU activation
 * - Hidden layer 2: 16 neurons with ReLU activation
 * - Output layer: 8 neurons (one for each SchedulerAction)
 * 
 * Uses softmax for probability distribution over actions.
 */
public final class NeuralNetworkPredictor implements PredictorModel, Serializable {

    private static final long serialVersionUID = 1L;

    // Network dimensions
    private static final int INPUT_SIZE = 8;
    private static final int HIDDEN_SIZE_1 = 32;
    private static final int HIDDEN_SIZE_2 = 16;
    private static final int OUTPUT_SIZE = 8;

    // Weights and biases
    private double[][] w1;  // Input -> Hidden1
    private double[] b1;
    private double[][] w2;  // Hidden1 -> Hidden2
    private double[] b2;
    private double[][] w3;  // Hidden2 -> Output
    private double[] b3;

    // Training hyperparameters
    private double learningRate = 0.01;
    private double regularization = 0.001;
    private int epochs = 100;
    private int batchSize = 32;

    // Statistics
    private int trainingSamples = 0;
    private double lastLoss = Double.MAX_VALUE;

    /**
     * Create untrained neural network with random initialization.
     */
    public NeuralNetworkPredictor() {
        initializeWeights();
    }

    /**
     * Initialize weights using Xavier initialization.
     */
    private void initializeWeights() {
        Random rand = new Random(42);

        // Layer 1: 8 -> 32
        w1 = new double[INPUT_SIZE][HIDDEN_SIZE_1];
        b1 = new double[HIDDEN_SIZE_1];
        double scale1 = Math.sqrt(2.0 / INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE_1; j++) {
                w1[i][j] = (rand.nextDouble() - 0.5) * 2 * scale1;
            }
        }
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            b1[j] = 0.0;
        }

        // Layer 2: 32 -> 16
        w2 = new double[HIDDEN_SIZE_1][HIDDEN_SIZE_2];
        b2 = new double[HIDDEN_SIZE_2];
        double scale2 = Math.sqrt(2.0 / HIDDEN_SIZE_1);
        for (int i = 0; i < HIDDEN_SIZE_1; i++) {
            for (int j = 0; j < HIDDEN_SIZE_2; j++) {
                w2[i][j] = (rand.nextDouble() - 0.5) * 2 * scale2;
            }
        }
        for (int j = 0; j < HIDDEN_SIZE_2; j++) {
            b2[j] = 0.0;
        }

        // Layer 3: 16 -> 8
        w3 = new double[HIDDEN_SIZE_2][OUTPUT_SIZE];
        b3 = new double[OUTPUT_SIZE];
        double scale3 = Math.sqrt(2.0 / HIDDEN_SIZE_2);
        for (int i = 0; i < HIDDEN_SIZE_2; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                w3[i][j] = (rand.nextDouble() - 0.5) * 2 * scale3;
            }
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            b3[j] = 0.0;
        }
    }

    /**
     * Forward pass through network.
     */
    @Override
    public Decision predict(double[] features) {
        if (features.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Expected " + INPUT_SIZE + " features, got " + features.length);
        }

        // Forward propagation
        double[] hidden1 = forward(features, w1, b1, true);
        double[] hidden2 = forward(hidden1, w2, b2, true);
        double[] output = forward(hidden2, w3, b3, false);

        // Apply softmax to get probabilities
        double[] probs = softmax(output);

        // Select action with highest probability
        int maxActionIdx = 0;
        double maxProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxActionIdx = i;
            }
        }

        SchedulerAction action = indexToAction(maxActionIdx);
        long targetId = (long)(features[0] * 1000);  // Use layer as target

        return new Decision(action, targetId, maxProb);
    }

    /**
     * Forward pass for single layer.
     */
    private double[] forward(double[] input, double[][] weights, double[] bias, boolean useReLU) {
        int outputSize = weights[0].length;
        double[] output = new double[outputSize];

        // Matrix multiplication
        for (int j = 0; j < outputSize; j++) {
            output[j] = bias[j];
            for (int i = 0; i < input.length; i++) {
                output[j] += input[i] * weights[i][j];
            }
        }

        // Apply activation
        if (useReLU) {
            for (int j = 0; j < outputSize; j++) {
                output[j] = Math.max(0.0, output[j]);  // ReLU
            }
        }

        return output;
    }

    /**
     * Train on collected samples.
     */
    public void train(List<TrainingSample> samples) {
        if (samples.isEmpty()) {
            System.out.println("No training samples available");
            return;
        }

        trainingSamples = samples.size();
        System.out.println("Starting training on " + samples.size() + " samples");

        FeatureExtractor extractor = new FeatureExtractor();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;

            // Mini-batch gradient descent
            for (int batch = 0; batch < samples.size(); batch += batchSize) {
                int batchEnd = Math.min(batch + batchSize, samples.size());

                for (int i = batch; i < batchEnd; i++) {
                    TrainingSample sample = samples.get(i);
                    double[] features = extractor.extractNormalized(sample.state());
                    Decision decision = sample.decision();

                    // Forward pass
                    double[] pred = predictRaw(features);

                    // Target: one-hot encoding for action
                    double[] target = new double[OUTPUT_SIZE];
                    target[actionToIndex(decision.action())] = 1.0;

                    // Calculate loss (cross-entropy)
                    double loss = crossEntropyLoss(pred, target);
                    epochLoss += loss;

                    // Backprop (simplified gradient update)
                    updateWeights(features, pred, target);
                }
            }

            epochLoss /= samples.size();
            lastLoss = epochLoss;

            if ((epoch + 1) % 10 == 0) {
                System.out.println("Epoch " + (epoch + 1) + "/" + epochs + " - Loss: " + String.format("%.6f", epochLoss));
            }
        }

        System.out.println("Training complete. Final loss: " + String.format("%.6f", lastLoss));
    }

    /**
     * Raw prediction output (before softmax).
     */
    private double[] predictRaw(double[] features) {
        double[] h1 = forward(features, w1, b1, true);
        double[] h2 = forward(h1, w2, b2, true);
        return forward(h2, w3, b3, false);
    }

    /**
     * Softmax activation for output.
     */
    private double[] softmax(double[] x) {
        double max = x[0];
        for (double v : x) {
            if (v > max) max = v;
        }

        double[] exp = new double[x.length];
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < exp.length; i++) {
            exp[i] /= sum;
        }

        return exp;
    }

    /**
     * Cross-entropy loss.
     */
    private double crossEntropyLoss(double[] pred, double[] target) {
        double loss = 0.0;
        double[] probs = softmax(pred);
        for (int i = 0; i < probs.length; i++) {
            if (target[i] == 1.0) {
                loss -= Math.log(Math.max(probs[i], 1e-10));
            }
        }
        return loss;
    }

    /**
     * Simplified weight update using gradient descent.
     */
    private void updateWeights(double[] features, double[] pred, double[] target) {
        double[] probs = softmax(pred);
        double[] outputError = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            outputError[i] = (probs[i] - target[i]) * learningRate;
        }

        // Update output layer (w3, b3)
        double[] h2 = forward(forward(features, w1, b1, true), w2, b2, true);
        for (int i = 0; i < HIDDEN_SIZE_2; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                w3[i][j] -= outputError[j] * h2[i] + regularization * w3[i][j];
            }
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            b3[j] -= outputError[j];
        }
    }

    /**
     * Convert action to network output index.
     */
    private int actionToIndex(SchedulerAction action) {
        switch (action) {
            case PREFETCH_LAYER:     return 0;
            case EVICT_LAYER:        return 1;
            case KEEP_LAYER:         return 2;
            case MOVE_KV_TO_RAM:     return 3;
            case MOVE_KV_TO_GPU:     return 4;
            case COMPRESS_KV:        return 5;
            case OFFLOAD_KV:         return 6;
            case NO_ACTION:          return 7;
            default:                 return 7;
        }
    }

    /**
     * Convert network output index to action.
     */
    private SchedulerAction indexToAction(int index) {
        switch (index) {
            case 0:  return SchedulerAction.PREFETCH_LAYER;
            case 1:  return SchedulerAction.EVICT_LAYER;
            case 2:  return SchedulerAction.KEEP_LAYER;
            case 3:  return SchedulerAction.MOVE_KV_TO_RAM;
            case 4:  return SchedulerAction.MOVE_KV_TO_GPU;
            case 5:  return SchedulerAction.COMPRESS_KV;
            case 6:  return SchedulerAction.OFFLOAD_KV;
            case 7:  return SchedulerAction.NO_ACTION;
            default: return SchedulerAction.NO_ACTION;
        }
    }

    /**
     * Get training statistics.
     */
    public String getStatistics() {
        return String.format(
            "NeuralNetworkPredictor{samples=%d, loss=%.6f, lr=%.4f, reg=%.6f}",
            trainingSamples, lastLoss, learningRate, regularization
        );
    }

    /**
     * Configure training hyperparameters.
     */
    public void setTrainingConfig(double lr, double reg, int epochs, int batchSize) {
        this.learningRate = lr;
        this.regularization = reg;
        this.epochs = epochs;
        this.batchSize = batchSize;
    }

    /**
     * Get network size info.
     */
    public String getNetworkInfo() {
        return String.format(
            "Network: %d -> %d -> %d -> %d (8 actions)",
            INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE
        );
    }
}
