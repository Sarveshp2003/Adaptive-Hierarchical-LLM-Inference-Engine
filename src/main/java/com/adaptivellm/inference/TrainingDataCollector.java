package com.adaptivellm.inference;

import com.adaptivellm.runtime.NativeInferenceEngine;
import java.io.*;
import java.util.*;

/**
 * Collect real inference data for training scheduler
 */
public class TrainingDataCollector {

    private NativeInferenceEngine engine;
    private List<DataPoint> dataset = new ArrayList<>();

    public static class DataPoint {
        public String prompt;
        public int tokenCount;
        public long inferenceTime;
        public int nextToken;
        public double perplexity;
        public long timestamp;

        @Override
        public String toString() {
            return String.format("%s,%d,%d,%d,%.4f,%d",
                    prompt.replace(",", ";"),
                    tokenCount,
                    inferenceTime,
                    nextToken,
                    perplexity,
                    timestamp
            );
        }
    }

    public TrainingDataCollector() {
        this.engine = new NativeInferenceEngine();
        engine.initialize();
    }

    /**
     * Collect data from a batch of prompts
     */
    public void collectData(List<String> prompts) {
        System.out.println("📊 Collecting inference data from " + prompts.size() + " prompts...\n");

        for (int i = 0; i < prompts.size(); i++) {
            String prompt = prompts.get(i);

            try {
                System.out.print("[" + (i + 1) + "/" + prompts.size() + "] Processing: " +
                        prompt.substring(0, Math.min(40, prompt.length())) + "... ");

                // Tokenize
                int[] tokens = engine.tokenize(prompt);

                // Measure inference time
                long start = System.nanoTime();
                NativeInferenceEngine.InferencePrediction pred = engine.infer(tokens);
                long duration = System.nanoTime() - start;

                // Compute perplexity
                double perplexity = 0;
                if (tokens.length >= 2) {
                    perplexity = engine.computePerplexity(tokens);
                }

                // Create data point
                DataPoint point = new DataPoint();
                point.prompt = prompt;
                point.tokenCount = tokens.length;
                point.inferenceTime = duration / 1_000_000;  // Convert to ms
                point.nextToken = pred.nextToken;
                point.perplexity = perplexity;
                point.timestamp = System.currentTimeMillis();

                dataset.add(point);

                System.out.println("✓ (" + point.tokenCount + " tokens, " +
                        point.inferenceTime + "ms)");

            } catch (Exception e) {
                System.err.println("✗ Error: " + e.getMessage());
            }
        }

        System.out.println("\n✅ Collected " + dataset.size() + " data points");
    }

    /**
     * Export data to CSV for analysis
     */
    public void exportToCSV(String filename) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            // Header
            writer.println("prompt,token_count,inference_time_ms,next_token,perplexity,timestamp");

            // Data rows
            for (DataPoint point : dataset) {
                writer.println(point);
            }
        }

        System.out.println("📁 Exported to: " + filename);
    }

    public void shutdown() {
        engine.shutdown();
    }

    public static void main(String[] args) throws IOException {
        TrainingDataCollector collector = new TrainingDataCollector();

        // 100 diverse prompts
        List<String> prompts = Arrays.asList(
                "Machine learning is",
                "Neural networks are",
                "Deep learning enables",
                "Transformers use attention to",
                "The model learns from data by",
                "Inference means running",
                "Tokens are",
                "Embeddings represent",
                "The transformer architecture has",
                "Large language models can",
                "Fine-tuning a model means",
                "Prompt engineering is the art of",
                "Batch processing helps",
                "Gradient descent optimizes",
                "Backpropagation computes",
                "Attention mechanisms allow",
                "Self-attention enables",
                "Cross-attention combines",
                "Layer normalization stabilizes",
                "Dropout prevents",
                "Regularization reduces",
                "Loss functions measure",
                "Optimization algorithms update",
                "Learning rates control",
                "Momentum accelerates",
                "Adaptive learning rates adapt",
                "Batch size affects",
                "Epochs repeat",
                "Validation sets test",
                "Cross-validation ensures",
                "Overfitting occurs when",
                "Underfitting means",
                "Hyperparameters are",
                "Transfer learning reuses",
                "Zero-shot learning",
                "Few-shot learning uses",
                "In-context learning means",
                "Prompt sensitivity refers to",
                "Chain of thought prompting",
                "System prompts provide"
        );

        // Collect data
        collector.collectData(prompts);

        // Export
        collector.exportToCSV("target/training_data.csv");

        collector.shutdown();
    }
}