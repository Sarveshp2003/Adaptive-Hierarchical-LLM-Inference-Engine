package com.adaptivellm.inference;

import java.io.*;
import java.util.*;

/**
 * Analyze inference metrics
 */
public class InferenceAnalyzer {

    public static void main(String[] args) throws IOException {
        String csvFile = "target/training_data.csv";

        List<Double> times = new ArrayList<>();
        List<Integer> tokenCounts = new ArrayList<>();
        List<Double> perplexities = new ArrayList<>();

        // Read CSV
        try (Scanner scanner = new Scanner(new File(csvFile))) {
            scanner.nextLine();  // Skip header

            while (scanner.hasNextLine()) {
                String[] parts = scanner.nextLine().split(",");
                times.add(Double.parseDouble(parts[2]));
                tokenCounts.add(Integer.parseInt(parts[1]));
                perplexities.add(Double.parseDouble(parts[4]));
            }
        }

        // Calculate statistics
        double avgTime = times.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgTokens = tokenCounts.stream().mapToDouble(Integer::doubleValue).average().orElse(0);
        double avgPerplexity = perplexities.stream().mapToDouble(Double::doubleValue).average().orElse(0);

        System.out.println("\n╔════════════════════════════════════════════════════╗");
        System.out.println("║         INFERENCE METRICS ANALYSIS                 ║");
        System.out.println("╚════════════════════════════════════════════════════╝");
        System.out.println("\n📊 STATISTICS:");
        System.out.println("  Samples collected:     " + times.size());
        System.out.println("  Avg inference time:    " + String.format("%.2f ms", avgTime));
        System.out.println("  Avg tokens per prompt: " + String.format("%.2f", avgTokens));
        System.out.println("  Avg perplexity:        " + String.format("%.4f", avgPerplexity));
        System.out.println("  Min time:              " + String.format("%.2f ms", Collections.min(times)));
        System.out.println("  Max time:              " + String.format("%.2f ms", Collections.max(times)));
        System.out.println();
    }
}