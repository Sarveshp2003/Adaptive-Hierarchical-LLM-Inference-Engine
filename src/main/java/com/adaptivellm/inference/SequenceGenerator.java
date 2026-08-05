package com.adaptivellm.inference;

import com.adaptivellm.runtime.NativeInferenceEngine;
import java.util.*;

/**
 * Generate full text sequences token by token
 */
public class SequenceGenerator {

    private NativeInferenceEngine engine;
    private static final int MAX_TOKENS = 50;  // Limit output

    public SequenceGenerator() {
        this.engine = new NativeInferenceEngine();
        engine.initialize();
    }

    /**
     * Generate text continuation
     */
    public String generate(String prompt, int maxTokens) {
        try {
            System.out.println("🚀 Generating sequence...");

            // Start with prompt tokens
            int[] tokens = engine.tokenize(prompt);
            List<Integer> generatedTokens = new ArrayList<>();
            for (int t : tokens) {
                generatedTokens.add(t);
            }

            // Generate tokens one by one
            for (int i = 0; i < maxTokens; i++) {
                // Inference
                NativeInferenceEngine.InferencePrediction pred =
                        engine.infer(Arrays.stream(generatedTokens.stream()
                                        .mapToInt(Integer::intValue).toArray()).boxed()
                                .mapToInt(Integer::intValue).toArray());

                int nextToken = pred.nextToken;
                generatedTokens.add(nextToken);

                System.out.print(".");
                if ((i + 1) % 10 == 0) {
                    System.out.println(" [" + (i + 1) + " tokens]");
                }
            }

            System.out.println("\n✓ Generation complete!");

            // Decode full sequence
            int[] allTokens = generatedTokens.stream()
                    .mapToInt(Integer::intValue).toArray();
            return engine.detokenize(allTokens);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            return "ERROR";
        }
    }

    public void shutdown() {
        engine.shutdown();
    }

    public static void main(String[] args) {
        SequenceGenerator gen = new SequenceGenerator();

        String prompt = "Machine learning is";
        String result = gen.generate(prompt, 30);

        System.out.println("\n📝 GENERATED TEXT:");
        System.out.println("─".repeat(50));
        System.out.println(result);
        System.out.println("─".repeat(50));

        gen.shutdown();
    }
}