package com.adaptivellm.inference;

import com.adaptivellm.runtime.NativeInferenceEngine;
import java.util.Scanner;

/**
 * Simple interactive client to chat with Llama model.
 * Direct Java API - Option 1
 */
public class LlamaInferenceClient {

    private static NativeInferenceEngine engine;

    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║       LLAMA-3.2-3B INFERENCE CLIENT (Java API)        ║");
        System.out.println("╚════════════════════════════════════════════════════════╝\n");

        try {
            // Initialize engine
            System.out.println("📦 Initializing Llama model...");
            engine = new NativeInferenceEngine();
            engine.initialize();
            System.out.println("✅ Model loaded!\n");

            // Interactive loop
            runInteractiveMode();

            // Cleanup
            engine.shutdown();

        } catch (Throwable e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Interactive prompt mode - talk to the model
     */
    private static void runInteractiveMode() throws Exception {
        Scanner scanner = new Scanner(System.in);

        System.out.println("📝 Enter prompts (type 'exit' to quit):\n");

        while (true) {
            System.out.print("You: ");
            String prompt = scanner.nextLine();

            if (prompt.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            if (prompt.trim().isEmpty()) {
                continue;
            }

            // Process prompt
            processPrompt(prompt);
            System.out.println();
        }

        scanner.close();
    }

    /**
     * Process a single prompt through the model and autoregress until EOS or max tokens.
     */
    private static void processPrompt(String prompt) {
        long startTime = System.currentTimeMillis();

        try {
            System.out.println("\n⏳ Processing...");

            // Step 1: Tokenize
            System.out.print("  [1/4] Tokenizing... ");
            int[] tokens = engine.tokenize(prompt);
            System.out.println("✓ " + tokens.length + " tokens");

            // Step 2: Autoregressive generation until EOS
            System.out.print("  [2/4] Generating... ");
            java.util.List<Integer> seq = new java.util.ArrayList<>();
            for (int t : tokens) seq.add(t);

            final int MAX_GEN = 256; // safety cap
            int eos = engine.getEosToken();
            if (eos <= 0) eos = 128009; // fallback
            int generated = 0;
            for (int i = 0; i < MAX_GEN; i++) {
                int[] cur = seq.stream().mapToInt(Integer::intValue).toArray();
                NativeInferenceEngine.InferencePrediction pred = engine.infer(cur);
                int nt = pred.nextToken;
                seq.add(nt);
                generated++;
                if (nt == eos) break;
            }
            System.out.println("✓ generated " + generated + " tokens");

            // Step 3: Decode full sequence (input + generated)
            System.out.print("  [3/4] Decoding... ");
            int[] seqArr = seq.stream().mapToInt(Integer::intValue).toArray();
            String response = engine.detokenize(seqArr);
            System.out.println("✓");

            // Step 4: Compute perplexity for input tokens
            System.out.print("  [4/4] Computing perplexity... ");
            double perplexity = engine.computePerplexity(tokens);
            System.out.println(String.format("✓ %.4f", perplexity));

            long duration = System.currentTimeMillis() - startTime;

            // Display results
            System.out.println("\n" + "═".repeat(56));
            System.out.println("📊 INFERENCE RESULTS");
            System.out.println("═".repeat(56));
            System.out.println("Input:      " + prompt);
            System.out.println("Tokens:     " + tokens.length);
            System.out.println("Generated:  " + generated);
            System.out.println("Perplexity: " + String.format("%.4f", perplexity));
            System.out.println("Time:       " + duration + "ms");
            System.out.println("Response:   " + response);
            System.out.println("═".repeat(56));

        } catch (Exception e) {
            System.err.println("\n❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}