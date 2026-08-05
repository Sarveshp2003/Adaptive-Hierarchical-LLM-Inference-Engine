package com.adaptivellm.inference;

import com.adaptivellm.runtime.NativeInferenceEngine;
import com.adaptivellm.scheduler.AdaptiveScheduler;
import com.adaptivellm.scheduler.Decision;
import com.adaptivellm.scheduler.FeatureExtractor;
import com.adaptivellm.scheduler.MemoryState;
import com.adaptivellm.scheduler.NeuralNetworkPredictor;
import com.adaptivellm.scheduler.NativeEngineAdapter;
import com.adaptivellm.scheduler.ScheduledDecision;
import com.adaptivellm.scheduler.SchedulerRuntimeController;
import com.adaptivellm.scheduler.TrainingDataCollector;
import java.io.BufferedReader;
import java.io.Console;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Simple interactive client to chat with Llama model.
 * Direct Java API - Option 1
 */
public class LlamaInferenceClient {

    private static NativeInferenceEngine engine;
    private static SchedulerRuntimeController schedulerController;
    private static AdaptiveScheduler adaptiveScheduler;
    private static NativeEngineAdapter schedulerAdapter;

    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║       LLAMA-3.2-3B INFERENCE CLIENT (Java API)        ║");
        System.out.println("╚════════════════════════════════════════════════════════╝\n");

        boolean nativeAvailable = false;
        try {
            // Initialize engine
            System.out.println("📦 Initializing Llama model...");
            engine = new NativeInferenceEngine();
            engine.initialize();
            System.out.println("✅ Model loaded!\n");
            nativeAvailable = true;

            FeatureExtractor extractor = new FeatureExtractor();
            NeuralNetworkPredictor predictor = new NeuralNetworkPredictor();
            TrainingDataCollector collector = new TrainingDataCollector();
            adaptiveScheduler = new AdaptiveScheduler(extractor, predictor, collector);
            schedulerAdapter = new NativeEngineAdapter(engine);
            schedulerController = new SchedulerRuntimeController(adaptiveScheduler, schedulerAdapter, () -> new MemoryState(
                    0, System.currentTimeMillis(), 0.2, 0.3, 5.0, 8, 2));
            schedulerController.setDecisionIntervalMs(50);
            schedulerController.start();
            System.out.println("🧠 Scheduler started\n");

            // Interactive loop
            runInteractiveMode();

            // Cleanup
            if (schedulerController != null) {
                schedulerController.stop();
            }
            engine.shutdown();

        } catch (UnsatisfiedLinkError | ExceptionInInitializerError e) {
            System.err.println("⚠️  Warning: Native library not available (" + e.getMessage() + ")");
            System.err.println("⚠️  Running in simulator mode - scheduler will work but inference is mocked\n");
            
            try {
                // Still run scheduler in simulation mode
                FeatureExtractor extractor = new FeatureExtractor();
                NeuralNetworkPredictor predictor = new NeuralNetworkPredictor();
                TrainingDataCollector collector = new TrainingDataCollector();
                adaptiveScheduler = new AdaptiveScheduler(extractor, predictor, collector);
                schedulerAdapter = new NativeEngineAdapter(null); // null engine = simulation mode
                schedulerController = new SchedulerRuntimeController(adaptiveScheduler, schedulerAdapter, () -> new MemoryState(
                        0, System.currentTimeMillis(), 0.2, 0.3, 5.0, 8, 2));
                schedulerController.setDecisionIntervalMs(50);
                schedulerController.start();
                System.out.println("🧠 Scheduler started (simulation mode)\n");
                
                runInteractiveMode();
                
                if (schedulerController != null) {
                    schedulerController.stop();
                }
            } catch (Throwable ex) {
                System.err.println("❌ Error in simulator mode: " + ex.getMessage());
                ex.printStackTrace();
                System.exit(1);
            }
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
        Console console = System.console();
        BufferedReader reader = console != null
                ? null
                : new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

        System.out.println("📝 Enter prompts (type 'exit' to quit):\n");

        while (true) {
            String prompt;
            if (console != null) {
                prompt = console.readLine("You: ");
            } else {
                System.out.print("You: ");
                System.out.flush();
                prompt = reader.readLine();
            }

            if (prompt == null) {
                System.out.println("\nNo more input detected. Exiting.");
                break;
            }

            if (prompt.equalsIgnoreCase("exit") || prompt.equalsIgnoreCase("quit")) {
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

        if (reader != null) {
            reader.close();
        }
    }

    /**
     * Process a single prompt through the model and autoregress until EOS or max tokens.
     */
    private static void processPrompt(String prompt) {
        long startTime = System.currentTimeMillis();

        try {
            System.out.println("\n⏳ Processing...");

            // Handle simulator mode (no native engine)
            if (engine == null) {
                simulateInference(prompt);
                return;
            }

            // Step 1: Tokenize
            System.out.print("  [1/4] Tokenizing... ");
            String chatPrompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
            int[] promptTokens = engine.tokenize(chatPrompt);
            int[] visiblePromptTokens = engine.tokenize(prompt);
            System.out.println("✓ " + promptTokens.length + " tokens (prompt=" + visiblePromptTokens.length + ")");

            // Step 2: Ask the scheduler for a decision before each generation step.
            System.out.print("  [2/4] Generating... ");
            final int MAX_GEN = 64;
            int eos = engine.getEosToken();
            if (eos <= 0) eos = 128009; // fallback
            int generated = 0;

            java.util.List<Integer> generatedTokens = new java.util.ArrayList<>();
            int currentToken = promptTokens.length > 0 ? promptTokens[promptTokens.length - 1] : 0;
            StringBuilder streamed = new StringBuilder();

            for (int i = 0; i < MAX_GEN; i++) {
                try {
                    MemoryState state = new MemoryState(
                            i,
                            generated,
                            0.2 + (0.01 * i),
                            0.3 + (0.01 * i),
                            5.0 + (0.5 * i),
                            Math.max(1, 8 + i),
                            Math.max(1, 2 + i / 4)
                    );
                    ScheduledDecision scheduled = adaptiveScheduler != null
                            ? adaptiveScheduler.evaluate(state)
                            : null;
                    if (scheduled != null) {
                        Decision decision = scheduled.decision();
                        if (schedulerAdapter != null) {
                            try {
                                schedulerAdapter.executeDecision(decision);
                            } catch (Exception ignored) {
                                // Scheduler adapter is currently lightweight; ignore execution failures.
                            }
                        }
                        System.out.print("[scheduler:" + decision.action() + "] ");
                    }

                    NativeInferenceEngine.InferencePrediction pred = engine.infer(new int[] { currentToken });
                    int nt = pred.nextToken;
                    generatedTokens.add(nt);
                    generated++;

                    int[] generatedArr = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
                    String currentText = engine.detokenize(generatedArr);
                    if (currentText.length() > streamed.length()) {
                        String delta = currentText.substring(streamed.length());
                        System.out.print(delta);
                        System.out.flush();
                        streamed.append(delta);
                    }

                    currentToken = nt;
                    if (nt == eos) break;
                } catch (RuntimeException ex) {
                    System.err.println("\n[LlamaInferenceClient] inference error: " + ex.getMessage());
                    break;
                }
            }
            System.out.println();
            System.out.println("✓ generated " + generated + " tokens");

            String response = streamed.toString();

            // Step 3: Decode generated tokens for final output text (optional)
            System.out.print("  [3/4] Decoding... ");
            int[] generatedArr = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
            response = engine.detokenize(generatedArr);
            System.out.println("✓");

            // Step 4: Compute perplexity for input tokens
            System.out.print("  [4/4] Computing perplexity... ");
            double perplexity = engine.computePerplexity(visiblePromptTokens);
            System.out.println(String.format("✓ %.4f", perplexity));

            long duration = System.currentTimeMillis() - startTime;

            // Display results
            System.out.println("\n" + "═".repeat(56));
            System.out.println("📊 INFERENCE RESULTS");
            System.out.println("═".repeat(56));
            System.out.println("Input:      " + prompt);
            System.out.println("Tokens:     " + visiblePromptTokens.length);
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

    /**
     * Simulate inference response when native library is unavailable.
     */
    private static void simulateInference(String prompt) {
        System.out.println("  [1/4] Tokenizing... ✓ 35 tokens (prompt=8)");
        System.out.print("  [2/4] Generating... ");
        
        // Simulate scheduler decisions
        final int MAX_GEN = 16;
        int generated = 0;
        String[] responses = {
                "Machine learning is a subset of artificial intelligence",
                "ML enables computers to learn from data without explicit programming",
                "It powers recommendation systems, image recognition, and language models",
                "Neural networks are a key ML technique inspired by biological neurons"
        };
        String response = responses[(int)(Math.random() * responses.length)];
        
        // Simulate token generation with scheduler decisions
        String[] schedulerActions = { "prefetch", "evict", "compress", "keep" };
        for (int i = 0; i < MAX_GEN && i < response.length(); i++) {
            if (i % 4 == 0) {
                System.out.print("[scheduler:" + schedulerActions[i % 4] + "] ");
            }
            System.out.print(response.charAt(i));
            generated++;
            try { Thread.sleep(5); } catch (InterruptedException ignored) {}
        }
        System.out.println();
        System.out.println("✓ generated " + generated + " tokens");
        
        System.out.println("  [3/4] Decoding... ✓");
        System.out.println("  [4/4] Computing perplexity... ✓ 2.1847");
        
        // Display results
        System.out.println("\n" + "═".repeat(56));
        System.out.println("📊 INFERENCE RESULTS (SIMULATED)");
        System.out.println("═".repeat(56));
        System.out.println("Input:      " + prompt);
        System.out.println("Tokens:     8");
        System.out.println("Generated:  " + generated);
        System.out.println("Perplexity: 2.1847");
        System.out.println("Time:       125ms");
        System.out.println("Response:   " + response.substring(0, Math.min(60, response.length())));
        System.out.println("═".repeat(56));
    }
}