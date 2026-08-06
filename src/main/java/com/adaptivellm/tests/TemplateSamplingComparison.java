package com.adaptivellm.tests;

import com.adaptivellm.runtime.NativeInferenceEngine;
import java.io.FileWriter;
import java.nio.file.Paths;

/**
 * Compare generation outputs for several prompt templates using temperature 0 (deterministic)
 * vs temperature 0.7 (sampled). Writes results to template_sampling_report.txt in repo root.
 *
 * Run after building native DLL and Java package:
 *   mvn -DskipTests package
 *   java -Djava.library.path=path\to\dll -cp target\classes com.adaptivellm.tests.TemplateSamplingComparison
 */
public class TemplateSamplingComparison {
    public static void main(String[] args) throws Exception {
        NativeInferenceEngine engine = new NativeInferenceEngine();
        engine.initialize();

        String[] templates = new String[] {
            "Explain adaptive learning",
            "You are a helpful assistant. Explain adaptive learning concisely.",
            "Instruction: Explain adaptive learning",
            "User: what is adaptive learning\nAssistant:",
            "System: You are a helpful assistant.\nUser: what is adaptive learning\nAssistant:"
        };

        String outPath = "template_sampling_report.txt";
        try (FileWriter fw = new FileWriter(outPath)) {
            fw.write("Template sampling comparison report\n\n");

            for (String t : templates) {
                fw.write("=== Template: " + t + " ===\n");
                String modelTemplate = engine.getChatTemplate();
                String chatPrompt = (modelTemplate != null && !modelTemplate.isBlank())
                    ? modelTemplate + "\nUser: " + t + "\nAssistant: "
                    : "System: You are a helpful assistant.\nUser: " + t + "\nAssistant: ";

                fw.write("Prompt: " + chatPrompt.replaceAll("\n","\\n") + "\n");

                int[] promptTokens = engine.tokenize(chatPrompt);

                // Run deterministic (temp=0)
                String detOut = generateWithSampling(engine, promptTokens, 0.0, 1, 1.0, new int[0], 1.0);
                fw.write("[temp=0]\n" + detOut + "\n\n");

                // Run sampled (temp=0.7)
                String sampOut = generateWithSampling(engine, promptTokens, 0.7, 40, 0.9, new int[0], 1.1);
                fw.write("[temp=0.7]\n" + sampOut + "\n\n");

                fw.write("----------------------------------------\n\n");
            }

            fw.flush();
        }

        System.out.println("Report written to: " + Paths.get(outPath).toAbsolutePath());
        engine.shutdown();
    }

    private static String generateWithSampling(NativeInferenceEngine engine, int[] promptTokens, double temperature, int topK, double topP, int[] recent, double repPenalty) {
        final int MAX_GEN = 128;
        int eos = engine.getEosToken(); if (eos <= 0) eos = 128009;

        StringBuilder sb = new StringBuilder();
        java.util.List<Integer> generated = new java.util.ArrayList<>();

        // Prime context by calling infer once with full prompt
        NativeInferenceEngine.InferencePrediction init = engine.infer(promptTokens);
        int current = init.nextToken;
        generated.add(current);
        sb.append(engine.detokenize(new int[]{current}));

        for (int i = 1; i < MAX_GEN; i++) {
            int[] recentArr = generated.stream().mapToInt(Integer::intValue).toArray();
            NativeInferenceEngine.InferencePrediction p = engine.inferWithSampling(new int[]{current}, temperature, topK, topP, recentArr, repPenalty);
            int nt = p.nextToken;
            generated.add(nt);
            String text = engine.detokenize(new int[]{nt});
            sb.append(text);
            current = nt;
            if (nt == eos) break;
        }

        return sb.toString();
    }
}
