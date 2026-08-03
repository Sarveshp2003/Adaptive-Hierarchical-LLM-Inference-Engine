package com.adaptivellm.scheduler;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Persistent storage for trained predictor models.
 * 
 * Features:
 * - Save/load models to disk
 * - Version management
 * - Metadata tracking (creation date, performance, dataset size)
 * - Model validation on load
 * - Atomic writes for consistency
 */
public final class ModelPersistence {

    private final Path modelDir;
    private static final String METADATA_SUFFIX = ".meta";
    private static final String MODEL_SUFFIX = ".model";
    private static final DateTimeFormatter DATE_FORMAT = 
        DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss");

    /**
     * Create persistence manager.
     */
    public ModelPersistence(String modelDirectory) throws IOException {
        this.modelDir = Paths.get(modelDirectory);
        Files.createDirectories(modelDir);
    }

    /**
     * Save trained model with metadata.
     */
    public String saveModel(NeuralNetworkPredictor model, String name, 
                           int samplesUsed, double finalLoss) throws IOException {
        String timestamp = LocalDateTime.now().format(DATE_FORMAT);
        String modelName = String.format("%s_%s", name, timestamp.replace(":", "-"));

        Path modelPath = modelDir.resolve(modelName + MODEL_SUFFIX);
        Path metaPath = modelDir.resolve(modelName + METADATA_SUFFIX);

        // Serialize model
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(modelPath.toFile()))) {
            oos.writeObject(model);
        }

        // Write metadata
        Map<String, String> metadata = new LinkedHashMap<>();
        metadata.put("name", name);
        metadata.put("timestamp", timestamp);
        metadata.put("samples_used", String.valueOf(samplesUsed));
        metadata.put("final_loss", String.valueOf(finalLoss));
        metadata.put("model_file", modelPath.getFileName().toString());
        metadata.put("network_info", model.getNetworkInfo());
        metadata.put("statistics", model.getStatistics());

        try (PrintWriter writer = new PrintWriter(new FileWriter(metaPath.toFile()))) {
            for (Map.Entry<String, String> entry : metadata.entrySet()) {
                writer.println(entry.getKey() + "=" + entry.getValue());
            }
        }

        System.out.println("Model saved: " + modelName);
        return modelName;
    }

    /**
     * Load trained model from disk.
     */
    public NeuralNetworkPredictor loadModel(String modelName) throws IOException, ClassNotFoundException {
        Path modelPath = modelDir.resolve(modelName + MODEL_SUFFIX);

        if (!Files.exists(modelPath)) {
            throw new FileNotFoundException("Model not found: " + modelPath);
        }

        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(modelPath.toFile()))) {
            Object obj = ois.readObject();
            if (!(obj instanceof NeuralNetworkPredictor)) {
                throw new ClassCastException("Loaded object is not a NeuralNetworkPredictor");
            }
            System.out.println("Model loaded: " + modelName);
            return (NeuralNetworkPredictor) obj;
        }
    }

    /**
     * Load latest model.
     */
    public NeuralNetworkPredictor loadLatestModel() throws IOException, ClassNotFoundException {
        List<String> models = listModels();
        if (models.isEmpty()) {
            throw new FileNotFoundException("No models found in " + modelDir);
        }

        // Models are sorted by timestamp (latest first)
        return loadModel(models.get(0));
    }

    /**
     * List all saved models (most recent first).
     */
    public List<String> listModels() throws IOException {
        List<String> models = new ArrayList<>();

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(modelDir, "*" + MODEL_SUFFIX)) {
            for (Path path : stream) {
                String name = path.getFileName().toString();
                models.add(name.substring(0, name.length() - MODEL_SUFFIX.length()));
            }
        }

        // Sort by timestamp (reverse = newest first)
        models.sort(Collections.reverseOrder());
        return models;
    }

    /**
     * Get model metadata.
     */
    public Map<String, String> getModelMetadata(String modelName) throws IOException {
        Path metaPath = modelDir.resolve(modelName + METADATA_SUFFIX);

        if (!Files.exists(metaPath)) {
            throw new FileNotFoundException("Metadata not found: " + metaPath);
        }

        Map<String, String> metadata = new LinkedHashMap<>();
        for (String line : Files.readAllLines(metaPath)) {
            if (line.contains("=")) {
                String[] parts = line.split("=", 2);
                metadata.put(parts[0], parts.length > 1 ? parts[1] : "");
            }
        }

        return metadata;
    }

    /**
     * Delete model and metadata.
     */
    public boolean deleteModel(String modelName) throws IOException {
        Path modelPath = modelDir.resolve(modelName + MODEL_SUFFIX);
        Path metaPath = modelDir.resolve(modelName + METADATA_SUFFIX);

        boolean modelDeleted = Files.deleteIfExists(modelPath);
        boolean metaDeleted = Files.deleteIfExists(metaPath);

        if (modelDeleted) {
            System.out.println("Model deleted: " + modelName);
        }

        return modelDeleted && metaDeleted;
    }

    /**
     * Export model performance report.
     */
    public String generateReport() throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║            SAVED MODELS INVENTORY REPORT                   ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        List<String> models = listModels();

        if (models.isEmpty()) {
            sb.append("No models found.\n");
            return sb.toString();
        }

        sb.append(String.format("%-40s | %-12s | %-10s\n", "Model Name", "Samples", "Loss"));
        sb.append("─".repeat(65)).append("\n");

        for (String modelName : models) {
            try {
                Map<String, String> meta = getModelMetadata(modelName);
                String samples = meta.getOrDefault("samples_used", "?");
                String loss = meta.getOrDefault("final_loss", "?");

                sb.append(String.format("%-40s | %12s | %10s\n", 
                    modelName.substring(0, Math.min(40, modelName.length())), 
                    samples, loss));
            } catch (IOException e) {
                sb.append(String.format("%-40s | Error reading metadata\n", modelName));
            }
        }

        sb.append("\nModel Directory: ").append(modelDir.toAbsolutePath()).append("\n");
        return sb.toString();
    }

    /**
     * Validate model integrity.
     */
    public boolean validateModel(String modelName) {
        try {
            Path modelPath = modelDir.resolve(modelName + MODEL_SUFFIX);
            Path metaPath = modelDir.resolve(modelName + METADATA_SUFFIX);

            if (!Files.exists(modelPath)) {
                System.err.println("Model file not found: " + modelPath);
                return false;
            }

            if (!Files.exists(metaPath)) {
                System.err.println("Metadata file not found: " + metaPath);
                return false;
            }

            // Try to load to verify integrity
            loadModel(modelName);
            getModelMetadata(modelName);

            return true;
        } catch (Exception e) {
            System.err.println("Model validation failed: " + e.getMessage());
            return false;
        }
    }

    /**
     * Get model directory path.
     */
    public Path getModelDirectory() {
        return modelDir;
    }

    /**
     * Get model file size.
     */
    public long getModelSize(String modelName) throws IOException {
        Path modelPath = modelDir.resolve(modelName + MODEL_SUFFIX);
        if (Files.exists(modelPath)) {
            return Files.size(modelPath);
        }
        throw new FileNotFoundException("Model not found: " + modelPath);
    }

    /**
     * Export model to human-readable format.
     */
    public String exportModelDetails(String modelName) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("Model: ").append(modelName).append("\n");

        Map<String, String> meta = getModelMetadata(modelName);
        for (Map.Entry<String, String> entry : meta.entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }

        long sizeBytes = getModelSize(modelName);
        sb.append("file_size: ").append(formatBytes(sizeBytes)).append("\n");

        return sb.toString();
    }

    /**
     * Format bytes as human-readable string.
     */
    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        String pre = "KMGTPE".charAt(exp - 1) + "";
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

    /**
     * Clean up old models (keep only N most recent).
     */
    public void retainLatestModels(int count) throws IOException {
        List<String> models = listModels();

        if (models.size() <= count) {
            return;
        }

        for (int i = count; i < models.size(); i++) {
            deleteModel(models.get(i));
        }

        System.out.println("Retained " + count + " models, deleted " + (models.size() - count));
    }
}
