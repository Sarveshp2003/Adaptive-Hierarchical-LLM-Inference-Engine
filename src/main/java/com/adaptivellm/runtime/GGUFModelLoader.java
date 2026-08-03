package com.adaptivellm.runtime;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * GGUF model loader for Java.
 * 
 * Loads GGUF format models and coordinates with native engine.
 */
public final class GGUFModelLoader {

    private final RuntimeBridgeClient nativeClient;
    private String modelPath;
    private ModelMetadata metadata;
    private boolean loaded;

    public GGUFModelLoader(RuntimeBridgeClient nativeClient) {
        this.nativeClient = Objects.requireNonNull(
            nativeClient,
            "Native client cannot be null"
        );
        this.loaded = false;
    }

    /**
     * Load GGUF model file.
     * 
     * This is a metadata-only operation - weights are streamed on demand.
     */
    public boolean loadModel(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IllegalArgumentException(
                "Model file not found: " + filePath
            );
        }

        this.modelPath = filePath;

        // Call native loader
        boolean success = nativeLoadModelGGUF(filePath);
        if (!success) {
            throw new java.lang.RuntimeException(
                "Failed to load GGUF model: " + filePath
            );
        }

        // Extract metadata from native side
        this.metadata = extractMetadata();
        this.loaded = true;

        System.out.println("GGUF model loaded: " + filePath);
        System.out.println("  Layers: " + metadata.numLayers);
        System.out.println("  Hidden: " + metadata.hiddenDim);
        System.out.println("  Heads: " + metadata.numHeads);

        return true;
    }

    /**
     * Load specific layer from model.
     * 
     * Streams weights from SSD to GPU via native engine.
     */
    public boolean loadLayer(int layerId) {
        if (!loaded) {
            throw new IllegalStateException("Model not loaded");
        }

        if (layerId < 0 || layerId >= metadata.numLayers) {
            throw new IllegalArgumentException(
                "Invalid layer ID: " + layerId
            );
        }

        return nativeLoadLayerFromGGUF(layerId);
    }

    /**
     * Prefetch multiple layers asynchronously.
     */
    public void prefetchLayers(int... layerIds) {
        for (int layerId : layerIds) {
            if (layerId >= 0 && layerId < metadata.numLayers) {
                // Async load - non-blocking
                new Thread(() -> loadLayer(layerId)).start();
            }
        }
    }

    /**
     * Get model metadata.
     */
    public ModelMetadata getMetadata() {
        if (!loaded) {
            throw new IllegalStateException("Model not loaded");
        }
        return metadata;
    }

    /**
     * Get model file path.
     */
    public String getModelPath() {
        return modelPath;
    }

    /**
     * Check if model is loaded.
     */
    public boolean isLoaded() {
        return loaded;
    }

    // ============ Native Calls ============

    private native boolean nativeLoadModelGGUF(String filePath);
    private native boolean nativeLoadLayerFromGGUF(int layerId);
    private native ModelMetadata nativeGetMetadata();

    private ModelMetadata extractMetadata() {
        return nativeGetMetadata();
    }

    // ============ Model Metadata ============

    public static class ModelMetadata {
        public int numLayers;
        public int hiddenDim;
        public int numHeads;
        public int vocabSize;
        public long totalSize;
        public String modelName;

        public ModelMetadata(int numLayers, int hiddenDim, int numHeads, 
                            int vocabSize, long totalSize, String modelName) {
            this.numLayers = numLayers;
            this.hiddenDim = hiddenDim;
            this.numHeads = numHeads;
            this.vocabSize = vocabSize;
            this.totalSize = totalSize;
            this.modelName = modelName;
        }

        @Override
        public String toString() {
            return String.format(
                "ModelMetadata{layers=%d, hidden=%d, heads=%d, vocab=%d, size=%dMB}",
                numLayers, hiddenDim, numHeads, vocabSize, totalSize / (1024*1024)
            );
        }
    }

    static {
        // Load native library
        try {
            System.loadLibrary("adaptive_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native library: " + e.getMessage());
        }
    }
}
