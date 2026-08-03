package com.adaptivellm.runtime;

import java.util.*;
import java.io.File;

/**
 * Model validation and health checking system.
 * 
 * Verifies:
 * - Model file integrity
 * - Configuration validity
 * - Weight ranges and NaN/Inf checks
 * - Layer count and dimensions
 * - Hardware compatibility
 */
public final class ModelValidator {

    private final List<ValidationError> errors;
    private final List<ValidationWarning> warnings;
    private boolean strictMode;

    public ModelValidator(boolean strictMode) {
        this.strictMode = strictMode;
        this.errors = new ArrayList<>();
        this.warnings = new ArrayList<>();
    }

    /**
     * Validate GGUF model file
     */
    public ValidationResult validateGGUFModel(String filePath) {
        errors.clear();
        warnings.clear();

        File file = new File(filePath);
        
        // Check file exists
        if (!file.exists()) {
            addError("Model file not found: " + filePath);
            return new ValidationResult(false, errors, warnings);
        }

        // Check file size
        long fileSize = file.length();
        if (fileSize < 1024) {  // Less than 1KB
            addError("Model file too small: " + fileSize + " bytes");
            return new ValidationResult(false, errors, warnings);
        }

        if (fileSize > 500L * 1024 * 1024 * 1024) {  // More than 500GB
            addWarning("Model file very large: " + formatBytes(fileSize));
        }

        // Validate via native call
        boolean isValid = nativeValidateModel(filePath);
        if (!isValid) {
            addError("Native model validation failed");
        }

        return new ValidationResult(isValid && errors.isEmpty(), errors, warnings);
    }

    /**
     * Validate model configuration
     */
    public ValidationResult validateConfig(ModelConfig config) {
        errors.clear();
        warnings.clear();

        // Check basic parameters
        if (config.numLayers <= 0 || config.numLayers > 1000) {
            addError("Invalid layer count: " + config.numLayers);
        }

        if (config.hiddenDim <= 0 || config.hiddenDim % 64 != 0) {
            addError("Invalid hidden dimension: " + config.hiddenDim);
        }

        if (config.numHeads <= 0 || config.hiddenDim % config.numHeads != 0) {
            addError("Hidden dim not divisible by num_heads");
        }

        if (config.vocabSize <= 0 || config.vocabSize > 1000000) {
            addError("Invalid vocab size: " + config.vocabSize);
        }

        // Check compatibility
        long requiredMemory = estimateMemoryRequirement(config);
        long availableMemory = getAvailableMemory();

        if (requiredMemory > availableMemory) {
            if (strictMode) {
                addError("Insufficient memory: need " + formatBytes(requiredMemory) + 
                        ", have " + formatBytes(availableMemory));
            } else {
                addWarning("Model may not fit in memory: need " + formatBytes(requiredMemory));
            }
        }

        return new ValidationResult(errors.isEmpty(), errors, warnings);
    }

    /**
     * Check weight tensor validity
     */
    public ValidationResult validateWeights(String tensorName, float[] weights) {
        errors.clear();
        warnings.clear();

        // Check for NaN
        int nanCount = 0;
        int infCount = 0;
        float minVal = Float.MAX_VALUE;
        float maxVal = -Float.MAX_VALUE;

        for (float w : weights) {
            if (Float.isNaN(w)) nanCount++;
            if (Float.isInfinite(w)) infCount++;
            minVal = Math.min(minVal, w);
            maxVal = Math.max(maxVal, w);
        }

        if (nanCount > 0) {
            addError(tensorName + " contains " + nanCount + " NaN values");
        }

        if (infCount > 0) {
            addError(tensorName + " contains " + infCount + " Inf values");
        }

        // Check value range
        float absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
        if (absMax > 100.0f) {
            addWarning(tensorName + " has large values: [" + minVal + ", " + maxVal + "]");
        }

        if (absMax < 1e-6f) {
            addWarning(tensorName + " has very small values (possible underflow)");
        }

        return new ValidationResult(errors.isEmpty(), errors, warnings);
    }

    /**
     * Hardware compatibility check
     */
    public ValidationResult validateHardware() {
        errors.clear();
        warnings.clear();

        // Check GPU
        boolean hasGPU = nativeCheckGPU();
        if (!hasGPU) {
            if (strictMode) {
                addError("No GPU available (CUDA not detected)");
            } else {
                addWarning("No GPU available (will use CPU)");
            }
        }

        // Check GPU memory
        long gpuMemory = nativeGetGPUMemory();
        if (gpuMemory < 2L * 1024 * 1024 * 1024) {  // Less than 2GB
            addWarning("Low GPU memory: " + formatBytes(gpuMemory));
        }

        // Check CUDA compute capability
        int computeCapability = nativeGetComputeCapability();
        if (computeCapability < 70) {  // Pre-Volta
            addWarning("GPU compute capability low: " + (computeCapability / 10) + "." + (computeCapability % 10));
        }

        return new ValidationResult(errors.isEmpty(), errors, warnings);
    }

    // ============ Native Methods ============

    private native boolean nativeValidateModel(String filePath);
    private native boolean nativeCheckGPU();
    private native long nativeGetGPUMemory();
    private native int nativeGetComputeCapability();

    // ============ Helper Methods ============

    private long estimateMemoryRequirement(ModelConfig config) {
        // Rough estimate: 4 bytes per parameter
        long numParams = (long) config.numLayers * config.hiddenDim * config.hiddenDim * 8;
        return numParams * 4;  // FP32
    }

    private long getAvailableMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory());
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        String pre = "KMGTPE".charAt(exp - 1) + "";
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

    private void addError(String message) {
        errors.add(new ValidationError(message));
    }

    private void addWarning(String message) {
        warnings.add(new ValidationWarning(message));
    }

    // ============ Data Classes ============

    public static class ValidationResult {
        public final boolean isValid;
        public final List<ValidationError> errors;
        public final List<ValidationWarning> warnings;

        public ValidationResult(boolean isValid, List<ValidationError> errors, 
                               List<ValidationWarning> warnings) {
            this.isValid = isValid;
            this.errors = new ArrayList<>(errors);
            this.warnings = new ArrayList<>(warnings);
        }

        public void print() {
            if (isValid) {
                System.out.println("✓ Validation passed");
            } else {
                System.out.println("✗ Validation failed");
                for (ValidationError err : errors) {
                    System.out.println("  ERROR: " + err.message);
                }
            }
            for (ValidationWarning warn : warnings) {
                System.out.println("  WARNING: " + warn.message);
            }
        }
    }

    public static class ValidationError {
        public final String message;
        public final long timestamp;

        public ValidationError(String message) {
            this.message = message;
            this.timestamp = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return "ERROR: " + message;
        }
    }

    public static class ValidationWarning {
        public final String message;
        public final long timestamp;

        public ValidationWarning(String message) {
            this.message = message;
            this.timestamp = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return "WARNING: " + message;
        }
    }

    public static class ModelConfig {
        public int numLayers;
        public int hiddenDim;
        public int numHeads;
        public int vocabSize;
        public String modelType;

        public ModelConfig(int layers, int hidden, int heads, int vocab) {
            this.numLayers = layers;
            this.hiddenDim = hidden;
            this.numHeads = heads;
            this.vocabSize = vocab;
            this.modelType = "llama";
        }
    }

    static {
        try {
            System.loadLibrary("adaptive_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native library: " + e.getMessage());
        }
    }
}
