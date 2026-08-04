package com.adaptivellm.scheduler;

import java.util.Objects;
import java.util.Random;

/**
 * Provides current runtime memory state.
 *
 * In production, this would fetch real state from the native runtime.
 * For now, it simulates realistic state variations.
 */
public final class RuntimeMemoryStateProvider implements SchedulerRuntimeController.MemoryStateProvider {

    private final int totalLayers;
    private final Random random = new Random(42); // Seeded for reproducibility
    private long lastUpdateMs = System.currentTimeMillis();
    private int currentLayer = 0;

    // Simulated state
    private long gpuMemoryUsedBytes = 2L * 1024 * 1024 * 1024; // 2GB
    private long ramMemoryUsedBytes = 4L * 1024 * 1024 * 1024; // 4GB
    private long kvCacheBytesGpu = 500 * 1024 * 1024; // 500MB
    private long kvCacheBytesRam = 200 * 1024 * 1024; // 200MB
    private double gpuUtilization = 0.75;
    private double ramUtilization = 0.80;

    public RuntimeMemoryStateProvider(int totalLayers) {
        if (totalLayers < 1 || totalLayers > 100) {
            throw new IllegalArgumentException("Total layers must be 1-100");
        }
        this.totalLayers = totalLayers;
    }

    @Override
    public MemoryState getCurrentState() {
        // Simulate time progression
        long now = System.currentTimeMillis();
        long elapsed = now - lastUpdateMs;
        lastUpdateMs = now;

        // Simulate sequential layer inference with KV cache accumulation
        currentLayer = (currentLayer + 1) % totalLayers;

        // Simulate memory pressure variations
        simulateMemoryPressure(elapsed);

        // Build state using actual MemoryState constructor
        return new MemoryState(
                currentLayer,           // currentLayer
                currentLayer * 128,     // currentToken (128 tokens per layer)
                gpuUtilization,         // gpuUsage
                ramUtilization,         // ramUsage
                5.0,                    // storageLatency (5ms typical)
                (currentLayer + 2) % totalLayers, // cachedLayers (2 prefetched)
                (int) (kvCacheBytesGpu / 4096)   // kvPages
        );
    }

    /**
     * Simulate realistic memory pressure over time.
     */
    private void simulateMemoryPressure(long elapsedMs) {
        // KV cache grows with each token
        double kvGrowthPerMs = 100 * 1024; // 100KB per ms
        kvCacheBytesGpu += (long) (elapsedMs * kvGrowthPerMs);
        kvCacheBytesRam += (long) (elapsedMs * kvGrowthPerMs * 0.5); // Half rate on RAM

        // Cap KV cache at realistic limits
        long maxKvGpu = 2L * 1024 * 1024 * 1024; // 2GB max
        long maxKvRam = 4L * 1024 * 1024 * 1024; // 4GB max

        if (kvCacheBytesGpu > maxKvGpu) {
            kvCacheBytesGpu = maxKvGpu;
        }
        if (kvCacheBytesRam > maxKvRam) {
            kvCacheBytesRam = maxKvRam;
        }

        // Simulate utilization trends
        double trend = 0.5 + 0.4 * Math.sin(System.currentTimeMillis() / 5000.0);
        gpuUtilization = 0.6 + trend * 0.3; // 0.6-0.9
        ramUtilization = 0.7 + trend * 0.2; // 0.7-0.9

        // Add randomness
        gpuUtilization += (random.nextDouble() - 0.5) * 0.1;
        ramUtilization += (random.nextDouble() - 0.5) * 0.1;

        // Clamp to valid range
        gpuUtilization = Math.max(0.0, Math.min(1.0, gpuUtilization));
        ramUtilization = Math.max(0.0, Math.min(1.0, ramUtilization));

        // Update layer memory usage based on current layer
        long layerMemory = 100 * 1024 * 1024; // ~100MB per layer for 3B model
        long prefetchedMemory = layerMemory * 2; // 2 prefetched layers

        gpuMemoryUsedBytes = prefetchedMemory + kvCacheBytesGpu;
        ramMemoryUsedBytes = kvCacheBytesRam + 500 * 1024 * 1024; // Base 500MB for other operations
    }

    /**
     * Reset to initial state (useful for testing).
     */
    public void reset() {
        currentLayer = 0;
        gpuMemoryUsedBytes = 2L * 1024 * 1024 * 1024;
        ramMemoryUsedBytes = 4L * 1024 * 1024 * 1024;
        kvCacheBytesGpu = 500 * 1024 * 1024;
        kvCacheBytesRam = 200 * 1024 * 1024;
        gpuUtilization = 0.75;
        ramUtilization = 0.80;
        lastUpdateMs = System.currentTimeMillis();
    }

    /**
     * Get current simulated layer.
     */
    public int getCurrentLayer() {
        return currentLayer;
    }

    /**
     * Get KV cache size (combined GPU + RAM).
     */
    public long getTotalKvCacheBytes() {
        return kvCacheBytesGpu + kvCacheBytesRam;
    }

    @Override
    public String toString() {
        return String.format(
                "RuntimeMemoryState{layer=%d, gpu=%.1f%%, ram=%.1f%%, kv=%.0f MB}",
                currentLayer,
                gpuUtilization * 100,
                ramUtilization * 100,
                getTotalKvCacheBytes() / (1024.0 * 1024.0)
        );
    }
}
