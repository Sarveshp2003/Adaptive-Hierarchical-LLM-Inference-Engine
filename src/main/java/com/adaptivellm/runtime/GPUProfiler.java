package com.adaptivellm.runtime.monitoring;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * GPU profiling and monitoring system.
 * 
 * Tracks:
 * - GPU memory usage
 * - Kernel execution times
 * - Data transfer rates
 * - Temperature
 * - Power consumption
 * - Performance metrics
 */
public final class GPUProfiler {

    private final Map<String, KernelMetrics> kernelMetrics;
    private final List<GPUSnapshot> snapshots;
    private final Timer profilerTimer;
    private volatile boolean isRunning;
    private final int samplingIntervalMs;

    public GPUProfiler(int samplingIntervalMs) {
        this.kernelMetrics = new ConcurrentHashMap<>();
        this.snapshots = new CopyOnWriteArrayList<>();
        this.samplingIntervalMs = samplingIntervalMs;
        this.isRunning = false;
        this.profilerTimer = new Timer("GPUProfiler", true);
    }

    /**
     * Start profiling
     */
    public void start() {
        if (isRunning) return;

        isRunning = true;
        
        // Sample GPU metrics periodically
        profilerTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                collectMetrics();
            }
        }, 0, samplingIntervalMs);

        System.out.println("GPU Profiler started (interval: " + samplingIntervalMs + "ms)");
    }

    /**
     * Stop profiling
     */
    public void stop() {
        isRunning = false;
        profilerTimer.cancel();
        System.out.println("GPU Profiler stopped");
    }

    /**
     * Record kernel execution time
     */
    public void recordKernel(String kernelName, long executionTimeMs, 
                            long inputBytes, long outputBytes) {
        KernelMetrics metrics = kernelMetrics.computeIfAbsent(kernelName, 
            k -> new KernelMetrics(kernelName));
        
        metrics.recordExecution(executionTimeMs, inputBytes, outputBytes);
    }

    /**
     * Collect current GPU metrics snapshot
     */
    private void collectMetrics() {
        GPUSnapshot snapshot = new GPUSnapshot();
        
        // Get GPU stats from native code
        snapshot.memoryUsed = nativeGetGPUMemoryUsed();
        snapshot.memoryTotal = nativeGetGPUMemoryTotal();
        snapshot.temperature = nativeGetGPUTemperature();
        snapshot.powerDraw = nativeGetGPUPowerDraw();
        snapshot.memoryBandwidth = nativeGetMemoryBandwidth();
        
        snapshots.add(snapshot);
        
        // Keep only last 1000 snapshots
        if (snapshots.size() > 1000) {
            snapshots.remove(0);
        }
    }

    /**
     * Get performance report
     */
    public String getReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║              GPU PROFILING REPORT                          ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        // Overall stats
        if (!snapshots.isEmpty()) {
            GPUSnapshot latest = snapshots.get(snapshots.size() - 1);
            sb.append("Current Status:\n");
            sb.append("  Memory: ").append(formatBytes(latest.memoryUsed)).append(" / ")
              .append(formatBytes(latest.memoryTotal)).append("\n");
            sb.append("  Temperature: ").append(latest.temperature).append("°C\n");
            sb.append("  Power Draw: ").append(latest.powerDraw).append("W\n");
            sb.append("  Memory Bandwidth: ").append(latest.memoryBandwidth).append(" GB/s\n");

            // Memory trend
            if (snapshots.size() > 1) {
                double avgMemory = snapshots.stream()
                    .mapToLong(s -> s.memoryUsed)
                    .average()
                    .orElse(0);
                sb.append("  Avg Memory: ").append(formatBytes((long)avgMemory)).append("\n");
            }
        }

        // Kernel statistics
        sb.append("\nKernel Statistics:\n");
        kernelMetrics.values().stream()
            .sorted(Comparator.comparingLong(KernelMetrics::getTotalTime).reversed())
            .limit(10)
            .forEach(metrics -> {
                sb.append("  ").append(metrics.name).append(":\n");
                sb.append("    Calls: ").append(metrics.executionCount).append("\n");
                sb.append("    Total Time: ").append(metrics.getTotalTime()).append("ms\n");
                sb.append("    Avg Time: ").append(metrics.getAverageTime()).append("ms\n");
                sb.append("    Throughput: ").append(metrics.getThroughput()).append(" GB/s\n");
            });

        return sb.toString();
    }

    /**
     * Get specific kernel metrics
     */
    public KernelMetrics getKernelMetrics(String kernelName) {
        return kernelMetrics.get(kernelName);
    }

    /**
     * Get all kernel metrics
     */
    public Collection<KernelMetrics> getAllKernelMetrics() {
        return kernelMetrics.values();
    }

    /**
     * Export metrics to CSV
     */
    public String exportCSV() {
        StringBuilder sb = new StringBuilder();
        sb.append("timestamp,memory_used,memory_total,temperature,power_draw,bandwidth\n");

        for (GPUSnapshot snapshot : snapshots) {
            sb.append(snapshot.timestamp).append(",")
              .append(snapshot.memoryUsed).append(",")
              .append(snapshot.memoryTotal).append(",")
              .append(snapshot.temperature).append(",")
              .append(snapshot.powerDraw).append(",")
              .append(snapshot.memoryBandwidth).append("\n");
        }

        return sb.toString();
    }

    // ============ Native Methods ============

    private native long nativeGetGPUMemoryUsed();
    private native long nativeGetGPUMemoryTotal();
    private native float nativeGetGPUTemperature();
    private native float nativeGetGPUPowerDraw();
    private native double nativeGetMemoryBandwidth();

    // ============ Helper Methods ============

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        String pre = "KMGTPE".charAt(exp - 1) + "";
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

    // ============ Inner Classes ============

    /**
     * Kernel execution metrics
     */
    public static class KernelMetrics {
        public final String name;
        private final AtomicLong executionCount = new AtomicLong(0);
        private final AtomicLong totalTime = new AtomicLong(0);
        private final AtomicLong totalInputBytes = new AtomicLong(0);
        private final AtomicLong totalOutputBytes = new AtomicLong(0);

        public KernelMetrics(String name) {
            this.name = name;
        }

        public void recordExecution(long timeMs, long inputBytes, long outputBytes) {
            executionCount.incrementAndGet();
            totalTime.addAndGet(timeMs);
            totalInputBytes.addAndGet(inputBytes);
            totalOutputBytes.addAndGet(outputBytes);
        }

        public long getExecutionCount() {
            return executionCount.get();
        }

        public long getTotalTime() {
            return totalTime.get();
        }

        public double getAverageTime() {
            long count = executionCount.get();
            return count == 0 ? 0 : (double) totalTime.get() / count;
        }

        public double getThroughput() {
            long totalBytes = totalInputBytes.get() + totalOutputBytes.get();
            long totalTimeMs = totalTime.get();
            if (totalTimeMs == 0) return 0;
            return (totalBytes / (1024.0 * 1024.0)) / (totalTimeMs / 1000.0);  // GB/s
        }

        @Override
        public String toString() {
            return String.format("%s: %d calls, %.2f ms avg, %.2f GB/s",
                name, executionCount.get(), getAverageTime(), getThroughput());
        }
    }

    /**
     * GPU metrics snapshot
     */
    public static class GPUSnapshot {
        public long timestamp = System.currentTimeMillis();
        public long memoryUsed;
        public long memoryTotal;
        public float temperature;
        public float powerDraw;
        public double memoryBandwidth;

        @Override
        public String toString() {
            return String.format("GPU[mem=%dMB, temp=%.1f°C, power=%.1fW]",
                memoryUsed / (1024*1024), temperature, powerDraw);
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
