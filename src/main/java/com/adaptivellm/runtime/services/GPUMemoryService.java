package com.adaptivellm.runtime.services;

import com.adaptivellm.runtime.RuntimeService;
import com.adaptivellm.runtime.RuntimeContext;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * GPU memory management service.
 * 
 * Coordinates between Java memory manager and GPU VRAM.
 */
public final class GPUMemoryService implements RuntimeService {

    static {
        try {
            System.loadLibrary("adaptive_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library not loaded in GPUMemoryService: " + e.getMessage());
        }
    }

    private final long totalGPUMemory;
    private volatile long allocatedMemory;
    private volatile boolean initialized;
    private volatile com.adaptivellm.runtime.ServiceState state = com.adaptivellm.runtime.ServiceState.CREATED;
    private final Map<String, Long> allocations;

    public GPUMemoryService(long gpuMemoryBytes) {
        this.totalGPUMemory = gpuMemoryBytes;
        this.allocatedMemory = 0;
        this.initialized = false;
        this.allocations = new ConcurrentHashMap<>();
    }

    @Override
    public String name() {
        return "GPUMemoryService";
    }

    @Override
    public void initialize(RuntimeContext context) throws com.adaptivellm.runtime.RuntimeException {
        try {
            this.state = com.adaptivellm.runtime.ServiceState.INITIALIZING;
            initializeGPU();
            this.initialized = true;
            this.state = com.adaptivellm.runtime.ServiceState.READY;
            System.out.println("GPU Memory Service initialized: " + formatBytes(totalGPUMemory) + " total");
        } catch (Throwable t) {
            this.state = com.adaptivellm.runtime.ServiceState.ERROR;
            throw new com.adaptivellm.runtime.RuntimeException(com.adaptivellm.runtime.ErrorCode.CUDA_ERROR, "Failed to initialize GPU Memory Service", t);
        }
    }

    @Override
    public void start() throws com.adaptivellm.runtime.RuntimeException {
        if (state == com.adaptivellm.runtime.ServiceState.CREATED) {
            initialize(new RuntimeContext());
        }
        this.state = com.adaptivellm.runtime.ServiceState.RUNNING;
        System.out.println("GPU Memory Service started: " + formatBytes(totalGPUMemory) + " total");
    }

    /**
     * Backwards-compatible start with context.
     */
    public void start(RuntimeContext context) {
        try {
            initialize(context);
            start();
        } catch (com.adaptivellm.runtime.RuntimeException e) {
            throw new java.lang.RuntimeException(e);
        }
    }

    @Override
    public void stop() {
        try {
            this.state = com.adaptivellm.runtime.ServiceState.STOPPING;
            shutdown();
            this.initialized = false;
            this.state = com.adaptivellm.runtime.ServiceState.STOPPED;
            System.out.println("GPU Memory Service stopped");
        } catch (Throwable t) {
            this.state = com.adaptivellm.runtime.ServiceState.ERROR;
            System.err.println("Error stopping GPU Memory Service: " + t.getMessage());
        }
    }

    @Override
    public com.adaptivellm.runtime.ServiceState state() {
        return state;
    }

    /**
     * Allocate GPU memory.
     */
    public long allocateGPUMemory(long bytes, String label) {
        if (allocatedMemory + bytes > totalGPUMemory) {
            throw new OutOfMemoryError(
                "GPU memory exhausted: need " + bytes + ", have " + 
                getAvailableMemory()
            );
        }

        long handle = nativeAllocateGPU(bytes);
        if (handle == 0) {
            throw new OutOfMemoryError("GPU allocation failed");
        }

        allocatedMemory += bytes;
        allocations.put(label, handle);

        System.out.println("GPU allocated: " + formatBytes(bytes) + " for " + label);
        return handle;
    }

    /**
     * Free GPU memory.
     */
    public void freeGPUMemory(String label) {
        Long handle = allocations.remove(label);
        if (handle != null) {
            nativeFreeGPU(handle);
            allocatedMemory -= nativeGetSize(handle);
            System.out.println("GPU freed: " + label);
        }
    }

    /**
     * Transfer data from RAM to GPU.
     */
    public void transferToGPU(long hostPtr, long gpuPtr, long bytes) {
        nativeMemcpyHostToDevice(hostPtr, gpuPtr, bytes);
    }

    /**
     * Transfer data from GPU to RAM.
     */
    public void transferFromGPU(long gpuPtr, long hostPtr, long bytes) {
        nativeMemcpyDeviceToHost(gpuPtr, hostPtr, bytes);
    }

    /**
     * Get GPU memory statistics.
     */
    public GPUMemoryStats getStats() {
        return new GPUMemoryStats(
            totalGPUMemory,
            allocatedMemory,
            getAvailableMemory(),
            allocations.size()
        );
    }

    /**
     * Check available GPU memory.
     */
    public long getAvailableMemory() {
        return totalGPUMemory - allocatedMemory;
    }

    /**
     * Check if GPU has available memory for allocation.
     */
    public boolean hasCapacity(long bytes) {
        return allocatedMemory + bytes <= totalGPUMemory;
    }

    // ============ Native Methods ============

    private native void initializeGPU();
    private native void shutdown();
    private native long nativeAllocateGPU(long bytes);
    private native void nativeFreeGPU(long handle);
    private native long nativeGetSize(long handle);
    private native void nativeMemcpyHostToDevice(long hostPtr, long gpuPtr, long bytes);
    private native void nativeMemcpyDeviceToHost(long gpuPtr, long hostPtr, long bytes);

    // ============ Helper Methods ============

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        String pre = "KMGTPE".charAt(exp - 1) + "";
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

    // ============ Statistics ============

    public static class GPUMemoryStats {
        public final long total;
        public final long allocated;
        public final long available;
        public final int allocationCount;

        public GPUMemoryStats(long total, long allocated, long available, int count) {
            this.total = total;
            this.allocated = allocated;
            this.available = available;
            this.allocationCount = count;
        }

        @Override
        public String toString() {
            return String.format(
                "GPU Memory: %d%% used (%s/%s), %d allocations",
                (allocated * 100) / total,
                formatMB(allocated),
                formatMB(total),
                allocationCount
            );
        }

        private static String formatMB(long bytes) {
            return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
        }
    }
}
