package com.adaptivellm.runtime.services;

import com.adaptivellm.runtime.RuntimeService;
import com.adaptivellm.runtime.RuntimeContext;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * KV Cache GPU Manager.
 * 
 * Manages KV cache pages between RAM and GPU memory.
 * Implements paging strategy for large sequence lengths.
 */
public final class KVCacheGPUManager implements RuntimeService {

    private final long gpuMemoryForKV;
    private final long pageSize;
    private final Map<Integer, KVCachePage> pages;
    private volatile boolean initialized;
    private volatile com.adaptivellm.runtime.ServiceState state = com.adaptivellm.runtime.ServiceState.CREATED;

    public KVCacheGPUManager(long gpuMemoryBytes, long pageSizeBytes) {
        this.gpuMemoryForKV = gpuMemoryBytes;
        this.pageSize = pageSizeBytes;
        this.pages = new ConcurrentHashMap<>();
        this.initialized = false;
    }

    @Override
    public String name() { return "KVCacheGPUManager"; }

    @Override
    public void initialize(RuntimeContext context) throws com.adaptivellm.runtime.RuntimeException {
        try {
            this.state = com.adaptivellm.runtime.ServiceState.INITIALIZING;
            initializeKVCache();
            this.initialized = true;
            this.state = com.adaptivellm.runtime.ServiceState.READY;
            System.out.println("KV Cache GPU Manager initialized");
        } catch (Throwable t) {
            this.state = com.adaptivellm.runtime.ServiceState.ERROR;
            throw new com.adaptivellm.runtime.RuntimeException(com.adaptivellm.runtime.ErrorCode.RUNTIME_ERROR, "Failed to initialize KV Cache GPU Manager", t);
        }
    }

    @Override
    public void start() throws com.adaptivellm.runtime.RuntimeException {
        if (state == com.adaptivellm.runtime.ServiceState.CREATED) {
            initialize(new RuntimeContext());
        }
        this.state = com.adaptivellm.runtime.ServiceState.RUNNING;
        System.out.println("KV Cache GPU Manager started");
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
        shutdownKVCache();
        pages.clear();
        this.initialized = false;
        this.state = com.adaptivellm.runtime.ServiceState.STOPPED;
        System.out.println("KV Cache GPU Manager stopped");
    }

    @Override
    public com.adaptivellm.runtime.ServiceState state() { return state; }

    /**
     * Transfer KV cache page from RAM to GPU.
     * 
     * @param pageId Logical page identifier
     * @param hostBuffer Host memory buffer
     * @param size Number of bytes
     * @return GPU pointer for this page
     */
    public long transferPageToGPU(int pageId, long hostBuffer, long size) {
        if (!initialized) {
            throw new IllegalStateException("KV Cache GPU Manager not initialized");
        }

        long gpuPtr = nativeTransferToGPU(hostBuffer, size, pageId);
        if (gpuPtr == 0) {
            throw new RuntimeException(
                "Failed to transfer KV page " + pageId + " to GPU"
            );
        }

        KVCachePage page = new KVCachePage(pageId, gpuPtr, size);
        pages.put(pageId, page);

        System.out.println("KV Page " + pageId + " transferred to GPU: " + 
                          size / 1024 + " KB");
        return gpuPtr;
    }

    /**
     * Transfer KV cache page from GPU to RAM.
     * 
     * @param pageId Logical page identifier
     * @param gpuPtr GPU memory pointer
     * @param hostBuffer Target host buffer
     * @param size Number of bytes
     */
    public void transferPageFromGPU(int pageId, long gpuPtr, long hostBuffer, long size) {
        if (!initialized) {
            throw new IllegalStateException("KV Cache GPU Manager not initialized");
        }

        boolean success = nativeTransferFromGPU(gpuPtr, hostBuffer, size);
        if (!success) {
            throw new RuntimeException(
                "Failed to transfer KV page " + pageId + " from GPU"
            );
        }

        pages.remove(pageId);
        System.out.println("KV Page " + pageId + " transferred from GPU");
    }

    /**
     * Mark page as recently used (for LRU eviction).
     */
    public void markPageUsed(int pageId, long tokenPosition) {
        KVCachePage page = pages.get(pageId);
        if (page != null) {
            page.markUsed(tokenPosition);
            nativeMarkPageUsed(pageId, tokenPosition);
        }
    }

    /**
     * Get page metadata.
     */
    public KVCachePage getPage(int pageId) {
        return pages.get(pageId);
    }

    /**
     * Get GPU memory usage for KV cache.
     */
    public long getUsedMemory() {
        return nativeGetUsedMemory();
    }

    /**
     * Get available GPU memory for KV cache.
     */
    public long getAvailableMemory() {
        return gpuMemoryForKV - getUsedMemory();
    }

    /**
     * Evict least recently used page if memory is full.
     */
    public boolean evictLRUPage() {
        return nativeEvictLRU();
    }

    // ============ Native Methods ============

    private native void initializeKVCache();
    private native void shutdownKVCache();
    private native long nativeTransferToGPU(long hostPtr, long size, int pageId);
    private native boolean nativeTransferFromGPU(long gpuPtr, long hostPtr, long size);
    private native void nativeMarkPageUsed(int pageId, long tokenPosition);
    private native long nativeGetUsedMemory();
    private native boolean nativeEvictLRU();

    // ============ Page Metadata ============

    public static class KVCachePage {
        private final int pageId;
        private final long gpuPtr;
        private final long size;
        private volatile long lastUsedToken;

        public KVCachePage(int pageId, long gpuPtr, long size) {
            this.pageId = pageId;
            this.gpuPtr = gpuPtr;
            this.size = size;
            this.lastUsedToken = 0;
        }

        public int getPageId() { return pageId; }
        public long getGPUPtr() { return gpuPtr; }
        public long getSize() { return size; }
        public long getLastUsedToken() { return lastUsedToken; }

        private void markUsed(long tokenPosition) {
            this.lastUsedToken = tokenPosition;
        }

        @Override
        public String toString() {
            return String.format("KVPage{id=%d, gpu=%d, size=%d, lastToken=%d}",
                    pageId, gpuPtr, size, lastUsedToken);
        }
    }
}
