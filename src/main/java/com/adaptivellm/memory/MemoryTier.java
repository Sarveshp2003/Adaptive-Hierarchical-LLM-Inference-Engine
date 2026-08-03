package com.adaptivellm.memory;


/**
 * Represents memory hierarchy levels.
 *
 * Data moves through these tiers:
 *
 * SSD
 *  |
 * RAM
 *  |
 * PINNED RAM
 *  |
 * GPU VRAM
 *
 */
public enum MemoryTier {


    /**
     * Persistent storage.
     *
     * Example:
     *
     * Model weights
     * Layer files
     * Archived KV cache
     */
    SSD,


    /**
     * Main system memory.
     *
     * Example:
     *
     * Layer cache
     * Active KV pages
     */
    RAM,


    /**
     * Page-locked host memory.
     *
     * Used for:
     *
     * Faster PCIe transfer
     * Async CUDA copies
     */
    PINNED_RAM,


    /**
     * GPU device memory.
     *
     * Example:
     *
     * Active tensors
     * CUDA buffers
     */
    GPU_VRAM;



    /**
     * Returns whether this tier
     * supports persistent storage.
     */
    public boolean isPersistent() {

        return this == SSD;
    }



    /**
     * Returns whether this tier
     * is GPU accessible.
     */
    public boolean isDeviceMemory() {

        return this == GPU_VRAM;
    }
}