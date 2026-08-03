package com.adaptivellm.scheduler;


/**
 * Actions available to
 * adaptive memory scheduler.
 *
 */
public enum SchedulerAction {


    /**
     * Load future layer.
     */
    PREFETCH_LAYER,


    /**
     * Remove unused layer.
     */
    EVICT_LAYER,


    /**
     * Keep current memory.
     */
    KEEP_LAYER,


    /**
     * Move KV GPU -> RAM.
     */
    MOVE_KV_TO_RAM,


    /**
     * Move KV RAM -> GPU.
     */
    MOVE_KV_TO_GPU,


    /**
     * Compress KV cache.
     */
    COMPRESS_KV,


    /**
     * Move KV to SSD.
     */
    OFFLOAD_KV,


    /**
     * No operation.
     */
    NO_ACTION;
}