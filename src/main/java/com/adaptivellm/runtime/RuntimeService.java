package com.adaptivellm.runtime;

/**
 * Base lifecycle contract for all runtime services.
 *
 * A runtime service represents a managed subsystem:
 *
 * Examples:
 *
 * - MemoryHierarchyManager
 * - Scheduler
 * - NativeRuntime
 * - ModelManager
 * - KVCacheManager
 *
 * Lifecycle:
 *
 * CREATED
 *    |
 * initialize()
 *    |
 * READY
 *    |
 * start()
 *    |
 * RUNNING
 *    |
 * stop()
 *    |
 * STOPPED
 *
 */
public interface RuntimeService {


    /**
     * Returns service name.
     *
     * Used for:
     * - logging
     * - metrics
     * - debugging
     */
    String name();



    /**
     * Initialize service resources.
     *
     * Examples:
     *
     * Memory Manager:
     * - allocate memory pools
     *
     * CUDA Runtime:
     * - initialize CUDA context
     *
     * Model Manager:
     * - read model metadata
     *
     */
    void initialize(RuntimeContext context)
            throws RuntimeException;



    /**
     * Start service execution.
     *
     * Examples:
     *
     * Scheduler:
     * - start worker threads
     *
     * Prefetch Engine:
     * - start background workers
     *
     */
    void start()
            throws RuntimeException;



    /**
     * Stop service safely.
     *
     * Must release:
     *
     * - threads
     * - memory
     * - native resources
     * - GPU resources
     *
     */
    void stop();



    /**
     * Current lifecycle state.
     */
    ServiceState state();



    /**
     * Health check.
     *
     * Used by:
     *
     * - monitoring
     * - runtime supervisor
     * - future distributed controller
     *
     */
    default boolean isHealthy() {

        return state() == ServiceState.RUNNING ||
                state() == ServiceState.READY;
    }



    /**
     * Optional service description.
     *
     * Useful for diagnostics.
     */
    default String description() {

        return name();
    }
}