package com.adaptivellm.runtime;

import java.util.Objects;


/**
 * Shared runtime environment.
 *
 * Provides access to common runtime services:
 *
 * - configuration
 * - metrics
 * - event system
 *
 * Additional managers will be added here later:
 *
 * - MemoryHierarchyManager
 * - NativeRuntime
 * - ModelManager
 * - Scheduler
 *
 */
public final class RuntimeContext {


    private final RuntimeConfiguration configuration;



    /**
     * Creates runtime context with provided configuration.
     */
    public RuntimeContext(
            RuntimeConfiguration configuration
    ) {

        this.configuration =
                Objects.requireNonNull(
                        configuration,
                        "Runtime configuration cannot be null"
                );
    }

    /**
     * Convenience constructor using default configuration.
     */
    public RuntimeContext() {
        this(new RuntimeConfiguration("models/default", Math.max(1, Runtime.getRuntime().availableProcessors()/2)));
    }



    /**
     * Runtime configuration.
     */
    public RuntimeConfiguration configuration() {

        return configuration;
    }
}