package com.adaptivellm.runtime;


/**
 * Runtime error categories.
 *
 * Used for:
 *
 * - logging
 * - monitoring
 * - debugging
 * - recovery decisions
 */
public enum ErrorCode {


    /**
     * Invalid configuration.
     */
    CONFIGURATION_ERROR,


    /**
     * Memory allocation or memory management failure.
     */
    MEMORY_ERROR,


    /**
     * Model loading or parsing failure.
     */
    MODEL_ERROR,


    /**
     * CUDA runtime failure.
     */
    CUDA_ERROR,


    /**
     * JNI communication failure.
     */
    JNI_ERROR,


    /**
     * Scheduler failure.
     */
    SCHEDULER_ERROR,


    /**
     * Runtime lifecycle failure.
     */
    RUNTIME_ERROR,


    /**
     * Shutdown failure.
     */
    SHUTDOWN_ERROR,


    /**
     * Unknown unexpected failure.
     */
    UNKNOWN_ERROR
}