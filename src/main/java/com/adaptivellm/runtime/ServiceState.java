package com.adaptivellm.runtime;

/**
 * Represents lifecycle states of runtime services.
 *
 * Every major runtime component should follow this lifecycle:
 *
 * CREATED
 *     |
 * INITIALIZING
 *     |
 * READY
 *     |
 * RUNNING
 *     |
 * STOPPING
 *     |
 * STOPPED
 *
 * ERROR can occur from any state.
 */
public enum ServiceState {

    /**
     * Service object has been created
     * but initialization has not started.
     */
    CREATED,


    /**
     * Service is loading resources.
     */
    INITIALIZING,


    /**
     * Service initialization completed.
     * Waiting to start.
     */
    READY,


    /**
     * Service is actively running.
     */
    RUNNING,


    /**
     * Service shutdown requested.
     */
    STOPPING,


    /**
     * Service completely stopped.
     */
    STOPPED,


    /**
     * Service encountered an unrecoverable error.
     */
    ERROR;


    /**
     * Checks whether this state represents
     * an active service.
     */
    public boolean isActive() {
        return this == RUNNING;
    }


    /**
     * Checks whether this state represents
     * a terminal state.
     */
    public boolean isTerminal() {
        return this == STOPPED ||
                this == ERROR;
    }


    /**
     * Checks whether the service can transition
     * into another state.
     */
    public boolean canTransitionTo(ServiceState next) {

        if (next == null) {
            return false;
        }


        return switch (this) {

            case CREATED ->
                    next == INITIALIZING ||
                            next == ERROR;


            case INITIALIZING ->
                    next == READY ||
                            next == ERROR;


            case READY ->
                    next == RUNNING ||
                            next == STOPPING ||
                            next == ERROR;


            case RUNNING ->
                    next == STOPPING ||
                            next == ERROR;


            case STOPPING ->
                    next == STOPPED ||
                            next == ERROR;


            case STOPPED,
                 ERROR ->
                    false;
        };
    }
}