package com.adaptivellm.memory;


/**
 * Lifecycle states of a memory object.
 *
 * A memory object can represent:
 *
 * - Model layer
 * - Tensor
 * - KV cache page
 * - Temporary buffer
 *
 */
public enum MemoryState {


    /**
     * Object metadata created.
     *
     * No memory allocated yet.
     */
    CREATED,


    /**
     * Memory allocation/loading in progress.
     */
    LOADING,


    /**
     * Object is loaded into a memory tier.
     *
     * Example:
     *
     * Layer 10 exists in RAM.
     */
    RESIDENT,


    /**
     * Object is currently being used.
     *
     * Example:
     *
     * GPU executing layer.
     */
    ACTIVE,


    /**
     * Object is scheduled for removal.
     */
    EVICTING,


    /**
     * Object removed from memory.
     */
    RELEASED,


    /**
     * Loading or movement failed.
     */
    ERROR;



    /**
     * Checks whether memory
     * can be accessed.
     */
    public boolean isAvailable() {

        return this == RESIDENT ||
                this == ACTIVE;
    }



    /**
     * Checks if memory is usable
     * for execution.
     */
    public boolean isExecutable() {

        return this == ACTIVE;
    }



    /**
     * Checks terminal states.
     */
    public boolean isFinal() {

        return this == RELEASED ||
                this == ERROR;
    }



    /**
     * Validates state transitions.
     */
    public boolean canTransitionTo(
            MemoryState next
    ) {


        if (next == null) {

            return false;
        }


        return switch(this) {


            case CREATED ->

                    next == LOADING ||
                            next == ERROR;



            case LOADING ->

                    next == RESIDENT ||
                            next == ERROR;



            case RESIDENT ->

                    next == ACTIVE ||
                            next == EVICTING ||
                            next == RELEASED;



            case ACTIVE ->

                    next == RESIDENT ||
                            next == EVICTING;



            case EVICTING ->

                    next == RELEASED ||
                            next == ERROR;



            case RELEASED,
                 ERROR ->

                    false;
        };
    }
}