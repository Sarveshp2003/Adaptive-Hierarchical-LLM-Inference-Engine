package com.adaptivellm.layer;


/**
 * Lifecycle of a model layer.
 *
 * A layer moves through:
 *
 * SSD
 *  |
 *  v
 * LOADING
 *  |
 *  v
 * RAM
 *  |
 *  v
 * GPU
 *
 */
public enum LayerState {


    /**
     * Metadata exists only.
     */
    CREATED,


    /**
     * Reading from storage.
     */
    LOADING,


    /**
     * Layer available in RAM.
     */
    CACHED,


    /**
     * Layer uploaded to GPU.
     */
    ACTIVE,


    /**
     * Layer removed from memory.
     */
    EVICTING,


    /**
     * Layer unavailable.
     */
    RELEASED,


    /**
     * Failure.
     */
    ERROR;



    public boolean canTransitionTo(
            LayerState next
    ) {


        return switch(this) {


            case CREATED ->

                    next == LOADING ||
                            next == ERROR;


            case LOADING ->

                    next == CACHED ||
                            next == ERROR;


            case CACHED ->

                    next == ACTIVE ||
                            next == EVICTING ||
                            next == RELEASED;


            case ACTIVE ->

                    next == CACHED ||
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