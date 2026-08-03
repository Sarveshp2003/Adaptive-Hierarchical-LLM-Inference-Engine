package com.adaptivellm.kv;


/**
 * KV cache page lifecycle.
 *
 * A page moves through:
 *
 * CREATED
 *
 * GPU
 *
 * RAM
 *
 * SSD
 *
 * RELEASED
 *
 */
public enum KVPageState {


    /**
     * Metadata exists.
     */
    CREATED,


    /**
     * Memory allocated.
     */
    ALLOCATED,


    /**
     * Active on GPU.
     */
    GPU_RESIDENT,


    /**
     * Stored in system RAM.
     */
    RAM_RESIDENT,


    /**
     * Compressed representation.
     */
    COMPRESSED,


    /**
     * Stored on SSD.
     */
    OFFLOADED,


    /**
     * Released.
     */
    RELEASED,


    /**
     * Failure.
     */
    ERROR;



    /**
     * Validate transition.
     */
    public boolean canTransitionTo(
            KVPageState next
    )
    {


        return switch(this)
        {

            case CREATED ->

                    next == ALLOCATED
                            ||
                            next == ERROR;



            case ALLOCATED ->

                    next == GPU_RESIDENT
                            ||
                            next == RAM_RESIDENT
                            ||
                            next == ERROR;



            case GPU_RESIDENT ->

                    next == RAM_RESIDENT
                            ||
                            next == COMPRESSED
                            ||
                            next == OFFLOADED;



            case RAM_RESIDENT ->

                    next == GPU_RESIDENT
                            ||
                            next == COMPRESSED
                            ||
                            next == OFFLOADED;



            case COMPRESSED ->

                    next == RAM_RESIDENT
                            ||
                            next == OFFLOADED;



            case OFFLOADED ->

                    next == RAM_RESIDENT
                            ||
                            next == RELEASED;



            case RELEASED,
                 ERROR ->

                    false;
        };
    }
}