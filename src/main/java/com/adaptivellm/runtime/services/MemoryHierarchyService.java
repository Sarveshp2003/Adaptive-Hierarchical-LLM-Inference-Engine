package com.adaptivellm.runtime.services;


import com.adaptivellm.memory.MemoryHierarchyManager;
import com.adaptivellm.runtime.RuntimeContext;
import com.adaptivellm.runtime.RuntimeException;
import com.adaptivellm.runtime.RuntimeService;
import com.adaptivellm.runtime.ServiceState;



/**
 * Runtime service wrapper for memory management.
 *
 * Controls:
 *
 * - RAM management
 * - GPU memory management
 * - SSD storage tier
 * - memory policies
 *
 */
public final class MemoryHierarchyService
        implements RuntimeService {



    private MemoryHierarchyManager manager;



    private ServiceState state =
            ServiceState.CREATED;



    /**
     * Service name.
     */
    @Override
    public String name() {

        return "MemoryHierarchyService";
    }



    /**
     * Initialize memory system.
     */
    @Override
    public void initialize(
            RuntimeContext context
    )
            throws RuntimeException {


        if(state != ServiceState.CREATED)
        {
            throw new RuntimeException(
                    com.adaptivellm.runtime.ErrorCode.RUNTIME_ERROR,
                    "Memory service already initialized"
            );
        }



        /*
         * Temporary configuration.
         *
         * Later:
         *
         * runtime.yaml
         *
         */
        manager =
                new MemoryHierarchyManager(

                        /*
                         * SSD
                         * 4 TB
                         */
                        4L *
                                1024 *
                                1024 *
                                1024 *
                                1024,


                        /*
                         * RAM
                         * 64 GB
                         */
                        64L *
                                1024 *
                                1024 *
                                1024,


                        /*
                         * Pinned RAM
                         * 8 GB
                         */
                        8L *
                                1024 *
                                1024 *
                                1024,


                        /*
                         * GPU VRAM
                         * 24 GB
                         */
                        24L *
                                1024 *
                                1024 *
                                1024
                );


        state =
                ServiceState.READY;
    }



    /**
     * Start service.
     */
    @Override
    public void start()
            throws RuntimeException {


        if(state != ServiceState.READY)
        {
            throw new RuntimeException(
                    com.adaptivellm.runtime.ErrorCode.RUNTIME_ERROR,
                    "Memory service not ready"
            );
        }


        state =
                ServiceState.RUNNING;
    }



    /**
     * Stop service.
     */
    @Override
    public void stop() {


        manager = null;


        state =
                ServiceState.STOPPED;
    }



    /**
     * Access memory manager.
     */
    public MemoryHierarchyManager manager()
    {

        if(manager == null)
        {
            throw new IllegalStateException(
                    "Memory manager unavailable"
            );
        }


        return manager;
    }



    /**
     * Current state.
     */
    @Override
    public ServiceState state()
    {

        return state;
    }
}