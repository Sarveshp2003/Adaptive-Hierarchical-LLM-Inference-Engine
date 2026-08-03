package com.adaptivellm.runtime;

import java.util.concurrent.atomic.AtomicReference;


/**
 * Main runtime coordinator.
 *
 * Controls:
 *
 * - initialization
 * - service startup
 * - runtime state
 * - shutdown
 *
 */
public final class RuntimeEngine {


    private final RuntimeContext context;


    private final ServiceRegistry registry;


    private final AtomicReference<ServiceState> state =
            new AtomicReference<>(
                    ServiceState.CREATED
            );



    /**
     * Creates runtime engine.
     */
    public RuntimeEngine(
            RuntimeContext context
    ) {

        if (context == null) {

            throw new IllegalArgumentException(
                    "Runtime context cannot be null"
            );
        }


        this.context = context;

        this.registry =
                new ServiceRegistry();
    }



    /**
     * Returns service registry.
     *
     * Used during bootstrap:
     *
     * engine.registry()
     *
     * .register(...)
     *
     */
    public ServiceRegistry registry() {

        return registry;
    }



    /**
     * Initializes runtime.
     *
     * Lifecycle:
     *
     * CREATED
     *      |
     * INITIALIZING
     *      |
     * READY
     *
     */
    public synchronized void initialize()
            throws RuntimeException {


        if (!state.compareAndSet(
                ServiceState.CREATED,
                ServiceState.INITIALIZING
        )) {

            throw new RuntimeException(
                    ErrorCode.RUNTIME_ERROR,
                    "Runtime already initialized"
            );
        }



        try {

            registry.initializeAll(
                    context
            );


            state.set(
                    ServiceState.READY
            );


        }
        catch(Exception e) {


            state.set(
                    ServiceState.ERROR
            );


            throw new RuntimeException(
                    ErrorCode.RUNTIME_ERROR,
                    "Runtime initialization failed",
                    e
            );
        }
    }



    /**
     * Starts runtime execution.
     *
     * Lifecycle:
     *
     * READY
     *
     *    |
     *
     * RUNNING
     */
    public synchronized void start()
            throws RuntimeException {


        if (!state.compareAndSet(
                ServiceState.READY,
                ServiceState.RUNNING
        )) {


            throw new RuntimeException(
                    ErrorCode.RUNTIME_ERROR,
                    "Runtime is not ready"
            );
        }



        try {

            registry.startAll();


        }
        catch(Exception e) {


            state.set(
                    ServiceState.ERROR
            );


            throw new RuntimeException(
                    ErrorCode.RUNTIME_ERROR,
                    "Runtime startup failed",
                    e
            );
        }
    }



    /**
     * Stops runtime safely.
     *
     * Lifecycle:
     *
     * RUNNING
     *
     *    |
     *
     * STOPPING
     *
     *    |
     *
     * STOPPED
     *
     */
    public synchronized void stop() {


        ServiceState current =
                state.get();



        if (current.isTerminal()) {

            return;
        }



        state.set(
                ServiceState.STOPPING
        );



        try {


            registry.stopAll();



            state.set(
                    ServiceState.STOPPED
            );


        }
        catch(Exception e) {


            state.set(
                    ServiceState.ERROR
            );
        }
    }



    /**
     * Current runtime state.
     */
    public ServiceState state() {

        return state.get();
    }



    /**
     * Checks runtime availability.
     */
    public boolean isRunning() {

        return state.get()
                == ServiceState.RUNNING;
    }
}