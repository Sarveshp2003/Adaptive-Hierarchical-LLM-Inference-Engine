package com.adaptivellm.runtime;
import com.adaptivellm.runtime.services.MemoryHierarchyService;

/**
 * Runtime startup coordinator.
 *
 * Responsible for:
 *
 * - creating runtime objects
 * - registering services
 * - starting runtime
 *
 */
public final class RuntimeBootstrap {


    private RuntimeBootstrap() {

        /*
         * Utility class.
         */
    }



    /**
     * Creates and starts runtime.
     *
     * @return running RuntimeEngine
     */
    public static RuntimeEngine start()
            throws RuntimeException {


        /*
         * Load runtime configuration.
         *
         * Later this will become:
         *
         * runtime.yaml
         *
         */
        RuntimeConfiguration configuration =
                new RuntimeConfiguration(
                        "models/default",
                        Runtime
                                .getRuntime()
                                .availableProcessors()
                );



        /*
         * Create shared context.
         */
        RuntimeContext context =
                new RuntimeContext(
                        configuration
                );



        /*
         * Create runtime engine.
         */
        RuntimeEngine engine =
                new RuntimeEngine(
                        context
                );



        /*
         * Register runtime services.
         *
         * Actual services will be added
         * in later phases.
         *
         */
        registerServices(engine);



        /*
         * Start lifecycle.
         */
        engine.initialize();

        engine.start();



        return engine;
    }



    /**
     * Registers all runtime services.
     *
     * Currently empty.
     *
     * Future:
     *
     * register MemoryHierarchyManager
     * register NativeRuntime
     * register Scheduler
     *
     */
    private static void registerServices(
            RuntimeEngine engine
    )
    {

        engine.registry()
                .register(
                        new MemoryHierarchyService()
                );

    }



    /**
     * Stops runtime.
     */
    public static void shutdown(
            RuntimeEngine engine
    ) {


        if (engine != null) {

            engine.stop();
        }
    }
}