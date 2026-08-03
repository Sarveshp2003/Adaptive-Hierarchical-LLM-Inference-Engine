package com.adaptivellm.runtime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


/**
 * Manages runtime services.
 *
 * Responsible for:
 *
 * - registering services
 * - initializing services
 * - starting services
 * - stopping services
 * - service lookup
 *
 */
public final class ServiceRegistry {


    /**
     * Maintains insertion order.
     *
     * Startup order matters.
     */
    private final Map<String, RuntimeService> services =
            new LinkedHashMap<>();



    /**
     * Register a runtime service.
     *
     * Example:
     *
     * registry.register(memoryManager);
     *
     */
    public synchronized void register(
            RuntimeService service
    ) {


        if (service == null) {

            throw new IllegalArgumentException(
                    "Service cannot be null"
            );
        }


        String name = service.name();


        if (services.containsKey(name)) {

            throw new IllegalStateException(
                    "Service already registered: " + name
            );
        }


        services.put(name, service);
    }



    /**
     * Get service by name.
     */
    public synchronized RuntimeService get(
            String name
    ) {

        return services.get(name);
    }



    /**
     * Returns all registered services.
     */
    public synchronized List<RuntimeService> services() {

        return Collections.unmodifiableList(
                new ArrayList<>(services.values())
        );
    }



    /**
     * Initialize all services.
     *
     * Services initialize in registration order.
     */
    public synchronized void initializeAll(
            RuntimeContext context
    )
            throws RuntimeException {


        for (RuntimeService service : services.values()) {

            service.initialize(context);
        }
    }



    /**
     * Start all services.
     *
     * Services start in registration order.
     */
    public synchronized void startAll()
            throws RuntimeException {


        for (RuntimeService service : services.values()) {

            service.start();
        }
    }



    /**
     * Stop all services.
     *
     * Stops in reverse order.
     *
     * This is important because:
     *
     * CUDA depends on memory.
     *
     * Scheduler depends on workers.
     *
     */
    public synchronized void stopAll() {


        List<RuntimeService> list =
                new ArrayList<>(services.values());


        Collections.reverse(list);



        for (RuntimeService service : list) {

            try {

                service.stop();

            }
            catch(Exception ignored) {

                /*
                 * Shutdown should continue
                 * even if one service fails.
                 */
            }
        }
    }



    /**
     * Number of registered services.
     */
    public synchronized int size() {

        return services.size();
    }
}