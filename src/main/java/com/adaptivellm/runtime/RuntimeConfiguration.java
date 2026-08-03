package com.adaptivellm.runtime;


/**
 * Runtime configuration.
 *
 * Loaded from:
 *
 * config/runtime.yaml
 *
 */
public final class RuntimeConfiguration {


    private final String modelPath;


    private final int workerThreads;



    public RuntimeConfiguration(
            String modelPath,
            int workerThreads
    ) {

        this.modelPath = modelPath;
        this.workerThreads = workerThreads;
    }



    public String modelPath() {

        return modelPath;
    }



    public int workerThreads() {

        return workerThreads;
    }
}