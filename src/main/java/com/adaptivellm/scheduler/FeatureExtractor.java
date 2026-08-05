package com.adaptivellm.scheduler;



import java.util.Arrays;



/**
 * Converts runtime state into
 * AI model input features.
 *
 */
public final class FeatureExtractor {



    /**
     * Feature names.
     *
     * Order is important.
     */
    public static final String[] FEATURES = {


            "current_layer",


            "current_token",


            "gpu_usage",


            "ram_usage",


            "storage_latency",


            "cached_layers",


            "kv_pages",


            "pressure_score"

    };




    /**
     * Extract numerical features.
     */
    public double[] extract(
            MemoryState state
    ) {


        return new double[]{


                state.currentLayer(),


                state.currentToken(),


                state.gpuUsage(),


                state.ramUsage(),


                state.storageLatency(),


                state.cachedLayers(),


                state.kvPages(),


                state.pressureScore()

        };
    }





    /**
     * Normalized features.
     *
     * Used by neural networks.
     */
    public double[] extractNormalized(
            MemoryState state
    ) {


        double normalizedLayer = normalizeLayer(state.currentLayer());
        double normalizedToken = normalizeToken(state.currentToken());
        double normalizedLatency = normalizeLatency(state.storageLatency());
        double normalizedCached = normalizeCount(state.cachedLayers());
        double normalizedKv = normalizeCount(state.kvPages());

        double gpuRamInteraction = Math.min(state.gpuUsage() * state.ramUsage(), 1.0);
        double cacheRatio = state.cachedLayers() > 0
                ? Math.min(state.cachedLayers() / (double) Math.max(1, state.kvPages()), 1.0)
                : 0.0;
        double layerTokenInteraction = Math.min(normalizedLayer * normalizedToken, 1.0);
        double latencyPressureInteraction = Math.min(normalizedLatency * state.pressureScore(), 1.0);

        return new double[]{
                normalizedLayer,
                normalizedToken,
                state.gpuUsage(),
                state.ramUsage(),
                normalizedLatency,
                normalizedCached,
                normalizedKv,
                state.pressureScore(),
                gpuRamInteraction,
                cacheRatio,
                layerTokenInteraction,
                latencyPressureInteraction
        };
    }




    private double normalizeLayer(
            int layer
    )
    {

        /*
         * Assuming maximum 256 layers.
         */
        return Math.min(
                layer / 256.0,
                1.0
        );
    }





    private double normalizeToken(
            long token
    )
    {

        /*
         * Assuming 1M context.
         */
        return Math.min(
                token / 1_000_000.0,
                1.0
        );
    }





    private double normalizeLatency(
            double latency
    )
    {

        return Math.min(
                latency / 1000.0,
                1.0
        );
    }





    private double normalizeCount(
            int value
    )
    {

        return Math.min(
                value / 10000.0,
                1.0
        );
    }





    /**
     * Debug output.
     */
    public String describe()
    {

        return Arrays.toString(
                FEATURES
        );
    }
}