package com.adaptivellm.layer;


import com.adaptivellm.memory.MemoryTier;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Objects;


/**
 * Main layer streaming controller.
 *
 * Provides layers for inference execution.
 *
 */
public final class LayerStreamer {


    private final LayerCache cache;


    private final LayerLoader loader;


    private final PrefetchEngine prefetch;



    /**
     * All model layers.
     *
     * Metadata registry.
     */
    private final Map<Integer, Layer> layers =
            new ConcurrentHashMap<>();



    public LayerStreamer(
            LayerCache cache,
            LayerLoader loader,
            PrefetchEngine prefetch
    ) {


        this.cache =
                Objects.requireNonNull(
                        cache
                );


        this.loader =
                Objects.requireNonNull(
                        loader
                );


        this.prefetch =
                Objects.requireNonNull(
                        prefetch
                );
    }



    /**
     * Registers model layer.
     */
    public void register(
            Layer layer
    ) {

        layers.put(
                layer.id(),
                layer
        );
    }



    /**
     * Gets layer ready for execution.
     *
     * Flow:
     *
     * Cache
     *
     * or
     *
     * Loader
     *
     */
    public synchronized Layer getLayer(
            int layerId
    ) {


        Layer layer =
                cache.get(
                        layerId
                );



        if(layer == null)
        {

            layer =
                    layers.get(
                            layerId
                    );


            if(layer == null)
            {
                throw new IllegalArgumentException(
                        "Unknown layer "
                                +
                                layerId
                );
            }



            loader.load(
                    layer
            );


            cache.put(
                    layer
            );
        }



        /*
         * Move to GPU.
         */
        if(
                layer.tier()
                        != MemoryTier.GPU_VRAM
        )
        {

            loader.loadToGPU(
                    layer
            );
        }



        layer.markUsed();



        /*
         * Predict next layer.
         */
        prefetchNext(
                layerId
        );



        return layer;
    }



    /**
     * Predict sequential next layer.
     */
    private void prefetchNext(
            int current
    )
    {

        Layer next =
                layers.get(
                        current + 1
                );


        if(next != null)
        {

            prefetch.prefetch(
                    next
            );
        }
    }



    /**
     * Release layer.
     */
    public synchronized void release(
            int layerId
    )
    {


        Layer layer =
                layers.get(
                        layerId
                );


        if(layer != null)
        {

            cache.remove(
                    layerId
            );


            loader.release(
                    layer
            );
        }
    }



    /**
     * Total layers.
     */
    public int size()
    {

        return layers.size();
    }
}