package com.adaptivellm.layer;


import com.adaptivellm.memory.MemoryStatistics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;


/**
 * RAM layer cache.
 *
 * Uses LRU eviction strategy.
 *
 */
public final class LayerCache {


    /**
     * Maximum number of cached layers.
     */
    private final int capacity;



    /**
     * LRU cache.
     *
     * accessOrder=true
     * means recently used entries move to end.
     */
    private final Map<Integer, Layer> cache;



    private final MemoryStatistics statistics;



    public LayerCache(
            int capacity,
            MemoryStatistics statistics
    ) {


        if(capacity <= 0)
        {
            throw new IllegalArgumentException(
                    "Cache capacity must be positive"
            );
        }


        this.capacity =
                capacity;



        this.statistics =
                Objects.requireNonNull(
                        statistics,
                        "Statistics cannot be null"
                );



        this.cache =
                new LinkedHashMap<>(
                        capacity,
                        0.75f,
                        true
                );
    }



    /**
     * Adds layer to cache.
     */
    public synchronized void put(
            Layer layer
    ) {


        Objects.requireNonNull(
                layer,
                "Layer cannot be null"
        );


        if(cache.size() >= capacity)
        {
            evictOldest();
        }



        cache.put(
                layer.id(),
                layer
        );
    }



    /**
     * Gets cached layer.
     */
    public synchronized Layer get(
            int layerId
    ) {


        Layer layer =
                cache.get(
                        layerId
                );


        if(layer != null)
        {

            statistics.recordCacheHit();

            layer.markUsed();

        }
        else
        {

            statistics.recordCacheMiss();
        }



        return layer;
    }



    /**
     * Check existence.
     */
    public synchronized boolean contains(
            int layerId
    ) {

        return cache.containsKey(
                layerId
        );
    }



    /**
     * Remove layer.
     */
    public synchronized Layer remove(
            int layerId
    ) {


        return cache.remove(
                layerId
        );
    }



    /**
     * Removes least recently used layer.
     */
    private void evictOldest()
    {

        Integer oldest =
                cache.keySet()
                        .iterator()
                        .next();



        Layer removed =
                cache.remove(
                        oldest
                );


        if(removed != null)
        {
            statistics.recordEviction();
        }
    }



    /**
     * Current size.
     */
    public synchronized int size()
    {

        return cache.size();
    }



    /**
     * Cache capacity.
     */
    public int capacity()
    {

        return capacity;
    }



    /**
     * Clear cache.
     */
    public synchronized void clear()
    {

        cache.clear();
    }
}