package com.adaptivellm.scheduler;


import java.time.Instant;


/**
 * Runtime snapshot used by
 * AI memory scheduler.
 *
 */
public final class MemoryState {


    /**
     * Current transformer layer.
     */
    private final int currentLayer;



    /**
     * Current generated token.
     */
    private final long currentToken;



    /**
     * GPU memory usage ratio.
     *
     * 0.0 - 1.0
     */
    private final double gpuUsage;



    /**
     * RAM usage ratio.
     */
    private final double ramUsage;



    /**
     * SSD latency milliseconds.
     */
    private final double storageLatency;



    /**
     * Active cached layers.
     */
    private final int cachedLayers;



    /**
     * Active KV pages.
     */
    private final int kvPages;



    /**
     * Timestamp.
     */
    private final long timestamp;



    public MemoryState(

            int currentLayer,

            long currentToken,

            double gpuUsage,

            double ramUsage,

            double storageLatency,

            int cachedLayers,

            int kvPages

    ) {


        this.currentLayer =
                currentLayer;


        this.currentToken =
                currentToken;


        this.gpuUsage =
                gpuUsage;


        this.ramUsage =
                ramUsage;


        this.storageLatency =
                storageLatency;


        this.cachedLayers =
                cachedLayers;


        this.kvPages =
                kvPages;


        this.timestamp =
                Instant.now()
                        .toEpochMilli();
    }



    public int currentLayer()
    {
        return currentLayer;
    }



    public long currentToken()
    {
        return currentToken;
    }



    public double gpuUsage()
    {
        return gpuUsage;
    }



    public double ramUsage()
    {
        return ramUsage;
    }



    public double storageLatency()
    {
        return storageLatency;
    }



    public int cachedLayers()
    {
        return cachedLayers;
    }



    public int kvPages()
    {
        return kvPages;
    }



    public long timestamp()
    {
        return timestamp;
    }



    /**
     * Memory pressure score.
     *
     * Higher means more urgent.
     */
    public double pressureScore()
    {


        return

                (gpuUsage * 0.45)

                        +

                        (ramUsage * 0.35)

                        +

                        Math.min(
                                storageLatency / 1000.0,
                                1.0
                        )
                                *
                                0.20;
    }



    @Override
    public String toString()
    {

        return "MemoryState{" +

                "layer=" +
                currentLayer +

                ", token=" +
                currentToken +

                ", gpu=" +
                gpuUsage +

                ", ram=" +
                ramUsage +

                ", pressure=" +
                pressureScore()

                +
                '}';
    }
}