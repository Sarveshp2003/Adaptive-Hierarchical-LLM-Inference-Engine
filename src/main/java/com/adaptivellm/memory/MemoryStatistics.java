package com.adaptivellm.memory;

import java.util.concurrent.atomic.AtomicLong;


/**
 * Runtime memory metrics collector.
 *
 * Tracks memory hierarchy behavior.
 *
 * Thread safe.
 *
 */
public final class MemoryStatistics {


    /*
     * Total allocated bytes.
     */
    private final AtomicLong allocatedBytes =
            new AtomicLong();



    /*
     * Total released bytes.
     */
    private final AtomicLong releasedBytes =
            new AtomicLong();



    /*
     * Number of memory transfers.
     */
    private final AtomicLong transferCount =
            new AtomicLong();



    /*
     * Number of cache hits.
     */
    private final AtomicLong cacheHits =
            new AtomicLong();



    /*
     * Number of cache misses.
     */
    private final AtomicLong cacheMisses =
            new AtomicLong();



    /*
     * Number of evictions.
     */
    private final AtomicLong evictionCount =
            new AtomicLong();



    /*
     * SSD read operations.
     */
    private final AtomicLong ssdReads =
            new AtomicLong();



    /*
     * GPU transfers.
     */
    private final AtomicLong gpuTransfers =
            new AtomicLong();



    /**
     * Record allocation.
     */
    public void recordAllocation(
            long bytes
    ) {

        allocatedBytes.addAndGet(bytes);
    }



    /**
     * Record release.
     */
    public void recordRelease(
            long bytes
    ) {

        releasedBytes.addAndGet(bytes);
    }



    /**
     * Record memory movement.
     */
    public void recordTransfer(
            MemoryTier source,
            MemoryTier destination
    ) {


        transferCount.incrementAndGet();


        if(source == MemoryTier.SSD) {

            ssdReads.incrementAndGet();
        }


        if(destination == MemoryTier.GPU_VRAM) {

            gpuTransfers.incrementAndGet();
        }
    }



    /**
     * Cache hit.
     */
    public void recordCacheHit() {

        cacheHits.incrementAndGet();
    }



    /**
     * Cache miss.
     */
    public void recordCacheMiss() {

        cacheMisses.incrementAndGet();
    }



    /**
     * Record eviction.
     */
    public void recordEviction() {

        evictionCount.incrementAndGet();
    }



    public long allocatedBytes() {

        return allocatedBytes.get();
    }



    public long releasedBytes() {

        return releasedBytes.get();
    }



    public long activeBytes() {

        return allocatedBytes.get()
                -
                releasedBytes.get();
    }



    public long transferCount() {

        return transferCount.get();
    }



    public long cacheHits() {

        return cacheHits.get();
    }



    public long cacheMisses() {

        return cacheMisses.get();
    }



    public long evictionCount() {

        return evictionCount.get();
    }



    public long ssdReads() {

        return ssdReads.get();
    }



    public long gpuTransfers() {

        return gpuTransfers.get();
    }



    /**
     * Cache hit ratio.
     */
    public double cacheHitRatio() {


        long total =
                cacheHits.get()
                        +
                        cacheMisses.get();



        if(total == 0) {

            return 0.0;
        }


        return
                (double) cacheHits.get()
                        /
                        total;
    }



    @Override
    public String toString() {

        return "MemoryStatistics{" +

                "allocatedBytes=" +
                allocatedBytes() +

                ", activeBytes=" +
                activeBytes() +

                ", transfers=" +
                transferCount() +

                ", cacheHitRatio=" +
                cacheHitRatio() +

                ", evictions=" +
                evictionCount() +

                '}';
    }
}