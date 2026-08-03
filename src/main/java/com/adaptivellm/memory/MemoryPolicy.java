package com.adaptivellm.memory;

import java.util.Comparator;
import java.util.List;


/**
 * Memory decision policy.
 *
 * Responsible for:
 *
 * - eviction decisions
 * - block ranking
 * - memory pressure handling
 *
 */
public class MemoryPolicy {


    /**
     * Minimum free memory ratio.
     *
     * When free memory falls below this,
     * eviction starts.
     */
    private final double minimumFreeRatio;



    public MemoryPolicy(
            double minimumFreeRatio
    ) {


        if(minimumFreeRatio < 0 ||
                minimumFreeRatio > 1) {

            throw new IllegalArgumentException(
                    "Invalid memory ratio"
            );
        }


        this.minimumFreeRatio =
                minimumFreeRatio;
    }



    /**
     * Checks memory pressure.
     */
    public boolean requiresEviction(
            MemoryAllocator allocator
    ) {


        double freeRatio =
                (double)
                        allocator.availableBytes()
                        /
                        (
                                allocator.usedBytes()
                                        +
                                        allocator.availableBytes()
                        );



        return freeRatio < minimumFreeRatio;
    }



    /**
     * Selects blocks for eviction.
     *
     * Current strategy:
     *
     * 1. Lowest priority
     * 2. Oldest access time
     *
     */
    public List<MemoryBlock> selectEvictionCandidates(
            List<MemoryBlock> blocks,
            int count
    ) {


        return blocks.stream()

                .sorted(
                        Comparator
                                .comparingInt(
                                        MemoryBlock::getPriority
                                )
                                .thenComparingLong(
                                        MemoryBlock::lastAccessTime
                                )
                )

                .limit(count)

                .toList();
    }



    /**
     * Calculates eviction priority.
     */
    public int evictionScore(
            MemoryBlock block
    ) {


        long age =
                block.ageMillis();



        /*
         * Higher score:
         *
         * more likely to evict
         */
        return
                (int)Math.min(
                        Integer.MAX_VALUE,
                        age / 1000
                )
                        -
                        block.priority();
    }
}