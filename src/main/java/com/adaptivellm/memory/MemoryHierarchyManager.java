package com.adaptivellm.memory;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


/**
 * Central memory management controller.
 *
 * Controls movement between:
 *
 * SSD
 * RAM
 * PINNED RAM
 * GPU VRAM
 *
 */
public final class MemoryHierarchyManager {


    private final Map<String, MemoryBlock> blocks =
            new ConcurrentHashMap<>();


    private final MemoryAllocator ssdAllocator;


    private final MemoryAllocator ramAllocator;


    private final MemoryAllocator pinnedAllocator;


    private final MemoryAllocator gpuAllocator;


    private final MemoryPolicy policy;


    private final MemoryStatistics statistics;



    public MemoryHierarchyManager(
            long ssdCapacity,
            long ramCapacity,
            long pinnedCapacity,
            long gpuCapacity
    ) {


        this.ssdAllocator =
                new MemoryAllocator(
                        MemoryTier.SSD,
                        ssdCapacity
                );


        this.ramAllocator =
                new MemoryAllocator(
                        MemoryTier.RAM,
                        ramCapacity
                );


        this.pinnedAllocator =
                new MemoryAllocator(
                        MemoryTier.PINNED_RAM,
                        pinnedCapacity
                );


        this.gpuAllocator =
                new MemoryAllocator(
                        MemoryTier.GPU_VRAM,
                        gpuCapacity
                );


        this.policy =
                new MemoryPolicy(
                        0.10
                );


        this.statistics =
                new MemoryStatistics();
    }



    /**
     * Allocate new memory object.
     */
    public synchronized MemoryBlock allocate(
            String name,
            long size,
            MemoryTier tier
    ) {


        MemoryAllocator allocator =
                allocatorFor(tier);



        MemoryBlock block =
                allocator.allocate(
                        name,
                        size
                );


        blocks.put(
                block.id(),
                block
        );


        statistics.recordAllocation(
                size
        );


        return block;
    }



    /**
     * Move memory between tiers.
     */
    public synchronized void move(
            MemoryBlock block,
            MemoryTier destination
    ) {


        MemoryTier source =
                block.tier();



        if(source == destination) {

            return;
        }



        MemoryAllocator target =
                allocatorFor(destination);



        MemoryBlock newBlock =
                target.allocate(
                        block.id(),
                        block.size()
                );



        block.transition(
                MemoryState.EVICTING
        );


        block.moveTo(
                newBlock.address()
        );


        block.transition(
                MemoryState.RESIDENT
        );


        statistics.recordTransfer(
                source,
                destination
        );
    }



    /**
     * Release memory.
     */
    public synchronized void release(
            MemoryBlock block
    ) {


        MemoryAllocator allocator =
                allocatorFor(
                        block.tier()
                );


        allocator.release(
                block
        );


        blocks.remove(
                block.id()
        );


        statistics.recordRelease(
                block.size()
        );
    }



    /**
     * Find memory block.
     */
    public MemoryBlock get(
            String id
    ) {

        return blocks.get(id);
    }



    /**
     * Returns all blocks.
     */
    public ArrayList<MemoryBlock> blocks()
    {

        return new ArrayList<>(
                blocks.values()
        );
    }



    /**
     * Memory statistics.
     */
    public MemoryStatistics statistics()
    {

        return statistics;
    }



    /**
     * Select eviction candidates.
     */
    public void evict(
            int count
    ) {


        var candidates =
                policy.selectEvictionCandidates(
                        blocks(),
                        count
                );


        for(MemoryBlock block : candidates)
        {

            block.transition(
                    MemoryState.EVICTING
            );


            release(block);
        }
    }



    /**
     * Select allocator.
     */
    private MemoryAllocator allocatorFor(
            MemoryTier tier
    ) {


        return switch(tier)
        {

            case SSD ->
                    ssdAllocator;


            case RAM ->
                    ramAllocator;


            case PINNED_RAM ->
                    pinnedAllocator;


            case GPU_VRAM ->
                    gpuAllocator;
        };
    }
}