package com.adaptivellm.memory;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;


/**
 * Memory allocator for a single memory tier.
 *
 * Manages:
 *
 * - allocation
 * - release
 * - capacity tracking
 *
 */
public final class MemoryAllocator {


    /**
     * Managed memory tier.
     */
    private final MemoryTier tier;



    /**
     * Maximum capacity in bytes.
     */
    private final long capacity;



    /**
     * Current allocated bytes.
     */
    private final AtomicLong usedBytes =
            new AtomicLong();



    /**
     * Active memory blocks.
     */
    private final Map<String, MemoryBlock> blocks =
            new ConcurrentHashMap<>();



    /**
     * Creates allocator.
     */
    public MemoryAllocator(
            MemoryTier tier,
            long capacity
    ) {


        if(capacity <= 0) {

            throw new IllegalArgumentException(
                    "Capacity must be positive"
            );
        }


        this.tier = tier;

        this.capacity = capacity;
    }



    /**
     * Allocates memory block.
     */
    public synchronized MemoryBlock allocate(
            String name,
            long size
    ) {


        if(size <= 0) {

            throw new IllegalArgumentException(
                    "Allocation size must be positive"
            );
        }



        if(usedBytes.get() + size > capacity) {


            throw new IllegalStateException(
                    "Insufficient memory in tier: "
                            + tier
            );
        }



        String id =
                name + "-"
                        +
                        UUID.randomUUID();



        MemoryAddress address =
                createAddress(id);



        MemoryBlock block =
                new MemoryBlock(
                        id,
                        size,
                        address
                );



        block.transition(
                MemoryState.LOADING
        );


        block.transition(
                MemoryState.RESIDENT
        );



        blocks.put(
                id,
                block
        );


        usedBytes.addAndGet(
                size
        );



        return block;
    }



    /**
     * Releases memory block.
     */
    public synchronized void release(
            MemoryBlock block
    ) {


        if(block == null) {

            return;
        }



        MemoryBlock removed =
                blocks.remove(
                        block.id()
                );



        if(removed != null) {


            usedBytes.addAndGet(
                    -removed.size()
            );


            removed.transition(
                    MemoryState.RELEASED
            );
        }
    }



    /**
     * Creates a memory address.
     *
     * Real implementation will use:
     *
     * SSD:
     * mmap offset
     *
     * RAM:
     * native pointer
     *
     * GPU:
     * CUDA pointer
     *
     */
    private MemoryAddress createAddress(
            String id
    ) {


        long fakeAddress =
                System.nanoTime();



        return new MemoryAddress(
                tier,
                fakeAddress,
                id
        );
    }



    /**
     * Current memory usage.
     */
    public long usedBytes() {

        return usedBytes.get();
    }



    /**
     * Remaining capacity.
     */
    public long availableBytes() {

        return capacity
                -
                usedBytes.get();
    }



    /**
     * Memory tier.
     */
    public MemoryTier tier() {

        return tier;
    }



    /**
     * Number of active blocks.
     */
    public int blockCount() {

        return blocks.size();
    }
}