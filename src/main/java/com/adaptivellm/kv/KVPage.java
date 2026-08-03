package com.adaptivellm.kv;


import com.adaptivellm.memory.MemoryBlock;
import com.adaptivellm.memory.MemoryTier;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;



/**
 * Represents one paged KV cache unit.
 *
 * Example:
 *
 * Tokens:
 *
 * 10000 - 10256
 *
 * Stored:
 *
 * GPU VRAM
 *
 */
public final class KVPage {


    /**
     * Unique page identifier.
     */
    private final long pageId;



    /**
     * First token index.
     */
    private final long startToken;



    /**
     * Last token index.
     */
    private final long endToken;



    /**
     * Page size in bytes.
     */
    private final long sizeBytes;



    /**
     * Data precision.
     *
     * FP16
     * INT8
     * BF16
     */
    private final String precision;



    /**
     * Current memory.
     */
    private final AtomicReference<MemoryBlock> memory;



    /**
     * Current lifecycle state.
     */
    private final AtomicReference<KVPageState> state;



    /**
     * Last access time.
     */
    private volatile long lastAccessTime;



    /**
     * Number of accesses.
     */
    private volatile long accessCount;



    public KVPage(
            long pageId,
            long startToken,
            long endToken,
            long sizeBytes,
            String precision
    ) {


        if(pageId < 0)
        {
            throw new IllegalArgumentException(
                    "Invalid page id"
            );
        }



        if(startToken < 0 ||
                endToken < startToken)
        {
            throw new IllegalArgumentException(
                    "Invalid token range"
            );
        }



        if(sizeBytes <= 0)
        {
            throw new IllegalArgumentException(
                    "Invalid page size"
            );
        }



        this.pageId =
                pageId;



        this.startToken =
                startToken;



        this.endToken =
                endToken;



        this.sizeBytes =
                sizeBytes;



        this.precision =
                Objects.requireNonNull(
                        precision
                );



        this.memory =
                new AtomicReference<>();



        this.state =
                new AtomicReference<>(
                        KVPageState.CREATED
                );
    }



    /**
     * Page identifier.
     */
    public long id()
    {
        return pageId;
    }



    /**
     * Token start.
     */
    public long startToken()
    {
        return startToken;
    }



    /**
     * Token end.
     */
    public long endToken()
    {
        return endToken;
    }



    /**
     * Size.
     */
    public long sizeBytes()
    {
        return sizeBytes;
    }



    /**
     * Precision.
     */
    public String precision()
    {
        return precision;
    }



    /**
     * Current state.
     */
    public KVPageState state()
    {
        return state.get();
    }



    /**
     * Current memory block.
     */
    public MemoryBlock memory()
    {
        return memory.get();
    }



    /**
     * Attach memory.
     */
    public void attachMemory(
            MemoryBlock block
    )
    {

        memory.set(
                Objects.requireNonNull(
                        block
                )
        );
    }



    /**
     * Remove memory reference.
     */
    public void detachMemory()
    {
        memory.set(null);
    }



    /**
     * Change lifecycle state.
     */
    public synchronized void transition(
            KVPageState next
    )
    {

        KVPageState current =
                state.get();



        if(!current.canTransitionTo(next))
        {

            throw new IllegalStateException(

                    "Invalid KV page transition "
                            +
                            current
                            +
                            " -> "
                            +
                            next
            );
        }


        state.set(next);
    }



    /**
     * Record usage.
     */
    public void touch()
    {

        accessCount++;

        lastAccessTime =
                System.currentTimeMillis();
    }



    /**
     * Access count.
     */
    public long accessCount()
    {
        return accessCount;
    }



    /**
     * Last usage.
     */
    public long lastAccessTime()
    {
        return lastAccessTime;
    }



    /**
     * Current memory tier.
     */
    public MemoryTier tier()
    {

        MemoryBlock block =
                memory.get();


        if(block == null)
        {
            return null;
        }


        return block.tier();
    }



    /**
     * Checks GPU residency.
     */
    public boolean isGpuResident()
    {

        return state.get()
                == KVPageState.GPU_RESIDENT;
    }



    @Override
    public String toString()
    {

        return "KVPage{" +

                "id=" +
                pageId +

                ", tokens=" +
                startToken +
                "-" +
                endToken +

                ", size=" +
                sizeBytes +

                ", state=" +
                state +

                ", tier=" +
                tier()

                +
                '}';
    }
}