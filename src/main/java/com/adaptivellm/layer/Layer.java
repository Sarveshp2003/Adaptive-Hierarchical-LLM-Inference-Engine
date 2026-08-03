package com.adaptivellm.layer;


import com.adaptivellm.memory.MemoryBlock;
import com.adaptivellm.memory.MemoryState;
import com.adaptivellm.memory.MemoryTier;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;


/**
 * Runtime model layer.
 *
 * Represents one transformer layer
 * managed by the streaming engine.
 *
 */
public final class Layer {


    /**
     * Static layer information.
     */
    private final LayerMetadata metadata;



    /**
     * Allocated memory.
     */
    private final AtomicReference<MemoryBlock> memory;



    /**
     * Layer lifecycle state.
     */
    private final AtomicReference<LayerState> state;



    /**
     * Execution count.
     */
    private long executionCount;



    /**
     * Last execution timestamp.
     */
    private long lastUsedTime;



    public Layer(
            LayerMetadata metadata
    ) {


        this.metadata =
                Objects.requireNonNull(
                        metadata,
                        "Layer metadata cannot be null"
                );


        this.memory =
                new AtomicReference<>();


        this.state =
                new AtomicReference<>(
                        LayerState.CREATED
                );
    }



    /**
     * Layer metadata.
     */
    public LayerMetadata metadata() {

        return metadata;
    }



    /**
     * Layer id.
     */
    public int id() {

        return metadata.layerId();
    }



    /**
     * Current state.
     */
    public LayerState state() {

        return state.get();
    }



    /**
     * Attached memory.
     */
    public MemoryBlock memory() {

        return memory.get();
    }



    /**
     * Attach memory block.
     */
    public void attachMemory(
            MemoryBlock block
    ) {


        this.memory.set(
                Objects.requireNonNull(
                        block
                )
        );
    }



    /**
     * Remove memory reference.
     */
    public void detachMemory() {

        memory.set(null);
    }



    /**
     * Move layer state.
     */
    public synchronized void transition(
            LayerState next
    ) {


        LayerState current =
                state.get();



        if(!current.canTransitionTo(next))
        {

            throw new IllegalStateException(

                    "Invalid layer transition "
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
     * Mark layer execution.
     */
    public synchronized void markUsed()
    {

        executionCount++;

        lastUsedTime =
                System.currentTimeMillis();
    }



    /**
     * Number of executions.
     */
    public long executionCount()
    {

        return executionCount;
    }



    /**
     * Last usage time.
     */
    public long lastUsedTime()
    {

        return lastUsedTime;
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
     * Checks if layer is GPU ready.
     */
    public boolean isGpuReady()
    {

        return state.get()
                == LayerState.ACTIVE
                &&
                tier()
                        == MemoryTier.GPU_VRAM;
    }



    @Override
    public String toString()
    {

        return "Layer{" +

                "id=" +
                id()

                +

                ", state=" +
                state()

                +

                ", tier=" +
                tier()

                +

                '}';
    }
}