package com.adaptivellm.memory;

import java.time.Instant;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;


/**
 * Represents a managed memory object.
 *
 * Examples:
 *
 * - Transformer layer
 * - Tensor
 * - KV cache page
 * - Temporary GPU buffer
 *
 */
public final class MemoryBlock {


    /**
     * Unique identifier.
     */
    private final String id;



    /**
     * Size in bytes.
     */
    private final long size;



    /**
     * Current memory location.
     */
    private final AtomicReference<MemoryAddress> address;



    /**
     * Current lifecycle state.
     */
    private final AtomicReference<MemoryState> state;



    /**
     * Last access timestamp.
     *
     * Used by:
     *
     * - LRU eviction
     * - adaptive caching
     */
    private volatile long lastAccessTime;



    /**
     * Creation time.
     */
    private final long createdTime;



    /**
     * Priority value.
     *
     * Higher means:
     *
     * more important to keep
     */
    private volatile int priority;



    public MemoryBlock(
            String id,
            long size,
            MemoryAddress address
    ) {


        if (size < 0) {

            throw new IllegalArgumentException(
                    "Memory size cannot be negative"
            );
        }


        this.id =
                Objects.requireNonNull(
                        id,
                        "Memory block id cannot be null"
                );


        this.size =
                size;


        this.address =
                new AtomicReference<>(
                        address
                );


        this.state =
                new AtomicReference<>(
                        MemoryState.CREATED
                );


        this.createdTime =
                Instant.now()
                        .toEpochMilli();


        this.lastAccessTime =
                createdTime;


        this.priority =
                0;
    }



    /**
     * Unique identifier.
     */
    public String id() {

        return id;
    }



    /**
     * Size in bytes.
     */
    public long size() {

        return size;
    }



    /**
     * Current memory address.
     */
    public MemoryAddress address() {

        return address.get();
    }



    /**
     * Current memory tier.
     */
    public MemoryTier tier() {

        MemoryAddress current =
                address.get();


        return current == null
                ? null
                : current.tier();
    }



    /**
     * Current state.
     */
    public MemoryState state() {

        return state.get();
    }



    /**
     * Update memory location.
     */
    public void moveTo(
            MemoryAddress newAddress
    ) {


        address.set(
                Objects.requireNonNull(
                        newAddress
                )
        );


        touch();
    }



    /**
     * Change lifecycle state.
     */
    public synchronized void transition(
            MemoryState next
    ) {


        MemoryState current =
                state.get();



        if (!current.canTransitionTo(next)) {

            throw new IllegalStateException(
                    "Invalid memory transition: "
                            + current
                            + " -> "
                            + next
            );
        }


        state.set(next);
    }



    /**
     * Marks memory as recently used.
     */
    public void touch() {

        lastAccessTime =
                Instant.now()
                        .toEpochMilli();
    }



    /**
     * Last access timestamp.
     */
    public long lastAccessTime() {

        return lastAccessTime;
    }



    /**
     * Priority getter.
     */
    public int priority() {

        return priority;
    }



    /**
     * Update priority.
     */
    public void priority(
            int value
    ) {

        this.priority =
                value;
    }

    /**
     * Returns block priority.
     */
    public int getPriority() {

        return priority;
    }



    /**
     * Age of this memory block.
     */
    public long ageMillis() {

        return Instant.now()
                .toEpochMilli()
                -
                lastAccessTime;
    }



    @Override
    public String toString() {

        return "MemoryBlock{" +
                "id='" + id + '\'' +
                ", size=" + size +
                ", tier=" + tier() +
                ", state=" + state() +
                '}';
    }
}