package com.adaptivellm.memory;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;


/**
 * Represents a memory transfer request.
 *
 * Examples:
 *
 * SSD -> RAM
 * RAM -> GPU_VRAM
 * GPU_VRAM -> RAM
 *
 * Used by:
 *
 * - MemoryHierarchyManager
 * - PrefetchEngine
 * - Scheduler
 *
 */
public final class MemoryRequest {


    /**
     * Unique request identifier.
     */
    private final String requestId;



    /**
     * Memory object being moved.
     */
    private final MemoryBlock block;



    /**
     * Source memory tier.
     */
    private final MemoryTier source;



    /**
     * Destination memory tier.
     */
    private final MemoryTier destination;



    /**
     * Request creation timestamp.
     */
    private final long createdTime;



    /**
     * Transfer priority.
     *
     * Higher values execute earlier.
     */
    private final int priority;



    /**
     * Whether operation is asynchronous.
     */
    private final boolean asynchronous;



    public MemoryRequest(
            MemoryBlock block,
            MemoryTier source,
            MemoryTier destination,
            int priority,
            boolean asynchronous
    ) {


        this.requestId =
                UUID.randomUUID()
                        .toString();



        this.block =
                Objects.requireNonNull(
                        block,
                        "Memory block cannot be null"
                );



        this.source =
                Objects.requireNonNull(
                        source,
                        "Source tier cannot be null"
                );



        this.destination =
                Objects.requireNonNull(
                        destination,
                        "Destination tier cannot be null"
                );



        this.priority =
                priority;



        this.asynchronous =
                asynchronous;



        this.createdTime =
                Instant.now()
                        .toEpochMilli();
    }



    /**
     * Request identifier.
     */
    public String requestId() {

        return requestId;
    }



    /**
     * Memory block.
     */
    public MemoryBlock block() {

        return block;
    }



    /**
     * Source tier.
     */
    public MemoryTier source() {

        return source;
    }



    /**
     * Destination tier.
     */
    public MemoryTier destination() {

        return destination;
    }



    /**
     * Priority.
     */
    public int priority() {

        return priority;
    }



    /**
     * Async transfer flag.
     */
    public boolean asynchronous() {

        return asynchronous;
    }



    /**
     * Request age.
     */
    public long ageMillis() {

        return Instant.now()
                .toEpochMilli()
                -
                createdTime;
    }



    /**
     * Checks if this is a valid movement.
     */
    public boolean isValid() {


        return source != destination;
    }



    @Override
    public String toString() {

        return "MemoryRequest{" +
                "requestId='" + requestId + '\'' +
                ", block=" + block.id() +
                ", source=" + source +
                ", destination=" + destination +
                ", priority=" + priority +
                ", asynchronous=" + asynchronous +
                '}';
    }
}