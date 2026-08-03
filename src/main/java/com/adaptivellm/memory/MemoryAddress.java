package com.adaptivellm.memory;

import java.util.Objects;


/**
 * Represents the physical location of memory data.
 *
 * Supports:
 *
 * SSD:
 *      file path + offset
 *
 * RAM:
 *      native memory address
 *
 * PINNED RAM:
 *      host pointer
 *
 * GPU VRAM:
 *      CUDA device pointer
 *
 */
public final class MemoryAddress {


    private final MemoryTier tier;


    /**
     * Physical address value.
     *
     * For:
     *
     * SSD:
     *     file offset
     *
     * RAM:
     *     native pointer
     *
     * GPU:
     *     CUDA pointer
     */
    private final long address;



    /**
     * Optional storage identifier.
     *
     * Examples:
     *
     * SSD:
     *     model.layers
     *
     * RAM:
     *     memory pool id
     *
     */
    private final String identifier;



    public MemoryAddress(
            MemoryTier tier,
            long address,
            String identifier
    ) {


        this.tier =
                Objects.requireNonNull(
                        tier,
                        "Memory tier cannot be null"
                );


        this.address = address;


        this.identifier =
                identifier;
    }



    /**
     * Memory tier.
     */
    public MemoryTier tier() {

        return tier;
    }



    /**
     * Physical address value.
     */
    public long address() {

        return address;
    }



    /**
     * Storage identifier.
     */
    public String identifier() {

        return identifier;
    }



    /**
     * Checks if address is valid.
     */
    public boolean isValid() {

        return address >= 0;
    }



    @Override
    public String toString() {

        return "MemoryAddress{" +
                "tier=" + tier +
                ", address=" + address +
                ", identifier='" +
                identifier +
                '\'' +
                '}';
    }
}