package com.adaptivellm.kv;


import java.util.Objects;



/**
 * KV cache compression manager.
 *
 * Converts older KV pages into
 * lower precision formats.
 *
 */
public final class KVCompressionEngine {


    /**
     * Age threshold for INT8.
     */
    private final long int8Threshold;



    /**
     * Age threshold for heavy compression.
     */
    private final long compressedThreshold;



    public KVCompressionEngine(
            long int8Threshold,
            long compressedThreshold
    ) {


        if(int8Threshold <= 0 ||
                compressedThreshold <= 0)
        {
            throw new IllegalArgumentException(
                    "Invalid compression thresholds"
            );
        }



        this.int8Threshold =
                int8Threshold;



        this.compressedThreshold =
                compressedThreshold;
    }



    /**
     * Compresses KV page if required.
     */
    public synchronized void compress(
            KVPage page
    ) {


        Objects.requireNonNull(
                page
        );



        long age =
                System.currentTimeMillis()
                        -
                        page.lastAccessTime();



        if(age >= compressedThreshold)
        {

            applyCompression(
                    page,
                    "COMPRESSED"
            );

        }

        else if(age >= int8Threshold)
        {

            applyCompression(
                    page,
                    "INT8"
            );
        }
    }



    /**
     * Applies compression state.
     */
    private void applyCompression(
            KVPage page,
            String precision
    )
    {


        /*
         * Real implementation:
         *
         * CUDA kernel
         *
         * FP16 -> INT8
         *
         */


        if(
                page.state()
                        == KVPageState.GPU_RESIDENT
        )
        {

            page.transition(
                    KVPageState.COMPRESSED
            );
        }
    }



    /**
     * Estimates compressed size.
     */
    public long estimateSize(
            long originalBytes,
            String precision
    )
    {


        return switch(precision)
        {


            case "FP16" ->

                    originalBytes;



            case "INT8" ->

                    originalBytes / 2;



            case "COMPRESSED" ->

                    originalBytes / 4;



            default ->

                    originalBytes;
        };
    }



    /**
     * Compression ratio.
     */
    public double compressionRatio(
            String precision
    )
    {


        return switch(precision)
        {

            case "INT8" ->

                    2.0;


            case "COMPRESSED" ->

                    4.0;


            default ->

                    1.0;
        };
    }
}