package com.adaptivellm.scheduler;



import com.adaptivellm.kv.KVCacheManager;
import com.adaptivellm.kv.KVCompressionEngine;
import com.adaptivellm.kv.KVOffloadManager;
import com.adaptivellm.layer.LayerStreamer;

import java.util.Objects;



/**
 * Executes scheduler decisions.
 *
 */
public final class SchedulerExecutor {


    private final LayerStreamer layerStreamer;


    private final KVCacheManager kvManager;


    private final KVCompressionEngine compressor;


    private final KVOffloadManager offloader;



    public SchedulerExecutor(

            LayerStreamer layerStreamer,

            KVCacheManager kvManager,

            KVCompressionEngine compressor,

            KVOffloadManager offloader

    )
    {


        this.layerStreamer =
                Objects.requireNonNull(
                        layerStreamer
                );


        this.kvManager =
                Objects.requireNonNull(
                        kvManager
                );


        this.compressor =
                Objects.requireNonNull(
                        compressor
                );


        this.offloader =
                Objects.requireNonNull(
                        offloader
                );
    }



    /**
     * Executes decision.
     */
    public void execute(
            Decision decision
    )
    {


        switch(
                decision.action()
        )
        {


            case PREFETCH_LAYER ->

                    prefetchLayer(
                            decision
                    );



            case EVICT_LAYER ->

                    evictLayer(
                            decision
                    );



            case COMPRESS_KV ->

                    compressKV(
                            decision
                    );



            case MOVE_KV_TO_RAM ->

                    moveKVToRam(
                            decision
                    );



            case OFFLOAD_KV ->

                    offloadKV(
                            decision
                    );



            case MOVE_KV_TO_GPU ->

                    moveKVToGPU(
                            decision
                    );



            case KEEP_LAYER,
                 NO_ACTION ->

            {
                // Nothing required
            }
        }
    }



    private void prefetchLayer(
            Decision decision
    )
    {

        int layerId =
                (int)
                        decision.targetId();


        /*
         * LayerStreamer internally
         * handles prefetch queue.
         */
        layerStreamer.getLayer(
                layerId
        );
    }





    private void evictLayer(
            Decision decision
    )
    {

        int layerId =
                (int)
                        decision.targetId();


        layerStreamer.release(
                layerId
        );
    }





    private void compressKV(
            Decision decision
    )
    {

        /*
         * targetId:
         *
         * KV Page ID
         *
         */


        // Lookup page implementation
        // will be connected with KV registry later

    }





    private void moveKVToRam(
            Decision decision
    )
    {

        /*
         * GPU pressure handling.
         *
         * KVOffloadManager
         *
         * GPU -> RAM
         */

    }





    private void offloadKV(
            Decision decision
    )
    {


        /*
         * RAM -> SSD
         */

    }





    private void moveKVToGPU(
            Decision decision
    )
    {


        /*
         * SSD/RAM -> GPU
         */

    }
}