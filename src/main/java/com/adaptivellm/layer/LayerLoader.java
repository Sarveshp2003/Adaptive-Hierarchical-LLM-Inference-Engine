package com.adaptivellm.layer;


import com.adaptivellm.memory.MemoryBlock;
import com.adaptivellm.memory.MemoryHierarchyManager;
import com.adaptivellm.memory.MemoryTier;

import java.util.Objects;



/**
 * Loads model layers from storage
 * into runtime memory.
 *
 */
public final class LayerLoader {


    private final MemoryHierarchyManager memoryManager;



    /**
     * Creates loader.
     */
    public LayerLoader(
            MemoryHierarchyManager memoryManager
    ) {

        this.memoryManager =
                Objects.requireNonNull(
                        memoryManager,
                        "Memory manager cannot be null"
                );
    }



    /**
     * Loads layer into RAM.
     *
     * Flow:
     *
     * SSD
     *  |
     *  v
     * RAM
     *
     */
    public synchronized void load(
            Layer layer
    ) {


        Objects.requireNonNull(
                layer,
                "Layer cannot be null"
        );



        LayerState current =
                layer.state();



        if(current != LayerState.CREATED
                &&
                current != LayerState.RELEASED)
        {

            throw new IllegalStateException(

                    "Layer cannot be loaded from state "
                            +
                            current
            );
        }



        try {


            layer.transition(
                    LayerState.LOADING
            );



            LayerMetadata metadata =
                    layer.metadata();



            /*
             * Allocate RAM for layer.
             */
            MemoryBlock block =
                    memoryManager.allocate(

                            "layer_"
                                    +
                                    metadata.layerId(),

                            metadata.sizeBytes(),

                            MemoryTier.RAM
                    );



            /*
             * Future implementation:
             *
             * read file:
             *
             * metadata.filePath()
             *
             * into block memory
             *
             */


            layer.attachMemory(
                    block
            );



            layer.transition(
                    LayerState.CACHED
            );



        }
        catch(Exception e)
        {

            /*
             * Failed loading.
             */
            if(layer.state()
                    == LayerState.LOADING)
            {

                layer.transition(
                        LayerState.ERROR
                );
            }


            throw e;
        }
    }



    /**
     * Loads directly to GPU.
     *
     * Future CUDA path:
     *
     * SSD
     *  |
     * RAM
     *  |
     * GPU
     *
     */
    public synchronized void loadToGPU(
            Layer layer
    ) {


        if(layer.state()
                != LayerState.CACHED)
        {

            throw new IllegalStateException(
                    "Layer must be cached first"
            );
        }



        memoryManager.move(
                layer.memory(),
                MemoryTier.GPU_VRAM
        );



        layer.transition(
                LayerState.ACTIVE
        );
    }



    /**
     * Releases layer.
     */
    public synchronized void release(
            Layer layer
    ) {


        if(layer.memory() != null)
        {

            memoryManager.release(
                    layer.memory()
            );
        }



        layer.detachMemory();



        if(layer.state()
                != LayerState.RELEASED)
        {

            layer.transition(
                    LayerState.RELEASED
            );
        }
    }
}