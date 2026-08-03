package com.adaptivellm.kv;


import com.adaptivellm.memory.MemoryHierarchyManager;
import com.adaptivellm.memory.MemoryTier;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;


/**
 * Handles KV cache movement
 * between memory tiers.
 *
 */
public final class KVOffloadManager {


    private final MemoryHierarchyManager memoryManager;



    /**
     * Simulated SSD index.
     *
     * Production version:
     *
     * mmap files
     * direct IO
     * async loading
     */
    private final Map<Long, KVPage> storageIndex =
            new ConcurrentHashMap<>();



    public KVOffloadManager(
            MemoryHierarchyManager memoryManager
    ) {


        this.memoryManager =
                Objects.requireNonNull(
                        memoryManager
                );
    }



    /**
     * GPU -> RAM
     */
    public synchronized void moveToRAM(
            KVPage page
    ) {


        move(
                page,
                MemoryTier.RAM
        );


        page.transition(
                KVPageState.RAM_RESIDENT
        );
    }



    /**
     * RAM -> SSD
     */
    public synchronized void offloadToSSD(
            KVPage page
    ) {


        /*
         * Future:
         *
         * Write tensor data
         *
         * mmap storage file
         *
         */


        storageIndex.put(
                page.id(),
                page
        );



        page.transition(
                KVPageState.OFFLOADED
        );
    }



    /**
     * SSD -> RAM
     */
    public synchronized void restoreFromSSD(
            KVPage page
    ) {


        if(
                !storageIndex.containsKey(
                        page.id()
                )
        )
        {
            throw new IllegalStateException(
                    "Page not found in storage"
            );
        }



        storageIndex.remove(
                page.id()
        );



        page.transition(
                KVPageState.RAM_RESIDENT
        );
    }



    /**
     * RAM -> GPU
     */
    public synchronized void moveToGPU(
            KVPage page
    ) {


        move(
                page,
                MemoryTier.GPU_VRAM
        );


        page.transition(
                KVPageState.GPU_RESIDENT
        );
    }



    private void move(
            KVPage page,
            MemoryTier target
    ) {


        if(page.memory() == null)
        {
            throw new IllegalStateException(
                    "Page has no memory allocation"
            );
        }


        memoryManager.move(
                page.memory(),
                target
        );
    }



    /**
     * Stored pages count.
     */
    public int storageSize()
    {
        return storageIndex.size();
    }
}