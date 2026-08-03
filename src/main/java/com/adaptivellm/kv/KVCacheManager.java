package com.adaptivellm.kv;


import com.adaptivellm.memory.MemoryHierarchyManager;
import com.adaptivellm.memory.MemoryTier;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;


/**
 * Central KV cache controller.
 *
 * Manages transformer attention memory.
 *
 */
public final class KVCacheManager {


    private final KVPageAllocator allocator;


    private final MemoryHierarchyManager memoryManager;



    /**
     * Active KV blocks.
     *
     * key:
     *
     * layer-head combination
     */
    private final Map<String, KVCacheBlock> blocks =
            new ConcurrentHashMap<>();



    public KVCacheManager(
            KVPageAllocator allocator,
            MemoryHierarchyManager memoryManager
    ) {


        this.allocator =
                Objects.requireNonNull(
                        allocator
                );


        this.memoryManager =
                Objects.requireNonNull(
                        memoryManager
                );
    }



    /**
     * Creates KV cache block.
     */
    public KVCacheBlock createBlock(
            int layerId,
            int headId
    ) {


        String key =
                blockKey(
                        layerId,
                        headId
                );


        KVCacheBlock block =
                new KVCacheBlock(
                        layerId,
                        headId
                );


        blocks.put(
                key,
                block
        );


        return block;
    }



    /**
     * Allocate KV pages for new tokens.
     */
    public synchronized void allocatePages(
            KVCacheBlock block,
            long startToken,
            long endToken,
            String precision
    ) {


        KVPage keyPage =
                allocator.allocate(
                        startToken,
                        endToken,
                        precision
                );


        KVPage valuePage =
                allocator.allocate(
                        startToken,
                        endToken,
                        precision
                );



        block.addKeyPage(
                keyPage
        );


        block.addValuePage(
                valuePage
        );
    }



    /**
     * Moves KV block to GPU.
     */
    public synchronized void moveToGPU(
            KVCacheBlock block
    ) {


        block.keyPages()
                .forEach(
                        page ->
                                movePage(
                                        page,
                                        MemoryTier.GPU_VRAM
                                )
                );



        block.valuePages()
                .forEach(
                        page ->
                                movePage(
                                        page,
                                        MemoryTier.GPU_VRAM
                                )
                );
    }



    /**
     * Move page between tiers.
     */
    private void movePage(
            KVPage page,
            MemoryTier target
    ) {


        if(page.memory() != null)
        {
            memoryManager.move(
                    page.memory(),
                    target
            );
        }


        if(target == MemoryTier.GPU_VRAM)
        {

            if(page.state()
                    == KVPageState.ALLOCATED)
            {

                page.transition(
                        KVPageState.GPU_RESIDENT
                );
            }
        }


        else if(target == MemoryTier.RAM)
        {

            page.transition(
                    KVPageState.RAM_RESIDENT
            );
        }
    }



    /**
     * Gets active block.
     */
    public KVCacheBlock getBlock(
            int layerId,
            int headId
    ) {


        return blocks.get(
                blockKey(
                        layerId,
                        headId
                )
        );
    }



    /**
     * Evicts block from GPU.
     *
     * GPU -> RAM
     */
    public synchronized void evict(
            KVCacheBlock block
    ) {


        block.keyPages()
                .forEach(
                        page ->
                                movePage(
                                        page,
                                        MemoryTier.RAM
                                )
                );


        block.valuePages()
                .forEach(
                        page ->
                                movePage(
                                        page,
                                        MemoryTier.RAM
                                )
                );
    }



    /**
     * Releases complete block.
     */
    public synchronized void release(
            KVCacheBlock block
    ) {


        block.keyPages()
                .forEach(
                        allocator::release
                );


        block.valuePages()
                .forEach(
                        allocator::release
                );


        blocks.remove(
                blockKey(
                        block.layerId(),
                        block.headId()
                )
        );
    }



    private String blockKey(
            int layer,
            int head
    ) {

        return layer
                +
                "-"
                +
                head;
    }



    /**
     * Active blocks.
     */
    public int blockCount()
    {
        return blocks.size();
    }
}