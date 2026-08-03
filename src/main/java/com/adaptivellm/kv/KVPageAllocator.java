package com.adaptivellm.kv;


import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;



/**
 * Allocator for KV cache pages.
 *
 * Provides fast fixed-size page allocation.
 *
 */
public final class KVPageAllocator {


    /**
     * Page size.
     */
    private final long pageSizeBytes;



    /**
     * Page id generator.
     */
    private final AtomicLong idGenerator =
            new AtomicLong();



    /**
     * Available pages.
     */
    private final Queue<KVPage> freePages =
            new ConcurrentLinkedQueue<>();



    /**
     * Active pages.
     */
    private final Map<Long, KVPage> activePages =
            new ConcurrentHashMap<>();



    public KVPageAllocator(
            long pageSizeBytes
    ) {


        if(pageSizeBytes <= 0)
        {
            throw new IllegalArgumentException(
                    "Invalid page size"
            );
        }


        this.pageSizeBytes =
                pageSizeBytes;
    }



    /**
     * Allocate new KV page.
     */
    public KVPage allocate(
            long startToken,
            long endToken,
            String precision
    ) {


        KVPage page =
                freePages.poll();



        if(page == null)
        {

            long id =
                    idGenerator
                            .incrementAndGet();



            page =
                    new KVPage(

                            id,

                            startToken,

                            endToken,

                            pageSizeBytes,

                            precision
                    );
        }



        else
        {

            /*
             * Reused page.
             *
             * Future:
             *
             * reset metadata
             */
        }



        page.transition(
                KVPageState.ALLOCATED
        );



        activePages.put(
                page.id(),
                page
        );


        return page;
    }



    /**
     * Release page.
     */
    public void release(
            KVPage page
    ) {


        if(page == null)
        {
            return;
        }



        activePages.remove(
                page.id()
        );



        page.transition(
                KVPageState.RELEASED
        );



        freePages.offer(
                page
        );
    }



    /**
     * Find active page.
     */
    public KVPage get(
            long pageId
    )
    {

        return activePages.get(
                pageId
        );
    }



    /**
     * Active pages count.
     */
    public int activeCount()
    {

        return activePages.size();
    }



    /**
     * Free pages count.
     */
    public int freeCount()
    {

        return freePages.size();
    }



    /**
     * Page size.
     */
    public long pageSizeBytes()
    {

        return pageSizeBytes;
    }
}