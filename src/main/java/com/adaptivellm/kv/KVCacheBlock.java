package com.adaptivellm.kv;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;


/**
 * Logical KV cache block.
 *
 * Represents:
 *
 * Layer + Attention Head
 *
 * containing key and value pages.
 *
 */
public final class KVCacheBlock {


    /**
     * Transformer layer.
     */
    private final int layerId;



    /**
     * Attention head.
     */
    private final int headId;



    /**
     * Key cache pages.
     */
    private final List<KVPage> keyPages;



    /**
     * Value cache pages.
     */
    private final List<KVPage> valuePages;



    /**
     * Usage statistics.
     */
    private long accessCount;



    private long lastAccessTime;



    public KVCacheBlock(
            int layerId,
            int headId
    ) {


        if(layerId < 0)
        {
            throw new IllegalArgumentException(
                    "Invalid layer id"
            );
        }



        if(headId < 0)
        {
            throw new IllegalArgumentException(
                    "Invalid head id"
            );
        }



        this.layerId =
                layerId;



        this.headId =
                headId;



        this.keyPages =
                new ArrayList<>();



        this.valuePages =
                new ArrayList<>();
    }



    /**
     * Layer id.
     */
    public int layerId()
    {
        return layerId;
    }



    /**
     * Attention head id.
     */
    public int headId()
    {
        return headId;
    }



    /**
     * Add key page.
     */
    public synchronized void addKeyPage(
            KVPage page
    )
    {

        keyPages.add(
                Objects.requireNonNull(
                        page
                )
        );
    }



    /**
     * Add value page.
     */
    public synchronized void addValuePage(
            KVPage page
    )
    {

        valuePages.add(
                Objects.requireNonNull(
                        page
                )
        );
    }



    /**
     * Key pages.
     */
    public List<KVPage> keyPages()
    {

        return Collections.unmodifiableList(
                keyPages
        );
    }



    /**
     * Value pages.
     */
    public List<KVPage> valuePages()
    {

        return Collections.unmodifiableList(
                valuePages
        );
    }



    /**
     * Total pages.
     */
    public int pageCount()
    {

        return keyPages.size()
                +
                valuePages.size();
    }



    /**
     * Total memory.
     */
    public long memorySize()
    {

        long size = 0;


        for(KVPage page:keyPages)
        {
            size += page.sizeBytes();
        }


        for(KVPage page:valuePages)
        {
            size += page.sizeBytes();
        }


        return size;
    }



    /**
     * Mark access.
     */
    public synchronized void touch()
    {

        accessCount++;


        lastAccessTime =
                System.currentTimeMillis();



        keyPages.forEach(
                KVPage::touch
        );


        valuePages.forEach(
                KVPage::touch
        );
    }



    /**
     * Usage count.
     */
    public long accessCount()
    {
        return accessCount;
    }



    /**
     * Last usage.
     */
    public long lastAccessTime()
    {
        return lastAccessTime;
    }



    /**
     * Checks all pages are GPU resident.
     */
    public boolean isGpuReady()
    {

        return keyPages
                .stream()
                .allMatch(
                        KVPage::isGpuResident
                )

                &&

                valuePages
                        .stream()
                        .allMatch(
                                KVPage::isGpuResident
                        );
    }



    @Override
    public String toString()
    {

        return "KVCacheBlock{" +

                "layer=" +
                layerId +

                ", head=" +
                headId +

                ", pages=" +
                pageCount()

                +
                ", memory="
                +
                memorySize()

                +
                '}';
    }
}