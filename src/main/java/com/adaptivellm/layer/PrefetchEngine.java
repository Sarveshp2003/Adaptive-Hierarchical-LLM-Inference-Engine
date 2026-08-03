package com.adaptivellm.layer;


import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;


/**
 * Background layer prefetch engine.
 *
 * Loads future layers before execution needs them.
 *
 */
public final class PrefetchEngine {


    private final LayerLoader loader;



    /**
     * Pending layers.
     */
    private final BlockingQueue<Layer> queue =
            new LinkedBlockingQueue<>();



    /**
     * Prevent duplicate requests.
     */
    private final Set<Integer> scheduled =
            ConcurrentHashMap.newKeySet();



    private volatile boolean running;



    private Thread worker;



    public PrefetchEngine(
            LayerLoader loader
    ) {

        this.loader = loader;
    }



    /**
     * Starts background worker.
     */
    public synchronized void start()
    {

        if(running)
        {
            return;
        }


        running = true;



        worker =
                new Thread(
                        this::process,
                        "layer-prefetch-worker"
                );


        worker.start();
    }



    /**
     * Stops worker.
     */
    public synchronized void stop()
    {

        running = false;


        if(worker != null)
        {
            worker.interrupt();
        }
    }



    /**
     * Schedule layer loading.
     */
    public void prefetch(
            Layer layer
    ) {


        if(layer == null)
        {
            return;
        }



        if(
                scheduled.add(
                        layer.id()
                )
        )
        {

            queue.offer(
                    layer
            );
        }
    }



    /**
     * Worker loop.
     */
    private void process()
    {


        while(running)
        {

            try
            {

                Layer layer =
                        queue.take();



                try
                {

                    loader.load(
                            layer
                    );

                }
                finally
                {

                    scheduled.remove(
                            layer.id()
                    );
                }


            }
            catch(
                    InterruptedException e
            )
            {

                Thread.currentThread()
                        .interrupt();

                break;
            }
            catch(Exception e)
            {

                /*
                 * Production version:
                 *
                 * log failure
                 * metrics
                 * retry policy
                 *
                 */
            }
        }
    }



    /**
     * Pending requests.
     */
    public int queueSize()
    {

        return queue.size();
    }
}