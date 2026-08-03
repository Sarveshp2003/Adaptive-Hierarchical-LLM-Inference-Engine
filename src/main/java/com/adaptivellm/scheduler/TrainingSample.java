package com.adaptivellm.scheduler;



import java.time.Instant;



/**
 * One scheduler experience record.
 *
 */
public final class TrainingSample {


    private final MemoryState state;


    private final Decision decision;



    /**
     * Execution result metrics.
     */
    private double latencyImprovement;


    private long memorySavedBytes;


    private final long timestamp;



    public TrainingSample(
            MemoryState state,
            Decision decision
    )
    {

        this.state =
                state;


        this.decision =
                decision;


        this.timestamp =
                Instant.now()
                        .toEpochMilli();
    }



    public MemoryState state()
    {
        return state;
    }



    public Decision decision()
    {
        return decision;
    }



    public long timestamp()
    {
        return timestamp;
    }



    public double latencyImprovement()
    {
        return latencyImprovement;
    }



    public long memorySavedBytes()
    {
        return memorySavedBytes;
    }



    public void updateResult(
            double latencyImprovement,
            long memorySavedBytes
    )
    {

        this.latencyImprovement =
                latencyImprovement;


        this.memorySavedBytes =
                memorySavedBytes;
    }
}