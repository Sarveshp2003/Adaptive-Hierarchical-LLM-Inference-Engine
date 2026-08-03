package com.adaptivellm.scheduler;



import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



/**
 * Collects scheduler experiences.
 *
 */
public final class TrainingDataCollector {


    private final List<TrainingSample> samples =
            new ArrayList<>();



    /**
     * Records decision.
     */
    public synchronized TrainingSample record(
            MemoryState state,
            Decision decision
    )
    {

        TrainingSample sample =
                new TrainingSample(
                        state,
                        decision
                );


        samples.add(
                sample
        );


        return sample;
    }




    /**
     * Updates result after execution.
     */
    public synchronized void updateResult(
            TrainingSample sample,
            double latencyImprovement,
            long memorySavedBytes
    )
    {

        sample.updateResult(
                latencyImprovement,
                memorySavedBytes
        );
    }




    /**
     * Returns dataset snapshot.
     */
    public List<TrainingSample> samples()
    {

        return Collections.unmodifiableList(
                samples
        );
    }




    /**
     * Dataset size.
     */
    public int size()
    {
        return samples.size();
    }



    /**
     * Clears collected data.
     */
    public synchronized void clear()
    {
        samples.clear();
    }
}