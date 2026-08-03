package com.adaptivellm.scheduler;



import java.util.Objects;



/**
 * Main adaptive memory scheduler.
 *
 */
public final class AdaptiveScheduler {


    private final FeatureExtractor extractor;


    private final PredictorModel predictor;


    private final TrainingDataCollector collector;



    public AdaptiveScheduler(

            FeatureExtractor extractor,

            PredictorModel predictor,

            TrainingDataCollector collector

    ) {


        this.extractor =
                Objects.requireNonNull(
                        extractor
                );


        this.predictor =
                Objects.requireNonNull(
                        predictor
                );


        this.collector =
                Objects.requireNonNull(
                        collector
                );
    }



    /**
     * Evaluates current runtime.
     *
     */
    public ScheduledDecision evaluate(
            MemoryState state
    )
    {


        double[] features =
                extractor.extractNormalized(
                        state
                );



        Decision decision =
                predictor.predict(
                        features
                );



        TrainingSample sample =
                collector.record(
                        state,
                        decision
                );



        return new ScheduledDecision(
                decision,
                sample
        );
    }





    /**
     * Reports execution result.
     */
    public void reportResult(

            ScheduledDecision scheduled,

            double latencyImprovement,

            long memorySavedBytes

    )
    {


        collector.updateResult(

                scheduled.sample(),

                latencyImprovement,

                memorySavedBytes

        );
    }



    /**
     * Training samples count.
     */
    public int trainingSamples()
    {
        return collector.size();
    }
}