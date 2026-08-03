package com.adaptivellm.scheduler;



/**
 * Decision together with
 * training reference.
 *
 */
public final class ScheduledDecision {


    private final Decision decision;


    private final TrainingSample sample;



    public ScheduledDecision(

            Decision decision,

            TrainingSample sample

    )
    {

        this.decision =
                decision;


        this.sample =
                sample;
    }



    public Decision decision()
    {
        return decision;
    }



    public TrainingSample sample()
    {
        return sample;
    }
}