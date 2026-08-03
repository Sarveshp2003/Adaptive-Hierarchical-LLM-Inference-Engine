package com.adaptivellm.scheduler;



import java.util.Objects;



/**
 * Scheduler output.
 *
 * Represents one memory action.
 *
 */
public final class Decision {


    /**
     * Action type.
     */
    private final SchedulerAction action;



    /**
     * Target resource id.
     *
     * Meaning depends on action:
     *
     * Layer id
     *
     * KV page id
     *
     */
    private final long targetId;



    /**
     * Confidence score.
     *
     * Used by AI models.
     *
     * 0.0 - 1.0
     */
    private final double confidence;



    public Decision(

            SchedulerAction action,

            long targetId,

            double confidence

    ) {


        this.action =
                Objects.requireNonNull(
                        action
                );


        if(confidence < 0 ||
                confidence > 1)
        {
            throw new IllegalArgumentException(
                    "Invalid confidence"
            );
        }


        this.targetId =
                targetId;


        this.confidence =
                confidence;
    }



    /**
     * Convenience constructor.
     */
    public Decision(

            SchedulerAction action,

            long targetId

    )
    {

        this(
                action,
                targetId,
                1.0
        );
    }



    public SchedulerAction action()
    {
        return action;
    }



    public long targetId()
    {
        return targetId;
    }



    public double confidence()
    {
        return confidence;
    }



    /**
     * Checks if decision requires action.
     */
    public boolean isActionRequired()
    {

        return action
                !=
                SchedulerAction.NO_ACTION;
    }



    @Override
    public String toString()
    {

        return "Decision{" +

                "action="
                +
                action

                +

                ", target="
                +
                targetId

                +

                ", confidence="
                +
                confidence

                +
                '}';
    }
}