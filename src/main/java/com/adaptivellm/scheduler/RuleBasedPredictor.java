package com.adaptivellm.scheduler;



/**
 * Initial scheduler implementation.
 *
 * Replaces AI model during early development.
 *
 */
public final class RuleBasedPredictor
        implements PredictorModel {



    /**
     * GPU pressure limit.
     */
    private static final double GPU_LIMIT =
            0.90;



    /**
     * RAM pressure limit.
     */
    private static final double RAM_LIMIT =
            0.90;



    /**
     * Prefetch threshold.
     */
    private static final double PREFETCH_LIMIT =
            0.70;





    @Override
    public Decision predict(
            double[] features
    ) {


        /*
         * Feature order:
         *
         * 0 layer
         * 1 token
         * 2 gpu
         * 3 ram
         * 4 storage latency
         * 5 cached layers
         * 6 kv pages
         * 7 pressure
         */


        double gpu =
                features[2];


        double ram =
                features[3];



        double pressure =
                features[7];



        /*
         * GPU memory emergency.
         */
        if(
                gpu >= GPU_LIMIT
        )
        {

            return new Decision(

                    SchedulerAction.MOVE_KV_TO_RAM,

                    -1,

                    0.95

            );
        }




        /*
         * RAM emergency.
         */
        if(
                ram >= RAM_LIMIT
        )
        {

            return new Decision(

                    SchedulerAction.COMPRESS_KV,

                    -1,

                    0.90

            );
        }




        /*
         * Normal operation.
         *
         * Keep pipeline full.
         */
        if(
                pressure < PREFETCH_LIMIT
        )
        {

            long currentLayer =
                    (long)features[0];


            return new Decision(

                    SchedulerAction.PREFETCH_LAYER,

                    currentLayer + 1,

                    0.80

            );
        }




        return new Decision(

                SchedulerAction.NO_ACTION,

                -1,

                0.50

        );
    }
}