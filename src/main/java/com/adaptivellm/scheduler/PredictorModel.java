package com.adaptivellm.scheduler;


/**
 * Prediction interface.
 *
 * Any AI model must implement this.
 *
 */
public interface PredictorModel {


    /**
     * Generates memory decision.
     *
     * @param features numerical input vector
     *
     * @return scheduler decision
     */
    Decision predict(
            double[] features
    );


}