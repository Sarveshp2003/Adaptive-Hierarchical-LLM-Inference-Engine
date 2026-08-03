package com.adaptivellm.layer;

import java.util.Objects;


/**
 * Static information about a model layer.
 *
 * This data normally comes from:
 *
 * model.json
 * metadata.bin
 * safetensors index
 *
 */
public final class LayerMetadata {


    /**
     * Layer index.
     *
     * Example:
     *
     * Transformer layer 20
     */
    private final int layerId;



    /**
     * Storage location.
     *
     * Example:
     *
     * layers/layer_20.bin
     */
    private final String filePath;



    /**
     * Layer size in bytes.
     */
    private final long sizeBytes;



    /**
     * Data precision.
     *
     * Examples:
     *
     * FP32
     * FP16
     * BF16
     * INT8
     */
    private final String dataType;



    /**
     * Creates layer metadata.
     */
    public LayerMetadata(
            int layerId,
            String filePath,
            long sizeBytes,
            String dataType
    ) {


        if(layerId < 0) {

            throw new IllegalArgumentException(
                    "Layer id cannot be negative"
            );
        }



        if(sizeBytes <= 0) {

            throw new IllegalArgumentException(
                    "Layer size must be positive"
            );
        }



        this.layerId =
                layerId;



        this.filePath =
                Objects.requireNonNull(
                        filePath,
                        "Layer path cannot be null"
                );



        this.sizeBytes =
                sizeBytes;



        this.dataType =
                Objects.requireNonNull(
                        dataType,
                        "Datatype cannot be null"
                );
    }



    /**
     * Layer number.
     */
    public int layerId() {

        return layerId;
    }



    /**
     * Storage path.
     */
    public String filePath() {

        return filePath;
    }



    /**
     * Size.
     */
    public long sizeBytes() {

        return sizeBytes;
    }



    /**
     * Tensor precision.
     */
    public String dataType() {

        return dataType;
    }



    @Override
    public String toString() {

        return "LayerMetadata{" +

                "layerId=" +
                layerId +

                ", filePath='" +
                filePath +
                '\'' +

                ", sizeBytes=" +
                sizeBytes +

                ", dataType='" +
                dataType +
                '\'' +

                '}';
    }
}