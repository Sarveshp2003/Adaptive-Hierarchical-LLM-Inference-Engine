#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "Tensor.h"


class LayerNorm
{

private:

    int size;
    Tensor gamma;
    Tensor beta;

public:

    LayerNorm(int hiddenSize);


    void forward(
        Tensor& input,
        Tensor& output
    );


};


#endif