#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Tensor.h"

class Softmax
{
public:

    Softmax();

    ~Softmax();

    void forward(
        Tensor& input,
        Tensor& output
    );
};

#endif