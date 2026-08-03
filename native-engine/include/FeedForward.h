#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "Tensor.h"
#include "Linear.h"

class FeedForward
{
private:
    int hiddenSize;
    Linear fc1;
    Linear fc2;
    Tensor hidden;
    bool initialized;

public:
    FeedForward(int hiddenSize);

    void forward(Tensor& input, Tensor& output);
};

#endif
