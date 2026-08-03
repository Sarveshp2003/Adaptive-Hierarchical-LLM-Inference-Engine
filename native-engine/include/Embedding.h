#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "Tensor.h"

class Embedding
{
private:
    int vocabSize;
    int hiddenSize;
    Tensor weights; // shape: [vocabSize, hiddenSize]

public:
    Embedding(int vocabSize, int hiddenSize);

    // tokenIds: Tensor with shape [N] (flattened token indices as floats)
    // output: Tensor with shape [N, hiddenSize] (must be allocated)
    void forward(Tensor& tokenIds, Tensor& output);
};

#endif

