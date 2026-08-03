#ifndef LM_HEAD_H
#define LM_HEAD_H

#include "Tensor.h"
#include "Linear.h"

class LMHead
{
private:
    Linear proj;

public:
    LMHead(int hiddenSize, int vocabSize);
    void forward(Tensor& hidden, Tensor& logits);
};

#endif

