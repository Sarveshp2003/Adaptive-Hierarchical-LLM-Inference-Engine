#include "LMHead.h"
#include <iostream>

LMHead::LMHead(int hiddenSize, int vocabSize)
: proj(hiddenSize, vocabSize)
{
    std::cout << "LMHead created: " << hiddenSize << " -> " << vocabSize << std::endl;
}

void LMHead::forward(Tensor& hidden, Tensor& logits)
{
    proj.forward(hidden, logits);
}
