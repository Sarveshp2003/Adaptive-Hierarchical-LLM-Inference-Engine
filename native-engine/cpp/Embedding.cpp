#include "Embedding.h"
#include <iostream>

Embedding::Embedding(int vocabSize, int hiddenSize)
: vocabSize(vocabSize), hiddenSize(hiddenSize), weights({vocabSize, hiddenSize}, DataType::FP32)
{
    weights.allocateCPU();
    weights.allocateGPU();

    // simple init
    for(size_t i=0;i<weights.elements();i++)
        weights.cpu()[i] = 0.01f;

    weights.upload();

    std::cout << "Embedding created: " << vocabSize << " x " << hiddenSize << std::endl;
}

void Embedding::forward(Tensor& tokenIds, Tensor& output)
{
    // tokenIds: CPU integers stored as floats
    tokenIds.download();
    weights.download();

    float* ids = tokenIds.cpu();
    float* w = weights.cpu();
    float* out = output.cpu();

    int N = static_cast<int>(tokenIds.elements());

    for(int i=0;i<N;i++)
    {
        int id = static_cast<int>(ids[i]);
        if(id < 0 || id >= vocabSize) id = 0;
        for(int j=0;j<hiddenSize;j++)
        {
            out[i * hiddenSize + j] = w[id * hiddenSize + j];
        }
    }

    output.upload();
}
