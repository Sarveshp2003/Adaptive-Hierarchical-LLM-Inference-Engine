#pragma once

#include "Tensor.h"
#include "Linear.h"
#include "KVCache.h"

#include <vector>


class MultiHeadAttention
{

private:

    int hiddenSize;
    int numHeads;
    int headDim;


    Linear qLinear;
    Linear kLinear;
    Linear vLinear;
    Linear outLinear;


public:

    MultiHeadAttention(
        int hiddenSize,
        int numHeads
    );


    // If cache is provided and useCache=true, forward will append K/V to cache and
    // compute attention over cached + current keys/values. Cache currently supports batch==1 only.
    void forward(
        Tensor& input,
        Tensor& output,
        KVCache* cache = nullptr,
        bool useCache = false
    );

    // CPU fallback for benchmarking/comparison
    void forwardCPU(
        Tensor& input,
        Tensor& output,
        KVCache* cache = nullptr,
        bool useCache = false
    );


private:

    void softmax(
        std::vector<float>& data
    );


};