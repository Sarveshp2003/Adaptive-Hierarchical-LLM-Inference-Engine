#ifndef TRANSFORMER_LAYER_H
#define TRANSFORMER_LAYER_H

#include "Tensor.h"
#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "KVCache.h"

class TransformerLayer
{
private:
    int hiddenSize;

    MultiHeadAttention attention;
    FeedForward ffn;

    LayerNorm layerNorm1;
    LayerNorm layerNorm2;

    Tensor workspace;

public:
    TransformerLayer(int hiddenSize);
    ~TransformerLayer();

    // If cache is provided and useCache=true, it is passed to the attention module.
    void forward(Tensor& input, Tensor& output, KVCache* cache = nullptr, bool useCache = false);
    int getHiddenSize() const;
};

#endif
