#ifndef TRANSFORMER_STACK_H
#define TRANSFORMER_STACK_H

#include <vector>
#include <iostream>

#include "TransformerLayer.h"
#include "Tensor.h"
#include "KVCache.h"

class TransformerStack
{
private:
    int numLayers;
    int hiddenSize;
    std::vector<TransformerLayer*> layers;

public:
    TransformerStack(int numLayers, int hiddenSize)
    : numLayers(numLayers), hiddenSize(hiddenSize)
    {
        layers.reserve(numLayers);
        for(int i=0;i<numLayers;i++)
        {
            layers.push_back(new TransformerLayer(hiddenSize));
        }
        std::cout << "TransformerStack created: " << numLayers << " x " << hiddenSize << std::endl;
    }

    ~TransformerStack()
    {
        for(auto p : layers) delete p;
    }

    void forward(Tensor& input, Tensor& output)
    {
        if(numLayers <= 0) return;

        // First layer consumes input -> output
        layers[0]->forward(input, output);

        // Remaining layers operate in-place on output
        for(int i=1;i<numLayers;i++)
        {
            layers[i]->forward(output, output);
        }
    }

    // Forward with per-layer KV caches. caches.size() must be numLayers; supports batch==1 caches only.
    void forward(Tensor& input, Tensor& output, std::vector<KVCache*>& caches, bool useCache)
    {
        if(numLayers <= 0) return;

        if((int)caches.size() != numLayers)
        {
            std::cout << "TransformerStack.forward: caches size mismatch" << std::endl;
            // fallback to normal forward
            forward(input, output);
            return;
        }

        // First layer
        layers[0]->forward(input, output, caches[0], useCache);

        for(int i=1;i<numLayers;i++)
        {
            layers[i]->forward(output, output, caches[i], useCache);
        }
    }

    int getNumLayers() const { return numLayers; }
};

#endif
