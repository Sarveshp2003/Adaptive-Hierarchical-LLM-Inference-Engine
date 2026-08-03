#include "TransformerLayer.h"
#include "TransformerStack.h"
#include "Embedding.h"
#include "LMHead.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <iostream>


int main()
{

    std::cout
        << "Transformer Layer Test"
        << std::endl;


    CUDAStream::initialize();


    RuntimeMemory::initializeGPU(
        1024 * 1024 * 1024
    );


    TransformerLayer layer(
        64
    );


    std::cout
        << "Transformer layer created successfully"
        << std::endl;


    TransformerStack stack(
        2,
        64
    );

    std::cout
        << "Transformer stack created successfully"
        << std::endl;


    // Simple generation demo (small sizes)
    {
        std::cout << "\nStarting simple generation demo" << std::endl;

        int vocab = 200;
        int hidden = 64;

        Embedding emb(vocab, hidden);
        LMHead head(hidden, vocab);
        TransformerStack smallStack(2, hidden);

        // per-layer KV caches
        int numLayers = smallStack.getNumLayers();
        std::vector<KVCache*> caches;
        caches.reserve(numLayers);
        std::vector<KVCache> cacheStorage(numLayers);
        for(int i=0;i<numLayers;i++) { caches.push_back(&cacheStorage[i]); }

        // single token batch
        Tensor tokenIds({1}, DataType::FP32);
        tokenIds.allocateCPU(); tokenIds.allocateGPU();
        tokenIds.cpu()[0] = 0;
        tokenIds.upload();

        Tensor embOut({1, hidden}, DataType::FP32);
        embOut.allocateCPU(); embOut.allocateGPU();

        Tensor modelOut({1, hidden}, DataType::FP32);
        modelOut.allocateCPU(); modelOut.allocateGPU();

        Tensor logits({1, vocab}, DataType::FP32);
        logits.allocateCPU(); logits.allocateGPU();

        int steps = 2;
        for(int step=0; step<steps; ++step)
        {
            emb.forward(tokenIds, embOut);

            // Run stack with KV caches for autoregressive generation (batch==1)
            smallStack.forward(embOut, modelOut, caches, true);

            head.forward(modelOut, logits);

            logits.download();
            float* lp = logits.cpu();

            int best = 0; float bestv = lp[0];
            for(int i=1;i<vocab;i++){
                if(lp[i] > bestv){ best = i; bestv = lp[i]; }
            }

            std::cout << "Generated token: " << best << std::endl;

            // set next input
            tokenIds.cpu()[0] = best;
            tokenIds.upload();
        }
    }

    RuntimeMemory::shutdown();


    CUDAStream::shutdown();


    std::cout
        << "TRANSFORMER TEST PASSED"
        << std::endl;


    return 0;

}
