#include "MultiHeadAttention.h"
#include "Linear.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// CPU fallback implementation for benchmarking/comparison
void MultiHeadAttention::forwardCPU(
    Tensor& input,
    Tensor& output,
    KVCache* cache,
    bool useCache
)
{
    Tensor Q(input.shape(), DataType::FP32);
    Tensor K(input.shape(), DataType::FP32);
    Tensor V(input.shape(), DataType::FP32);

    Q.allocateCPU(); Q.allocateGPU();
    K.allocateCPU(); K.allocateGPU();
    V.allocateCPU(); V.allocateGPU();

    qLinear.forward(input, Q);
    kLinear.forward(input, K);
    vLinear.forward(input, V);

    // download projections to host
    Q.download();
    K.download();
    V.download();

    float* q = Q.cpu();
    float* k = K.cpu();
    float* v = V.cpu();

    Tensor context(input.shape(), DataType::FP32);
    context.allocateCPU(); context.allocateGPU();
    float* ctx = context.cpu();
    memset(ctx, 0, context.bytes());

    int batch = input.shape()[0];
    int seqLen = input.shape()[1];

    // Use cache only for batch==1 and when requested
    int cacheLen = 0;
    if(cache && useCache)
    {
        cacheLen = cache->seqLen;
        if(batch != 1 && cacheLen > 0)
            cacheLen = 0;
    }

    int totalSeq = seqLen + cacheLen;
    float scale = 1.0f / sqrtf((float)headDim);

    // Prepare pointers for cached K/V if present
    std::vector<float> k_total;
    std::vector<float> v_total;
    if(cacheLen > 0)
    {
        k_total.resize(totalSeq * hiddenSize);
        v_total.resize(totalSeq * hiddenSize);

        if(!cache->keys.empty())
            memcpy(k_total.data(), cache->keys.data(), sizeof(float) * cache->keys.size());
        if(!cache->vals.empty())
            memcpy(v_total.data(), cache->vals.data(), sizeof(float) * cache->vals.size());

        // append current K/V (assume batch==1)
        for(int i=0;i<seqLen;i++)
        {
            for(int h=0;h<hiddenSize;h++)
            {
                k_total[(cacheLen + i) * hiddenSize + h] = k[i * hiddenSize + h];
                v_total[(cacheLen + i) * hiddenSize + h] = v[i * hiddenSize + h];
            }
        }
    }

    for(int b=0;b<batch;b++)
    {
        for(int h=0;h<numHeads;h++)
        {
            int headOffset = h * headDim;
            std::vector<float> scores(seqLen * totalSeq);

            for(int i=0;i<seqLen;i++)
            {
                float maxScore = -1e9f;
                for(int j=0;j<totalSeq;j++)
                {
                    float score = 0.f;
                    for(int d=0;d<headDim;d++)
                    {
                        int qIndex = b*seqLen*hiddenSize + i*hiddenSize + headOffset + d;
                        float qv = q[qIndex];

                        float kv;
                        if(cacheLen>0)
                        {
                            kv = k_total[j * hiddenSize + headOffset + d];
                        }
                        else
                        {
                            int kIndex = b*seqLen*hiddenSize + j*hiddenSize + headOffset + d;
                            kv = k[kIndex];
                        }
                        score += qv * kv;
                    }
                    score *= scale;
                    scores[i*totalSeq + j] = score;
                    maxScore = std::max(maxScore, score);
                }

                // Softmax across totalSeq
                float sum = 0.f;
                for(int j=0;j<totalSeq;j++)
                {
                    float e = expf(scores[i*totalSeq + j] - maxScore);
                    scores[i*totalSeq + j] = e;
                    sum += e;
                }
                for(int j=0;j<totalSeq;j++)
                {
                    scores[i*totalSeq + j] /= sum;
                }
            }

            // Attention x V -> ctx (only produce outputs for current seq positions)
            for(int i=0;i<seqLen;i++)
            {
                for(int d=0;d<headDim;d++)
                {
                    float value = 0.f;
                    for(int j=0;j<totalSeq;j++)
                    {
                        float vv;
                        if(cacheLen>0)
                        {
                            vv = v_total[j * hiddenSize + headOffset + d];
                        }
                        else
                        {
                            int vIndex = b*seqLen*hiddenSize + j*hiddenSize + headOffset + d;
                            vv = v[vIndex];
                        }
                        value += scores[i*totalSeq + j] * vv;
                    }
                    int index = b*seqLen*hiddenSize + i*hiddenSize + headOffset + d;
                    ctx[index] = value;
                }
            }
        }
    }

    // Output projection
    context.upload();
    outLinear.forward(context, output);

    // Update cache if requested and supported (batch==1)
    if(cache && useCache && cacheLen >= 0)
    {
        if(cacheLen == 0)
        {
            cache->keys.resize(seqLen * hiddenSize);
            cache->vals.resize(seqLen * hiddenSize);
            for(int i=0;i<seqLen;i++)
            {
                for(int h=0;h<hiddenSize;h++)
                {
                    cache->keys[i * hiddenSize + h] = k[i * hiddenSize + h];
                    cache->vals[i * hiddenSize + h] = v[i * hiddenSize + h];
                }
            }
        }
        else
        {
            int old = cache->seqLen;
            cache->keys.resize((old + seqLen) * hiddenSize);
            cache->vals.resize((old + seqLen) * hiddenSize);
            for(int i=0;i<seqLen;i++)
            {
                for(int h=0;h<hiddenSize;h++)
                {
                    cache->keys[(old + i) * hiddenSize + h] = k[i * hiddenSize + h];
                    cache->vals[(old + i) * hiddenSize + h] = v[i * hiddenSize + h];
                }
            }
        }
        cache->seqLen += seqLen;
    }

}
