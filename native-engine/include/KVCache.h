#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <vector>

// Simple placeholder KV cache structure. Not integrated into MHA yet.
struct KVCache
{
    int seqLen = 0;
    std::vector<float> keys;
    std::vector<float> vals;
};

#endif
