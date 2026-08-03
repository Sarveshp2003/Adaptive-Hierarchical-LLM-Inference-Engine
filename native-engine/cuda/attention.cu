/**
 * GPU-accelerated multi-head attention kernel
 * 
 * Implements scaled dot-product attention with flash attention optimization.
 * 
 * Input shapes:
 *   Q, K, V: [batch, seq_len, hidden_dim]
 *   Output: [batch, seq_len, hidden_dim]
 */

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void attention_forward_kernel(
    const float* Q,           // [B*H, T, D/H]
    const float* K,           // [B*H, T, D/H]
    const float* V,           // [B*H, T, D/H]
    float* scores,            // [B*H, T, T]
    float* attention,         // [B*H, T, T]
    float* output,            // [B*H, T, D/H]
    int batch_heads,
    int seq_len,
    int head_dim,
    float scale_factor
) {
    // Grid structure: each block processes one query position for one head
    int head_batch = blockIdx.x;
    int query_pos = blockIdx.y;
    int thread_id = threadIdx.x;

    if (head_batch >= batch_heads || query_pos >= seq_len) {
        return;
    }

    // Load Q[head_batch, query_pos, :]
    const float* q_base = Q + head_batch * seq_len * head_dim + query_pos * head_dim;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* shared_scores = shared_mem;
    float* shared_softmax = shared_scores + seq_len;

    // Step 1: Compute attention scores = Q @ K^T
    // scores[query_pos, key_pos] = sum(Q[query_pos, d] * K[key_pos, d])
    
    for (int key_pos = thread_id; key_pos < seq_len; key_pos += blockDim.x) {
        float score = 0.0f;
        const float* k_base = K + head_batch * seq_len * head_dim + key_pos * head_dim;
        
        for (int d = 0; d < head_dim; d++) {
            score += q_base[d] * k_base[d];
        }
        
        shared_scores[key_pos] = score * scale_factor;
    }
    __syncthreads();

    // Step 2: Softmax normalization
    // Find max for numerical stability
    float max_score = -1e9f;
    for (int i = thread_id; i < seq_len; i += blockDim.x) {
        max_score = fmax(max_score, shared_scores[i]);
    }

    // Reduce to get global max
    __shared__ float max_scores[WARP_SIZE];
    if (thread_id < WARP_SIZE) max_scores[thread_id] = -1e9f;
    max_scores[thread_id % WARP_SIZE] = max_score;
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (thread_id + i < WARP_SIZE) {
            max_scores[thread_id] = fmax(max_scores[thread_id], max_scores[thread_id + i]);
        }
        __syncthreads();
    }
    max_score = max_scores[0];

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = thread_id; i < seq_len; i += blockDim.x) {
        float exp_val = expf(shared_scores[i] - max_score);
        shared_softmax[i] = exp_val;
        sum_exp += exp_val;
    }

    // Reduce sum
    __shared__ float sum_exps[WARP_SIZE];
    if (thread_id < WARP_SIZE) sum_exps[thread_id] = 0.0f;
    sum_exps[thread_id % WARP_SIZE] = sum_exp;
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (thread_id + i < WARP_SIZE) {
            sum_exps[thread_id] += sum_exps[thread_id + i];
        }
        __syncthreads();
    }
    sum_exp = sum_exps[0] + 1e-9f; // Add epsilon for stability

    // Normalize
    for (int i = thread_id; i < seq_len; i += blockDim.x) {
        shared_softmax[i] /= sum_exp;
        // Store attention weights for output
        attention[head_batch * seq_len * seq_len + query_pos * seq_len + i] = shared_softmax[i];
    }
    __syncthreads();

    // Step 3: Apply attention to values
    // output[query_pos, d] = sum(attention[query_pos, key_pos] * V[key_pos, d])
    
    for (int d = thread_id; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int key_pos = 0; key_pos < seq_len; key_pos++) {
            float attn_weight = shared_softmax[key_pos];
            const float* v_base = V + head_batch * seq_len * head_dim + key_pos * head_dim;
            out_val += attn_weight * v_base[d];
        }
        
        output[head_batch * seq_len * head_dim + query_pos * head_dim + d] = out_val;
    }
}

extern "C" void cuda_attention_forward(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_attention,
    float* d_output,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    int batch_heads = batch * num_heads;
    float scale_factor = 1.0f / sqrtf((float)head_dim);

    // Grid: one block per (head, query_position)
    dim3 grid(batch_heads, seq_len);
    dim3 block(BLOCK_SIZE);

    // Shared memory: scores + softmax
    size_t shared_size = 2 * seq_len * sizeof(float);

    attention_forward_kernel<<<grid, block, shared_size, stream>>>(
        d_Q, d_K, d_V,
        d_scores, d_attention, d_output,
        batch_heads, seq_len, head_dim, scale_factor
    );
}
