#include "TransformerLayer.h"
#include "TensorOps.h"
#include "CUDAStream.h"
#include "CUDAError.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include "Logger.h"

TransformerLayer::TransformerLayer(int hiddenSize)
:
hiddenSize(hiddenSize),
attention(hiddenSize, 16),
ffn(hiddenSize),
layerNorm1(hiddenSize),
layerNorm2(hiddenSize),
workspace({ hiddenSize }, DataType::FP32)
{
    workspace.allocateGPU();

    std::cout << "Transformer Layer created. Hidden size: " << hiddenSize << std::endl;
}

TransformerLayer::~TransformerLayer()
{
}

void TransformerLayer::forward(
    Tensor& input,
    Tensor& output,
    KVCache* cache,
    bool useCache
)
{
    const bool rank2Input = input.rank() == 2;
    const std::vector<int> logicalShape = rank2Input ? std::vector<int>{1, 1, input.shape()[1]} : input.shape();

    Tensor input3(logicalShape, DataType::FP32);
    Tensor output3(logicalShape, DataType::FP32);
    Tensor norm1(logicalShape, DataType::FP32);
    Tensor attentionOut(logicalShape, DataType::FP32);
    Tensor norm2(logicalShape, DataType::FP32);
    Tensor mlpOut(logicalShape, DataType::FP32);

    // Ensure CPU backing is available for safe copy (some producers write only GPU)
    if(!input.cpu()) {
        try { input.download(); }
        catch(const std::exception &e) {
            LOG_ERROR_STREAM("TransformerLayer::forward failed to download input: " << e.what());
            throw;
        }
    }

    input3.allocateCPU(); input3.allocateGPU();
    output3.allocateCPU(); output3.allocateGPU();
    norm1.allocateCPU(); norm1.allocateGPU();
    attentionOut.allocateCPU(); attentionOut.allocateGPU();
    norm2.allocateCPU(); norm2.allocateGPU();
    mlpOut.allocateCPU(); mlpOut.allocateGPU();

    // copy available elements (defensive) and zero any remainder
    size_t copyElems = std::min(input.elements(), input3.elements());
    std::copy(input.cpu(), input.cpu() + copyElems, input3.cpu());
    if(copyElems < input3.elements()) {
        std::fill(input3.cpu() + copyElems, input3.cpu() + input3.elements(), 0.0f);
    }
    input3.upload();

    try {
        LOG_INFO_STREAM("TransformerLayer::forward - before layerNorm1.forward");
        layerNorm1.forward(input3, norm1);
        LOG_INFO_STREAM("TransformerLayer::forward - after layerNorm1.forward");
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_INFO_STREAM("TransformerLayer::forward - after cudaDeviceSynchronize (layerNorm1)");
    } catch(const std::exception &e) {
        LOG_ERROR_STREAM("LayerNorm::forward threw exception: " << e.what()
                         << " input.rank=" << input.rank()
                         << " input.elements=" << input.elements());
        try { std::ofstream ofs("out/transformer_state_layernorm_fail.json"); ofs << "{\"error\":\"layernorm_fail\"}"; ofs.close(); } catch(...) {}
        throw;
    }

    try {
        LOG_INFO_STREAM("TransformerLayer::forward - before attention.forward");
        attention.forward(norm1, attentionOut, cache, useCache);
        LOG_INFO_STREAM("TransformerLayer::forward - after attention.forward");
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_INFO_STREAM("TransformerLayer::forward - after cudaDeviceSynchronize (attention)");
    } catch(const std::exception &e) {
        LOG_ERROR_STREAM("Attention::forward threw exception: " << e.what());
        try { std::ofstream ofs("out/transformer_state_attention_fail.json"); ofs << "{\"error\":\"attention_fail\"}"; ofs.close(); } catch(...) {}
        throw;
    }

    LOG_INFO_STREAM("TransformerLayer::forward - before add(input3, attentionOut, output3)");
    add(input3, attentionOut, output3);
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_INFO_STREAM("TransformerLayer::forward - after add and sync");

    try {
        LOG_INFO_STREAM("TransformerLayer::forward - before layerNorm2.forward");
        layerNorm2.forward(output3, norm2);
        LOG_INFO_STREAM("TransformerLayer::forward - after layerNorm2.forward");
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch(const std::exception &e) {
        LOG_ERROR_STREAM("LayerNorm2::forward threw exception: " << e.what());
        try { std::ofstream ofs("out/transformer_state_layernorm2_fail.json"); ofs << "{\"error\":\"layernorm2_fail\"}"; ofs.close(); } catch(...) {}
        throw;
    }

    try {
        LOG_INFO_STREAM("TransformerLayer::forward - before ffn.forward");
        ffn.forward(norm2, mlpOut);
        LOG_INFO_STREAM("TransformerLayer::forward - after ffn.forward");
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_INFO_STREAM("TransformerLayer::forward - after cudaDeviceSynchronize (ffn)");
    } catch(const std::exception &e) {
        LOG_ERROR_STREAM("FFN::forward threw exception: " << e.what());
        try { std::ofstream ofs("out/transformer_state_ffn_fail.json"); ofs << "{\"error\":\"ffn_fail\"}"; ofs.close(); } catch(...) {}
        throw;
    }

    LOG_INFO_STREAM("TransformerLayer::forward - before add(output3, mlpOut, output3)");
    add(output3, mlpOut, output3);
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_INFO_STREAM("TransformerLayer::forward - after final add and sync");

    output.allocateCPU();
    output.allocateGPU();
    std::copy(output3.cpu(), output3.cpu() + output3.elements(), output.cpu());
    output.upload();
    LOG_INFO_STREAM("TransformerLayer::forward - finished output.upload");
}

int TransformerLayer::getHiddenSize() const
{
    return hiddenSize;
}
