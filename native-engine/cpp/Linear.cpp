#include "Linear.h"

#include "TensorOps.h"

#include <algorithm>
#include <stdexcept>
#include "Logger.h"



Linear::Linear(
    int inputSize,
    int outputSize
)

:

inputSize(inputSize),

outputSize(outputSize),

weights(
    {
        inputSize,
        outputSize
    },
    DataType::FP32
)

{

    LOG_INFO_STREAM("Linear layer created " << inputSize << " -> " << outputSize);



    weights.allocateCPU();

    weights.allocateGPU();



    /*
        Temporary initialization:

        Identity-like weights

    */


    for(size_t i=0;i<weights.elements();i++)
    {
        weights.cpu()[i]=0.01f;
    }


    weights.upload();

}



Linear::~Linear()
{

}


const Tensor& Linear::getWeights() const
{
    return weights;
}

void Linear::loadWeights(const std::vector<float>& values)
{
    if(values.size() != weights.elements())
    {
        throw std::runtime_error("Linear::loadWeights size mismatch");
    }

    weights.allocateCPU();
    std::copy(values.begin(), values.end(), weights.cpu());
    weights.allocateGPU();
    weights.upload();
}

void Linear::forward(
    Tensor& input,

    Tensor& output

)

{
    // Support rank-2 [batch, hidden] and rank-3 [batch, seq, hidden] layouts.
    if(input.rank() != 2 && input.rank() != 3)
    {
        throw std::runtime_error("Linear::forward expected input rank 2 or 3");
    }

    int batch = 1;
    int seq = 1;
    int inHidden = 0;

    if(input.rank() == 2)
    {
        batch = input.shape()[0];
        inHidden = input.shape()[1];
    }
    else
    {
        batch = input.shape()[0];
        seq = input.shape()[1];
        inHidden = input.shape()[2];
    }

    if(inHidden != inputSize)
    {
        throw std::runtime_error("Linear::forward input hidden dimension does not match layer input size");
    }

    int outHidden = outputSize;
    int rows = batch * seq;

    if(output.rank() != input.rank())
    {
        throw std::runtime_error("Linear::forward expected output rank to match input rank");
    }

    if(output.rank() == 2)
    {
        if(output.shape()[0] != batch || output.shape()[1] != outHidden)
        {
            throw std::runtime_error("Linear::forward expected output shape [batch,outHidden]");
        }
    }
    else if(output.rank() == 3)
    {
        if(output.shape()[0] != batch || output.shape()[1] != seq || output.shape()[2] != outHidden)
        {
            throw std::runtime_error("Linear::forward expected output shape [batch,seq,outHidden]");
        }
    }

    if(!input.cpu() || !weights.cpu())
    {
        throw std::runtime_error("Linear::forward requires CPU buffers for the fallback path");
    }

    output.allocateCPU();
    std::fill(output.cpu(), output.cpu() + output.elements(), 0.0f);

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < outHidden; ++c)
        {
            float sum = 0.0f;
            const float* rowInput = input.cpu() + r * inHidden;
            for(int k = 0; k < inHidden; ++k)
            {
                sum += rowInput[k] * weights.cpu()[k * outHidden + c];
            }
            output.cpu()[r * outHidden + c] = sum;
        }
    }

    output.allocateGPU();
    output.upload();

}