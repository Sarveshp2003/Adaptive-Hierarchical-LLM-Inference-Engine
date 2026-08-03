#include "Attention.h"

#include "TensorOps.h"
#include "Logger.h"
#include <stdexcept>


Attention::Attention(
    int hiddenSize
)
:
hiddenSize(hiddenSize)

{
   LOG_INFO_STREAM("Attention created. Hidden size: " << hiddenSize);
}



Attention::~Attention()
{

}



void Attention::forward(

    Tensor& query,
    Tensor& key,
    Tensor& value,
    Tensor& output

)

{
    LOG_INFO_STREAM("Attention forward");

    if(query.rank() != 2 || key.rank() != 2 || value.rank() != 2 || output.rank() != 2)
    {
       throw std::runtime_error("Attention expects rank-2 query/key/value/output tensors");
    }

    if(query.dim(1) != key.dim(1) || key.dim(0) != value.dim(0) || query.dim(0) != output.dim(0) || value.dim(1) != output.dim(1))
    {
       throw std::runtime_error("Attention input/output dimensions are incompatible");
    }

    /*
        Step 1:

        Q * K^T

    */


    Tensor scores(

        {
            query.shape()[0],
            key.shape()[0]

        },

        DataType::FP32

    );


    scores.allocateGPU();



    matmulTranspose(

        query,

        key,

        scores

    );



    /*
        Step 2:

        Softmax

    */


    Tensor attentionWeights(

        scores.shape(),

        DataType::FP32

    );


    attentionWeights.allocateGPU();



    softmax(

        scores,
        attentionWeights

    );



    /*
        Step 3:

        Attention * V

    */


    matmul(

        attentionWeights,
        value,
        output

    );


}