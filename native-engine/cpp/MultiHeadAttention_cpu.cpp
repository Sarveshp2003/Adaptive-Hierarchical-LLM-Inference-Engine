#include "MultiHeadAttention.h"
#include "Linear.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>


MultiHeadAttention::MultiHeadAttention(
    int hiddenSize,
    int numHeads
)
:
hiddenSize(hiddenSize),
numHeads(numHeads),
headDim(hiddenSize / numHeads),
qLinear(hiddenSize, hiddenSize),
kLinear(hiddenSize, hiddenSize),
vLinear(hiddenSize, hiddenSize),
outLinear(hiddenSize, hiddenSize)

{

    if(hiddenSize % numHeads != 0)
    {
        throw std::runtime_error(
            "hiddenSize must be divisible by numHeads"
        );
    }


    std::cout
        << "MultiHeadAttention created "
        << hiddenSize
        << " hidden size "
        << numHeads
        << " heads"
        << std::endl;

}





void MultiHeadAttention::forwardCPU(
    Tensor& input,
    Tensor& output,
    KVCache* cache,
    bool useCache
)
{

    std::cout
        << "MHA forward start"
        << std::endl;



    /*
        Q K V tensors
    */

    Tensor Q(
        input.shape(),
        DataType::FP32
    );

    Tensor K(
        input.shape(),
        DataType::FP32
    );

    Tensor V(
        input.shape(),
        DataType::FP32
    );



    Q.allocateCPU();
    Q.allocateGPU();

    K.allocateCPU();
    K.allocateGPU();

    V.allocateCPU();
    V.allocateGPU();



    qLinear.forward(
        input,
        Q
    );


    kLinear.forward(
        input,
        K
    );


    vLinear.forward(
        input,
        V
    );



    Q.download();
    K.download();
    V.download();



    float* q = Q.cpu();
    float* k = K.cpu();
    float* v = V.cpu();



    /*
        Attention output before projection
    */

    Tensor context(
        input.shape(),
        DataType::FP32
    );


    context.allocateCPU();
    context.allocateGPU();



    float* ctx =
        context.cpu();



    memset(
        ctx,
        0,
        context.bytes()
    );



    int batch =
        input.shape()[0];


    int seqLen =
        input.shape()[1];



    float scale =
        1.0f /
        sqrtf(
            (float)headDim
        );



    std::cout
        << "Attention "
        << "batch="
        << batch
        << " seq="
        << seqLen
        << " heads="
        << numHeads
        << std::endl;




    for(int b=0;b<batch;b++)
    {


        for(int h=0;h<numHeads;h++)
        {


            int headOffset =
                h * headDim;



            std::vector<float> scores(
                seqLen * seqLen
            );



            /*
                QK^T
            */


            for(int i=0;i<seqLen;i++)
            {

                float maxScore =
                    -1e9f;


                for(int j=0;j<seqLen;j++)
                {

                    float score=0;



                    for(int d=0;d<headDim;d++)
                    {

                        int qIndex =
                            b*seqLen*hiddenSize
                            +
                            i*hiddenSize
                            +
                            headOffset
                            +
                            d;


                        int kIndex =
                            b*seqLen*hiddenSize
                            +
                            j*hiddenSize
                            +
                            headOffset
                            +
                            d;



                        score +=
                            q[qIndex]
                            *
                            k[kIndex];

                    }


                    score*=scale;


                    scores[
                        i*seqLen+j
                    ] = score;


                    maxScore =
                        std::max(
                            maxScore,
                            score
                        );

                }




                /*
                    Softmax
                */


                float sum=0;


                for(int j=0;j<seqLen;j++)
                {

                    float e =
                        expf(
                            scores[i*seqLen+j]
                            -
                            maxScore
                        );


                    scores[i*seqLen+j]=e;

                    sum+=e;

                }



                for(int j=0;j<seqLen;j++)
                {

                    scores[i*seqLen+j]
                    /=
                    sum;

                }


            }




            /*
                Attention x V
            */


            for(int i=0;i<seqLen;i++)
            {


                for(int d=0;d<headDim;d++)
                {


                    float value=0;



                    for(int j=0;j<seqLen;j++)
                    {

                        int vIndex =
                            b*seqLen*hiddenSize
                            +
                            j*hiddenSize
                            +
                            headOffset
                            +
                            d;



                        value +=
                            scores[i*seqLen+j]
                            *
                            v[vIndex];

                    }



                    int index =
                        b*seqLen*hiddenSize
                        +
                        i*hiddenSize
                        +
                        headOffset
                        +
                        d;



                    ctx[index]=value;


                }


            }


        }


    }




    /*
        Output projection
    */


    context.upload();



    outLinear.forward(
        context,
        output
    );



    output.download();



    /*
        Residual connection

        output = input + attention
    */


    float* src =
        input.cpu();


    float* dst =
        output.cpu();



    for(int i=0;i<input.elements();i++)
    {
        dst[i]+=src[i];
    }



    output.upload();



    std::cout
        << "MHA output projection + residual complete"
        << std::endl;


}


void MultiHeadAttention::forward(
    Tensor& input,
    Tensor& output,
    KVCache* cache,
    bool useCache
)
{
    // In CPU-only build, forward delegates to CPU implementation
    forwardCPU(input, output, cache, useCache);
}