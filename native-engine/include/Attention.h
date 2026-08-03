#ifndef ATTENTION_H
#define ATTENTION_H

#include "Tensor.h"


class Attention
{

private:

    int hiddenSize;


public:

    Attention(
        int hiddenSize
    );


    ~Attention();


    void forward(

        Tensor& query,
        Tensor& key,
        Tensor& value,
        Tensor& output

    );


};


#endif