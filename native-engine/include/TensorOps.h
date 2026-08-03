#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H


#include "Tensor.h"


void matmul(
    Tensor& A,
    Tensor& B,
    Tensor& C
);


void matmulTranspose(
    Tensor& A,
    Tensor& B,
    Tensor& C
);


void relu(
    Tensor& input,
    Tensor& output
);

void gelu(
    Tensor& input,
    Tensor& output
);

void add(
    Tensor& A,
    Tensor& B,
    Tensor& C
);


void softmax(
    Tensor& input,
    Tensor& output
);


#endif