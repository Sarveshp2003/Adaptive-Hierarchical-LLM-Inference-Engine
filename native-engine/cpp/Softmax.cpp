#include "Softmax.h"

extern void softmax(
    Tensor& input,
    Tensor& output
);

Softmax::Softmax()
{
}

Softmax::~Softmax()
{
}

void Softmax::forward(
    Tensor& input,
    Tensor& output
)
{
    softmax(
        input,
        output
    );
}