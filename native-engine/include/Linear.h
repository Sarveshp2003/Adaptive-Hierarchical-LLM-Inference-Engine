#ifndef LINEAR_H
#define LINEAR_H


#include "Tensor.h"

#include <vector>

class Linear
{


private:

    int inputSize;

    int outputSize;


    Tensor weights;



public:


    Linear(
        int input,
        int output
    );


    ~Linear();



    void forward(

        Tensor& input,

        Tensor& output

    );

    // Accessor for weights (GPU tensor)
    const Tensor& getWeights() const;

    // Load a flattened weight vector into the internal weight tensor.
    void loadWeights(const std::vector<float>& values);

};



#endif