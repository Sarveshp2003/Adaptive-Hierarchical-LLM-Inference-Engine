#include "TensorOps.h"
#include "CUDAError.h"
#include "Logger.h"
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>


extern "C"
void launchMatmul(
    float* A,
    float* B,
    float* C,
    int rowsA,
    int colsA,
    int colsB
);


extern "C"
void launchMatmulTranspose(
    float* A,
    float* B,
    float* C,
    int M,
    int N,
    int K
);


extern "C"
void launchRelu(
    float* input,
    float* output,
    int elements
);


extern "C"
void launchAdd(
    float* A,
    float* B,
    float* C,
    int elements
);


extern "C"
void launchSoftmax(
    float* input,
    float* output,
    int rows,
    int cols
);

// GELU GPU launcher defined in cuda/gelu.cu
extern "C" void launchGelu(const float* input, float* output, size_t n);





void matmul(
    Tensor& A,
    Tensor& B,
    Tensor& C
)
{
    if(A.rank() < 2 || B.rank() < 2)
    {
        throw std::runtime_error("matmul requires rank >= 2 tensors");
    }

    const std::vector<int>& a_shape = A.shape();
    const std::vector<int>& b_shape = B.shape();

    int colsA = a_shape.back();
    int rowsA = 1;
    for(size_t i=0;i+1<a_shape.size();i++) rowsA *= a_shape[i];

    int rowsB = 1;
    for(size_t i=0;i+1<b_shape.size();i++) rowsB *= b_shape[i];
    int colsB = b_shape.back();

    if(colsA != rowsB)
    {
        throw std::runtime_error("matmul dimensions are incompatible");
    }

    if(C.elements() != static_cast<size_t>(rowsA) * static_cast<size_t>(colsB))
    {
        throw std::runtime_error("matmul output element count does not match expected result shape");
    }

    A.allocateGPU();
    B.allocateGPU();
    C.allocateGPU();

    LOG_INFO_STREAM("Matmul " << rowsA << "x" << colsA << " * " << rowsB << "x" << colsB);

    launchMatmul(
        A.gpu(),
        B.gpu(),
        C.gpu(),
        rowsA,
        colsA,
        colsB
    );

}




void matmulTranspose(
    Tensor& A,
    Tensor& B,
    Tensor& C
)
{
    if(A.rank() < 2 || B.rank() < 2)
    {
        throw std::runtime_error("matmulTranspose requires rank >= 2 tensors");
    }

    const std::vector<int>& a_shape = A.shape();
    const std::vector<int>& b_shape = B.shape();

    int K = a_shape.back();
    int M = 1;
    for(size_t i=0;i+1<a_shape.size();i++) M *= a_shape[i];
    int N = 1;
    for(size_t i=0;i+1<b_shape.size();i++) N *= b_shape[i];

    if(C.elements() != static_cast<size_t>(M) * static_cast<size_t>(N))
    {
        throw std::runtime_error("matmulTranspose output element count does not match expected result shape");
    }

    A.allocateGPU();
    B.allocateGPU();
    C.allocateGPU();

    LOG_INFO_STREAM("Matmul Transpose " << M << "x" << K << " * " << N << "x" << K);

    launchMatmulTranspose(
        A.gpu(),
        B.gpu(),
        C.gpu(),
        M,
        N,
        K
    );

}





void relu(
    
    Tensor& input,
    
    Tensor& output
    
)
{
   if(input.elements() != output.elements())
   {
       throw std::runtime_error("relu input/output element count mismatch");
   }

   input.allocateGPU();
   output.allocateGPU();
   LOG_INFO_STREAM("Launching ReLU kernel");

   launchRelu(
 
        input.gpu(),
 
        output.gpu(),
 
        static_cast<int>(
            input.elements()
        )
 
    );
 
}


void gelu(
    Tensor& input,
    Tensor& output
)
{
    // GPU implementation using CUDA kernel
    // Ensure GPU memory is present
    input.allocateGPU();
    output.allocateGPU();

    size_t n = input.elements();
    if(n == 0) return;

    // launch device kernel
    launchGelu(input.gpu(), output.gpu(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}





void add(
    
    Tensor& A,
    
    Tensor& B,
    
    Tensor& C
    
)
{
   if(A.elements() != B.elements() || B.elements() != C.elements())
   {
       throw std::runtime_error("add requires tensors with matching element counts");
   }

   A.allocateGPU();
   B.allocateGPU();
   C.allocateGPU();

   launchAdd(
 
        A.gpu(),
 
        B.gpu(),
 
        C.gpu(),
 
        static_cast<int>(
            A.elements()
        )
 
    );
 
}




void softmax(
    Tensor& input,
    Tensor& output
)
{
    if(input.rank() < 2)
    {
        throw std::runtime_error("softmax requires rank >= 2 tensors");
    }

    if(input.elements() != output.elements())
    {
        throw std::runtime_error("softmax input/output element count mismatch");
    }

    const std::vector<int>& shape = input.shape();
    int cols = shape.back();
    int rows = 1;
    for(size_t i=0;i+1<shape.size();i++) rows *= shape[i];

    input.allocateGPU();
    output.allocateGPU();
    LOG_INFO_STREAM("Launching Softmax " << rows << "x" << cols);

    launchSoftmax(
        input.gpu(),
        output.gpu(),
        rows,
        cols
    );

}