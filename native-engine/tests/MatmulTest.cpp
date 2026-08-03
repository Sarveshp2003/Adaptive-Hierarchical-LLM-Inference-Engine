#include "TensorOps.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <algorithm>
#include <iostream>
#include <vector>

int main()
{
    RuntimeMemory::initializeGPU(1024ULL * 1024ULL * 1024ULL);
    CUDAStream::initialize();

    std::cout << "CUDA Matmul Test\n";

    const int N = 4;

    Tensor A({N, N}, DataType::FP32);
    Tensor B({N, N}, DataType::FP32);
    Tensor C({N, N}, DataType::FP32);

    A.allocateCPU();
    B.allocateCPU();
    C.allocateCPU();
    A.allocateGPU();
    B.allocateGPU();
    C.allocateGPU();

    std::vector<float> h_A = {
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    };
    std::vector<float> h_B = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    std::copy(h_A.begin(), h_A.end(), A.cpu());
    std::copy(h_B.begin(), h_B.end(), B.cpu());
    A.upload();
    B.upload();

    matmul(A, B, C);
    CUDAStream::synchronize();
    C.download();

    bool passed = true;
    for(int i = 0; i < N * N; ++i)
    {
        if(C.cpu()[i] != h_A[i])
        {
            passed = false;
            break;
        }
    }

    RuntimeMemory::shutdown();
    CUDAStream::shutdown();

    std::cout << (passed ? "MATMUL TEST PASSED" : "MATMUL TEST FAILED") << std::endl;
    return passed ? 0 : 1;
}
