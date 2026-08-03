#include "../runtime/tensor_ops.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <functional>

using namespace runtime;

std::vector<float> rand_vec(size_t n) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> d(-1.0f,1.0f);
    std::vector<float> v(n);
    for (size_t i=0;i<n;++i) v[i]=d(rng);
    return v;
}

int main() {
    const size_t N = 512; // moderate size for local benchmark
    const size_t M = 512;
    const size_t K = 512;

    auto A = rand_vec(N*K);
    auto B = rand_vec(K*M);

    auto time_and_run = [&](const std::string &name, std::function<std::vector<float>()> job) {
        auto start = std::chrono::steady_clock::now();
        auto C = job();
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << " time_ms=" << ms << " C[0]=" << (C.size()?C[0]:0) << "\n";
    };

    time_and_run("matmul", [&](){ return TensorOps::matmul(A,B,N,K,M); });
    time_and_run("matmul_tiled", [&](){ return TensorOps::matmul_tiled(A,B,N,K,M,64); });
    time_and_run("matmul_parallel", [&](){ return TensorOps::matmul_parallel(A,B,N,K,M,0); });

    return 0;
}
