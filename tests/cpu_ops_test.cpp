#include "../runtime/tensor_ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    using namespace runtime;

    bool ok = true;

    // Test matmul 2x3 * 3x2
    std::vector<float> A = {1,2,3,4,5,6}; // 2x3
    std::vector<float> B = {7,8,9,10,11,12}; // 3x2
    auto C = TensorOps::matmul(A, B, 2, 3, 2);
    std::vector<float> C_expected = {58,64,139,154};
    for (size_t i = 0; i < C.size(); ++i) {
        if (std::abs(C[i] - C_expected[i]) > 1e-3) {
            std::cerr << "matmul mismatch at " << i << ": got " << C[i] << " expected " << C_expected[i] << "\n";
            ok = false;
        }
    }

    // Test softmax row-wise
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 2.0f}; // 2x2
    auto s = TensorOps::softmax(logits, 2, 2);
    auto soft = [](float a, float b){ float ma = std::max(a,b); float ea = std::exp(a-ma); float eb = std::exp(b-ma); float sum = ea+eb; return std::pair<float,float>(ea/sum, eb/sum); };
    auto p0 = soft(1.0f,2.0f);
    auto p1 = soft(3.0f,2.0f);
    if (std::abs(s[0]-p0.first)>1e-3) { std::cerr<<"softmax[0] mismatch: "<<s[0]<<" vs "<<p0.first<<"\n"; ok=false; }
    if (std::abs(s[1]-p0.second)>1e-3) { std::cerr<<"softmax[1] mismatch: "<<s[1]<<" vs "<<p0.second<<"\n"; ok=false; }
    if (std::abs(s[2]-p1.first)>1e-3) { std::cerr<<"softmax[2] mismatch: "<<s[2]<<" vs "<<p1.first<<"\n"; ok=false; }
    if (std::abs(s[3]-p1.second)>1e-3) { std::cerr<<"softmax[3] mismatch: "<<s[3]<<" vs "<<p1.second<<"\n"; ok=false; }

    // Test attention simple: seq_len=2 dim=2
    std::vector<float> Q = {1,0, 0,1};
    std::vector<float> K = {1,0, 0,1};
    std::vector<float> V = {1,0, 0,1};
    auto out = TensorOps::attention(Q,K,V,2,2);
    // Expect each row to weight its matching key more strongly than the other
    if (!(out[0] > out[1])) { std::cerr<<"att row0 not weighted to self: "<<out[0]<<","<<out[1]<<"\n"; ok=false; }
    if (!(out[3] > out[2])) { std::cerr<<"att row1 not weighted to self: "<<out[2]<<","<<out[3]<<"\n"; ok=false; }
    // Also check rows sum approximately to 1 (V is identity so the two elements of each output row are softmax weights)
    float sum0 = out[0] + out[1];
    float sum1 = out[2] + out[3];
    if (std::abs(sum0 - 1.0f) > 1e-3) { std::cerr<<"att row0 sum "<<sum0<<"\n"; ok=false; }
    if (std::abs(sum1 - 1.0f) > 1e-3) { std::cerr<<"att row1 sum "<<sum1<<"\n"; ok=false; }

    std::cout << (ok ? "CPU ops tests: PASS\n" : "CPU ops tests: FAIL\n");
    return ok ? 0 : 1;
}
