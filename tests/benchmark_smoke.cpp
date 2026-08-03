#include <chrono>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> values(100000, 1.0f);
    auto start = std::chrono::steady_clock::now();
    float sum = 0.0f;
    for (float v : values) {
        sum += v;
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "benchmark_ms=" << ms << " sum=" << sum << "\n";
    return 0;
}
