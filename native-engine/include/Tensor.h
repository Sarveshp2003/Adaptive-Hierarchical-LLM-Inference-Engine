#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <algorithm>

#include "RuntimeMemory.h"
// CUDA headers removed from header to allow CPU-only builds in this environment.
// Ensure Windows min/max macros don't break std::numeric_limits::max()
#ifdef max
#undef max
#endif

enum class DataType
{
    FP32,
    INT32
};


class Tensor
{

private:

    std::vector<int> shape_;

    DataType dtype_;

    float* cpuData_;

    float* gpuData_;

    size_t elements_;


public:


    Tensor(
        std::vector<int> shape,
        DataType dtype = DataType::FP32
    );

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();



    void allocateCPU();

    void allocateGPU();



    void upload();

    void download();



    void release();



    float* cpu();

    float* gpu();


    const float* cpu() const;

    const float* gpu() const;



    size_t elements() const;

    size_t bytes() const;


    const std::vector<int>& shape() const;


    int dim(size_t index) const;


    DataType dtype() const;

    // Rank (number of dimensions)
    int rank() const;

    // Compute flat element offset for [batch, sequence, hidden_index]
    // Throws if rank < 3 or indices out of range.
    size_t offset(int batch, int sequence, int hidden_index) const;


};

// Inline implementations to make Tensor header-only for test targets
inline Tensor::Tensor(std::vector<int> shape, DataType dtype)
    : shape_(std::move(shape)), dtype_(dtype), cpuData_(nullptr), gpuData_(nullptr), elements_(1) {
    if (shape_.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    for (int dim : shape_) {
        if (dim <= 0) throw std::runtime_error("Tensor dimensions must be positive");
        if (elements_ > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) throw std::runtime_error("Tensor size overflow");
        elements_ *= static_cast<size_t>(dim);
    }
}

inline Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), cpuData_(nullptr), gpuData_(nullptr), elements_(other.elements_) {
    if (other.cpuData_) {
        allocateCPU();
        std::copy(other.cpuData_, other.cpuData_ + elements_, cpuData_);
    }
    if (other.gpuData_) {
        gpuData_ = nullptr;
    }
}

inline Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        release();
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        elements_ = other.elements_;
        cpuData_ = nullptr;
        gpuData_ = nullptr;
        if (other.cpuData_) {
            allocateCPU();
            std::copy(other.cpuData_, other.cpuData_ + elements_, cpuData_);
        }
        if (other.gpuData_) {
            gpuData_ = nullptr;
        }
    }
    return *this;
}

inline Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), cpuData_(other.cpuData_), gpuData_(other.gpuData_), elements_(other.elements_) {
    other.cpuData_ = nullptr;
    other.gpuData_ = nullptr;
    other.elements_ = 0;
}

inline Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        cpuData_ = other.cpuData_;
        gpuData_ = other.gpuData_;
        elements_ = other.elements_;
        other.cpuData_ = nullptr;
        other.gpuData_ = nullptr;
        other.elements_ = 0;
    }
    return *this;
}

inline Tensor::~Tensor() { release(); }

inline void Tensor::allocateCPU() {
    if (cpuData_) return;
    if (elements_ == 0) throw std::runtime_error("Tensor has zero elements");
    cpuData_ = new float[elements_];
}

inline void Tensor::allocateGPU() {
    if (gpuData_) return;
    if (elements_ == 0) throw std::runtime_error("Tensor has zero elements");
    gpuData_ = static_cast<float*>(RuntimeMemory::allocateGPU(bytes()));
    if (!gpuData_) throw std::runtime_error("GPU allocation failed");
}

inline void Tensor::upload() {
    if (!cpuData_ || !gpuData_) throw std::runtime_error("Tensor upload failed: memory not allocated");
    if (bytes() == 0) return;
    // GPU upload not supported in this build environment (CUDA headers unavailable)
    throw std::runtime_error("Tensor::upload - GPU operations not available in this build");
}

inline void Tensor::download() {
    if (!cpuData_ || !gpuData_) throw std::runtime_error("Tensor download failed: memory not allocated");
    if (bytes() == 0) return;
    // GPU download not supported in this build environment (CUDA headers unavailable)
    throw std::runtime_error("Tensor::download - GPU operations not available in this build");
}

inline void Tensor::release() {
    if (cpuData_) { delete[] cpuData_; cpuData_ = nullptr; }
    if (gpuData_) {
        // In this reconstruction build we avoid invoking RuntimeMemory::releaseGPU to
        // reduce inter-target link dependencies. In full build, RuntimeMemory should
        // be used to free GPU pointers.
        gpuData_ = nullptr;
    }
}

inline float* Tensor::gpu() { return gpuData_; }
inline float* Tensor::cpu() { return cpuData_; }
inline const float* Tensor::gpu() const { return gpuData_; }
inline const float* Tensor::cpu() const { return cpuData_; }
inline size_t Tensor::elements() const { return elements_; }
inline size_t Tensor::bytes() const { return elements_ * sizeof(float); }
inline const std::vector<int>& Tensor::shape() const { return shape_; }
inline int Tensor::dim(size_t index) const { return shape_.at(index); }
inline DataType Tensor::dtype() const { return dtype_; }
inline int Tensor::rank() const { return static_cast<int>(shape_.size()); }
inline size_t Tensor::offset(int batch, int sequence, int hidden_index) const {
    if (shape_.size() < 3) throw std::runtime_error("Tensor::offset requires tensor of rank >= 3");
    int bDim = shape_.at(0); int sDim = shape_.at(1); int hDim = shape_.at(2);
    if (batch < 0 || batch >= bDim || sequence < 0 || sequence >= sDim || hidden_index < 0 || hidden_index >= hDim) throw std::out_of_range("Tensor::offset index out of range");
    return static_cast<size_t>(batch) * static_cast<size_t>(sDim) * static_cast<size_t>(hDim) + static_cast<size_t>(sequence) * static_cast<size_t>(hDim) + static_cast<size_t>(hidden_index);
}

#endif