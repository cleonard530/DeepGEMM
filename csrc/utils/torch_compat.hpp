#pragma once

// ATen ops with no curated `torch::stable::ops` wrapper yet, called via `torch_call_dispatcher`.

#include <array>
#include <map>
#include <limits>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cstdint>
#include <optional>
#include <vector>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>

namespace deep_gemm::torch_compat {

namespace detail {

inline void call_dispatcher(const char* op_name, const char* overload_name, StableIValue* stack) {
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(op_name, overload_name, stack, TORCH_ABI_VERSION));
}

// `torch_call_dispatcher` can't marshal `Scalar`-typed args; wrap in a 0-dim tensor instead.
inline torch::stable::Tensor scalar_tensor_like(const torch::stable::Tensor& self, double value) {
    return torch::stable::full({}, value, self.scalar_type(), std::nullopt, self.device());
}

} // namespace detail

// `aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor`
inline torch::stable::Tensor bitwise_and(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_and", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_and(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_and(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor`
inline torch::stable::Tensor bitwise_or(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_or", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_or(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_or(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor`
inline torch::stable::Tensor bitwise_left_shift(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_left_shift", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_left_shift(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_left_shift(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor`
inline torch::stable::Tensor bitwise_right_shift(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_right_shift", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_right_shift(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_right_shift(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::index_select(Tensor self, int dim, Tensor index) -> Tensor`
inline torch::stable::Tensor index_select(const torch::stable::Tensor& self, int64_t dim,
                                          const torch::stable::Tensor& index) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(index)};
    detail::call_dispatcher("aten::index_select", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::floor_divide(Tensor self, Tensor other) -> Tensor`
inline torch::stable::Tensor floor_divide(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::floor_divide", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor floor_divide(const torch::stable::Tensor& self, int64_t other) {
    return floor_divide(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// Build integer indices entirely on-device so layout conversion remains graph-capturable.
inline torch::stable::Tensor arange(int64_t end, torch::stable::Device device) {
    auto ones = torch::stable::full({end}, 1, torch::headeronly::ScalarType::Long, std::nullopt, device);
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(ones), torch::stable::detail::from(int64_t{0}),
        torch::stable::detail::from(std::optional<torch::headeronly::ScalarType>{})};
    detail::call_dispatcher("aten::cumsum", "", stack.data());
    const auto indices = torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
    return torch::stable::subtract(indices, ones);
}

// `Tensor::nbytes()` has no stable ABI equivalent; reconstruct from `numel()` and `element_size()`.
inline size_t nbytes(const torch::stable::Tensor& self) {
    return static_cast<size_t>(self.numel()) * self.element_size();
}

// `aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)`
inline torch::stable::Tensor permute(const torch::stable::Tensor& self, std::vector<int64_t> dims) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dims)};
    detail::call_dispatcher("aten::permute", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::view.dtype`: a different overload than the shape-based `view` in `torch::stable::ops`.
inline torch::stable::Tensor view_dtype(const torch::stable::Tensor& self, torch::headeronly::ScalarType dtype) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dtype)};
    detail::call_dispatcher("aten::view", "dtype", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `c10::elementSize(ScalarType)` has no stable ABI equivalent; the C shim exposes the same table.
inline size_t element_size(torch::headeronly::ScalarType dtype) {
    const int32_t shim_dtype = torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
    return aoti_torch_dtype_element_size(shim_dtype);
}

} // namespace deep_gemm::torch_compat

namespace deep_gemm::torch_compat {
inline cudaStream_t current_stream(const torch::stable::Tensor& tensor) {
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream));
    return static_cast<cudaStream_t>(stream);
}
using StreamKey = std::pair<int32_t, cudaStream_t>;
inline StreamKey stream_key(const torch::stable::Tensor& tensor) {
    return {tensor.get_device_index(), current_stream(tensor)};
}
inline bool is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    const auto error = cudaStreamIsCapturing(stream, &status);
    STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    return status != cudaStreamCaptureStatusNone;
}
inline torch::stable::Tensor zero_(torch::stable::Tensor self) {
    return torch::stable::zero_(self);
}
inline torch::stable::Tensor slice(const torch::stable::Tensor& self, int64_t dim,
                                  int64_t start, int64_t end, int64_t step = 1) {
    std::array<StableIValue, 5> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(std::optional<int64_t>(start)),
        torch::stable::detail::from(std::optional<int64_t>(end)), torch::stable::detail::from(step)};
    detail::call_dispatcher("aten::slice", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}
inline torch::stable::Tensor from_blob(void* data, std::vector<int64_t> sizes,
                                     torch::stable::Device device, torch::headeronly::ScalarType dtype) {
    std::vector<int64_t> strides(sizes.size());
    int64_t stride = 1;
    for (size_t i = sizes.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= std::max<int64_t>(sizes[i], 1);
    }
    return torch::stable::from_blob(data, sizes, strides, device, dtype);
}
inline cudaDataType scalar_type_to_cuda(torch::headeronly::ScalarType type) {
    switch (type) {
        case torch::headeronly::ScalarType::Float: return CUDA_R_32F;
        case torch::headeronly::ScalarType::Half: return CUDA_R_16F;
        case torch::headeronly::ScalarType::BFloat16: return CUDA_R_16BF;
        case torch::headeronly::ScalarType::Float8_e4m3fn: return CUDA_R_8F_E4M3;
        default: STD_TORCH_CHECK(false, "Unsupported cuBLASLt scalar type");
    }
}
}

namespace deep_gemm::torch_compat {
// Stable wrappers take mutable handles even when only tensor storage changes.
inline torch::stable::Tensor copy_(torch::stable::Tensor self, const torch::stable::Tensor& src) {
    return torch::stable::copy_(self, src);
}
inline torch::stable::Tensor narrow(torch::stable::Tensor self, int64_t dim, int64_t start, int64_t length) {
    return torch::stable::narrow(self, dim, start, length);
}
inline uint64_t storage_nbytes(const torch::stable::Tensor& self) {
    int64_t bytes = 0;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(self.get(), &bytes));
    return static_cast<uint64_t>(bytes);
}
inline torch::stable::Tensor from_blob(void* data, std::vector<int64_t> sizes, std::vector<int64_t> strides,
                                     torch::stable::Device device, torch::headeronly::ScalarType dtype) {
    return torch::stable::from_blob(data, sizes, strides, device, dtype);
}
}
