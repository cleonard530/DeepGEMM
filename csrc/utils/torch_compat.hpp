#pragma once

// ATen ops with no curated `torch::stable::ops` wrapper yet, called via `torch_call_dispatcher`.

#include <array>
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
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(op_name, overload_name, stack, TORCH_ABI_VERSION));
#else
    TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(op_name, overload_name, stack));
#endif
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

// `aten::arange`: no `Tensor`-only overload to fall back on, so build the sequence host-side.
inline torch::stable::Tensor arange(int64_t end, torch::stable::Device device) {
    std::vector<int64_t> host_data(end > 0 ? static_cast<size_t>(end) : 0);
    for (int64_t i = 0; i < end; ++i)
        host_data[static_cast<size_t>(i)] = i;
    const auto cpu_tensor = torch::stable::from_blob(
        host_data.data(), {end}, {1}, torch::stable::Device(torch::stable::DeviceType::CPU),
        torch::headeronly::ScalarType::Long);
    return torch::stable::to(cpu_tensor, device, /*non_blocking=*/false, /*copy=*/true);
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
