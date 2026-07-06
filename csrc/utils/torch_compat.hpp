#pragma once

#include <ATen/ATen.h>

// torch/library.h declares `namespace torch` for op registration but does not
// re-export ATen types. DeepGEMM csrc uses torch::Tensor throughout; under
// Py_LIMITED_API we cannot include torch/python.h or torch/types.h (autograd
// pulls the full Python C-API). Re-export at:: into torch:: instead.
namespace torch {
using namespace at;

constexpr auto kUInt8 = at::kByte;
constexpr auto kInt8 = at::kChar;
constexpr auto kInt16 = at::kShort;
constexpr auto kInt32 = at::kInt;
constexpr auto kInt64 = at::kLong;
constexpr auto kFloat16 = at::kHalf;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;
}  // namespace torch
