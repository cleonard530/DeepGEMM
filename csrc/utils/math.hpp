// TODO: merge this file with `math.cuh` (the device part)
#pragma once

#include <vector>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

#include "exception.hpp"
#include "torch_compat.hpp"

namespace deep_gemm {

// TODO: use `torch::kFloat4_e2m1fn_x2`
constexpr auto kPackedFP4 = torch::headeronly::ScalarType::Char;

template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

static torch::stable::Tensor fp4_repack_to_k_major(const torch::stable::Tensor& a, int logical_mn) {
    DG_HOST_ASSERT(a.scalar_type() == kPackedFP4);
    const int ndim = a.dim();
    DG_HOST_ASSERT(ndim == 2 or ndim == 3);
    const int mn_packed = a.size(-2);
    const int k = a.size(-1);
    DG_HOST_ASSERT(mn_packed * 2 == logical_mn and k % 2 == 0);

    auto lo = torch_compat::bitwise_and(a, 0x0F);
    auto hi = torch_compat::bitwise_and(
        torch::stable::to(torch_compat::bitwise_right_shift(
                              torch::stable::to(a, torch::headeronly::ScalarType::Byte), 4),
                          torch::headeronly::ScalarType::Char),
        0x0F);

    auto shape_full = a.sizes().vec();
    shape_full[ndim - 2] = logical_mn;
    auto codes = torch::stable::new_empty(a, shape_full, kPackedFP4);

    // No stable `index_put_` with a strided slice; interleave via reshape + `select` + `copy_`.
    std::vector<int64_t> interleaved_shape(shape_full.begin(), shape_full.begin() + (ndim - 2));
    interleaved_shape.push_back(mn_packed);
    interleaved_shape.push_back(2);
    interleaved_shape.push_back(k);
    auto codes_view = torch::stable::view(codes, interleaved_shape);
    auto codes_lo_slots = torch::stable::select(codes_view, ndim - 1, 0);
    auto codes_hi_slots = torch::stable::select(codes_view, ndim - 1, 1);
    torch::stable::copy_(codes_lo_slots, lo);
    torch::stable::copy_(codes_hi_slots, hi);

    auto shape_view = shape_full;
    shape_view[ndim - 1] = k / 2;
    shape_view.push_back(2);
    auto codes2 = torch::stable::view(codes, shape_view);
    auto result = torch_compat::bitwise_or(
        torch_compat::bitwise_and(torch::stable::select(codes2, -1, 0), 0x0F),
        torch::stable::to(torch_compat::bitwise_left_shift(
                              torch::stable::to(torch_compat::bitwise_and(torch::stable::select(codes2, -1, 1), 0x0F),
                                                torch::headeronly::ScalarType::Byte),
                              4),
                          torch::headeronly::ScalarType::Char));
    return torch::stable::contiguous(result);
}

} // namespace deep_gemm
