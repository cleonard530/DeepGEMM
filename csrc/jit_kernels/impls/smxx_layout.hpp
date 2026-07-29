#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/torch_compat.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

class TransposeFP32Runtime final: public LaunchRuntime<TransposeFP32Runtime> {
public:
    struct Args {
        int mn, sf_k;
        int block_mn;
        void *sf, *out;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_fp32<
        {}, {}, {}
    >);
}};
)", args.launch_args.num_threads, args.block_mn, args.sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn)));
    }
};

class TransposeAndPackFP32IntoUE8M0Runtime final: public LaunchRuntime<TransposeAndPackFP32IntoUE8M0Runtime> {
public:
    struct Args {
        int mn, sf_k;
        int num_psum_groups, m_alignment;
        bool use_psum_layout;
        int block_mn;
        void *sf, *out, *grouped_layout;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<
        {}, {}, {}, {}, {}
    >);
}};
)", args.launch_args.num_threads, args.block_mn, args.sf_k,
    args.num_psum_groups, args.use_psum_layout ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn),
            args.grouped_layout, static_cast<uint32_t>(args.m_alignment)));
    }
};

class PackFP32IntoUE8M0Runtime final: public LaunchRuntime<PackFP32IntoUE8M0Runtime> {
public:
    struct Args {
        int num_groups, mn, sf_k, packed_sf_k, gran_k, k_alignment;
        bool use_psum_layout;
        int block_mn, block_packed_sf_k;
        void *sf, *out, *grouped_layout;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pack_fp32_into_ue8m0<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", args.num_groups, args.launch_args.num_threads, args.block_mn, args.block_packed_sf_k, "true", args.use_psum_layout ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sf, args.out, args.grouped_layout, args.mn, args.sf_k, args.packed_sf_k, args.gran_k, args.k_alignment));
    }
};

static std::tuple<int, int, int, int, int, torch::stable::Tensor> preprocess_sf(const torch::stable::Tensor& sf) {
    // NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    const auto dim = sf.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    DG_HOST_ASSERT(sf.scalar_type() == torch::headeronly::ScalarType::Float);
    const auto batched_sf = dim == 2 ? torch::stable::unsqueeze(sf, 0) : sf;

    const auto [num_sf_batches, mn, sf_k] = get_shape<3>(batched_sf);
    const auto tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::stable::Tensor get_mn_major_tma_aligned_tensor(const torch::stable::Tensor& sf) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1 and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? torch::stable::squeeze(batched_sf, 0) : batched_sf;

    // No stable `new_empty_strided`: allocate transposed-shape, then transpose + narrow to `mn`.
    auto out_storage = torch::stable::new_empty(batched_sf, {num_sf_batches, sf_k, tma_aligned_mn}, batched_sf.scalar_type());
    auto out_transposed = torch::stable::transpose(out_storage, 1, 2);
    auto out = torch::stable::narrow(out_transposed, 1, 0, mn);

    if (not batched_sf.is_contiguous()) {
        // Fallback to PyTorch's slow copy if not contiguous
        // ReSharper disable once CppExpressionWithoutSideEffects
        torch::stable::copy_(out, batched_sf);
    } else {
        constexpr int block_mn = 64;
        constexpr int num_threads = 512;
        const auto smem_size = block_mn * (sf_k + (1 - (sf_k % 2))) * static_cast<int>(sizeof(float));
        const TransposeFP32Runtime::Args& args = {
            .mn = mn,
            .sf_k = sf_k,
            .block_mn = block_mn,
            .sf = batched_sf.mutable_data_ptr(),
            .out = out.mutable_data_ptr(),
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_sf_batches}, num_threads, smem_size)
        };

        const auto code = TransposeFP32Runtime::generate(args);
        const auto runtime = compiler->build("transpose_fp32", code);
        TransposeFP32Runtime::launch(runtime, args);
    }
    return (dim == 2) ? torch::stable::squeeze(out, 0) : out;
}

static torch::stable::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(const torch::stable::Tensor& sf) {
    const auto sf_reshaped = (sf.dim() == 2) ? torch::stable::unsqueeze(sf, 0) : sf;

    // First, convert into UE8M0 `uint8_t`
    const auto ue8m0_tensor = torch::stable::to(
        torch_compat::bitwise_right_shift(torch_compat::view_dtype(sf_reshaped, torch::headeronly::ScalarType::Int), 23),
        torch::headeronly::ScalarType::Byte);

    // Second, make padded packed tensors
    const auto [num_sf_batches, mn, k] = get_shape<3>(sf_reshaped);
    const auto aligned_mn = get_tma_aligned_size(mn, 4);
    const auto aligned_k  = align(k, 4);

    auto padded = torch::stable::new_zeros(sf, {num_sf_batches, aligned_mn, aligned_k}, torch::headeronly::ScalarType::Byte);
    // ReSharper disable once CppExpressionWithoutSideEffects
    auto padded_narrow_mn = torch::stable::narrow(padded, 1, 0, mn);
    auto padded_narrow_mn_k = torch::stable::narrow(padded_narrow_mn, 2, 0, k);
    torch::stable::copy_(padded_narrow_mn_k, ue8m0_tensor);
    auto padded_flat = torch::stable::view(padded, -1);
    auto padded_as_int = torch_compat::view_dtype(padded_flat, torch::headeronly::ScalarType::Int);
    padded = torch::stable::view(padded_as_int, {num_sf_batches, aligned_mn, aligned_k / 4});

    // No stable `new_empty_strided`: allocate transposed-shape, then transpose.
    auto out_storage = torch::stable::new_empty(sf, {num_sf_batches, aligned_k / 4, aligned_mn}, torch::headeronly::ScalarType::Int);
    auto out = torch::stable::transpose(out_storage, 1, 2);
    torch::stable::copy_(out, padded);
    auto out_narrow = torch::stable::narrow(out, 1, 0, mn);
    return (sf.dim() == 2) ? torch::stable::squeeze(out_narrow, 0) : out_narrow;
}

static torch::stable::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::stable::Tensor& sf,
                                                                  const std::optional<torch::stable::Tensor>& psum_layout = std::nullopt) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);
    const auto packed_sf_k = ceil_div(sf_k, 4);
    // No stable `new_empty_strided`: allocate transposed-shape, then transpose + narrow.
    auto out_storage = torch::stable::new_empty(batched_sf, {num_sf_batches, packed_sf_k, tma_aligned_mn}, torch::headeronly::ScalarType::Int);
    auto out_transposed = torch::stable::transpose(out_storage, 1, 2);
    auto out = torch::stable::narrow(out_transposed, 1, 0, mn);

    // PSUM layout (always 2D contiguous) lets the pack kernel skip uninitialized MN gap rows
    const auto use_psum_layout = psum_layout.has_value();
    if (use_psum_layout) {
        DG_HOST_ASSERT(num_sf_batches == 1 and batched_sf.is_contiguous());
        DG_HOST_ASSERT(psum_layout->scalar_type() == torch::headeronly::ScalarType::Int and psum_layout->is_contiguous());
        DG_HOST_ASSERT(psum_layout->numel() > 0);
    }
    const auto m_alignment = use_psum_layout ? heuristics_runtime->get_mk_alignment_for_contiguous_layout() : 0;
    const auto num_psum_groups = use_psum_layout ? static_cast<int>(psum_layout->numel()) : 1;

    // Launch the kernel
    if (batched_sf.is_contiguous()) {
        if ((mn * sf_k) % 4 != 0 and num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);

        const int max_smem = static_cast<int>(device_runtime->get_prop()->sharedMemPerBlockOptin);
        const int block_mn = std::min(48, (max_smem / (sf_k * 4) / 4) * 4);
        DG_HOST_ASSERT(block_mn >= 4);
        constexpr int num_threads = 512;
        const auto psum_smem_elems = use_psum_layout ? align(num_psum_groups * 2, 4) : 0;
        const auto smem_size = block_mn * sf_k * 4 + psum_smem_elems * 4;
        const TransposeAndPackFP32IntoUE8M0Runtime::Args& args = {
            .mn = mn,
            .sf_k = sf_k,
            .num_psum_groups = num_psum_groups,
            .m_alignment = m_alignment,
            .use_psum_layout = use_psum_layout,
            .block_mn = block_mn,
            .sf = batched_sf.mutable_data_ptr(),
            .out = out.mutable_data_ptr(),
            .grouped_layout = use_psum_layout ? psum_layout->mutable_data_ptr() : nullptr,
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_sf_batches}, num_threads, smem_size)
        };

        const auto code = TransposeAndPackFP32IntoUE8M0Runtime::generate(args);
        const auto runtime = compiler->build("transpose_and_pack_fp32_into_ue8m0", code);
        TransposeAndPackFP32IntoUE8M0Runtime::launch(runtime, args);
    } else {
        DG_HOST_ASSERT(not use_psum_layout);
        if (mn % 4 != 0 or num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);
        DG_HOST_ASSERT(batched_sf.stride(1) == 1 and batched_sf.stride(2) == mn);

        constexpr int block_mn = 128;
        constexpr int block_packed_sf_k = 16;
        constexpr int num_threads = 512;
        const PackFP32IntoUE8M0Runtime::Args& args = {
            .num_groups = 1,
            .mn = mn,
            .sf_k = sf_k,
            .packed_sf_k = packed_sf_k,
            // NOTES: `gran_k`/`k_alignment` are only used by k-grouped psum, dummy here
            .gran_k = 128,
            .k_alignment = 128,
            .block_mn = block_mn,
            .block_packed_sf_k = block_packed_sf_k,
            .sf = batched_sf.mutable_data_ptr(),
            .out = out.mutable_data_ptr(),
            .grouped_layout = nullptr,
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k)}, num_threads)
        };

        const auto code = PackFP32IntoUE8M0Runtime::generate(args);
        const auto runtime = compiler->build("pack_fp32_into_ue8m0", code);
        PackFP32IntoUE8M0Runtime::launch(runtime, args);
    }
    return (dim == 2) ? torch::stable::squeeze(out, 0) : out;
}

static torch::stable::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::stable::Tensor& sf,
                                                                            const torch::stable::Tensor& grouped_layout,
                                                                            const std::optional<std::vector<int>>& ks_cpu,
                                                                            const int gran_k,
                                                                            const int k_alignment,
                                                                            const bool& use_psum_layout) {
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(k_alignment % 32 == 0);
    const auto [sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(grouped_layout.numel());

    DG_HOST_ASSERT(sf.is_contiguous());
    DG_HOST_ASSERT(num_groups <= 128 and mn % 4 == 0);
    DG_HOST_ASSERT(grouped_layout.is_contiguous() and grouped_layout.scalar_type() == torch::headeronly::ScalarType::Int);

    const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
    if (has_synced_ks) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
    } else {
        DG_HOST_ASSERT(use_psum_layout);
    }

    int packed_sf_k = 0;
    if (has_synced_ks) {
        int ref_sf_k = 0;
        for (const auto k: ks_cpu.value()) {
            ref_sf_k += ceil_div(k, gran_k);
            packed_sf_k += ceil_div(k, gran_k * 4);
        }
        DG_HOST_ASSERT(use_psum_layout or ref_sf_k == sf_k);
    } else {
        packed_sf_k = (sf_k + num_groups * 3) / 4;
    }
    if (packed_sf_k == 0)
        return torch::stable::new_empty(sf, {0, mn}, torch::headeronly::ScalarType::Int);

    const auto out = torch::stable::new_empty(sf, {packed_sf_k, mn}, torch::headeronly::ScalarType::Int);

    constexpr int block_mn = 128;
    constexpr int block_packed_sf_k = 16;
    constexpr int num_threads = 512;
    const PackFP32IntoUE8M0Runtime::Args args = {
        .num_groups = num_groups,
        .mn = mn,
        .sf_k = sf_k,
        .packed_sf_k = packed_sf_k,
        .gran_k = gran_k,
        .k_alignment = k_alignment,
        .use_psum_layout = use_psum_layout,
        .block_mn = block_mn,
        .block_packed_sf_k = block_packed_sf_k,
        .sf = sf.mutable_data_ptr(),
        .out = out.mutable_data_ptr(),
        .grouped_layout = grouped_layout.mutable_data_ptr(),
        .launch_args = LaunchArgs({ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k)}, num_threads)
    };

    const auto code = PackFP32IntoUE8M0Runtime::generate(args);
    const auto runtime = compiler->build("pack_fp32_into_ue8m0", code);
    PackFP32IntoUE8M0Runtime::launch(runtime, args);
    return out;
}

// Validate a user-provided, already packed UE8M0 (`int32`) SF tensor.
static torch::stable::Tensor check_k_grouped_packed_ue8m0_tensor(const torch::stable::Tensor& sf,
                                                         const torch::stable::Tensor& grouped_layout,
                                                         const std::optional<std::vector<int>>& ks_cpu,
                                                         const int gran_k,
                                                         const int k_alignment,
                                                         const bool& use_psum_layout) {
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(k_alignment % 32 == 0);
    DG_HOST_ASSERT(sf.scalar_type() == torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(sf.is_contiguous());

    const auto [packed_sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(grouped_layout.numel());
    DG_HOST_ASSERT(mn % 4 == 0);
    DG_HOST_ASSERT(sf.stride(0) == mn and sf.stride(1) == 1);
    DG_HOST_ASSERT(grouped_layout.is_contiguous() and grouped_layout.scalar_type() == torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(packed_sf_k > 0);

    const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
    if (has_synced_ks) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
        if (not use_psum_layout) {
            int aligned_packed_sf_k = 0;
            for (const auto k: ks_cpu.value())
                aligned_packed_sf_k += ceil_div(k, gran_k * 4);
            DG_HOST_ASSERT(packed_sf_k >= aligned_packed_sf_k);
        }
    } else {
        DG_HOST_ASSERT(use_psum_layout);
    }
    return sf;
}

} // namespace deep_gemm
