#pragma once

#include <string>
#include <tuple>

#include "../utils/torch_compat.hpp"
#include <deep_gemm/common/types.cuh>
#include "../utils/math.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../torch_library_macros.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static std::pair<int, int> get_ring_limit_for_mega_moe(
    const int& num_max_tokens_per_rank, const int& num_experts_per_rank, const int& num_topk, const int& num_ranks) {
    return {
        get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, 1, layout::kLCMCandidateBlockM),
        get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, num_experts_per_rank, layout::kLCMCandidateBlockM)
    };
}

struct SymmBufferLayoutInfo {
    int64_t num_bytes = 0;
    int64_t input_token_base = 0;
    int64_t input_sf_base = 0;
    int64_t input_topk_idx_base = 0;
    int64_t input_topk_weights_base = 0;
    int64_t l1_token_base = 0;
    int64_t l1_sf_base = 0;
    int64_t l2_token_base = 0;
    int64_t l2_sf_base = 0;
    bool with_sf = false;
    int num_max_tokens_per_rank = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    int num_ring_tokens = 0;
    int num_sf_ring_tokens = 0;
};

static SymmBufferLayoutInfo build_symm_buffer_layout(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu");

    // Pool capacity must fit at least one full wave (one expert per wave) and aligned to block size
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto [num_min_ring_tokens, num_max_ring_tokens] =
        get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts_per_rank, num_topk, num_ranks);
    DG_HOST_ASSERT(num_ring_tokens % layout::kLCMCandidateBlockM == 0);
    DG_HOST_ASSERT(num_min_ring_tokens <= num_ring_tokens and num_ring_tokens <= num_max_ring_tokens);

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);
    const auto with_sf = is_mma_with_sf(mma_kind);

    // Workspace
    const auto workspace = layout::Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_ring_tokens);

    // Layouts
    const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
    const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0);
    const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        input_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        input_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Padded SF pool tokens
    int num_sf_ring_tokens = 0;
    for (int block_m: layout::kCandidateBlockM) {
        num_sf_ring_tokens = std::max(
            num_sf_ring_tokens,
            layout::get_num_sf_ring_tokens(num_ring_tokens, block_m)
        );
    }

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        input_token_layout, 1, num_ring_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        input_sf_layout, 1, num_sf_ring_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_ring_tokens,
        with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        intermediate_token_layout, 1, num_ring_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        intermediate_sf_layout, 1, num_sf_ring_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    SymmBufferLayoutInfo layout_info;
    layout_info.num_bytes = reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr());
    layout_info.input_token_base = reinterpret_cast<int64_t>(input_token_buffer.base);
    layout_info.input_sf_base = reinterpret_cast<int64_t>(input_sf_buffer.base);
    layout_info.input_topk_idx_base = reinterpret_cast<int64_t>(input_topk_idx_buffer.base);
    layout_info.input_topk_weights_base = reinterpret_cast<int64_t>(input_topk_weights_buffer.base);
    layout_info.l1_token_base = reinterpret_cast<int64_t>(l1_token_buffer.base);
    layout_info.l1_sf_base = reinterpret_cast<int64_t>(l1_sf_buffer.base);
    layout_info.l2_token_base = reinterpret_cast<int64_t>(l2_token_buffer.base);
    layout_info.l2_sf_base = reinterpret_cast<int64_t>(l2_sf_buffer.base);
    layout_info.with_sf = with_sf;
    layout_info.num_max_tokens_per_rank = num_max_tokens_per_rank;
    layout_info.num_topk = num_topk;
    layout_info.hidden = hidden;
    layout_info.intermediate_hidden = intermediate_hidden;
    layout_info.num_ring_tokens = num_ring_tokens;
    layout_info.num_sf_ring_tokens = num_sf_ring_tokens;
    return layout_info;
}

static int64_t get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens) {
    return build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_ring_tokens).num_bytes;
}

using SymmBufferSlice = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                   at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

static SymmBufferSlice slice_symm_buffer_from_layout(
    const torch::Tensor& buffer,
    const SymmBufferLayoutInfo& layout_info) {
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto x = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_token_base),
        {layout_info.num_max_tokens_per_rank, layout_info.hidden},
        torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
    auto x_sf = layout_info.with_sf ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_sf_base),
        {layout_info.num_max_tokens_per_rank, layout_info.hidden / 128},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    auto topk_idx = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_idx_base),
        {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
        torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
    auto topk_weights = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_weights_base),
        {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
    auto l1_acts = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l1_token_base),
        {layout_info.num_ring_tokens, layout_info.hidden},
        torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
    auto l1_acts_sf = layout_info.with_sf ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l1_sf_base),
        {layout_info.num_sf_ring_tokens, layout_info.hidden / 128},
        {1, layout_info.num_sf_ring_tokens},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    auto l2_acts = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l2_token_base),
        {layout_info.num_ring_tokens, layout_info.intermediate_hidden},
        torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
    auto l2_acts_sf = layout_info.with_sf ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l2_sf_base),
        {layout_info.num_sf_ring_tokens, layout_info.intermediate_hidden / 128},
        {1, layout_info.num_sf_ring_tokens},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
}

static SymmBufferSlice slice_symm_buffer_for_mega_moe(
    const torch::Tensor& buffer,
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens) {
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_ring_tokens);
    return slice_symm_buffer_from_layout(buffer, layout_info);
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes and slice views from one shared layout plan.
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_ring_tokens);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes and slice views from one shared layout plan.
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_ring_tokens);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, _x_sf, topk_idx, topk_weights, l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts, 
                            l1_weights, l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

}  // namespace deep_gemm::mega

namespace deep_gemm::torch_registration {

static int64_t get_token_alignment_for_mega_moe() {
    return static_cast<int64_t>(mega::get_token_alignment_for_mega_moe());
}

static std::tuple<int64_t, int64_t> get_ring_limit_for_mega_moe(
    const int64_t& num_max_tokens_per_rank, const int64_t& num_experts_per_rank,
    const int64_t& num_topk, const int64_t& num_ranks) {
    const auto [a, b] = mega::get_ring_limit_for_mega_moe(
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(num_ranks));
    return {static_cast<int64_t>(a), static_cast<int64_t>(b)};
}

static int64_t get_symm_buffer_size_for_mega_moe(
    const int64_t& num_ranks, const int64_t& num_experts,
    const int64_t& num_max_tokens_per_rank, const int64_t& num_topk,
    const int64_t& hidden, const int64_t& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int64_t& num_ring_tokens) {
    return mega::get_symm_buffer_size_for_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        mma_type, activation, static_cast<int>(num_ring_tokens));
}

static mega::SymmBufferSlice slice_symm_buffer_for_mega_moe(
    const torch::Tensor& buffer,
    const int64_t& num_ranks, const int64_t& num_experts,
    const int64_t& num_max_tokens_per_rank, const int64_t& num_topk,
    const int64_t& hidden, const int64_t& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int64_t& num_ring_tokens) {
    return mega::slice_symm_buffer_for_mega_moe(
        buffer,
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        mma_type, activation, static_cast<int>(num_ring_tokens));
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const c10::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const c10::List<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts, const int64_t& num_topk,
    const c10::List<int64_t>& recipe,
    const std::string& activation,
    const c10::optional<double>& activation_clamp,
    const bool& fast_math,
    const int64_t& num_ring_tokens) {
    mega::fp8_fp4_mega_moe(
        y,
        std::make_tuple(l1_weights, l1_weights_sf),
        std::make_tuple(l2_weights, l2_weights_sf),
        cumulative_local_expert_recv_stats,
        sym_buffer,
        std::vector<int64_t>(sym_buffer_ptrs.begin(), sym_buffer_ptrs.end()),
        static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        list_to_tuple3(recipe),
        activation,
        activation_clamp.has_value()
            ? std::make_optional(static_cast<float>(activation_clamp.value()))
            : std::nullopt,
        fast_math,
        static_cast<int>(num_ring_tokens));
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const c10::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const c10::List<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts, const int64_t& num_topk,
    const std::string& activation,
    const c10::optional<double>& activation_clamp,
    const bool& fast_math,
    const int64_t& num_ring_tokens) {
    mega::bf16_mega_moe(
        y, l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        std::vector<int64_t>(sym_buffer_ptrs.begin(), sym_buffer_ptrs.end()),
        static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        activation,
        activation_clamp.has_value()
            ? std::make_optional(static_cast<float>(activation_clamp.value()))
            : std::nullopt,
        fast_math,
        static_cast<int>(num_ring_tokens));
}

}  // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def(
        "get_token_alignment_for_mega_moe() -> int",
        DEEP_GEMM_IMPL(get_token_alignment_for_mega_moe));
    m.def(
        "get_ring_limit_for_mega_moe(int num_max_tokens_per_rank, int num_experts_per_rank, int num_topk, int num_ranks) -> (int, int)",
        DEEP_GEMM_IMPL(get_ring_limit_for_mega_moe));
    m.def(
        "get_symm_buffer_size_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, str mma_type, str activation, int num_ring_tokens) -> int",
        DEEP_GEMM_IMPL(get_symm_buffer_size_for_mega_moe));
    m.def(
        "slice_symm_buffer_for_mega_moe(Tensor buffer, int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, str mma_type, str activation, int num_ring_tokens) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "fp8_fp4_mega_moe(Tensor(y!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, Tensor l2_weights_sf, Tensor? cumulative_local_expert_recv_stats, Tensor(sym_buffer!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, int[] recipe, str activation, float? activation_clamp, bool fast_math, int num_ring_tokens) -> ()");
    m.def(
        "bf16_mega_moe(Tensor(y!) y, Tensor l1_weights, Tensor l2_weights, Tensor? cumulative_local_expert_recv_stats, Tensor(sym_buffer!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, str activation, float? activation_clamp, bool fast_math, int num_ring_tokens) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

#if DG_TENSORMAP_COMPATIBLE
    m.impl("slice_symm_buffer_for_mega_moe", TORCH_FN(slice_symm_buffer_for_mega_moe));
    m.impl("fp8_fp4_mega_moe", TORCH_FN(fp8_fp4_mega_moe));
    m.impl("bf16_mega_moe", TORCH_FN(bf16_mega_moe));
#endif
}
