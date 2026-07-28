# Stubs for module: _C

from typing import Any, Optional
import torch


def fp8_gemm_nt_skip_head_mid(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    head_splits: tuple[int, int, int],
    recipe: Optional[tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False
) -> None: ...


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, Optional[torch.Tensor]],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = True,
    max_seqlen_k: int = 0,
    logits_dtype: Any = None
) -> torch.Tensor: ...


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None
) -> torch.Tensor: ...


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, Optional[torch.Tensor]],
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    logits_dtype: Any = None,
    indices: Optional[torch.Tensor] = None
) -> torch.Tensor: ...


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = True,
    max_seqlen_k: int = 0
) -> torch.Tensor: ...


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    indices: Optional[torch.Tensor] = None
) -> torch.Tensor: ...


def einsum(
    expr: str,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    use_cublaslt: bool = False
) -> None: ...


def fp8_einsum(
    expr: str,
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: tuple[int, int, int] = (1, 128, 128)
) -> None: ...


def fp8_fp4_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False
) -> None: ...


def fp8_fp4_gemm_nn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False
) -> None: ...


def fp8_fp4_gemm_tn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False
) -> None: ...


def fp8_fp4_gemm_tt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False
) -> None: ...


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None
) -> None: ...


def m_grouped_fp8_fp4_gemm_nn_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = True
) -> None: ...


def m_grouped_fp8_fp4_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False
) -> None: ...


def k_grouped_fp8_gemm_tn_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: tuple[int, int, int] = (1, 1, 128),
    compiled_dims: str = "mn",
    use_psum_layout: bool = False
) -> None: ...


def k_grouped_fp8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: tuple[int, int, int] = (1, 1, 128),
    compiled_dims: str = "mn",
    use_psum_layout: bool = False
) -> None: ...


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk"
) -> None: ...


def bf16_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk"
) -> None: ...


def bf16_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn"
) -> None: ...


def bf16_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn"
) -> None: ...


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None
) -> None: ...


def m_grouped_bf16_gemm_nn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = True
) -> None: ...


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk"
) -> None: ...


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    use_psum_layout: bool = False
) -> None: ...


def cublaslt_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None: ...


def cublaslt_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None: ...


def cublaslt_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None: ...


def cublaslt_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None: ...


def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: Optional[int] = None
) -> None: ...


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: int,
    num_groups: Optional[int] = None,
    is_sfa: Optional[bool] = None,
    disable_ue8m0_cast: bool = False,
    psum_layout: Optional[torch.Tensor] = None
) -> torch.Tensor: ...


def get_tma_aligned_size(
    x: int,
    element_size: int
) -> int: ...


def get_mn_major_tma_aligned_tensor(
    sf: torch.Tensor
) -> torch.Tensor: ...


def get_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    psum_layout: Optional[torch.Tensor] = None
) -> torch.Tensor: ...


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    grouped_layout: torch.Tensor,
    ks_cpu: Optional[list[int]],
    gran_k: int,
    k_alignment: int,
    use_psum_layout: bool = False
) -> torch.Tensor: ...


def set_mk_alignment_for_contiguous_layout(*args, **kwargs) -> Any: ...


def get_mk_alignment_for_contiguous_layout(*args, **kwargs) -> Any: ...


def get_theoretical_mk_alignment_for_contiguous_layout(*args, **kwargs) -> Any: ...


def get_token_alignment_for_mega_moe() -> int: ...


def get_block_m_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    mma_type: str
) -> int: ...


def get_symm_buffer_size_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: str,
    activation: str,
    num_shared_experts: int
) -> tuple[int, torch.Tensor]: ...


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights_tuple: tuple[torch.Tensor, torch.Tensor],
    l2_weights_tuple: tuple[torch.Tensor, torch.Tensor],
    shared_l1_weights_tuple_opt: Optional[tuple[torch.Tensor, torch.Tensor]],
    shared_l2_weights_tuple_opt: Optional[tuple[torch.Tensor, torch.Tensor]],
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    sym_buffer: torch.Tensor,
    sym_buffer_ptrs: list[int],
    rank_idx: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    recipe: tuple[int, int, int],
    activation: str,
    activation_clamp_opt: Optional[float],
    fast_math: bool
) -> None: ...


def bf16_mega_moe(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    shared_l1_weights_opt: Optional[torch.Tensor],
    shared_l2_weights_opt: Optional[torch.Tensor],
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    sym_buffer: torch.Tensor,
    sym_buffer_ptrs: list[int],
    rank_idx: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    activation: str,
    activation_clamp_opt: Optional[float],
    fast_math: bool
) -> None: ...


def set_num_sms(*args, **kwargs) -> Any: ...


def get_num_sms(*args, **kwargs) -> Any: ...


def set_tc_util(*args, **kwargs) -> Any: ...


def get_tc_util(*args, **kwargs) -> Any: ...


def set_pdl(*args, **kwargs) -> Any: ...


def get_pdl(*args, **kwargs) -> Any: ...


def set_ignore_compile_dims(*args, **kwargs) -> Any: ...


def set_block_size_multiple_of(*args, **kwargs) -> Any: ...


def init(*args, **kwargs) -> Any: ...

