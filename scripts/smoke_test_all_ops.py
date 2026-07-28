#!/usr/bin/env python3
"""Smoke-test every deep_gemm op after the TORCH_LIBRARY migration.

Checks registration, one minimal CUDA call per exported op, and (via TORCH
schema ``alias_info.is_write``) that immutable tensor args stay bitwise-unchanged.

Usage:
  python scripts/smoke_test_all_ops.py
  python scripts/smoke_test_all_ops.py --registration-only
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Callable

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.join(_REPO_ROOT, 'tests')

_C_ONLY_OPS = frozenset({
    'init',
    'get_token_alignment_for_mega_moe', 'get_block_m_for_mega_moe',
    'get_symm_buffer_size_for_mega_moe', '_slice_symm_buffer_for_mega_moe',
})

# Keep in sync with TORCH_LIBRARY m.def inventory under csrc/apis/.
ALL_OPS: dict[str, tuple[str, ...]] = {
    'runtime': (
        'init', 'set_num_sms', 'get_num_sms', 'set_tc_util', 'get_tc_util',
        'set_pdl', 'get_pdl', 'set_ignore_compile_dims', 'set_block_size_multiple_of',
    ),
    'layout': (
        'set_mk_alignment_for_contiguous_layout',
        'get_mk_alignment_for_contiguous_layout',
        'get_theoretical_mk_alignment_for_contiguous_layout',
        'transform_sf_into_required_layout', 'get_tma_aligned_size',
        'get_mn_major_tma_aligned_tensor',
        'get_mn_major_tma_aligned_packed_ue8m0_tensor',
        'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor',
    ),
    'gemm': (
        'cublaslt_gemm_nt', 'cublaslt_gemm_nn', 'cublaslt_gemm_tn', 'cublaslt_gemm_tt',
        'fp8_fp4_gemm_nt', 'fp8_fp4_gemm_nn', 'fp8_fp4_gemm_tn', 'fp8_fp4_gemm_tt',
        'm_grouped_fp8_fp4_gemm_nt_contiguous', 'm_grouped_fp8_fp4_gemm_nn_contiguous',
        'm_grouped_fp8_fp4_gemm_nt_masked',
        'k_grouped_fp8_gemm_nt_contiguous', 'k_grouped_fp8_gemm_tn_contiguous',
        'bf16_gemm_nt', 'bf16_gemm_nn', 'bf16_gemm_tn', 'bf16_gemm_tt',
        'm_grouped_bf16_gemm_nt_contiguous', 'm_grouped_bf16_gemm_nn_contiguous',
        'm_grouped_bf16_gemm_nt_masked', 'k_grouped_bf16_gemm_tn_contiguous',
    ),
    'einsum': ('einsum', 'fp8_einsum'),
    'attention': (
        'fp8_gemm_nt_skip_head_mid',
        'fp8_fp4_mqa_logits', 'get_paged_mqa_logits_metadata', 'fp8_fp4_paged_mqa_logits',
        'fp8_mqa_logits', 'fp8_paged_mqa_logits',
    ),
    'hyperconnection': ('tf32_hc_prenorm_gemm',),
    'mega': (
        'get_token_alignment_for_mega_moe', 'get_block_m_for_mega_moe',
        'get_symm_buffer_size_for_mega_moe', '_slice_symm_buffer_for_mega_moe',
        'fp8_fp4_mega_moe', 'bf16_mega_moe',
    ),
}

_LAYOUTS = ('nt', 'nn', 'tn', 'tt')


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')


def _api(name: str):
    import deep_gemm
    from deep_gemm import _C
    return getattr(deep_gemm, name, None) or getattr(_C, name)


def _skip(reason: str) -> None:
    raise RuntimeError(f'SKIP: {reason}')


def _need_sm100(reason: str) -> None:
    from deep_gemm.testing import get_arch_major
    if get_arch_major() < 10:
        _skip(reason)


def _import_generators():
    if _TESTS_DIR not in sys.path:
        sys.path.insert(0, _TESTS_DIR)
    import generators as g  # type: ignore
    return g


# ---- schema immutable-tensor guards ---------------------------------------

def _is_write(arg) -> bool:
    alias = getattr(arg, 'alias_info', None)
    return bool(alias is not None and getattr(alias, 'is_write', False))


def _bitwise_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    return bool(torch.equal(
        a.detach().contiguous().view(torch.uint8),
        b.detach().contiguous().view(torch.uint8),
    ))


def _bind_args(schema, args: tuple, kwargs: dict) -> dict[str, Any]:
    kwargs, positional, bound = dict(kwargs), list(args), {}
    for sa in schema.arguments:
        if sa.name in kwargs:
            bound[sa.name] = kwargs.pop(sa.name)
        elif positional:
            bound[sa.name] = positional.pop(0)
    if positional or kwargs:
        raise TypeError(f'bad args for {schema}: leftover={positional} kwargs={kwargs}')
    return bound


def _snapshot_immutables(op_name: str, op, args: tuple, kwargs: dict):
    bound = _bind_args(op._schema, args, kwargs)
    write_ptrs = {
        bound[sa.name].data_ptr()
        for sa in op._schema.arguments
        if _is_write(sa) and isinstance(bound.get(sa.name), torch.Tensor)
    }
    snaps = []
    for sa in op._schema.arguments:
        if _is_write(sa):
            continue
        val = bound.get(sa.name)
        if isinstance(val, torch.Tensor) and val.data_ptr() not in write_ptrs:
            snaps.append((sa.name, val, val.detach().clone()))
    return snaps


def _check_immutables(op_name: str, snaps) -> None:
    for name, tensor, clone in snaps:
        if not _bitwise_eq(tensor, clone):
            raise AssertionError(
                f'{op_name}: schema-immutable tensor {name!r} changed bitwise '
                f'(dtype={tensor.dtype}, shape={tuple(tensor.shape)})'
            )


class _GuardingOpPacket:
    def __init__(self, name: str, packet):
        self._name, self._packet = name, packet

    def _guarded(self, op, args, kwargs):
        snaps = _snapshot_immutables(self._name, op, args, kwargs)
        result = op(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _check_immutables(self._name, snaps)
        return result

    def __call__(self, *args, **kwargs):
        return self._guarded(self._packet.default, args, kwargs)

    def __getattr__(self, item):
        attr = getattr(self._packet, item)
        if item != 'default':
            return attr
        return lambda *a, **k: self._guarded(self._packet.default, a, k)


class _GuardingTorchOps:
    def __init__(self, real_ns):
        object.__setattr__(self, '_real_ns', real_ns)
        object.__setattr__(self, '_cache', {})

    def __getattr__(self, name: str):
        cache = object.__getattribute__(self, '_cache')
        if name in cache:
            return cache[name]
        attr = getattr(object.__getattribute__(self, '_real_ns'), name)
        if hasattr(attr, 'default'):
            cache[name] = wrapped = _GuardingOpPacket(name, attr)
            return wrapped
        return attr


def install_immutable_tensor_guards(deep_gemm, _C) -> None:
    proxy = _GuardingTorchOps(torch.ops.deep_gemm)
    _C._torch_ops = proxy
    for name in set(getattr(_C, '__all__', [])) | {
        'init', 'set_num_sms', 'get_num_sms', 'set_tc_util', 'get_tc_util',
        'set_pdl', 'get_pdl', 'set_ignore_compile_dims',
        'get_mk_alignment_for_contiguous_layout',
        'get_theoretical_mk_alignment_for_contiguous_layout',
        'cublaslt_gemm_nt', 'cublaslt_gemm_nn', 'cublaslt_gemm_tn', 'cublaslt_gemm_tt',
    }:
        if hasattr(_C, name) and type(getattr(_C, name)).__name__ == 'OpOverloadPacket':
            setattr(_C, name, getattr(proxy, name))
    for ops in ALL_OPS.values():
        for name in ops:
            if hasattr(deep_gemm, name) and hasattr(_C, name):
                setattr(deep_gemm, name, getattr(_C, name))


# ---- smoke callers --------------------------------------------------------

def _bf16_abcd(layout: str):
    _require_cuda()
    m = n = k = 128
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    d = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    return {
        'nt': (a, b, d),
        'nn': (a, b.T.contiguous(), d),
        'tn': (a.T.contiguous(), b.T.contiguous(), d),
        'tt': (a.T.contiguous(), b, d),
    }[layout]


def _fp8_ab(layout: str, g):
    from deep_gemm.testing import get_arch_major
    kt = g.KernelType.Kernel1D2D if get_arch_major() == 9 else g.KernelType.Kernel1D1D
    use_ue8m0 = g.get_ue8m0_usage(kt)
    a, b, _, d, _ = g.generate_normal(
        128, 128, 128, g.MajorTypeAB.KMajor, g.MajorTypeAB.KMajor,
        False, torch.bfloat16, kt, use_ue8m0=use_ue8m0,
    )
    if layout == 'nn':
        b = (b[0].T, b[1].T)
    elif layout == 'tn':
        a, b = (a[0].T, a[1].T), (b[0].T, b[1].T)
    elif layout == 'tt':
        a = (a[0].T, a[1].T)
    return a, b, d, use_ue8m0


def _set_mk_align(expected_m=None) -> None:
    align = _api('get_theoretical_mk_alignment_for_contiguous_layout')(expected_m)
    _api('set_mk_alignment_for_contiguous_layout')(align)


def _pack_fp8_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    num_blocks, block_kv, _, head_dim = kv_cache.shape
    x_amax = kv_cache.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (kv_cache * (1.0 / sf)).to(torch.float8_e4m3fn)
    out = torch.empty((num_blocks, block_kv * (head_dim + 4)), device='cuda', dtype=torch.uint8)
    out[:, : block_kv * head_dim] = x_scaled.view(num_blocks, block_kv * head_dim).view(torch.uint8)
    out[:, block_kv * head_dim :] = sf.view(num_blocks, block_kv).view(torch.uint8)
    return out.view(num_blocks, block_kv, 1, head_dim + 4)


def _mqa_inputs(seq_len=64, seq_len_kv=64, num_heads=32, head_dim=128):
    from deep_gemm.utils import per_custom_dims_cast_to_fp8
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
    ks = torch.zeros(seq_len, dtype=torch.int32, device='cuda')
    ke = torch.arange(seq_len, dtype=torch.int32, device='cuda') + (seq_len_kv - seq_len)
    return q.to(torch.float8_e4m3fn), per_custom_dims_cast_to_fp8(kv, (0,), False), weights, ks, ke


def _paged_inputs(batch_size=4, next_n=1, num_heads=32, head_dim=128, block_kv=64, avg_kv=128):
    context_lens = torch.full((batch_size, next_n), avg_kv, device='cuda', dtype=torch.int32)
    q = torch.randn(batch_size, next_n, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(batch_size * next_n, num_heads, device='cuda', dtype=torch.float32)
    num_blocks = batch_size * ((avg_kv + block_kv - 1) // block_kv)
    kv = torch.randn(num_blocks, block_kv, 1, head_dim, device='cuda', dtype=torch.bfloat16)
    kv_fp8 = _pack_fp8_kv_cache(kv)
    block_table = torch.arange(num_blocks, device='cuda', dtype=torch.int32).view(batch_size, -1)
    schedule_meta = _api('get_paged_mqa_logits_metadata')(
        context_lens, block_kv=block_kv, num_sms=_api('get_num_sms')(),
    )
    return q, kv_fp8, weights, context_lens, block_table, schedule_meta, avg_kv


def _mega_common_args():
    from deep_gemm import _C
    tokens = int(_C.get_token_alignment_for_mega_moe())
    return _C, tokens


def build_op_calls() -> dict[str, Callable[[], None]]:
    g = _import_generators()
    calls: dict[str, Callable[[], None]] = {}

    # runtime
    def _init():
        import deep_gemm
        from deep_gemm import _C
        _C.init(os.path.dirname(deep_gemm.__file__),
                os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '')

    calls['init'] = _init
    calls['get_num_sms'] = lambda: _api('get_num_sms')()
    calls['set_num_sms'] = lambda: _api('set_num_sms')(max(1, _api('get_num_sms')()))
    calls['get_tc_util'] = lambda: _api('get_tc_util')()
    calls['set_tc_util'] = lambda: _api('set_tc_util')(_api('get_tc_util')())
    calls['get_pdl'] = lambda: _api('get_pdl')()
    calls['set_pdl'] = lambda: _api('set_pdl')(_api('get_pdl')())
    calls['set_ignore_compile_dims'] = lambda: _api('set_ignore_compile_dims')(False)
    calls['set_block_size_multiple_of'] = lambda: _api('set_block_size_multiple_of')(1)

    # layout
    calls['get_mk_alignment_for_contiguous_layout'] = (
        lambda: _api('get_mk_alignment_for_contiguous_layout')())
    calls['set_mk_alignment_for_contiguous_layout'] = (
        lambda: _api('set_mk_alignment_for_contiguous_layout')(
            _api('get_mk_alignment_for_contiguous_layout')()))
    calls['get_theoretical_mk_alignment_for_contiguous_layout'] = lambda: (
        _api('get_theoretical_mk_alignment_for_contiguous_layout')(),
        _api('get_theoretical_mk_alignment_for_contiguous_layout')(None),
    )
    calls['get_tma_aligned_size'] = lambda: _api('get_tma_aligned_size')(128, 4)

    def _sf_tensor(use_ue8m0: bool, mn=64, k=256):
        from deep_gemm.utils import per_token_cast_to_fp8
        _require_cuda()
        _, sf = per_token_cast_to_fp8(
            torch.randn(mn, k, device='cuda', dtype=torch.bfloat16), use_ue8m0=use_ue8m0)
        return sf

    calls['transform_sf_into_required_layout'] = lambda: _api(
        'transform_sf_into_required_layout')(_sf_tensor(False, 128, 256), 128, 256, (1, 128))
    calls['get_mn_major_tma_aligned_tensor'] = lambda: _api(
        'get_mn_major_tma_aligned_tensor')(_sf_tensor(False))

    def _packed_ue8m0():
        _need_sm100('packed ue8m0 layout smoke needs SM100+')
        _api('get_mn_major_tma_aligned_packed_ue8m0_tensor')(_sf_tensor(True))

    def _k_grouped_packed():
        _need_sm100('k-grouped packed ue8m0 smoke needs SM100+')
        from deep_gemm.utils import per_channel_cast_to_fp8
        ks, gran_k = [128, 128], 128
        x = torch.randn((sum(ks), 64), dtype=torch.bfloat16, device='cuda')
        _, sf = per_channel_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)
        _api('get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor')(
            sf, torch.tensor(ks, dtype=torch.int32, device='cuda'), ks, gran_k, gran_k)

    calls['get_mn_major_tma_aligned_packed_ue8m0_tensor'] = _packed_ue8m0
    calls['get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor'] = _k_grouped_packed

    # gemm — layout variants
    for layout in _LAYOUTS:
        def _cublas(layout=layout):
            a, b, d = _bf16_abcd(layout)
            _api(f'cublaslt_gemm_{layout}')(a, b, d)

        def _bf16(layout=layout):
            a, b, d = _bf16_abcd(layout)
            _api(f'bf16_gemm_{layout}')(a, b, d)

        def _fp8(layout=layout):
            a, b, d, use_ue8m0 = _fp8_ab(layout, g)
            _api(f'fp8_fp4_gemm_{layout}')(a, b, d, disable_ue8m0_cast=not use_ue8m0)

        calls[f'cublaslt_gemm_{layout}'] = _cublas
        calls[f'bf16_gemm_{layout}'] = _bf16
        calls[f'fp8_fp4_gemm_{layout}'] = _fp8

    for layout in ('nt', 'nn'):
        def _m_bf16(layout=layout):
            _set_mk_align()
            major_b = g.MajorTypeAB.KMajor if layout[1] == 't' else g.MajorTypeAB.MNMajor
            _, a, b, gl, d, _ = g.generate_m_grouped_contiguous(
                2, 128, 128, 128, g.MajorTypeAB.KMajor, major_b, use_bf16=True)
            if not major_b.is_k_major():
                b = b.mT.contiguous()
            _api(f'm_grouped_bf16_gemm_{layout}_contiguous')(a, b, d, gl)

        def _m_fp8(layout=layout):
            from deep_gemm.testing import get_arch_major
            kt = g.KernelType.Kernel1D2D if get_arch_major() == 9 else g.KernelType.Kernel1D1D
            use_ue8m0 = g.get_ue8m0_usage(kt)
            _set_mk_align()
            _, a, b, gl, d, _ = g.generate_m_grouped_contiguous(
                2, 128, 128, 128, g.MajorTypeAB.KMajor, g.MajorTypeAB.KMajor,
                use_ue8m0=use_ue8m0)
            if layout == 'nn':
                b = (b[0].mT, b[1].mT)
            _api(f'm_grouped_fp8_fp4_gemm_{layout}_contiguous')(
                a, b, d, gl, disable_ue8m0_cast=not use_ue8m0)

        calls[f'm_grouped_bf16_gemm_{layout}_contiguous'] = _m_bf16
        calls[f'm_grouped_fp8_fp4_gemm_{layout}_contiguous'] = _m_fp8

    def _m_bf16_masked():
        _set_mk_align(int(128 * 1.2))
        a, b, masked_m, _, d, _ = g.generate_m_grouped_masked(
            2, 256, 128, 128, 128, use_bf16=True)
        _api('m_grouped_bf16_gemm_nt_masked')(a, b, d, masked_m, 128)

    def _m_fp8_masked():
        from deep_gemm.testing import get_arch_major
        kt = g.KernelType.Kernel1D2D if get_arch_major() == 9 else g.KernelType.Kernel1D1D
        use_ue8m0 = g.get_ue8m0_usage(kt)
        _set_mk_align(int(128 * 1.2))
        a, b, masked_m, _, d, _ = g.generate_m_grouped_masked(
            2, 256, 128, 128, 128, use_ue8m0=use_ue8m0)
        _api('m_grouped_fp8_fp4_gemm_nt_masked')(
            a, b, d, masked_m, 128, disable_ue8m0_cast=not use_ue8m0)

    def _k_bf16():
        _, a, b, c, d, _, gl, ks = g.generate_k_grouped_contiguous(
            2, 128, 128, g.MajorTypeAB.MNMajor, g.MajorTypeAB.MNMajor, [128, 128],
            use_bf16=True)
        _api('k_grouped_bf16_gemm_tn_contiguous')(a, b, d, ks, gl, c=c)

    def _k_fp8(layout: str):
        from deep_gemm.testing import get_arch_major
        if layout == 'tn' and get_arch_major() < 10:
            _skip('k_grouped_fp8_gemm_tn_contiguous needs SM100+')
        use_ue8m0 = get_arch_major() >= 10
        _, a, b, c, d, _, gl, ks = g.generate_k_grouped_contiguous(
            2, 128, 128, g.MajorTypeAB.MNMajor, g.MajorTypeAB.MNMajor, [128, 128],
            use_ue8m0=use_ue8m0)
        if layout == 'nt':
            a, b = (a[0].T.contiguous(), a[1].T.contiguous()), (b[0].T.contiguous(), b[1].T.contiguous())
        _api(f'k_grouped_fp8_gemm_{layout}_contiguous')(a, b, d, ks, gl, c=c)

    calls['m_grouped_bf16_gemm_nt_masked'] = _m_bf16_masked
    calls['m_grouped_fp8_fp4_gemm_nt_masked'] = _m_fp8_masked
    calls['k_grouped_bf16_gemm_tn_contiguous'] = _k_bf16
    calls['k_grouped_fp8_gemm_tn_contiguous'] = lambda: _k_fp8('tn')
    calls['k_grouped_fp8_gemm_nt_contiguous'] = lambda: _k_fp8('nt')

    # einsum
    def _einsum():
        _require_cuda()
        a = torch.randn(2, 128, 128, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(2, 128, 128, device='cuda', dtype=torch.bfloat16)
        _api('einsum')('bmk,bnk->mn', a, b, torch.empty(128, 128, device='cuda', dtype=torch.bfloat16))

    def _fp8_einsum():
        from deep_gemm.utils.math import ceil_div, per_block_cast_to_fp8, per_token_cast_to_fp8
        from deep_gemm.testing import get_arch_major
        _require_cuda()
        use_ue8m0, b, h, r, d_dim = get_arch_major() >= 10, 4, 8, 256, 128
        x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
        y = torch.randn((h, d_dim, r), device='cuda', dtype=torch.bfloat16)
        xf = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=use_ue8m0)
        x_fp8 = xf[0].view(b, h, r), xf[1].view(b, h, ceil_div(r, 128))
        y_fp8 = (
            torch.empty_like(y, dtype=torch.float8_e4m3fn),
            torch.empty((h, ceil_div(d_dim, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float),
        )
        for i in range(h):
            y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
        _api('fp8_einsum')(
            'bhr,hdr->bhd', x_fp8, y_fp8,
            torch.empty((b, h, d_dim), device='cuda', dtype=torch.bfloat16))

    calls['einsum'], calls['fp8_einsum'] = _einsum, _fp8_einsum

    # attention
    def _skip_head_mid():
        from deep_gemm.testing import get_arch_major
        _require_cuda()
        head_splits = (64, 32, 64)
        left, mid, right = head_splits
        kt = g.KernelType.Kernel1D2D if get_arch_major() == 9 else g.KernelType.Kernel1D1D
        use_ue8m0 = g.get_ue8m0_usage(kt)
        m, n, k = 128, left + right, 128
        a, b, _, _, _ = g.generate_normal(
            m, n, k, g.MajorTypeAB.KMajor, g.MajorTypeAB.KMajor,
            False, torch.bfloat16, kt, use_ue8m0=use_ue8m0)
        d = torch.empty((m, (n // (left + right)) * (left + mid + right)),
                        device='cuda', dtype=torch.bfloat16)
        _api('fp8_gemm_nt_skip_head_mid')(
            a, b, d, list(head_splits), disable_ue8m0_cast=not use_ue8m0)

    def _mqa(name: str, q_as_pair: bool):
        _require_cuda()
        q, kv, weights, ks, ke = _mqa_inputs()
        q_in = (q, None) if q_as_pair else q
        _api(name)(q_in, kv, weights, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke)

    def _paged(name: str, q_as_pair: bool):
        _require_cuda()
        q, kv_fp8, weights, ctx, table, meta, avg_kv = _paged_inputs()
        q_in = (q.to(torch.float8_e4m3fn), None) if q_as_pair else q.to(torch.float8_e4m3fn)
        _api(name)(q_in, kv_fp8, weights, ctx, table, meta, avg_kv)

    calls['fp8_gemm_nt_skip_head_mid'] = _skip_head_mid
    calls['fp8_mqa_logits'] = lambda: _mqa('fp8_mqa_logits', False)
    calls['fp8_fp4_mqa_logits'] = lambda: _mqa('fp8_fp4_mqa_logits', True)
    calls['get_paged_mqa_logits_metadata'] = lambda: (
        _require_cuda(),
        _api('get_paged_mqa_logits_metadata')(
            torch.tensor([[32]], device='cuda', dtype=torch.int32),
            block_kv=64, num_sms=_api('get_num_sms')()),
    )
    calls['fp8_paged_mqa_logits'] = lambda: _paged('fp8_paged_mqa_logits', False)
    calls['fp8_fp4_paged_mqa_logits'] = lambda: _paged('fp8_fp4_paged_mqa_logits', True)

    # hyperconnection
    def _hc():
        _require_cuda()
        m, n, k = 16, 24, 256
        _api('tf32_hc_prenorm_gemm')(
            torch.randn(m, k, device='cuda', dtype=torch.bfloat16),
            torch.randn(n, k, device='cuda', dtype=torch.float32),
            torch.empty(m, n, device='cuda', dtype=torch.float32),
            torch.empty(m, device='cuda', dtype=torch.float32),
        )

    calls['tf32_hc_prenorm_gemm'] = _hc

    # mega
    def _token_align():
        from deep_gemm import _C
        _C.get_token_alignment_for_mega_moe()

    def _block_m():
        _C, tokens = _mega_common_args()
        _C.get_block_m_for_mega_moe(1, 8, tokens, tokens, 2, 'bf16xbf16')

    def _symm_size():
        _C, tokens = _mega_common_args()
        _C.get_symm_buffer_size_for_mega_moe(
            1, 8, tokens, 2, 256, 512, 'bf16xbf16', 'swiglu')

    def _slice_buf():
        _require_cuda()
        _C, tokens = _mega_common_args()
        num_bytes, layout_info = _C.get_symm_buffer_size_for_mega_moe(
            1, 8, tokens, 2, 256, 512, 'bf16xbf16', 'swiglu')
        _C._slice_symm_buffer_for_mega_moe(
            torch.empty(num_bytes, device='cuda', dtype=torch.uint8), layout_info)

    calls['get_token_alignment_for_mega_moe'] = _token_align
    calls['get_block_m_for_mega_moe'] = _block_m
    calls['get_symm_buffer_size_for_mega_moe'] = _symm_size
    calls['_slice_symm_buffer_for_mega_moe'] = _slice_buf
    for name in ('fp8_fp4_mega_moe', 'bf16_mega_moe'):
        calls[name] = (
            lambda n=name: _skip(
                f'{n} requires multi-rank symmetric memory (see tests/test_mega_moe.py)'))

    return calls


def check_registration(deep_gemm, ops: tuple[str, ...], family: str) -> list[str]:
    from deep_gemm import _C
    errors, exported, available = [], set(_C.__all__), False
    for op in ops:
        if op not in exported:
            continue
        available = True
        if not hasattr(torch.ops.deep_gemm, op):
            errors.append(f'{op}: in _C.__all__ but missing from torch.ops.deep_gemm')
        if op not in _C_ONLY_OPS and not hasattr(deep_gemm, op):
            errors.append(f'{op}: in _C.__all__ but missing from deep_gemm package namespace')
        if not hasattr(_C, op):
            errors.append(f'{op}: in _C.__all__ but missing from deep_gemm._C module')
    if not available:
        errors.append(f'{family}: no representative ops exported on _C (guard mismatch?)')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-only', action='store_true')
    args = parser.parse_args()

    print('==> import deep_gemm')
    try:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        import deep_gemm
        from deep_gemm import _C
    except Exception as exc:
        print(f'FAIL import: {exc}')
        traceback.print_exc()
        return 1
    print(f'    ok (__all__ exports {len(_C.__all__)} ops)')

    failed = False
    print('\n==> registration checks (per csrc/apis family)')
    for family, ops in ALL_OPS.items():
        errors = check_registration(deep_gemm, ops, family)
        if errors:
            failed = True
            print(f'FAIL {family}')
            for err in errors:
                print(f'    - {err}')
        else:
            print(f'ok   {family}')

    if args.registration_only:
        print('\n==> op calls skipped (--registration-only)')
    else:
        print('\n==> installing schema immutable-tensor guards on torch.ops')
        install_immutable_tensor_guards(deep_gemm, _C)
        print('    ok (non-write Tensor args must be bitwise-unchanged)')

        print('\n==> op calls (one minimal call per exported op)')
        op_calls, exported, seen = build_op_calls(), set(_C.__all__), 0
        for family, ops in ALL_OPS.items():
            print(f'-- {family}')
            for op in ops:
                if op not in exported:
                    print(f'skip {op}: not exported (compile guard)')
                    continue
                seen += 1
                fn = op_calls.get(op)
                if fn is None:
                    failed = True
                    print(f'FAIL {op}: no smoke caller defined')
                    continue
                try:
                    fn()
                    print(f'ok   {op}')
                except Exception as exc:
                    msg = str(exc)
                    if msg.startswith('SKIP:'):
                        print(f'skip {op}: {msg[len("SKIP:"):].strip()}')
                    else:
                        failed = True
                        print(f'FAIL {op}: {exc}')
                        traceback.print_exc()
        print(f'\n    attempted {seen} exported ops')

    print('\nRESULT: FAILED' if failed else '\nRESULT: PASSED')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
