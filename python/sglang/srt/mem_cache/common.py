from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_hip, support_triton
from sglang.srt.utils.common import ceil_align

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # The page is guaranteed to be full except the last page.
    if page_size == 1:
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    # Under CP, the caller passes mirror coords; the triton kernel treats them
    # as ordinary LOCAL positions and writes `out_cache_loc` at LOCAL
    # `[prefix_lens, seq_lens)` per request.
    if support_triton(get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    attn_backend = get_global_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def _get_last_loc_safe_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result_i32,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_DTYPE_IS_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    if PREFIX_DTYPE_IS_I64:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    else:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices.to(tl.int64) * req_to_token_stride + (
            prefix_lens.to(tl.int64) - 1
        )

    token_mask = mask & (prefix_lens > 0)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)
    # Result stays int32 (req_to_token dtype); caller promotes after return.
    tl.store(result_i32 + offset, tokens, mask=mask)


def get_last_loc_triton_safe(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Fused `last_loc` Triton kernel whose in-kernel result buffer is int32
    (the dtype of req_to_token). The consumer-dtype promotion happens in
    torch after the kernel returns, so Triton never issues a mixed-width
    store — avoiding the HIP int32->int64 store bug hit by the legacy kernel.
    """
    num_tokens = prefix_lens_tensor.shape[0]
    BLOCK_SIZE = 256
    result_i32 = torch.empty(
        num_tokens, dtype=torch.int32, device=prefix_lens_tensor.device
    )
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
    _get_last_loc_safe_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result_i32,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        PREFIX_DTYPE_IS_I64=(prefix_lens_tensor.dtype == torch.int64),
    )
    return result_i32.to(prefix_lens_tensor.dtype)


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    from sglang.srt.layers.utils.cp_utils import is_prefill_cp_round_robin_split

    cp_active = is_prefill_cp_round_robin_split()
    if cp_active:
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_rank,
            get_attention_cp_size,
        )

    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    # Under CP, `r.prefix_indices` holds this rank's LOCAL cached slots
    # (len = global_prefix / cp_size); CPRadixCache produces them.
    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu_global = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    seq_lens_cpu_global = batch.seq_lens_cpu

    if cp_active:
        cp_size = get_attention_cp_size()
        cp_rank = get_attention_cp_rank()
        # Mirror coords: ⌈global / cp⌉ — the heaviest rank's local count.
        prefix_lens_cpu = (prefix_lens_cpu_global + cp_size - 1) // cp_size
        seq_lens_cpu = (seq_lens_cpu_global + cp_size - 1) // cp_size
        extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu
        alloc_extend_num_tokens = int(extend_lens_cpu.sum().item())
    else:
        prefix_lens_cpu = prefix_lens_cpu_global
        seq_lens_cpu = seq_lens_cpu_global
        extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
        alloc_extend_num_tokens = batch.extend_num_tokens

    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)
    seq_lens_device = seq_lens_cpu.to(batch.device, non_blocking=True) if cp_active else batch.seq_lens

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    # CP: `tree_cache.page_size` is the larger cp-aligned KEY page size, so read
    # the allocator's page size instead.
    pool_page_size = (
        batch.token_to_kv_pool_allocator.page_size
        if cp_active
        else batch.tree_cache.page_size
    )
    if pool_page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, alloc_extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens_device,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=alloc_extend_num_tokens,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        seq_lens_device,
        seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    if not cp_active:
        return out_cache_loc, req_pool_indices_device, req_pool_indices

    # Under CP, mirror-alloc wrote a superset of this rank's REAL extend
    # range into r2t at LOCAL positions [mirror_prefix, mirror_seq). Extract
    # this rank's REAL new slots — `n_local_real - mirror_prefix` per request,
    # starting at LOCAL offset `mirror_prefix` (= P/cp, this rank's real
    # cached count since the cached prefix is cp-aligned) — for set_kv_buffer.
    n_local_real_cpu = (
        (seq_lens_cpu_global - cp_rank + cp_size - 1) // cp_size
    ).clamp_min(0)
    real_extend_lens_cpu = (n_local_real_cpu - prefix_lens_cpu).clamp_min(0).to(
        torch.int64
    )
    assert int(real_extend_lens_cpu.sum().item()) == batch.extend_num_tokens, (
        f"CP real_extend mismatch: r2t-derived "
        f"{int(real_extend_lens_cpu.sum().item())} vs "
        f"batch.extend_num_tokens={batch.extend_num_tokens}"
    )
    real_pieces = []
    r2t = batch.req_to_token_pool.req_to_token
    for r in range(len(req_pool_indices)):
        n_real_r = int(real_extend_lens_cpu[r].item())
        pre_r = int(prefix_lens_cpu[r].item())
        if n_real_r > 0:
            real_pieces.append(
                r2t[req_pool_indices[r], pre_r : pre_r + n_real_r]
            )
    if real_pieces:
        out_cache_loc_real = torch.cat(real_pieces).to(torch.int64)
    else:
        out_cache_loc_real = torch.empty(0, dtype=torch.int64, device=batch.device)
    return out_cache_loc_real, req_pool_indices_device, req_pool_indices


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations

    Under CP, applies the mirror-alloc rule; the returned `[n_owned]` slots
    are the OWNER's actual slots for its real K/V write (NOT the
    mirror-alloc'd superset).
    """
    from sglang.srt.layers.utils.cp_utils import is_prefill_cp_round_robin_split

    cp_active = is_prefill_cp_round_robin_split()
    if cp_active:
        assert not batch.model_config.is_encoder_decoder, (
            "Encoder-decoder + CP is not supported."
        )
        owned_idx = batch.cp_owned_indices
        assert owned_idx is not None, (
            "alloc_for_decode under CP expected batch.cp_owned_indices to be set."
        )
        assert token_per_req == 1, "token_per_req > 1 + CP decode not implemented."

    batch.maybe_evict_swa()

    bs = batch.seq_lens.shape[0]

    if cp_active:
        from sglang.srt.layers.dp_attention import get_attention_cp_size
        cp_size = get_attention_cp_size()
        page_size = batch.token_to_kv_pool_allocator.page_size
        device = batch.seq_lens.device

        # Mirror coords: pre/post-increment ⌈T/cp⌉; identical on every rank.
        # `batch.seq_lens` here is PRE-increment (the new token's position).
        mirror_pre = (batch.seq_lens + cp_size - 1) // cp_size
        mirror_post = (batch.seq_lens + 1 + cp_size - 1) // cp_size
        mirror_pre_cpu = (batch.seq_lens_cpu + cp_size - 1) // cp_size
        mirror_post_cpu = (batch.seq_lens_cpu + 1 + cp_size - 1) // cp_size

        # Requests where mirror count grew this step need a fresh slot/page;
        # the rest reuse a slot pre-allocated on an earlier advance step.
        advance_mask_cpu = mirror_post_cpu > mirror_pre_cpu
        advance_idx_cpu = advance_mask_cpu.nonzero(as_tuple=True)[0]
        n_advance = int(advance_idx_cpu.numel())

        if n_advance > 0:
            advance_idx = advance_idx_cpu.to(device, non_blocking=True)
            req_pool_advance = batch.req_pool_indices.index_select(0, advance_idx)
            mirror_post_advance = mirror_post.index_select(0, advance_idx)
            mirror_pre_advance = mirror_pre.index_select(0, advance_idx)
            mirror_post_advance_cpu = mirror_post_cpu[advance_idx_cpu]

            if page_size == 1:
                new_slots = alloc_token_slots(
                    batch.tree_cache, n_advance * token_per_req
                )
            else:
                last_loc = batch.req_to_token_pool.req_to_token[
                    req_pool_advance,
                    (mirror_pre_advance - 1).clamp_min(0),
                ]
                # Worst-case 1 new page per advance request; matches non-CP.
                evict_from_tree_cache(batch.tree_cache, n_advance * page_size)
                new_slots = batch.token_to_kv_pool_allocator.alloc_decode(
                    seq_lens=mirror_post_advance,
                    seq_lens_cpu=mirror_post_advance_cpu,
                    last_loc=last_loc,
                )
                if new_slots is None:
                    raise RuntimeError(
                        "Decode OOM under CP page>1 mirror-alloc."
                    )

            # Every rank writes the new slot at `mirror_post - 1` so
            # subsequent owner-steps for the same LOCAL position can
            # look it up.
            batch.req_to_token_pool.write(
                (req_pool_advance, mirror_post_advance - 1),
                new_slots.to(torch.int32),
            )

        # Owner's slot per owned request = r2t[req_pool, mirror_post - 1]
        # (which equals owner's real LOCAL pos under round-robin).
        n_owned = int(owned_idx.numel())
        if n_owned == 0:
            return torch.zeros(0, dtype=torch.int64, device=device)
        owned_req_pool = batch.req_pool_indices.index_select(0, owned_idx)
        owned_local_pos = mirror_post.index_select(0, owned_idx) - 1
        out_cache_loc = batch.req_to_token_pool.req_to_token[
            owned_req_pool, owned_local_pos
        ]
        return out_cache_loc.to(torch.int64)

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        seq_lens_next = batch.seq_lens + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # and bookkeeping flag sync internally, then sets req_pool_idx = None.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
