"""Context-parallel attention backend, prefill round-robin layout.

Rank `i` (of cp_size `c`) holds global tokens {i, i+c, i+2c, ...}. Prefill
is a c-stage pass-KV ring; decode is owner-only pass-Q ring. Both reuse
FA4's native causal / window-size fast paths via a 3-case mask dispatch
(prefill) or single-causal per-stage call (decode).

See `cp_design_notes.md` for the full architecture rationale (split point,
field semantics, ring math, sync-storm fix, mirror-alloc rule).

Inter-rank P2P routes through a single helper `_cp_p2p_exchange` (today:
NCCL `batch_isend_irecv`; future: drop-in custom intra-node CUDA-IPC kernel
or inter-node NVSHMEM kernel, both cudagraph-capturable without NCCL's
per-call kernel-launch overhead). The interface is the swap-in point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.interface import flash_attn_combine

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import (
    get_attention_cp_group,
    get_attention_cp_rank,
    get_attention_cp_size,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# Per-ForwardBatch metadata for the CP attention path. Built once and
# reused by every layer + the sampler.

@dataclass
class _CPDecodeMeta:
    """Layer-invariant decode metadata: fully GPU-resident, fixed-shape per
    capture bucket `(P, L)` (P = padded owned-request count, L = padded
    per-request local-KV length), so the whole build + pass-Q ring is
    CUDA-graph capturable. Every per-step-varying value is a GPU tensor; the
    only host values are the bucket constants `(P, L)`.

    Pad-and-run-all: every ring stage runs FA on all `P` query rows; inactive
    rows (padding, or a real req with `n_local==0` on this rank) get
    `seqused_k == 0` → FA emits `-inf` lse → `flash_attn_combine` drops them.

    KV is read in-place from the paged pool via FA4's `page_table` mode — no
    explicit gather buffer."""
    # Sampler fields, [B]-shaped (B = padded request count): the sampler
    # indexes gathered[owner_ranks, request_to_local_slot] over the real bs.
    owner_ranks: torch.Tensor             # [B]             int64
    request_to_local_slot: torch.Tensor   # [B]             int64
    # Per-stage paged-KV tables. `page_table[s, slot]` holds the KV-pool slot
    # indices for query row `slot` at ring stage `s` (row-per-query, padded
    # with 0 past `seqused_k`); FA4 reads KV from the pool directly via it.
    page_table: torch.Tensor              # [cp_size, P, L] int32
    seqused_k: torch.Tensor               # [cp_size, P]    int32 — per-row KV len
    cu_seqlens_q: torch.Tensor            # [P+1]           int32 — arange (Sq=1/row)
    # Bucket constants (Python ints, baked into the captured graph).
    P: int
    L: int


@dataclass
class _CPExtendMeta:
    n_local_per_rank: torch.Tensor        # [cp_size, bs]     int32
    cu_seqlens_k_per_rank: torch.Tensor   # [cp_size, bs+1]   int32
    local_q_lens: torch.Tensor            # [bs]              int32
    cu_seqlens_q: torch.Tensor            # [bs+1]            int32
    # Per-kv_owner slice (None ⇒ no slicing).
    slice_idx_per_owner: List[Optional[torch.Tensor]]
    slice_new_cu_per_owner: List[Optional[torch.Tensor]]
    # Shared shape with _CPDecodeMeta so the sampler is one code path.
    # For extend, local_slot = arange(bs).
    owner_ranks: torch.Tensor             # [bs]              int64
    request_to_local_slot: torch.Tensor   # [bs]              int64
    max_seqlen_q: int
    max_seqlen_k_per_owner: List[int]
    n_local_total_per_rank: List[int]     # for ring-shift recv_seqlen
    bs: int
    # Partial prefill (prefix cache): the cached prefix KV already lives in the
    # pool. `self_kv_flat_idx` gathers this rank's FULL local KV (cached ++ new)
    # for the ring; the slice formulas drop an extra `prefix_local` trailing KV
    # tokens per request so causal=True still yields the round-robin mask
    # (the causal diagonal shifts by P/cp_size).
    prefix_local_per_req: torch.Tensor    # [bs]   int32 — P_global // cp_size
    self_kv_flat_idx: Optional[torch.Tensor]  # [Σ n_local_self] int64, or None
    total_prefix_local: int               # Σ prefix_local; 0 ⇒ no gather


# ---------------------------------------------------------------------------
# Triton kernels for GPU-native decode metadata (mirror sglang's
# `create_flashinfer_kv_indices_triton` pattern at
# `srt/layers/attention/utils.py:16`).
# ---------------------------------------------------------------------------


@triton.jit
def _cp_build_meta_fused_kernel(
    seq_lens_ptr,                  # [bs] (int32 in sglang)
    owner_ranks_ptr,               # [bs]    int64 OUT
    n_local_ptr,                   # [bs]    int64 OUT
    cu_local_ptr,                  # [bs+1]  int32 OUT
    stage_req_ids_ptr,             # [cp_size, max_n_owned] int64 OUT (pre-filled −1)
    request_to_local_slot_ptr,     # [bs]    int64 OUT
    bs,
    max_n_owned,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    BS_BLOCK: tl.constexpr,
):
    """One program per request `pid`. Each program redundantly scans
    `seq_lens[0..bs)` to derive its prefix (cu_local[pid+1]) and its
    position within its owner bucket (request_to_local_slot[pid]) —
    O(bs) work per program, O(bs²) total but fully parallel.

    Replaces ~15 small torch ops (cast/sub/mod/floordiv/clamp/cumsum/sort/
    bincount/scatter) with one Triton launch. Saves ~180 μs at any bs
    (the OLD ops cost ~230 μs almost regardless of size — dominated by
    launch overhead, not data). Worth it because metadata-build runs
    once per decode step and the constant overhead is otherwise visible
    against an MoE-style 6 ms/step ceiling.

    Padding requests (`seq_len == 0`, present when the request buffer is
    padded to a fixed `B` for CUDA-graph capture) are skipped from
    `stage_req_ids` placement: otherwise all padding reqs collapse to owner
    `cp−1` and overflow that owner's `[max_n_owned]`-wide row. `n_local` is
    already 0 for them, so `cu_local` / `slot_flat` skip them naturally.

    Pre-conditions (host): `cu_local_ptr[0]` is 0 (we write [1..bs+1));
    `stage_req_ids_ptr` is pre-filled with −1.
    """
    pid = tl.program_id(axis=0)

    my_s = tl.load(seq_lens_ptr + pid).to(tl.int32)
    my_real = my_s > 0
    my_owner = (my_s - 1) % cp_size
    my_n_local = tl.maximum((my_s - cp_rank + cp_size - 1) // cp_size, 0)

    # Redundant scan: each program totals n_local[0..pid) and counts
    # owner-matches among REAL reqs in [0..pid).
    acc_nl_prefix = tl.zeros((), tl.int32)
    acc_rtls = tl.zeros((), tl.int32)
    num_blocks = tl.cdiv(bs, BS_BLOCK)
    for b in range(num_blocks):
        offs = b * BS_BLOCK + tl.arange(0, BS_BLOCK)
        mask = offs < pid
        all_s = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
        all_nl = tl.maximum((all_s - cp_rank + cp_size - 1) // cp_size, 0)
        all_owner = (all_s - 1) % cp_size
        all_real = all_s > 0
        acc_nl_prefix += tl.sum(tl.where(mask, all_nl, 0))
        acc_rtls += tl.sum(
            tl.where(mask & all_real & (all_owner == my_owner), 1, 0)
        )

    tl.store(owner_ranks_ptr + pid, my_owner.to(tl.int64))
    tl.store(n_local_ptr + pid, my_n_local.to(tl.int64))
    tl.store(cu_local_ptr + pid + 1, acc_nl_prefix + my_n_local)
    tl.store(request_to_local_slot_ptr + pid, acc_rtls.to(tl.int64))
    # Only real reqs go into stage_req_ids; padding reqs leave it at −1.
    # Masked store: the offset for a padding req may be OOB (its `acc_rtls`
    # can reach `max_n_owned`), but `mask=my_real` suppresses the access.
    tl.store(
        stage_req_ids_ptr + my_owner * max_n_owned + acc_rtls,
        pid.to(tl.int64),
        mask=my_real,
    )


@triton.jit
def _cp_build_slot_flat_kernel(
    req_to_token_ptr,       # [num_reqs+1, max_ctx] int32
    req_pool_indices_ptr,   # [bs] int (int32 or int64)
    n_local_ptr,            # [bs] int64
    cu_local_ptr,           # [bs+1] int32
    slot_flat_ptr,          # [Σ n_local] int64 — OUTPUT
    req_to_token_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per request. Copy `req_to_token[req_pool[r], :n_local[r]]`
    into `slot_flat[cu_local[r] : cu_local[r] + n_local[r]]`. Output is int64
    to match the downstream `index_select` callsites."""
    pid = tl.program_id(axis=0)
    req_pool = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    n_r = tl.load(n_local_ptr + pid)
    dst_start = tl.load(cu_local_ptr + pid).to(tl.int64)

    num_blocks = tl.cdiv(n_r, BLOCK_SIZE)
    for b in range(num_blocks):
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offs < n_r
        data = tl.load(
            req_to_token_ptr + req_pool * req_to_token_stride + offs,
            mask=mask, other=0,
        )
        tl.store(slot_flat_ptr + dst_start + offs, data.to(tl.int64), mask=mask)


@triton.jit
def _cp_build_decode_page_table_kernel(
    stage_req_ids_ptr,      # [cp_size, P] int64, −1 = padding
    n_local_ptr,            # [P] int64
    cu_local_ptr,           # [P+1] int32 — cumsum into slot_flat
    slot_flat_ptr,          # [Σ n_local] int64 — flat KV-pool slots, per-req order
    page_table_ptr,         # [cp_size, P, L] int32 — OUTPUT (row-per-query slots)
    P: tl.constexpr,
    L: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per `(stage, slot)`. For ring stage `stage`, query row
    `slot` maps to global request `rid = stage_req_ids[origin, slot]` where
    `origin = (cp_rank − stage) % cp_size`. Write that request's KV-pool slots
    `slot_flat[cu_local[rid] : +n_local[rid]]` into the page-table row
    `page_table[stage, slot, 0 : n_local[rid]]` (the tail stays the pre-filled
    dummy 0; FA reads only `seqused_k[stage, slot] = n_local[rid]` entries).
    Padding rows (`rid < 0`) write nothing → `seqused_k == 0` → FA emits
    `-inf` lse for that query row.

    Fixed-shape (grid `(cp_size, P)`, output `[cp_size, P, L]`) → CUDA-graph
    capturable. FA4 reads KV in-place from the pool via this table — no
    explicit gather buffer."""
    stage = tl.program_id(axis=0)
    slot = tl.program_id(axis=1)
    origin = (cp_rank - stage + cp_size) % cp_size
    rid = tl.load(stage_req_ids_ptr + origin * P + slot)
    valid = rid >= 0
    rid_safe = tl.maximum(rid, 0)
    n_r = tl.where(valid, tl.load(n_local_ptr + rid_safe), 0)
    src_start = tl.load(cu_local_ptr + rid_safe).to(tl.int64)
    row_base = (stage.to(tl.int64) * P + slot) * L

    num_blocks = tl.cdiv(n_r, BLOCK_SIZE)
    for b in range(num_blocks):
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offs < n_r
        src_slots = tl.load(slot_flat_ptr + src_start + offs, mask=mask, other=0)
        tl.store(page_table_ptr + row_base + offs, src_slots.to(tl.int32), mask=mask)


def _build_decode_meta(
    seq_lens: torch.Tensor,            # [B] int (GLOBAL per-request length, padded to B)
    req_to_token: torch.Tensor,        # [num_reqs, max_ctx] int32 — the r2t pool
    req_pool_indices: torch.Tensor,    # [B] int — GLOBAL req-pool indices, padded to B
    *,
    cp_size: int,
    cp_rank: int,
    P: int,
    L: int,
) -> _CPDecodeMeta:
    """Build layer-invariant decode metadata — fully GPU-resident and
    fixed-shape for the capture bucket. Every per-step-varying value is
    produced by a Triton kernel reading the GPU `seq_lens`; the only host
    values are the bucket constants. No `.tolist()` / `.item()` /
    host-derived shapes → CUDA-graph capturable inside or outside a capture.

    Three bucket dims (equal only at cp_size=1):
    - `B = seq_lens.shape[0]` — padded GLOBAL request count.
    - `P` — padded per-rank owned-request count (Q-buffer rows /
      `stage_req_ids` width). Must be `>= max_r count(owner_ranks == r)`.
    - `L` — padded per-request local-KV length; must be `>= max_r n_local[r]`.

    Contract: `seq_lens` / `req_pool_indices` arrive padded to length `B`
    (padding entries have `seq_len == 0` → `n_local == 0` → inactive rows)."""
    device = seq_lens.device
    B = seq_lens.shape[0]
    assert req_pool_indices.shape[0] == B, (
        f"req_pool_indices must be [B]={B}, got {req_pool_indices.shape}"
    )

    # ----- Per-request metadata: one fused Triton launch (grid (B,)) -----
    owner_ranks = torch.empty(B, dtype=torch.int64, device=device)
    n_local_per_req = torch.empty(B, dtype=torch.int64, device=device)
    request_to_local_slot = torch.empty(B, dtype=torch.int64, device=device)
    cu_local = torch.zeros(B + 1, dtype=torch.int32, device=device)
    stage_req_ids = torch.full((cp_size, P), -1, dtype=torch.int64, device=device)
    _cp_build_meta_fused_kernel[(B,)](
        seq_lens,
        owner_ranks, n_local_per_req, cu_local,
        stage_req_ids, request_to_local_slot,
        B, P,  # bs=B, max_n_owned=P (padding reqs are owner-(cp-1), n_local 0)
        cp_size=cp_size, cp_rank=cp_rank,
        BS_BLOCK=128,
    )

    # ----- slot_flat: gather req_to_token -> flat KV-pool slots, per-req order -----
    slot_flat = torch.zeros(B * L, dtype=torch.int64, device=device)
    _cp_build_slot_flat_kernel[(B,)](
        req_to_token,
        req_pool_indices,
        n_local_per_req,
        cu_local,
        slot_flat,
        req_to_token_stride=req_to_token.shape[1],
        BLOCK_SIZE=512,
    )

    # ----- Per-stage paged-KV tables (batched over the cp_size ring stages) -----
    # seqused_k[stage, slot] = n_local[ stage_req_ids[origin, slot] ], 0 if padding.
    origins = (cp_rank - torch.arange(cp_size, device=device)) % cp_size  # [cp_size]
    sri_by_stage = stage_req_ids.index_select(0, origins)                 # [cp_size, P]
    valid = sri_by_stage >= 0
    seqused_k = torch.where(
        valid,
        n_local_per_req[sri_by_stage.clamp_min(0)],
        torch.zeros_like(sri_by_stage),
    ).to(torch.int32)                                                    # [cp_size, P]

    # page_table[stage, slot, :] = that query row's KV-pool slots (row-per-query,
    # tail padded with 0). FA4 reads KV in-place via this table — no gather.
    page_table = torch.zeros(cp_size, P, L, dtype=torch.int32, device=device)
    _cp_build_decode_page_table_kernel[(cp_size, P)](
        stage_req_ids, n_local_per_req, cu_local, slot_flat, page_table,
        P=P, L=L, cp_size=cp_size, cp_rank=cp_rank, BLOCK_SIZE=512,
    )

    cu_seqlens_q = torch.arange(P + 1, dtype=torch.int32, device=device)

    return _CPDecodeMeta(
        owner_ranks=owner_ranks,
        request_to_local_slot=request_to_local_slot,
        page_table=page_table,
        seqused_k=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        P=P,
        L=L,
    )


@triton.jit
def _cp_build_extend_meta_fused_kernel(
    seq_lens_ptr,                      # [bs]              int32
    extend_seq_lens_ptr,               # [bs]              int32 (LOCAL Q)
    prefix_local_ptr,                  # [bs]              int32 (P_global//cp)
    n_local_per_rank_ptr,              # [cp_size, bs]     int32 OUT
    cu_seqlens_k_per_rank_ptr,         # [cp_size, bs+1]   int32 OUT (host pre-fills [:,0]=0)
    new_n_k_per_owner_ptr,             # [cp_size, bs]     int32 OUT
    new_cu_per_owner_ptr,              # [cp_size, bs+1]   int32 OUT (host pre-fills [:,0]=0)
    owner_ranks_ptr,                   # [bs]              int64 OUT
    cu_seqlens_q_ptr,                  # [bs+1]            int32 OUT (host pre-fills [0]=0)
    bs,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    BS_BLOCK: tl.constexpr,
):
    """One program per request `pid`. For each potential KV owner `i` in
    [0, cp_size), redundantly scans `seq_lens[0..pid)` and
    `extend_seq_lens[0..pid)` to derive cu_seqlens_k_per_rank and
    new_cu_per_owner prefixes. Also writes per-pid owner_ranks and
    cu_seqlens_q.

    Replaces ~10 small torch ops + the per-owner Python idx_pieces.append
    loop with one Triton launch. The per-owner slice_idx expansion is a
    separate kernel below."""
    pid = tl.program_id(axis=0)
    my_s = tl.load(seq_lens_ptr + pid).to(tl.int32)
    my_q = tl.load(extend_seq_lens_ptr + pid).to(tl.int32)
    my_prefix = tl.load(prefix_local_ptr + pid).to(tl.int32)

    tl.store(owner_ranks_ptr + pid, ((my_s - 1) % cp_size).to(tl.int64))

    num_blocks = tl.cdiv(bs, BS_BLOCK)

    # cu_seqlens_q scan over r' < pid.
    acc_q = tl.zeros((), tl.int32)
    for b in range(num_blocks):
        offs_b = b * BS_BLOCK + tl.arange(0, BS_BLOCK)
        mask_b = offs_b < pid
        all_q = tl.load(extend_seq_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int32)
        acc_q += tl.sum(tl.where(mask_b, all_q, 0))
    tl.store(cu_seqlens_q_ptr + pid + 1, acc_q + my_q)

    # Per-rank loop (compile-time unrolled). For each i, redundantly scan
    # r' < pid to accumulate n_local_per_rank[i] and new_n_k_per_owner[i].
    for i in tl.static_range(cp_size):
        n_local_i = tl.maximum((my_s - i + cp_size - 1) // cp_size, 0)
        # Partial prefill: drop `my_prefix` extra trailing KV tokens so the
        # bottom-right causal diagonal lands at `+P/cp` (self collapses to a
        # plain causal=True with full KV).
        if i < cp_rank:
            slice_i = tl.maximum(n_local_i - my_q - my_prefix, 0)
        elif i > cp_rank:
            slice_i = tl.maximum(n_local_i - my_q - my_prefix + 1, 0)
        else:
            slice_i = tl.zeros((), tl.int32)
        new_n_k_i = n_local_i - slice_i

        acc_n_local_i = tl.zeros((), tl.int32)
        acc_new_n_k_i = tl.zeros((), tl.int32)
        for b in range(num_blocks):
            offs_b = b * BS_BLOCK + tl.arange(0, BS_BLOCK)
            mask_b = offs_b < pid
            all_s_b = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int32)
            all_q_b = tl.load(extend_seq_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int32)
            all_prefix_b = tl.load(
                prefix_local_ptr + offs_b, mask=mask_b, other=0
            ).to(tl.int32)
            all_n_l = tl.maximum((all_s_b - i + cp_size - 1) // cp_size, 0)
            if i < cp_rank:
                all_slice = tl.maximum(all_n_l - all_q_b - all_prefix_b, 0)
            elif i > cp_rank:
                all_slice = tl.maximum(all_n_l - all_q_b - all_prefix_b + 1, 0)
            else:
                all_slice = tl.zeros_like(all_n_l)
            all_new_n_k = all_n_l - all_slice
            acc_n_local_i += tl.sum(tl.where(mask_b, all_n_l, 0))
            acc_new_n_k_i += tl.sum(tl.where(mask_b, all_new_n_k, 0))

        tl.store(n_local_per_rank_ptr + i * bs + pid, n_local_i)
        tl.store(cu_seqlens_k_per_rank_ptr + i * (bs + 1) + pid + 1, acc_n_local_i + n_local_i)
        tl.store(new_n_k_per_owner_ptr + i * bs + pid, new_n_k_i)
        tl.store(new_cu_per_owner_ptr + i * (bs + 1) + pid + 1, acc_new_n_k_i + new_n_k_i)


@triton.jit
def _cp_build_extend_slice_idx_kernel(
    cu_per_rank_owner_ptr,      # [bs+1] int32 — cu_seqlens_k_per_rank[kv_owner]
    new_n_k_owner_ptr,          # [bs]   int32 — new_n_k_per_owner[kv_owner]
    new_cu_owner_ptr,           # [bs+1] int32 — new_cu_per_owner[kv_owner]
    slice_idx_ptr,              # [Σ new_n_k] int64 OUT
    BLOCK_SIZE: tl.constexpr,
):
    """One program per request. Write `[cu_per_rank[r], cu_per_rank[r]+new_n_k[r])`
    into `slice_idx[new_cu[r] : new_cu[r]+new_n_k[r]]`. Replaces the Python
    `for j in range(new_n_k[r]): idx_pieces.append(start+j)` loop that
    scaled with total local KV (≈ ms at large bs/T)."""
    pid = tl.program_id(axis=0)
    n_r = tl.load(new_n_k_owner_ptr + pid)
    src_start = tl.load(cu_per_rank_owner_ptr + pid).to(tl.int64)
    dst_start = tl.load(new_cu_owner_ptr + pid).to(tl.int64)
    n_blocks = tl.cdiv(n_r, BLOCK_SIZE)
    for b in range(n_blocks):
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offs < n_r
        tl.store(slice_idx_ptr + dst_start + offs, src_start + offs, mask=mask)


def _build_extend_meta(
    forward_batch: "ForwardBatch",
    cp_size: int,
    cp_rank: int,
) -> _CPExtendMeta:
    """Build layer-invariant extend metadata. Per-(rank, request) tensors
    via one fused Triton kernel; per-owner `slice_idx_per_owner` via one
    extra Triton kernel per non-self owner that actually needs slicing.
    Scalar shapes (max_seqlen_q, max_seqlen_k_per_owner, n_local_total_per_rank)
    are derived from CPU mirrors (no GPU→CPU sync)."""
    device = forward_batch.seq_lens.device
    seq_lens_gpu = forward_batch.seq_lens
    bs = seq_lens_gpu.shape[0]
    seq_lens_list: List[int] = forward_batch.seq_lens_cpu.tolist()
    extend_seq_lens_cpu: List[int] = forward_batch.extend_seq_lens_cpu or [0] * bs
    # GLOBAL prefix lengths → this rank's LOCAL cached count per request.
    # Page-aligned to cp_size * page_size by CPRadixCache, so // cp is exact.
    extend_prefix_lens_cpu: List[int] = (
        forward_batch.extend_prefix_lens_cpu or [0] * bs
    )
    prefix_local_cpu: List[int] = [p // cp_size for p in extend_prefix_lens_cpu]

    # ----- CPU-mirror scalars (no GPU sync) -----
    # `max_seqlen_q`, `n_local_total_per_rank`, `max_seqlen_k_per_owner`,
    # and a per-owner needs-slicing flag.
    n_local_per_rank_lists: List[List[int]] = [
        [max(0, (s - i + cp_size - 1) // cp_size) for s in seq_lens_list]
        for i in range(cp_size)
    ]
    n_local_total_per_rank: List[int] = [sum(row) for row in n_local_per_rank_lists]
    max_seqlen_q = max(extend_seq_lens_cpu) if extend_seq_lens_cpu else 0

    max_seqlen_k_per_owner: List[int] = [0] * cp_size
    needs_slice: List[bool] = [False] * cp_size
    total_new_n_k: List[int] = [0] * cp_size
    for kv_owner in range(cp_size):
        n_k_list = n_local_per_rank_lists[kv_owner]
        if kv_owner == cp_rank:
            max_seqlen_k_per_owner[kv_owner] = max(n_k_list) if n_k_list else 0
            continue
        # Slice formula: lower drops `n_k − q − prefix_local`; upper drops
        # `n_k − q − prefix_local + 1`. `prefix_local` is the cached-prefix
        # length (0 ⇒ full prefill, original formula).
        if cp_rank > kv_owner:
            scs = [
                max(0, n_k_list[r] - extend_seq_lens_cpu[r] - prefix_local_cpu[r])
                for r in range(bs)
            ]
        else:
            scs = [
                max(0, n_k_list[r] - extend_seq_lens_cpu[r] - prefix_local_cpu[r] + 1)
                for r in range(bs)
            ]
        if sum(scs) == 0:
            max_seqlen_k_per_owner[kv_owner] = max(n_k_list) if n_k_list else 0
            continue
        new_n_k_list = [n_k_list[r] - scs[r] for r in range(bs)]
        max_seqlen_k_per_owner[kv_owner] = max(new_n_k_list) if new_n_k_list else 0
        needs_slice[kv_owner] = True
        total_new_n_k[kv_owner] = sum(new_n_k_list)

    # ----- Per-(rank, request) GPU tensors via one fused Triton launch -----
    n_local_per_rank_dev = torch.empty(cp_size, bs, dtype=torch.int32, device=device)
    cu_seqlens_k_per_rank_dev = torch.zeros(cp_size, bs + 1, dtype=torch.int32, device=device)
    new_n_k_per_owner_dev = torch.empty(cp_size, bs, dtype=torch.int32, device=device)
    new_cu_per_owner_dev = torch.zeros(cp_size, bs + 1, dtype=torch.int32, device=device)
    owner_ranks_dev = torch.empty(bs, dtype=torch.int64, device=device)
    cu_seqlens_q_dev = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    # local_q_lens is just extend_seq_lens_cpu uploaded as int32. One H2D —
    # used directly by the fused kernel for both reads and as a downstream
    # input (no separate "write" needed inside the kernel).
    local_q_lens_dev = torch.tensor(extend_seq_lens_cpu, dtype=torch.int32).to(
        device=device, non_blocking=True
    )
    prefix_local_dev = torch.tensor(prefix_local_cpu, dtype=torch.int32).to(
        device=device, non_blocking=True
    )

    if bs > 0:
        _cp_build_extend_meta_fused_kernel[(bs,)](
            seq_lens_gpu, local_q_lens_dev, prefix_local_dev,
            n_local_per_rank_dev, cu_seqlens_k_per_rank_dev,
            new_n_k_per_owner_dev, new_cu_per_owner_dev,
            owner_ranks_dev, cu_seqlens_q_dev,
            bs, cp_size=cp_size, cp_rank=cp_rank, BS_BLOCK=128,
        )

    # ----- Per-owner slice_idx -----
    slice_idx_per_owner: List[Optional[torch.Tensor]] = [None] * cp_size
    slice_new_cu_per_owner: List[Optional[torch.Tensor]] = [None] * cp_size
    for kv_owner in range(cp_size):
        if not needs_slice[kv_owner]:
            continue
        n_total = total_new_n_k[kv_owner]
        slice_idx_dev = torch.empty(n_total, dtype=torch.int64, device=device)
        if bs > 0:
            _cp_build_extend_slice_idx_kernel[(bs,)](
                cu_seqlens_k_per_rank_dev[kv_owner],
                new_n_k_per_owner_dev[kv_owner],
                new_cu_per_owner_dev[kv_owner],
                slice_idx_dev,
                BLOCK_SIZE=512,
            )
        slice_idx_per_owner[kv_owner] = slice_idx_dev
        slice_new_cu_per_owner[kv_owner] = new_cu_per_owner_dev[kv_owner]

    request_to_local_slot_dev = torch.arange(bs, dtype=torch.int64, device=device)

    # Partial prefill: gather this rank's FULL local KV (cached prefix ++ new)
    # for the ring. n_local_self[r] consecutive slots from r2t per request.
    total_prefix_local = sum(prefix_local_cpu)
    self_kv_flat_idx: Optional[torch.Tensor] = None
    if total_prefix_local > 0:
        n_local_self = n_local_per_rank_lists[cp_rank]
        r2t = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        pieces = [
            r2t[req_pool_indices[r], : n_local_self[r]].to(torch.int64)
            for r in range(bs)
            if n_local_self[r] > 0
        ]
        self_kv_flat_idx = (
            torch.cat(pieces)
            if pieces
            else torch.empty(0, dtype=torch.int64, device=device)
        )

    return _CPExtendMeta(
        n_local_per_rank=n_local_per_rank_dev,
        cu_seqlens_k_per_rank=cu_seqlens_k_per_rank_dev,
        local_q_lens=local_q_lens_dev,
        cu_seqlens_q=cu_seqlens_q_dev,
        slice_idx_per_owner=slice_idx_per_owner,
        slice_new_cu_per_owner=slice_new_cu_per_owner,
        owner_ranks=owner_ranks_dev,
        request_to_local_slot=request_to_local_slot_dev,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_per_owner=max_seqlen_k_per_owner,
        n_local_total_per_rank=n_local_total_per_rank,
        bs=bs,
        prefix_local_per_req=prefix_local_dev,
        self_kv_flat_idx=self_kv_flat_idx,
        total_prefix_local=total_prefix_local,
    )


# ---------------------------------------------------------------------------
# Ring kernel (pure torch + FA4; reusable from tests)
# ---------------------------------------------------------------------------

def _right_offset_pass_kv(stage: int, cp_rank: int, cp_size: int, sq: int, skv: int) -> int:
    """FA4 `window_size=(None, right)` offset for this ring stage."""
    kv_owner = (cp_rank - stage) % cp_size
    if cp_rank >= kv_owner:        # self or lower
        return sq - skv
    return sq - skv - 1            # upper


def _run_stage(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, right_offset: int):
    """One ring stage. right==0 hits FA4's causal=True fast path."""
    if right_offset == 0:
        return flash_attn_func(q, k, v, causal=True, return_lse=True)
    return flash_attn_func(q, k, v, window_size=(None, right_offset), return_lse=True)


def _lse_merge(out_acc, lse_acc, out_new, lse_new):
    """Online LSE merge. out_*: [B,S,H,D] fp32 accum; lse_*: [B,H,S] fp32."""
    out_new = out_new.float()
    lse_new = lse_new.float()
    if out_acc is None:
        return out_new, lse_new
    new_lse = torch.logaddexp(lse_acc, lse_new)
    a = torch.exp(lse_acc - new_lse).unsqueeze(-1).transpose(1, 2)
    b = torch.exp(lse_new - new_lse).unsqueeze(-1).transpose(1, 2)
    return a * out_acc + b * out_new, new_lse


# ---------------------------------------------------------------------------
# CP P2P boundary. Single chokepoint for inter-rank exchange. Today: NCCL
# `batch_isend_irecv`. Future: drop-in replacement with an intra-node CUDA-IPC
# kernel (peer-to-peer `cudaMemcpyAsync` via IPC handles) or an inter-node
# NVSHMEM kernel — both can run inside CUDA-graph capture without NCCL's
# per-call kernel-launch overhead. The interface (caller-provided send/recv
# tensors + rank pairs) is the swap-in point.
# ---------------------------------------------------------------------------

def _cp_p2p_exchange(
    sends: List[Tuple[torch.Tensor, int]],   # (buf, dst_rank)
    recvs: List[Tuple[torch.Tensor, int]],   # (buf, src_rank)
    group,
):
    """Batched P2P. All sends + recvs are issued in one `batch_isend_irecv`
    so NCCL can fuse them where it can. Caller passes pre-allocated recv
    buffers — required for the future cudagraph path (same buffers across
    replays) and for the IPC/NVSHMEM swap (target memory must be registered).

    Returns work handles; pair with `_cp_p2p_wait` to block until done."""
    ops = []
    for buf, dst in sends:
        ops.append(dist.P2POp(dist.isend, buf.contiguous(), dst, group=group))
    for buf, src in recvs:
        ops.append(dist.P2POp(dist.irecv, buf, src, group=group))
    return dist.batch_isend_irecv(ops)


def _cp_p2p_wait(works):
    """Stream-side wait on the handles from `_cp_p2p_exchange`. For the
    future inline-kernel variant this becomes a no-op (the kernel runs
    on-stream and ordering is implicit)."""
    for w in works:
        w.wait()


def _ring_shift_kv_launch(
    k_buf, v_buf, *, recv_seqlen: int, group, world_size: int, rank: int,
    k_recv: Optional[torch.Tensor] = None,
    v_recv: Optional[torch.Tensor] = None,
):
    """Enqueue ring-shift K/V. Recv buffers may be caller-allocated (for
    cudagraph-friendly reuse); if not provided we allocate fresh ones of
    `recv_seqlen` length in the seq dim. The recv seq dim must match the
    SENDER's KV length (round-robin's `n_local` differs by ≤1 across ranks)."""
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    if k_recv is None or v_recv is None:
        # Support both 3-D [total, H, D] (varlen) and 4-D [B, S, H, D] (legacy).
        k_recv_shape = (recv_seqlen,) + tuple(k_buf.shape[1:]) if k_buf.dim() == 3 \
            else (k_buf.shape[0], recv_seqlen) + tuple(k_buf.shape[2:])
        v_recv_shape = (recv_seqlen,) + tuple(v_buf.shape[1:]) if v_buf.dim() == 3 \
            else (v_buf.shape[0], recv_seqlen) + tuple(v_buf.shape[2:])
        k_recv = torch.empty(*k_recv_shape, dtype=k_buf.dtype, device=k_buf.device)
        v_recv = torch.empty(*v_recv_shape, dtype=v_buf.dtype, device=v_buf.device)
    works = _cp_p2p_exchange(
        sends=[(k_buf, next_rank), (v_buf, next_rank)],
        recvs=[(k_recv, prev_rank), (v_recv, prev_rank)],
        group=group,
    )
    return k_recv, v_recv, works


# ---------------------------------------------------------------------------
# Varlen (multi-request, bs >= 1) pass-KV ring
# ---------------------------------------------------------------------------

def pass_kv_ring_attention_varlen(
    q: torch.Tensor,         # [total_local_q, H_q, D]
    k: torch.Tensor,         # [total_local_kv_self, H_kv, D]
    v: torch.Tensor,         # [total_local_kv_self, H_kv, D]
    *,
    meta: _CPExtendMeta,
    cp_size: int,
    cp_rank: int,
    cp_group,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """c-stage pass-KV ring (bs≥1). Per-stage partials are scattered into a
    `[cp_size, 1, total_q, H_q, D]` fp32 buffer and combined with one
    `flash_attn_combine` call at the end — replaces ~9 elementwise kernels
    per stage (logaddexp/exp/mul/add/transpose) with one fused kernel.
    Stages whose `max_seqlen_k==0` simply leave their lse slot at `-inf`,
    so combine ignores them."""
    total_q, H_q, D = q.shape
    device, dtype = q.device, q.dtype
    max_seqlen_q = meta.max_seqlen_q
    cu_seqlens_q = meta.cu_seqlens_q
    cu_seqlens_k_per_rank = meta.cu_seqlens_k_per_rank   # [cp_size, bs+1] int32

    # Partials buffers (fp32). lse=-inf for unwritten slots → combine drops them.
    partials_out = torch.zeros(
        cp_size, 1, total_q, H_q, D, dtype=torch.float32, device=device,
    )
    partials_lse_native = torch.full(
        (cp_size, 1, H_q, total_q), -float("inf"), dtype=torch.float32, device=device,
    )
    # combine wants seqlen in the innermost stride-1 dim → transpose view.
    partials_lse = partials_lse_native.transpose(2, 3)

    k_buf, v_buf = k, v
    _push, _pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop

    _push("pass_kv_ring")
    # Pre-launch stage 1 comm (overlap with stage 0 compute).
    k_next, v_next, comm_works = None, None, None
    if cp_size > 1:
        next_kv_owner = (cp_rank - 1) % cp_size
        _push("p2p_kv_launch_stage1")
        k_next, v_next, comm_works = _ring_shift_kv_launch(
            k_buf, v_buf,
            recv_seqlen=meta.n_local_total_per_rank[next_kv_owner],
            group=cp_group, world_size=cp_size, rank=cp_rank,
        )
        _pop()

    for stage in range(cp_size):
        _push(f"stage_{stage}")
        kv_owner = (cp_rank - stage) % cp_size
        max_seqlen_k = meta.max_seqlen_k_per_owner[kv_owner]
        slice_idx = meta.slice_idx_per_owner[kv_owner]
        slice_new_cu = meta.slice_new_cu_per_owner[kv_owner]

        if slice_idx is None:
            k_in, v_in = k_buf, v_buf
            cu_k_in = cu_seqlens_k_per_rank[kv_owner]
        else:
            k_in = k_buf.index_select(0, slice_idx)
            v_in = v_buf.index_select(0, slice_idx)
            cu_k_in = slice_new_cu

        if max_seqlen_k > 0 and max_seqlen_q > 0:
            _push("fa_compute")
            out_p, lse_p = flash_attn_varlen_func(
                q, k_in, v_in,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_k_in,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                return_lse=True,
            )
            # Cast happens during copy_ (bf16/fp16 → fp32). One kernel each.
            partials_out[stage, 0].copy_(out_p)
            partials_lse_native[stage, 0].copy_(lse_p)
            _pop()

        # Advance ring.
        if stage < cp_size - 1:
            _push("p2p_wait")
            _cp_p2p_wait(comm_works)
            _pop()
            k_buf, v_buf = k_next, v_next
            if stage < cp_size - 2:
                next_next = (cp_rank - stage - 2) % cp_size
                _push(f"p2p_kv_launch_stage{stage+2}")
                k_next, v_next, comm_works = _ring_shift_kv_launch(
                    k_buf, v_buf,
                    recv_seqlen=meta.n_local_total_per_rank[next_next],
                    group=cp_group, world_size=cp_size, rank=cp_rank,
                )
                _pop()
        _pop()  # stage_

    out_buf = torch.empty(1, total_q, H_q, D, dtype=dtype, device=device)
    if max_seqlen_q > 0:
        _push("combine")
        flash_attn_combine(
            partials_out, partials_lse, out=out_buf, return_lse=False,
        )
        _pop()
    else:
        out_buf.zero_()
    _pop()  # pass_kv_ring
    return out_buf.squeeze(0), None


# ---------------------------------------------------------------------------
# Pass-Q decode. Q + accumulator rotate; KV stays local.
# ---------------------------------------------------------------------------

def _attn_decode_call(
    q: torch.Tensor,                # [P, H_q, D] — Sq=1 per row
    key_cache: torch.Tensor,        # [num_pages, page_size, H_kv, D] — raw paged pool
    value_cache: torch.Tensor,      # same layout
    *,
    cu_seqlens_q: torch.Tensor,     # [P+1] int32 — arange (Sq=1 per row)
    seqused_k: torch.Tensor,        # [P]   int32 — per-row KV length (0 = inactive)
    page_table: torch.Tensor,       # [P, L] int32 — per-row KV-pool slot indices
    max_seqlen_k: int,
):
    """FA4 decode call, paged-KV mode — FA reads KV directly from the pool via
    `page_table`, no explicit gather buffer. `page_table` mode requires
    `seqused_k` (not `cu_seqlens_k`) and the pool as
    `[num_pages, page_size, H_kv, D]`; inactive rows get `seqused_k == 0` → FA
    emits `-inf` lse.

    FA4 (cute DSL) unifies prefill and decode under `flash_attn_varlen_func`;
    the decode-optimized path is reached by `num_splits > 1` (split-K along
    KV) — for Sq=1 with few active rows this is critical, else only a handful
    of SMs work and the kernel is launch-bound. SplitKV is sm100-only;
    elsewhere fall back to num_splits=1."""
    cap = torch.cuda.get_device_capability(q.device)
    if cap[0] == 10 and max_seqlen_k > 256:
        # sm100 split-K heuristic: scale with KV length, cap at 16.
        num_splits = max(1, min(16, (max_seqlen_k + 511) // 512))
    else:
        num_splits = 1

    return flash_attn_varlen_func(
        q, key_cache, value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        causal=True,
        return_lse=True,
        num_splits=num_splits,
    )

def pass_q_ring_decode_owner_only(
    q: torch.Tensor,                # [P, H_q, D] — this rank's owned-Q, padded to P
    key_cache: torch.Tensor,        # [num_slots, H_kv, D] or 4-D [num_slots, 1, H_kv, D]
    value_cache: torch.Tensor,      # same layout as key_cache
    *,
    meta: _CPDecodeMeta,
    cp_size: int,
    cp_rank: int,
    cp_group,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Owner-only c-stage pass-Q ring for decode — fixed-shape & CUDA-graph
    capturable (`P` is a bucket constant). Every ring stage runs FA on all `P`
    query rows; inactive rows get `seqused_k == 0` → FA emits `-inf` lse →
    `flash_attn_combine` drops them. KV is read in-place from the paged pool
    via FA4 `page_table` mode — no gather buffer.

    Each rank's per-stage partial against its local KV is shipped to the Q's
    owner via reverse-ring P2P; Q rotates forward. After `cp_size` stages each
    rank holds all `cp_size` partials for its own owned Q, combined in one
    `flash_attn_combine` call. NCCL P2P is CUDA-graph capturable in this stack,
    so the whole function captures as one region."""
    P, H_q, D = q.shape
    device, dtype = q.device, q.dtype
    assert P == meta.P, f"q is [P={P}] but meta.P={meta.P}"

    # FA4 page_table mode wants the pool as [num_pages, page_size, H_kv, D].
    # The token-KV pool is either 4-D [num_slots, 1, H_kv, D] (page_size 1) or
    # 3-D [num_slots, H_kv, D] — normalize to 4-D.
    if key_cache.dim() == 3:
        key_cache = key_cache.unsqueeze(1)
        value_cache = value_cache.unsqueeze(1)

    # Final partials for OUR Q (filled across all cp_size stages). −inf lse =
    # "unwritten / inactive" → flash_attn_combine drops it.
    partials_out = torch.zeros(cp_size, 1, P, H_q, D, dtype=torch.float32, device=device)
    partials_lse_native = torch.full(
        (cp_size, 1, H_q, P), -float("inf"), dtype=torch.float32, device=device,
    )
    partials_lse = partials_lse_native.transpose(2, 3)  # seqlen-inner

    q_buf = q.contiguous()
    next_rank = (cp_rank + 1) % cp_size
    prev_rank = (cp_rank - 1) % cp_size
    _push, _pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop

    _push("pass_q_ring")
    for stage in range(cp_size):
        _push(f"stage_{stage}")
        q_origin = (cp_rank - stage) % cp_size

        # FA on all P rows, paged-KV mode (KV read in-place from the pool —
        # no gather). Rows with seqused_k == 0 yield −inf lse.
        _push("fa_compute")
        out_p, lse_p = _attn_decode_call(
            q_buf, key_cache, value_cache,
            cu_seqlens_q=meta.cu_seqlens_q,
            seqused_k=meta.seqused_k[stage],
            page_table=meta.page_table[stage],
            max_seqlen_k=meta.L,
        )
        _pop()

        if q_origin == cp_rank:
            # Own Q is here this stage: slot the partial directly. No comm.
            _push("partial_local_slot")
            partials_out[stage, 0].copy_(out_p)            # bf16 → fp32 cast on copy
            partials_lse_native[stage, 0].copy_(lse_p)
            _pop()
        else:
            # Ship partial to Q's owner; recv the partial for our Q from the
            # rank that currently holds it (= (cp_rank + stage) % cp_size).
            partial_dst = q_origin
            src = (cp_rank + stage) % cp_size
            cur_p_out = out_p.float().contiguous()
            cur_p_lse_native = lse_p.float().contiguous()
            _push("p2p_partial")
            works = _cp_p2p_exchange(
                sends=[(cur_p_out, partial_dst), (cur_p_lse_native, partial_dst)],
                recvs=[
                    (partials_out[stage, 0], src),
                    (partials_lse_native[stage, 0], src),
                ],
                group=cp_group,
            )
            _cp_p2p_wait(works)
            _pop()

        # Rotate Q forward for next stage.
        if stage < cp_size - 1:
            _push("p2p_q_rotate")
            q_recv = torch.empty_like(q_buf)
            works = _cp_p2p_exchange(
                sends=[(q_buf, next_rank)],
                recvs=[(q_recv, prev_rank)],
                group=cp_group,
            )
            _cp_p2p_wait(works)
            q_buf = q_recv
            _pop()
        _pop()  # stage_

    _push("combine")
    out_final = torch.empty(1, P, H_q, D, dtype=dtype, device=device)
    flash_attn_combine(partials_out, partials_lse, out=out_final, return_lse=False)
    _pop()
    _pop()  # pass_q_ring
    return out_final.squeeze(0), None


def pass_kv_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_locals: List[int],
    cp_size: int,
    cp_rank: int,
    cp_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Non-varlen c-stage pass-KV ring. Inputs `[B=1, S, H, D]`, returns
    `(out [B,S,H_q,D] in q.dtype, lse [B,H_q,S] fp32)`. Kept for `_self_test`
    only; production varlen path is `pass_kv_ring_attention_varlen`.

    `n_locals[i]` is rank `i`'s KV seq length — every CP rank can derive it
    from global `T` and `cp_size` without comm, so we make the caller pass it
    (no per-call all-gather, no GPU↔CPU sync, cudagraph-safe).

    Compute/comm overlap: each stage's compute runs while NCCL ships the
    next stage's K/V (`cudaStreamWaitEvent` under the hood)."""
    sq = q.shape[1]

    k_cur, v_cur = k, v
    k_next, v_next, comm_works = None, None, None
    out_acc, lse_acc = None, None

    if cp_size > 1:
        next_kv_owner = (cp_rank - 1) % cp_size
        k_next, v_next, comm_works = _ring_shift_kv_launch(
            k_cur, v_cur,
            recv_seqlen=n_locals[next_kv_owner],
            group=cp_group, world_size=cp_size, rank=cp_rank,
        )

    for stage in range(cp_size):
        kv_owner = (cp_rank - stage) % cp_size
        skv = n_locals[kv_owner]
        right_offset = _right_offset_pass_kv(stage, cp_rank, cp_size, sq, skv)

        if sq > 0 and skv > 0:
            out, lse = _run_stage(q, k_cur, v_cur, right_offset)
            out_acc, lse_acc = _lse_merge(out_acc, lse_acc, out, lse)

        if stage < cp_size - 1:
            _cp_p2p_wait(comm_works)
            k_cur, v_cur = k_next, v_next
            if stage < cp_size - 2:
                next_next_kv_owner = (cp_rank - stage - 2) % cp_size
                k_next, v_next, comm_works = _ring_shift_kv_launch(
                    k_cur, v_cur,
                    recv_seqlen=n_locals[next_next_kv_owner],
                    group=cp_group, world_size=cp_size, rank=cp_rank,
                )

    if out_acc is None:
        return q.new_zeros(*q.shape), q.new_zeros(q.shape[0], q.shape[2], q.shape[1])
    return out_acc.to(q.dtype), lse_acc


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class CPRoundRobinBackend(AttentionBackend):
    """Round-robin context-parallel attention backend.
    See cp_design_notes.md for the full design rationale."""

    def __init__(self, model_runner):
        super().__init__()
        self.cp_size = get_attention_cp_size()
        self.cp_rank = get_attention_cp_rank()
        self.cp_group = get_attention_cp_group().device_group
        self._context_len = model_runner.model_config.context_len
        self.cg_L = None  # set by `init_cuda_graph_state`

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Precompute layer-invariant CP metadata once per forward batch.

        Decode metadata is *not* built here — it is deferred to the first
        `forward_decode` call (lazy, gated on `cp_decode_meta is None`) so the
        metadata kernels land inside the captured `model.forward()` region
        under CUDA graph; eager and cudagraph then share one path."""
        if forward_batch.forward_mode.is_extend():
            forward_batch.cp_extend_meta = _build_extend_meta(
                forward_batch, self.cp_size, self.cp_rank
            )

    # ---- CUDA graph hooks (decode) -----------------------------------------
    # Decode metadata is built lazily inside `forward_decode` (captured as part
    # of `model.forward()`), so the capture/replay hooks need no per-replay
    # refresh. `init_cuda_graph_state` only pins the fixed `cg_L`.
    #
    # `cg_L` = the per-request local-KV length bound used under capture (shape
    # must be static). Set to `⌈context_len / cp_size⌉`: since
    # `n_local = ⌈seq_len/cp_size⌉` and `seq_len ≤ context_len`, this covers
    # every admissible request — so no `can_run` n_local gate is needed and
    # every decode batch is cudagraph-eligible. Not capped — what scales with
    # `cg_L` is only the small `page_table` / `slot_flat` index buffers.
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cg_L = max(1, (self._context_len + self.cp_size - 1) // self.cp_size)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs) -> None:
        pass

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs) -> None:
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v,
                layer.k_scale, layer.v_scale,
            )

        meta = forward_batch.cp_extend_meta
        if meta.total_prefix_local > 0:
            # Partial prefill: the ring needs this rank's FULL local KV
            # (cached prefix ++ the new tokens just written above), gathered
            # from the per-layer KV pool. `k`/`v` above are the NEW tokens only.
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            idx = meta.self_kv_flat_idx
            k_full = key_cache.index_select(0, idx)
            v_full = value_cache.index_select(0, idx)
            if k_full.dim() == 4:  # [N, 1, H_kv, D] paged pool → [N, H_kv, D]
                k_full = k_full.squeeze(1)
                v_full = v_full.squeeze(1)
        else:
            k_full, v_full = k, v

        out, _ = pass_kv_ring_attention_varlen(
            q, k_full, v_full,
            meta=meta,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            cp_group=self.cp_group,
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _ensure_decode_meta(self, q, forward_batch: "ForwardBatch") -> "_CPDecodeMeta":
        """Lazily build `_CPDecodeMeta` on the first `forward_decode` of a
        forward (gated on `cp_decode_meta is None`); layers 1..N-1 reuse it.
        Building here rather than in `init_forward_metadata` places the
        metadata kernels inside the captured `model.forward()` under CUDA
        graph, so no per-replay metadata hook is needed.

        `P` (= the model's owned-batch dim) comes from `q.shape[0]`; `B` from
        `cp_global_seq_lens`. `L` is the fixed `cg_L` under capture (shape must
        be static), else the exact per-batch max."""
        meta = forward_batch.cp_decode_meta
        if meta is not None:
            return meta

        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        seq_lens = forward_batch.cp_global_seq_lens
        P = q.shape[0]
        if get_is_capture_mode():
            # Under capture L must be a constant; cg_L covers any request.
            L = self.cg_L
        else:
            max_seq = int(seq_lens.max().item())  # eager-only, layer-0-only
            L = max(1, (max_seq - self.cp_rank + self.cp_size - 1) // self.cp_size)
        meta = _build_decode_meta(
            seq_lens,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.cp_global_req_pool_indices,
            cp_size=self.cp_size, cp_rank=self.cp_rank, P=P, L=L,
        )
        forward_batch.cp_decode_meta = meta
        return meta

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        """Owner-only pass-Q ring decode. q/k/v arrive `[P, H_*, D]` — this
        rank's owned requests + zero-padding, NOT TP-replicated. The pass-Q
        ring is shape-static and CUDA-graph capturable."""
        q = q.view(q.shape[0], layer.tp_q_head_num, layer.head_dim)
        meta = self._ensure_decode_meta(q, forward_batch)
        P = meta.P

        k = k.view(P, layer.tp_k_head_num, layer.head_dim)
        v = v.view(P, layer.tp_v_head_num, layer.v_head_dim)

        # First `n_owned` rows are real, the rest padding. Eager: `out_cache_loc`
        # is [n_owned] from `alloc_for_decode` and can be 0 on a rank that owns
        # nothing at small bs — skip the empty `set_kv_buffer`. Cudagraph:
        # `out_cache_loc` is the [P] static buffer (padding rows → slot 0), so
        # `n_owned == P > 0` and the guard / slice are no-ops.
        if save_kv_cache:
            n_owned = forward_batch.out_cache_loc.shape[0]
            if n_owned > 0:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k[:n_owned], v[:n_owned],
                    layer.k_scale, layer.v_scale,
                )

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        out, _ = pass_q_ring_decode_owner_only(
            q, key_cache, value_cache,
            meta=meta,
            cp_size=self.cp_size, cp_rank=self.cp_rank, cp_group=self.cp_group,
        )
        return out.reshape(P, layer.tp_q_head_num * layer.v_head_dim)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0
