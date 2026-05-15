from __future__ import annotations

"""
Radix prefix cache for the prefill round-robin context-parallel layout.

Under round-robin CP, global token `g` lives on rank `g % cp_size` at local
slot `g // cp_size` — a pure function of `g`, independent of the request or
its total length `T`. So a cached prefix is cross-request reusable on every
rank, and the radix tree can stay structurally identical across CP ranks.

Design: each CP rank runs its OWN `CPRadixCache`. The trees stay in lockstep
because every rank sees the same GLOBAL token-id stream and makes the same
deterministic decisions. They differ only in the per-node `value` payload:

- `key`   — GLOBAL token ids (identical on every rank), as in `RadixCache`.
- `value` — THIS rank's LOCAL KV-pool slot tensor, length `len(key)//cp_size`.

`self.page_size` is set to `cp_size * pool_page_size` so every node's global
range is `cp_size`-aligned (⇒ `len(key)//cp_size` is always exact) and lands
as whole local pool pages on every rank. The pool/allocator keeps its own
`page_size` — `mem_cache/common.py` reads `allocator.page_size` for slot math.

Only the methods where a GLOBAL key length indexes into the LOCAL `value` are
overridden; all matching / splitting / eviction / lock-ref logic is inherited
unchanged. See controller.md for the full rationale.
"""

import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    split_node_hash_value,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class CPRadixCache(RadixCache):
    def __init__(self, params: CacheInitParams):
        super().__init__(params)
        self.cp_size = params.attn_cp_size
        assert self.cp_size > 1, "CPRadixCache requires attn_cp_size > 1"
        # Pool/allocator page size stays `params.page_size`; the KEY-alignment
        # page size becomes `cp_size * pool` so cached node ranges are
        # cp-aligned and land as whole local pages. RadixCache reads
        # `self.page_size` for all key ops, so overriding it is sufficient.
        self.pool_page_size = params.page_size
        self.page_size = self.cp_size * params.page_size

    ##### Public API #####

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            # `value` holds LOCAL slots; `len(key)` is GLOBAL.
            value = value[: len(key) // self.cp_size]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: "Req", is_insert: bool = True):
        """Cache request when it finishes."""
        if self.disable_finished_insert:
            is_insert = False

        cp = self.cp_size
        kv_committed_len = req.pop_committed_kv_cache()  # GLOBAL
        n_local = (kv_committed_len + cp - 1) // cp  # mirror-alloc local count

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :n_local
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_local]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)  # GLOBAL, multiple of self.page_size
        local_key_len = key_len // cp
        values = kv_indices[:local_key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            # Free the duplicates that were already in the tree.
            # `cache_protected_len` is LOCAL; `result.prefix_len` is GLOBAL.
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len // cp]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : local_key_len]
            )

        # free the unaligned LOCAL tail (incl. phantom slots)
        self.token_to_kv_pool_allocator.free(kv_indices[local_key_len:])

        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: "Req", chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        cp = self.cp_size
        token_ids = req.fill_ids  # GLOBAL
        n_local = (len(token_ids) + cp - 1) // cp
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_local]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        local_key_len = len(radix_key) // cp
        values = kv_indices[:local_key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len  # GLOBAL

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len // cp]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        # `new_indices` is LOCAL; `radix_key` is GLOBAL.
        assert len(new_indices) == len(radix_key) // cp, (
            f"{len(new_indices)=}, {len(radix_key) // cp=}"
        )

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        req.cache_protected_len = len(new_indices)  # LOCAL

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` stays LOCAL: cached prefix slots ++ unaligned tail.
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def evictable_size(self):
        # `evictable_size_` accumulates GLOBAL key lengths; report LOCAL slots.
        return self.evictable_size_ // self.cp_size

    def protected_size(self):
        return self.protected_size_ // self.cp_size

    ##### Internal Helper Functions #####

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # `split_len` is GLOBAL (a multiple of self.page_size); `child.value`
        # holds LOCAL slots (len == global_key_len // cp_size).
        local_split = split_len // self.cp_size
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:local_split].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[local_split:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len // self.cp_size :]  # CP: value is LOCAL

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            # GLOBAL units — consistent with the `// cp_size` getters.
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            self._record_store_event(new_node)
        return total_prefix_length
