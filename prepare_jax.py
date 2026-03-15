"""
JAX-compatible data loading and evaluation utilities for autoresearch.
Reuses prepare.py's tokenizer, download, and document iteration logic.
Replaces PyTorch tensor operations with numpy/JAX arrays for TPU compatibility.

Usage: imported by train_jax.py
"""

import os
import math
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants (same as prepare.py, duplicated to avoid torch import)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288
VOCAB_SIZE = 8192

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Tokenizer (same as prepare.py, no torch dependency)
# ---------------------------------------------------------------------------

import tiktoken

class Tokenizer:
    """Minimal tokenizer wrapper."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)

# ---------------------------------------------------------------------------
# Token bytes lookup (rebuilt from tokenizer, no torch dependency)
# ---------------------------------------------------------------------------

_token_bytes_cache = None

def get_token_bytes():
    """Build token_bytes array from tokenizer. Cached after first call."""
    global _token_bytes_cache
    if _token_bytes_cache is not None:
        return _token_bytes_cache
    tokenizer = Tokenizer.from_directory()
    enc = tokenizer.enc
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    _token_bytes_cache = np.array(token_bytes_list, dtype=np.int32)
    return _token_bytes_cache

# ---------------------------------------------------------------------------
# Data pipeline (framework-agnostic document iteration)
# ---------------------------------------------------------------------------

def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Same algorithm as prepare.py but yields numpy arrays instead of torch tensors.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate numpy buffer
    row_buffer = np.empty((B, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = row_buffer[:, :-1].copy()
        targets = row_buffer[:, 1:].copy()
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (JAX version of evaluate_bpb from prepare.py)
# ---------------------------------------------------------------------------

def evaluate_bpb(model_apply_fn, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Same logic as prepare.py's evaluate_bpb but using JAX arrays.

    model_apply_fn: callable(x, y, reduction='none') -> per-token losses
    """
    token_bytes = get_token_bytes()
    token_bytes_jax = jnp.array(token_bytes)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x_np, y_np, _ = next(val_loader)
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        loss_flat = model_apply_fn(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes_jax[y_flat]
        mask = nbytes > 0
        total_nats += float(jnp.sum(loss_flat * mask))
        total_bytes += int(jnp.sum(nbytes))
    return total_nats / (math.log(2) * total_bytes)
