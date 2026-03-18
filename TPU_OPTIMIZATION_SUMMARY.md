# TPU Optimization Summary

## Overview

This document summarizes the optimization work done on the autoresearch JAX/TPU port (`train_jax.py`), targeting TPU v4-8 (single host, 4 chips, 32GB HBM each).

## Baseline

The starting point was a working JAX/TPU port with the following performance:

| Metric | Value |
|---|---|
| val_bpb | 1.136 |
| MFU | 8.71% |
| Throughput | ~400k tok/sec |
| Total tokens | 125.8M |
| Steps | 240 |
| Step time | ~1.3s |
| Peak HBM | 1,743 MB / 32,768 MB (5.3%) |
| Model | 50.3M params, DEPTH=8, n_embd=512 |

Key bottlenecks identified:
- **Gradient accumulation overhead**: `grad_accum_steps=2` required two separate `compute_grads` JIT calls per step, each with a Python-level sync (`float(loss)`) and tree_map accumulation.
- **Unnecessary data copy**: `jnp.array(x_np)` before `jax.device_put()` created a redundant host-side JAX array.
- **Underutilized HBM**: Only 5.3% of available memory was used, suggesting room for larger batches.

## Optimizations Applied

### 1. Increased DEVICE_BATCH_SIZE (32 -> 64)

- Doubles the per-device batch from 32 to 64 sequences
- Eliminates gradient accumulation entirely (`grad_accum_steps` drops from 2 to 1)
- Removes the micro-step loop, the `float(loss)` mid-step sync, and the `jax.tree.map` gradient averaging
- Net effect: ~10% faster step time at the same total batch size (524K tokens/step)

### 2. Gradient Checkpointing (`@nnx.remat` on Block)

- Added `@nnx.remat` decorator to `Block.__call__` to enable activation recomputation during backward pass
- Required to fit the larger batch size in HBM — without it, attention score tensors (`f32[64,4,2048,2048]` = 4GB each) cause OOM
- Trades some compute (recomputing block activations during backward) for memory savings
- Peak HBM increased only modestly: 1,743 MB -> 1,870 MB

### 3. Direct numpy-to-device transfer

- Removed unnecessary `jnp.array()` wrapping in the data pipeline
- `jax.device_put(x_np, data_sharding)` accepts numpy arrays directly

### 4. Simplified training loop

- No micro-step loop needed with `grad_accum=1`
- Loss read-back happens after `jax.block_until_ready()` instead of mid-step

## Final Results

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| **val_bpb** | 1.136 | **1.115** | -0.021 (better) |
| **MFU** | 8.71% | **9.80%** | +12.5% |
| **Throughput** | ~400k tok/sec | ~440k tok/sec | +10% |
| **Total tokens** | 125.8M | 141.0M | +12.1% |
| **Steps** | 240 | 269 | +12.1% |
| **Step time** | ~1.3s | ~1.2s | -8% |
| **Peak HBM** | 1,743 MB | 1,870 MB | +7% |

The val_bpb improvement (1.136 -> 1.115) comes primarily from processing 12% more tokens within the same 300s time budget.

## Experiments Attempted But Not Kept

### Fused train_step (single JIT for forward+backward+optimizer)

Combining `compute_grads` and `apply_optimizer` into a single `@jax.jit` function caused OOM (31.78G program memory vs 30.75G available). XLA could not free intermediate activations when the entire forward+backward+optimizer was a single compiled program. The attention score matrices (`f32[64,4,2048,2048]` = 4GB each) dominated memory. Keeping separate JIT calls acts as a natural memory checkpoint boundary.

### Larger models (DEPTH=10, DEPTH=12)

| Config | val_bpb | MFU | Tokens | Params |
|---|---|---|---|---|
| DEPTH=8, batch 64 | **1.115** | 9.8% | 141.0M | 50.3M |
| DEPTH=10, batch 64 | 1.143 | 12.4% | 101.2M | 85.9M |
| DEPTH=12, batch 32 | 1.232 | 14.3% | 73.9M | 135.3M |

Larger models achieved higher MFU (better hardware utilization) but worse val_bpb within the 300s budget. The larger models process fewer tokens per step and need more training time to converge. For this time budget, 50M params is near-optimal — larger models would need proportionally longer training.

## Remaining Observations

- **MFU ceiling**: At ~10% MFU, the 50M param model is fundamentally too small to saturate TPU v4-8 compute (1.1 PFLOPS bf16). This is a model-size limitation, not a code optimization issue.
- **HBM headroom**: 1.87 GB / 32 GB used — significant room remains, but the bottleneck is activation memory (attention scores) rather than parameter/optimizer state storage.
- **Scaling sweet spot**: For longer time budgets (e.g., 30 min+), larger models (DEPTH=12+) would likely outperform DEPTH=8 as they have time to converge.
