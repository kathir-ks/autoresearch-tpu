"""
Autoresearch pretraining script — JAX/TPU version.
Single-host TPU (v4-8), single-file.
Usage: uv run train_jax.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax.nnx as nnx

from prepare_jax import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)
    sin = jnp.sin(freqs).astype(jnp.bfloat16)
    # Shape: (1, seq_len, 1, head_dim//2) for broadcasting with (B, T, H, D)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def compute_window_sizes(config):
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern)
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": long_window, "S": short_window}
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = long_window  # last layer always full
    return window_sizes


def precompute_attention_masks(config):
    """Precompute causal + sliding window masks for each layer."""
    window_sizes = compute_window_sizes(config)
    T = config.sequence_len
    positions = jnp.arange(T)
    causal = positions[:, None] >= positions[None, :]
    masks = []
    for ws in window_sizes:
        if ws >= T:
            masks.append(None)
        else:
            window_mask = (positions[:, None] - positions[None, :]) < ws
            mask = causal & window_mask
            masks.append(mask)
    return masks


class CausalSelfAttention(nnx.Module):
    def __init__(self, config, layer_idx, *, rngs):
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nnx.Linear(self.n_embd, self.n_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(self.n_embd, self.n_embd, use_bias=False, rngs=rngs)
        self.ve_gate_channels = 32
        self.has_ve = has_ve(layer_idx, config.n_layer)
        if self.has_ve:
            self.ve_gate = nnx.Linear(self.ve_gate_channels, self.n_kv_head, use_bias=False, rngs=rngs)

    def __call__(self, x, ve, cos_sin, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * jax.nn.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate[..., None] * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # GQA: repeat k,v if n_kv_head < n_head
        if self.n_kv_head < self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        if mask is None:
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            y = jax.nn.dot_product_attention(q, k, v, mask=mask[None, None, :, :])

        y = y.reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nnx.Module):
    def __init__(self, config, *, rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.relu(x) ** 2
        x = self.c_proj(x)
        return x


class Block(nnx.Module):
    def __init__(self, config, layer_idx, *, rngs):
        self.attn = CausalSelfAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x, ve, cos_sin, mask):
        x = x + self.attn(norm(x), ve, cos_sin, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nnx.Module):
    def __init__(self, config, *, rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.blocks = [Block(config, i, rngs=rngs) for i in range(config.n_layer)]
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)
        self.resid_lambdas = nnx.Param(jnp.ones(config.n_layer))
        self.x0_lambdas = nnx.Param(jnp.zeros(config.n_layer))
        # Value embeddings (alternating layers)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self.value_embeds[str(i)] = nnx.Embed(config.vocab_size, kv_dim, rngs=rngs)
        # Precomputed RoPE (non-trainable)
        rotary_seq_len = config.sequence_len * 10
        cos, sin = precompute_rotary_embeddings(rotary_seq_len, head_dim)
        self.rope_cos = nnx.Variable(cos)
        self.rope_sin = nnx.Variable(sin)
        # Precomputed attention masks (non-trainable)
        self.attn_masks = precompute_attention_masks(config)

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape
        cos_sin = self.rope_cos.value[:, :T], self.rope_sin.value[:, :T]

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas.value[i] * x + self.x0_lambdas.value[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.attn_masks[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.astype(jnp.float32)
        logits = softcap * jnp.tanh(logits / softcap)

        if targets is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
            per_token_loss = -log_probs[jnp.arange(logits_flat.shape[0]), targets_flat]
            mask = targets_flat != -1
            per_token_loss = per_token_loss * mask
            if reduction == 'mean':
                return jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(mask), 1)
            else:
                return per_token_loss.reshape(B, T)
        return logits


def init_weights(model, config, key):
    """Initialize weights to match the PyTorch version exactly."""
    n_embd = config.n_embd
    s = 3**0.5 * n_embd**-0.5

    key, k1, k2 = jax.random.split(key, 3)

    # Embedding: normal(0, 1), stored in bf16
    model.wte.embedding.value = jax.random.normal(k1, model.wte.embedding.value.shape, dtype=jnp.float32).astype(jnp.bfloat16)

    # lm_head: normal(0, 0.001)
    model.lm_head.kernel.value = jax.random.normal(k2, model.lm_head.kernel.value.shape, dtype=jnp.float32) * 0.001

    # Transformer blocks
    for block in model.blocks:
        key, *bkeys = jax.random.split(key, 6)
        block.attn.c_q.kernel.value = jax.random.uniform(bkeys[0], block.attn.c_q.kernel.value.shape, minval=-s, maxval=s)
        block.attn.c_k.kernel.value = jax.random.uniform(bkeys[1], block.attn.c_k.kernel.value.shape, minval=-s, maxval=s)
        block.attn.c_v.kernel.value = jax.random.uniform(bkeys[2], block.attn.c_v.kernel.value.shape, minval=-s, maxval=s)
        block.attn.c_proj.kernel.value = jnp.zeros_like(block.attn.c_proj.kernel.value)
        block.mlp.c_fc.kernel.value = jax.random.uniform(bkeys[3], block.mlp.c_fc.kernel.value.shape, minval=-s, maxval=s)
        block.mlp.c_proj.kernel.value = jnp.zeros_like(block.mlp.c_proj.kernel.value)
        if block.attn.has_ve:
            block.attn.ve_gate.kernel.value = jnp.zeros_like(block.attn.ve_gate.kernel.value)

    # Per-layer scalars
    model.resid_lambdas.value = jnp.ones(config.n_layer)
    model.x0_lambdas.value = jnp.full(config.n_layer, 0.1)

    # Value embeddings: uniform(-s, s), stored in bf16
    for ve_key_str in model.value_embeds:
        key, vkey = jax.random.split(key)
        ve = model.value_embeds[ve_key_str]
        ve.embedding.value = jax.random.uniform(vkey, ve.embedding.value.shape, minval=-s, maxval=s).astype(jnp.bfloat16)


def estimate_flops(config, window_sizes):
    """Estimated FLOPs per token (forward + backward)."""
    head_dim = config.n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim
    per_block = (config.n_embd * config.n_head * head_dim +
                 config.n_embd * kv_dim +
                 config.n_embd * kv_dim +
                 config.n_embd * config.n_embd +
                 config.n_embd * 4 * config.n_embd +
                 4 * config.n_embd * config.n_embd)
    nparams_matrix = per_block * config.n_layer
    nparams_matrix += config.n_embd * config.vocab_size

    h = config.n_head
    q = head_dim
    t = config.sequence_len
    attn_flops = 0
    for window in window_sizes:
        effective_seq = t if window >= t else min(window, t)
        attn_flops += 12 * h * q * effective_seq
    return 6 * nparams_matrix + attn_flops


def count_params(model, config):
    """Count parameters by group."""
    wte = model.wte.embedding.value.size
    value_embeds = sum(model.value_embeds[k].embedding.value.size for k in model.value_embeds)
    lm_head = model.lm_head.kernel.value.size
    transformer_matrices = sum(
        block.attn.c_q.kernel.value.size + block.attn.c_k.kernel.value.size +
        block.attn.c_v.kernel.value.size + block.attn.c_proj.kernel.value.size +
        block.mlp.c_fc.kernel.value.size + block.mlp.c_proj.kernel.value.size +
        (block.attn.ve_gate.kernel.value.size if block.attn.has_ve else 0)
        for block in model.blocks
    )
    scalars = model.resid_lambdas.value.size + model.x0_lambdas.value.size
    total = wte + value_embeds + lm_head + transformer_matrices + scalars
    return {
        'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
        'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
    }


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, functional JAX version)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def newton_schulz_rows_gt_cols(X):
    """Newton-Schulz polar decomposition when rows > cols (raw shape m > n), 5 steps."""
    for a, b, c in polar_express_coeffs:
        A = jnp.swapaxes(X, -2, -1) @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
    return X


def newton_schulz_cols_ge_rows(X):
    """Newton-Schulz polar decomposition when cols >= rows (raw shape m <= n), 5 steps."""
    for a, b, c in polar_express_coeffs:
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def muon_update(stacked_params, stacked_grads, momentum_buffer, second_momentum_buffer,
                momentum, lr, wd, beta2, is_tall):
    """
    Muon update for a group of same-shape matrix params.
    is_tall: True if out_features >= in_features (semantic "tall").
      For Flax kernels (in, out): is_tall=True means shape[-1] >= shape[-2],
      so the raw array has cols >= rows.
    """
    # Nesterov momentum (lerp equivalent: buf += (1-m) * (grad - buf))
    new_momentum_buffer = momentum_buffer + (1 - momentum) * (stacked_grads - momentum_buffer)
    g = stacked_grads + momentum * (new_momentum_buffer - stacked_grads)

    # Polar express orthogonalization
    X = g.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) * 1.02 + 1e-6)

    # Choose NS path based on raw array shape:
    # is_tall (out >= in) means Flax shape (in, out) has cols >= rows -> use cols_ge_rows
    # not is_tall (in > out) means Flax shape (in, out) has rows > cols -> use rows_gt_cols
    if is_tall:
        X = newton_schulz_cols_ge_rows(X)
    else:
        X = newton_schulz_rows_gt_cols(X)
    g = X

    # NorMuon variance reduction
    # PyTorch uses red_dim=-1 when out >= in (reduce along in dim of (out, in) shape).
    # In Flax (in, out): when is_tall (out >= in), we reduce along in dim = axis -2.
    # When not is_tall (in > out), we reduce along out dim = axis -1.
    red_dim = -2 if is_tall else -1
    v_mean = (g.astype(jnp.float32) ** 2).mean(axis=red_dim, keepdims=True)
    red_dim_size = g.shape[red_dim]
    v_norm_sq = v_mean.sum(axis=(-2, -1), keepdims=True) * red_dim_size
    v_norm = jnp.sqrt(v_norm_sq)

    new_second_momentum_buffer = second_momentum_buffer + (1 - beta2) * (v_mean.astype(second_momentum_buffer.dtype) - second_momentum_buffer)
    step_size = jax.lax.rsqrt(jnp.maximum(new_second_momentum_buffer, 1e-10))
    scaled_sq_sum = (v_mean * red_dim_size) * (step_size.astype(jnp.float32) ** 2)
    v_norm_new = jnp.sqrt(scaled_sq_sum.sum(axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))
    g = g * final_scale.astype(g.dtype)

    # Cautious weight decay + parameter update
    mask = (g * stacked_params) >= 0
    new_params = stacked_params - lr * g.astype(stacked_params.dtype) - lr * wd * stacked_params * mask.astype(stacked_params.dtype)

    return new_params, new_momentum_buffer, new_second_momentum_buffer


def adamw_update(param_val, grad, exp_avg, exp_avg_sq, step_count, lr, beta1, beta2, eps, wd):
    """Single AdamW parameter update."""
    new_step = step_count + 1
    new_exp_avg = exp_avg * beta1 + grad * (1 - beta1)
    new_exp_avg_sq = exp_avg_sq * beta2 + grad ** 2 * (1 - beta2)
    bias1 = 1 - beta1 ** new_step
    bias2 = 1 - beta2 ** new_step
    denom = jnp.sqrt(new_exp_avg_sq / bias2) + eps
    new_param = (1 - lr * wd) * param_val - (lr / bias1) * new_exp_avg / denom
    return new_param, new_exp_avg, new_exp_avg_sq, new_step


# ---------------------------------------------------------------------------
# Parameter group management
# ---------------------------------------------------------------------------

def build_param_registry(model, config):
    """
    Build a registry mapping each trainable parameter to its optimizer group.
    Returns:
        param_id_to_group: dict mapping id(param_variable) -> group_info
        muon_shape_groups: dict mapping shape -> list of param_variables
    """
    dmodel_lr_scale = (config.n_embd / 768) ** -0.5

    # Each entry: (nnx_variable, group_name, extra_config)
    param_id_to_group = {}

    # AdamW groups
    adamw_config = {
        'lm_head': dict(base_lr=0.004 * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, wd=0.0),
        'embed': dict(base_lr=0.6 * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, wd=0.0),
        'resid_scalar': dict(base_lr=0.5 * 0.01, betas=(0.8, 0.95), eps=1e-10, wd=0.0),
        'x0_scalar': dict(base_lr=0.5, betas=(0.96, 0.95), eps=1e-10, wd=0.0),
    }

    param_id_to_group[id(model.lm_head.kernel)] = ('lm_head', model.lm_head.kernel)
    param_id_to_group[id(model.wte.embedding)] = ('embed', model.wte.embedding)
    for k in model.value_embeds:
        param_id_to_group[id(model.value_embeds[k].embedding)] = ('embed', model.value_embeds[k].embedding)
    param_id_to_group[id(model.resid_lambdas)] = ('resid_scalar', model.resid_lambdas)
    param_id_to_group[id(model.x0_lambdas)] = ('x0_scalar', model.x0_lambdas)

    # VE gate weights -> resid_scalar group (small params, AdamW)
    for block in model.blocks:
        if block.attn.has_ve:
            param_id_to_group[id(block.attn.ve_gate.kernel)] = ('resid_scalar', block.attn.ve_gate.kernel)

    # Muon groups keyed by shape
    muon_shape_groups = {}
    for block in model.blocks:
        for linear in [block.attn.c_q, block.attn.c_k, block.attn.c_v,
                       block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
            shape = linear.kernel.value.shape
            if shape not in muon_shape_groups:
                muon_shape_groups[shape] = []
            muon_shape_groups[shape].append(linear.kernel)

    return param_id_to_group, muon_shape_groups, adamw_config


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

# Optimization
TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 8
DEVICE_BATCH_SIZE = 64  # per-device batch size (TPU v4 has 32GB HBM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
key = jax.random.PRNGKey(42)

# TPU/device setup
num_devices = jax.device_count()
print(f"JAX devices: {num_devices} x {jax.devices()[0].platform}")
mesh = Mesh(np.array(jax.devices()), axis_names=('data',))
data_sharding = NamedSharding(mesh, P('data'))
replicated = NamedSharding(mesh, P())

# Peak FLOPS for MFU calculation
TPU_V4_BF16_PEAK_FLOPS = 275e12
SYSTEM_PEAK_FLOPS = TPU_V4_BF16_PEAK_FLOPS * num_devices

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

# Create model
model = GPT(config, rngs=nnx.Rngs(0))
key, init_key = jax.random.split(key)
init_weights(model, config, init_key)

param_counts = count_params(model, config)
print("Parameter counts:")
for k, v in param_counts.items():
    print(f"  {k:24s}: {v:,}")
num_params = param_counts['total']

window_sizes = compute_window_sizes(config)
num_flops_per_token = estimate_flops(config, window_sizes)
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Batch sizing
total_device_batch = DEVICE_BATCH_SIZE * num_devices
tokens_per_fwdbwd = total_device_batch * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, \
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by tokens_per_fwdbwd ({tokens_per_fwdbwd})"
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Build parameter registry
param_id_to_group, muon_shape_groups, adamw_config = build_param_registry(model, config)
dmodel_lr_scale = (config.n_embd / 768) ** -0.5
print(f"Scaling AdamW LRs by 1/sqrt({config.n_embd}/768) = {dmodel_lr_scale:.6f}")

# Initialize optimizer states
# AdamW states: keyed by id(variable)
adamw_states = {}
for pid, (group_name, var) in param_id_to_group.items():
    adamw_states[pid] = {
        'exp_avg': jnp.zeros_like(var.value),
        'exp_avg_sq': jnp.zeros_like(var.value),
        'step': 0,
    }

# Muon states: keyed by shape tuple
muon_states = {}
for shape, params_list in muon_shape_groups.items():
    n = len(params_list)
    stacked_shape = (n,) + shape
    # Flax kernel shape is (in, out) — PyTorch is (out, in).
    # "tall" in the Muon sense means out_features >= in_features,
    # which is shape[-1] >= shape[-2] in Flax convention.
    is_tall = shape[-1] >= shape[-2]
    if is_tall:
        # reduce along in dim (axis -2 in Flax (in, out)) -> smb_shape = (N, 1, out)
        smb_shape_tuple = (n, 1, shape[1])
    else:
        # reduce along out dim (axis -1 in Flax (in, out)) -> smb_shape = (N, in, 1)
        smb_shape_tuple = (n, shape[0], 1)
    muon_states[shape] = {
        'momentum_buffer': jnp.zeros(stacked_shape, dtype=jnp.float32),
        'second_momentum_buffer': jnp.zeros(smb_shape_tuple, dtype=jnp.float32),
    }

train_loader = make_dataloader(tokenizer, total_device_batch, MAX_SEQ_LEN, "train")

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Devices: {num_devices}, Device batch size: {DEVICE_BATCH_SIZE}, Total batch: {total_device_batch}")

# Schedules

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# JIT-compiled forward/backward
# ---------------------------------------------------------------------------

# Split model into: graphdef (static), params (nnx.Param), rest (nnx.Variable etc.)
graphdef, params, rest = nnx.split(model, nnx.Param, ...)

@jax.jit
def compute_grads(params, rest, x, y):
    """Forward + backward for a single micro-batch. Differentiates only params."""
    def loss_fn(params):
        mdl = nnx.merge(graphdef, params, rest)
        loss = mdl(x, y, reduction='mean')
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return grads, loss

@jax.jit
def eval_forward(params, rest, x, y):
    """Eval forward pass returning per-token losses."""
    mdl = nnx.merge(graphdef, params, rest)
    return mdl(x, y, reduction='none')

# ---------------------------------------------------------------------------
# Gradient extraction helper
# ---------------------------------------------------------------------------

def build_grad_map(params, grads):
    """
    Build a mapping from id(param_leaf) -> gradient array.
    Both params and grads are NNX State objects (nnx.Param filter) with identical structure.
    The param leaves are the same array objects as model.xxx.value for Param variables.
    """
    param_flat = jax.tree.leaves(params)
    grad_flat = jax.tree.leaves(grads)
    assert len(param_flat) == len(grad_flat), f"Leaf count mismatch: {len(param_flat)} vs {len(grad_flat)}"
    grad_map = {}
    for p_leaf, g_leaf in zip(param_flat, grad_flat):
        grad_map[id(p_leaf)] = g_leaf
    return grad_map

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    # Gradient accumulation
    accumulated_grads = None
    total_loss = 0.0

    for micro_step in range(grad_accum_steps):
        x_np, y_np, epoch = next(train_loader)
        x = jax.device_put(jnp.array(x_np), data_sharding)
        y = jax.device_put(jnp.array(y_np), data_sharding)

        grads, loss = compute_grads(params, rest, x, y)

        if accumulated_grads is None:
            accumulated_grads = jax.tree.map(lambda g: g / grad_accum_steps, grads)
        else:
            accumulated_grads = jax.tree.map(
                lambda a, g: a + g / grad_accum_steps, accumulated_grads, grads
            )
        total_loss += float(loss) / grad_accum_steps

    # Build gradient map: maps id(param_leaf) -> gradient_array
    grad_map = build_grad_map(params, accumulated_grads)

    # Compute schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    # --- AdamW updates ---
    for pid, (group_name, var) in param_id_to_group.items():
        cfg = adamw_config[group_name]
        lr = cfg['base_lr'] * lrm
        beta1, beta2 = cfg['betas']
        # Find gradient for this variable's current value
        grad_val = grad_map.get(id(var.value))
        if grad_val is None:
            continue
        new_val, new_ea, new_eas, new_step = adamw_update(
            var.value, grad_val,
            adamw_states[pid]['exp_avg'],
            adamw_states[pid]['exp_avg_sq'],
            adamw_states[pid]['step'],
            lr=lr, beta1=beta1, beta2=beta2, eps=cfg['eps'], wd=cfg['wd'],
        )
        var.value = new_val
        adamw_states[pid] = {'exp_avg': new_ea, 'exp_avg_sq': new_eas, 'step': new_step}

    # --- Muon updates ---
    for shape, params_list in muon_shape_groups.items():
        # Flax kernel: (in, out). "tall" = out >= in = shape[-1] >= shape[-2]
        is_tall = shape[-1] >= shape[-2]
        # LR scaling: max(1, out/in)^0.5 = max(1, shape[-1]/shape[-2])^0.5
        lr_scaled = MATRIX_LR * lrm * max(1.0, shape[-1] / shape[-2]) ** 0.5

        stacked_params = jnp.stack([p.value for p in params_list])
        stacked_grads = jnp.stack([grad_map[id(p.value)] for p in params_list])

        new_params, new_mb, new_smb = muon_update(
            stacked_params, stacked_grads,
            muon_states[shape]['momentum_buffer'],
            muon_states[shape]['second_momentum_buffer'],
            momentum=muon_momentum,
            lr=lr_scaled,
            wd=muon_weight_decay,
            beta2=0.95,
            is_tall=is_tall,
        )
        muon_states[shape] = {'momentum_buffer': new_mb, 'second_momentum_buffer': new_smb}

        for idx, p in enumerate(params_list):
            p.value = new_params[idx]

    # Re-split model state after param updates (rest is unchanged)
    _, params, rest = nnx.split(model, nnx.Param, ...)

    train_loss_f = total_loss

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    # Timing
    jax.block_until_ready(params)
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / SYSTEM_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
def eval_fn(x, y, reduction='none'):
    return eval_forward(params, rest, x, y)

val_bpb = evaluate_bpb(eval_fn, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / SYSTEM_PEAK_FLOPS if total_training_time > 0 else 0

try:
    peak_hbm_bytes = max(d.memory_stats()['peak_bytes_in_use'] for d in jax.local_devices())
    peak_hbm_mb = peak_hbm_bytes / 1024 / 1024
except Exception:
    peak_hbm_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_hbm_mb:      {peak_hbm_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
