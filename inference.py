import jax
import jax.numpy as jnp
from jax import nn
import numpy as np
from typing import Dict, Tuple, Optional, List, NamedTuple
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import functools

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class QwenConfig:
    """Configuration for Qwen2.5 0.5B"""
    def __init__(self):
        self.vocab_size = 151936
        self.hidden_dim = 896
        self.num_layers = 24
        self.num_query_heads = 14
        self.num_key_value_heads = 2
        self.intermediate_dim = 4864
        self.rope_theta = 1000000.0
        self.layer_norm_epsilon = 1e-6
        self.max_seq_length = 32768


def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(tensor.detach().cpu().numpy())


class AttentionWeights(NamedTuple):
    """Weights for attention layer."""
    q_proj: jnp.ndarray
    k_proj: jnp.ndarray
    v_proj: jnp.ndarray
    o_proj: jnp.ndarray
    q_bias: Optional[jnp.ndarray]
    k_bias: Optional[jnp.ndarray]
    v_bias: Optional[jnp.ndarray]


class MLPWeights(NamedTuple):
    """Weights for MLP layer."""
    gate_proj: jnp.ndarray
    up_proj: jnp.ndarray
    down_proj: jnp.ndarray


class LayerWeights(NamedTuple):
    """Weights for a single transformer layer."""
    attention: AttentionWeights
    mlp: MLPWeights
    ln1_weight: jnp.ndarray
    ln2_weight: jnp.ndarray


class ModelWeights(NamedTuple):
    """All model weights as a pytree."""
    embed_tokens: jnp.ndarray
    layers: Tuple[LayerWeights, ...]
    norm_weight: jnp.ndarray
    lm_head: jnp.ndarray


def apply_rotary_embedding(x, seq_len, rope_theta=1000000.0):
    """Apply rotary position embeddings to input tensors."""
    batch_size, num_heads, seq_length, head_dim = x.shape

    position_ids = jnp.arange(seq_len)

    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    freqs = jnp.outer(position_ids, inv_freq)

    emb = jnp.concatenate([freqs, freqs], axis=-1)
    
    cos_cached = jnp.cos(emb)
    sin_cached = jnp.sin(emb)

    cos_cached = cos_cached[None, None, :, :]
    sin_cached = sin_cached[None, None, :, :]

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotate_half = jnp.concatenate([-x2, x1], axis=-1)

    x_out = x * cos_cached + x_rotate_half * sin_cached
    
    return x_out


def rms_norm(x, weight, epsilon=1e-6):
    """Root Mean Square Layer Normalization."""
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + epsilon)
    return x * weight


def attention_forward(query, key, value, mask=None):
    """Multi-head attention mechanism."""
    batch_size, num_heads, seq_len, head_dim = query.shape

    scores = jnp.einsum('bhqd,bhkd->bhqk', query, key) / jnp.sqrt(head_dim)

    if mask is not None:
        scores = scores + mask

    attention_weights = nn.softmax(scores, axis=-1)

    output = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, value)

    return output


def mlp_forward(x, mlp_weights: MLPWeights):
    """Feed-forward network computation."""
    gate = jnp.dot(x, mlp_weights.gate_proj)
    up = jnp.dot(x, mlp_weights.up_proj)
    gate = gate * nn.sigmoid(gate)
    hidden = gate * up
    output = jnp.dot(hidden, mlp_weights.down_proj)
    return output


def attention_forward_layer(hidden_states, attn_weights: AttentionWeights, config, mask=None):
    """Multi-head attention layer computation."""
    batch_size, seq_len = hidden_states.shape[:2]
    head_dim = config.hidden_dim // config.num_query_heads

    query = jnp.dot(hidden_states, attn_weights.q_proj)
    if attn_weights.q_bias is not None:
        query = query + attn_weights.q_bias

    key = jnp.dot(hidden_states, attn_weights.k_proj)
    if attn_weights.k_bias is not None:
        key = key + attn_weights.k_bias

    value = jnp.dot(hidden_states, attn_weights.v_proj)
    if attn_weights.v_bias is not None:
        value = value + attn_weights.v_bias

    query = query.reshape(batch_size, seq_len, config.num_query_heads, head_dim)
    key = key.reshape(batch_size, seq_len, config.num_key_value_heads, head_dim)
    value = value.reshape(batch_size, seq_len, config.num_key_value_heads, head_dim)

    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))

    query = apply_rotary_embedding(query, seq_len, config.rope_theta)
    key = apply_rotary_embedding(key, seq_len, config.rope_theta)

    if config.num_key_value_heads < config.num_query_heads:
        repeat_factor = config.num_query_heads // config.num_key_value_heads
        key = jnp.repeat(key, repeat_factor, axis=1)
        value = jnp.repeat(value, repeat_factor, axis=1)

    attn_output = attention_forward(query, key, value, mask)

    attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
    attn_output = attn_output.reshape(batch_size, seq_len, config.hidden_dim)

    output = jnp.dot(attn_output, attn_weights.o_proj)

    return output


def decoder_layer_forward(hidden_states, layer_weights: LayerWeights, config, mask=None):
    """Single transformer decoder layer computation."""

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights.ln1_weight, config.layer_norm_epsilon)

    hidden_states = attention_forward_layer(hidden_states, layer_weights.attention, config, mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights.ln2_weight, config.layer_norm_epsilon)

    hidden_states = mlp_forward(hidden_states, layer_weights.mlp)
    hidden_states = residual + hidden_states

    return hidden_states


def model_forward(input_ids, weights: ModelWeights, config):
    """Full model forward pass - pure function."""

    hidden_states = weights.embed_tokens[input_ids]

    seq_len = input_ids.shape[1]
    causal_mask = jnp.triu(jnp.ones((seq_len, seq_len)) * -1e9, k=1)
    causal_mask = causal_mask[None, None, :, :]

    for layer_weights in weights.layers:
        hidden_states = decoder_layer_forward(hidden_states, layer_weights, config, causal_mask)

    hidden_states = rms_norm(hidden_states, weights.norm_weight, config.layer_norm_epsilon)

    logits = jnp.dot(hidden_states, weights.lm_head.T)

    return logits


def load_qwen_from_hf(model_name="Qwen/Qwen2.5-0.5B"):
    """Load Qwen model weights from Hugging Face into pytree structure."""
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = QwenConfig()
    config.vocab_size = torch_model.config.vocab_size
    config.hidden_dim = torch_model.config.hidden_size
    config.num_layers = torch_model.config.num_hidden_layers
    config.num_query_heads = torch_model.config.num_attention_heads
    config.num_key_value_heads = torch_model.config.num_key_value_heads
    config.intermediate_dim = torch_model.config.intermediate_size
    config.rope_theta = torch_model.config.rope_theta
    config.layer_norm_epsilon = torch_model.config.rms_norm_eps

    embed_tokens = torch_to_jax(torch_model.model.embed_tokens.weight)

    layer_weights_list = []
    for i in range(config.num_layers):
        torch_layer = torch_model.model.layers[i]

        attn_weights = AttentionWeights(
            q_proj=torch_to_jax(torch_layer.self_attn.q_proj.weight.T),
            k_proj=torch_to_jax(torch_layer.self_attn.k_proj.weight.T),
            v_proj=torch_to_jax(torch_layer.self_attn.v_proj.weight.T),
            o_proj=torch_to_jax(torch_layer.self_attn.o_proj.weight.T),
            q_bias=torch_to_jax(torch_layer.self_attn.q_proj.bias) if torch_layer.self_attn.q_proj.bias is not None else None,
            k_bias=torch_to_jax(torch_layer.self_attn.k_proj.bias) if torch_layer.self_attn.k_proj.bias is not None else None,
            v_bias=torch_to_jax(torch_layer.self_attn.v_proj.bias) if torch_layer.self_attn.v_proj.bias is not None else None,
        )

        mlp_weights = MLPWeights(
            gate_proj=torch_to_jax(torch_layer.mlp.gate_proj.weight.T),
            up_proj=torch_to_jax(torch_layer.mlp.up_proj.weight.T),
            down_proj=torch_to_jax(torch_layer.mlp.down_proj.weight.T),
        )

        layer_weights = LayerWeights(
            attention=attn_weights,
            mlp=mlp_weights,
            ln1_weight=torch_to_jax(torch_layer.input_layernorm.weight),
            ln2_weight=torch_to_jax(torch_layer.post_attention_layernorm.weight),
        )
        layer_weights_list.append(layer_weights)

    norm_weight = torch_to_jax(torch_model.model.norm.weight)
    lm_head = torch_to_jax(torch_model.lm_head.weight)

    weights = ModelWeights(
        embed_tokens=embed_tokens,
        layers=tuple(layer_weights_list),
        norm_weight=norm_weight,
        lm_head=lm_head,
    )

    return weights, tokenizer, config


@functools.partial(jax.jit, static_argnums=(2,))
def compiled_next_token_greedy(input_ids, weights: ModelWeights, config):
    """JIT-compiled greedy next token prediction."""
    logits = model_forward(input_ids, weights, config)
    last_logits = logits[0, -1, :]
    return jnp.argmax(last_logits)


@functools.partial(jax.jit, static_argnums=(2, 4))
def compiled_next_token_sample(input_ids, weights: ModelWeights, config, key, temperature):
    """JIT-compiled sampling next token prediction."""
    logits = model_forward(input_ids, weights, config)
    last_logits = logits[0, -1, :] / temperature
    log_probs = jax.nn.log_softmax(last_logits)
    return jax.random.categorical(key, log_probs)


def generate_text(weights, config, tokenizer, prompt, max_new_tokens=50, temperature=1.0, do_sample=False):

    input_ids = tokenizer(prompt, return_tensors="np").input_ids

    generated_ids = input_ids[0].tolist()

    for i in range(max_new_tokens):
        current_ids = jnp.array([generated_ids])

        if do_sample and temperature > 0:
            key = jax.random.PRNGKey(i)
            next_token = compiled_next_token_sample(current_ids, weights, config, key, temperature)
        else:
            next_token = compiled_next_token_greedy(current_ids, weights, config)

        generated_ids.append(int(next_token))

        if next_token == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def main():

    weights, tokenizer, config = load_qwen_from_hf("Qwen/Qwen2.5-0.5B")

    prompts = [
        "Once upon a time",
        "Sorry, I can't assist with that request because",
    ]

    for prompt in prompts:
        generated = generate_text(weights, config, tokenizer, prompt, max_new_tokens=30, do_sample=False)
        print(f"Generated: {generated}")



if __name__ == "__main__":
    main()