import torch

from einops import rearrange
from torch import nn

import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    self.attention_type = getattr(config, "attention_type", "full")
    self.attention_window = getattr(config, "attention_window", 32)
    self.linformer_k = getattr(config, "linformer_k", 64)

    if self.attention_type == "linformer":
      self.linformer_key = nn.Linear(config.max_position_embeddings, self.linformer_k)
      self.linformer_value = nn.Linear(config.max_position_embeddings, self.linformer_k)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key:   [bs, num_heads, seq_len, head_dim]
    query: [bs, num_heads, seq_len, head_dim]
    value: [bs, num_heads, seq_len, head_dim]
    attention_mask: [bs, 1, 1, seq_len]

    output: [bs, seq_len, hidden_size]
    """

    # flash version
    # using PyTorch optimized scaled dot-product attention instead of CUDA self written stuff
    if self.attention_type == "flash":
      seq_len = query.size(-2)

      # causal mask: future positions get -inf
      causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
        diagonal=1
      )

      causal_bias = torch.zeros(seq_len, seq_len, device=query.device, dtype=query.dtype)
      causal_bias = causal_bias.masked_fill(causal_mask, float("-inf"))

      # attention_mask: [bs, 1, 1, seq_len]
      # causal_bias:    [seq_len, seq_len]
      # combined mask broadcasts to [bs, heads, seq_len, seq_len]
      combined_mask = attention_mask.to(dtype=query.dtype) + causal_bias

      context = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=combined_mask,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=False
      )

      context = rearrange(context, "b h t d -> b t (h d)")
      return context

    if self.attention_type == "linformer":
      b, h, t, d = query.shape
      k_proj = min(self.linformer_k, t)

      # key/value: [b, h, t, d] -> [b, h, d, t]
      key_t = key.transpose(-1, -2)
      value_t = value.transpose(-1, -2)

      # use only first t input weights and first k_proj output dimensions
      E = self.linformer_key.weight[:k_proj, :t]  # [k_proj, t]
      F_proj = self.linformer_value.weight[:k_proj, :t]

      # project sequence length t -> k_proj
      key_proj = torch.matmul(key_t, E.T).transpose(-1, -2)
      value_proj = torch.matmul(value_t, F_proj.T).transpose(-1, -2)

      # key_proj/value_proj: [b, h, k_proj, d]
      attn_scores = torch.matmul(query, key_proj.transpose(-1, -2))
      attn_scores = attn_scores / (self.attention_head_size ** 0.5)

      attn_probs = torch.softmax(attn_scores, dim=-1)
      attn_probs = self.dropout(attn_probs)

      context = torch.matmul(attn_probs, value_proj)
      context = rearrange(context, "b h t d -> b t (h d)")
      return context

    # manual, full / sliding window attention
    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_scores = attn_scores / (self.attention_head_size ** 0.5)

    seq_len = query.size(-2)

    # causal mask: block future tokens
    causal_mask = torch.triu(
      torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
      diagonal=1
    )
    attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # padding mask
    attn_scores = attn_scores + attention_mask

    # sliding window mask, token i can only attend to tokens in [i-window, i]
    if self.attention_type == "sliding":
      w = self.attention_window

      positions = torch.arange(seq_len, device=query.device)
      i = positions[:, None]
      j = positions[None, :]

      sliding_mask = (i - j) > w
      sliding_mask = sliding_mask[None, None, :, :]  # [1, 1, seq_len, seq_len]

      attn_scores = attn_scores.masked_fill(sliding_mask, float("-inf"))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = self.dropout(attn_probs)

    context = torch.matmul(attn_probs, value)
    context = rearrange(context, "b h t d -> b t (h d)")

    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
