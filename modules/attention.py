import torch

from einops import rearrange
from torch import nn


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

    # 1. Raw attention scores: Q K^T
    # shape: [bs, num_heads, seq_len, seq_len]
    attn_scores = torch.matmul(query, key.transpose(-1, -2))

    # 2. Scale by sqrt(d_k)
    attn_scores = attn_scores / (self.attention_head_size ** 0.5)

    # 3. Add attention mask
    # Usually attention_mask contains 0 for keep and large negative values for mask.
    attn_scores = attn_scores + attention_mask

    # 4. Causal mask: prevent attending to future tokens
    seq_len = query.size(-2)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
        diagonal=1
    )

    attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # 5. Softmax over keys
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # 6. Dropout on attention probabilities
    attn_probs = self.dropout(attn_probs)

    # 7. Weighted sum of values
    # shape: [bs, num_heads, seq_len, head_dim]
    context = torch.matmul(attn_probs, value)

    # 8. Put heads back together
    # [bs, num_heads, seq_len, head_dim] -> [bs, seq_len, hidden_size]
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
