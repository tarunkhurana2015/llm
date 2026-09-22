# transformer_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Self-Attention Head
# ---------------------------------------------------------------------
# A single attention head lets the model look at other tokens in the sequence
# and decide which ones are relevant to the current token. The attention mask
# ensures that each token only sees the tokens before it (causal masking),
# which is the standard setup for autoregressive language models.


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        # Project each token into three learned views: keys, queries, and values.
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        # Lower triangular matrix: each position can only attend to itself and
        # earlier positions, never future ones.
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Attention scores compare every query against every key.
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)

        # Apply causal masking: future tokens become impossible by setting their
        # scores to negative infinity before softmax.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

# ---------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------
# Multiple heads let the model attend to different kinds of relationships at
# the same time. One head may focus on nearby words, while another tracks
# broader context or repeated patterns.


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(
            embedding_dim, block_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# ---------------------------------------------------------------------
# Feed Forward Network
# ---------------------------------------------------------------------
# After attention, each token is passed through a small MLP to mix information
# nonlinearly. This gives the block more expressive power than attention alone.


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------
# A full transformer block combines attention and a feed-forward network with
# residual connections and layer normalization. This is the basic building unit
# used in modern language models.


class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Residual connection around self-attention.
        x = x + self.sa(self.ln1(x))

        # Residual connection around feed-forward MLP.
        x = x + self.ffwd(self.ln2(x))
        return x
