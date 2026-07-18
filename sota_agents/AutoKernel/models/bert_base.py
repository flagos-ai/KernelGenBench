# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal BERT-base implementation for AutoKernel profiling.

Self-contained -- no transformers library needed.

Usage:
    uv run profile.py --model models/bert_base.py --class-name BertModel --input-shape 8,512 --dtype float16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.dropout(self.out_proj(y))


class BertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense2(self.gelu(self.dense1(x))))


class BertLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x


class BertModel(nn.Module):
    """
    BERT-base: hidden_size=768, num_layers=12, num_heads=12, intermediate=3072 (110M params).
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        self.pooler = nn.Linear(hidden_size, hidden_size)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"BertModel: {n_params / 1e6:.1f}M parameters")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        token_types = torch.zeros_like(input_ids)

        x = self.word_embeddings(input_ids) + self.position_embeddings(positions) + self.token_type_embeddings(token_types)
        x = self.embed_dropout(self.embed_norm(x))

        for layer in self.layers:
            x = layer(x)

        # Pooled output from [CLS] token
        pooled = torch.tanh(self.pooler(x[:, 0]))
        return pooled
