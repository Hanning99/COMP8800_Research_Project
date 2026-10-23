"""
Simple word-level text tokeniser and vocabulary for SecureCourier.

The vocabulary is built from all instance surface strings seen during
offline data collection.  Unknown tokens map to <UNK>.

Sequences are padded/truncated to a fixed max_len.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD_ID = 0
UNK_ID = 1
_SPECIAL = ["<PAD>", "<UNK>"]


class Vocab:
    def __init__(self, max_size: int = 2000) -> None:
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(_SPECIAL)}
        self.id2token: List[str] = list(_SPECIAL)
        self.max_size = max_size

    # ---- tokenisation ----

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lower-case word split, keeping alphanumerics and common punctuation."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def add_text(self, text: str) -> None:
        """Add unseen tokens from *text* to vocabulary (up to max_size)."""
        for tok in self.tokenize(text):
            if tok not in self.token2id and len(self.token2id) < self.max_size:
                self.token2id[tok] = len(self.id2token)
                self.id2token.append(tok)

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        """Encode text to a padded/truncated list of token IDs."""
        ids = [self.token2id.get(t, UNK_ID) for t in self.tokenize(text)]
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    @property
    def size(self) -> int:
        return len(self.id2token)

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Neural text encoder
# ---------------------------------------------------------------------------

class TextEncoder(torch.nn.Module):
    """
    GRU-based encoder for variable-length token sequences.

    encode(token_ids: LongTensor[batch, seq_len]) -> FloatTensor[batch, hidden_dim]
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.gru   = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, L] → [B, hidden_dim]"""
        emb = self.embed(token_ids)           # [B, L, embed]
        _, h = self.gru(emb)                  # h: [1, B, hidden]
        return h.squeeze(0)                   # [B, hidden]


class ActionEncoder(torch.nn.Module):
    """
    Simple mean-pooling encoder for action display strings.

    encode(token_ids: LongTensor[B, N, L]) -> FloatTensor[B, N, hidden_dim]
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.proj  = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, N, L] → [B, N, hidden_dim]"""
        B, N, L = token_ids.shape
        flat    = token_ids.reshape(B * N, L)
        emb     = self.embed(flat)            # [B*N, L, embed]
        # mean pool over non-padding tokens
        mask    = (flat != PAD_ID).float().unsqueeze(-1)  # [B*N, L, 1]
        denom   = mask.sum(dim=1).clamp(min=1.0)           # [B*N, 1]
        pooled  = (emb * mask).sum(dim=1) / denom          # [B*N, embed]
        projected = self.proj(pooled)                       # [B*N, hidden]
        return projected.view(B, N, -1)                    # [B, N, hidden]
