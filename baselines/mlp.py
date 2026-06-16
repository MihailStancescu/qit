import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Position-blind MLP baseline.

    Flattens all token embeddings into a single vector, then applies a
    2-layer MLP. Has no mechanism for sequential or relational reasoning —
    it can only learn functions that are permutation-invariant by accident.

    Default config (embed_dim=2, hidden=8):
        Embedding(2,2)=4  +  Linear(8→8)=72  +  Linear(8→2)=18  =  94 params
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_tokens: int,
        embed_dim: int = 2,
        hidden: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(n_tokens * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)   # (batch, n_tokens, embed_dim)
        x = x.flatten(1)               # (batch, n_tokens * embed_dim)
        return self.net(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
