import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    GRU-based sequence classifier.

    Processes tokens left-to-right; the final hidden state is decoded to
    class logits. Has explicit sequential inductive bias, which is helpful
    for order-dependent tasks but provides no theoretical advantage on parity
    (a permutation-invariant function).

    Default config (embed_dim=2, hidden=4):
        Embedding(2,2)=4  +  GRU(2→4)=96  +  Linear(4→2)=10  =  110 params
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_tokens: int,          # unused at runtime, kept for interface parity
        embed_dim: int = 2,
        hidden: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden, num_layers=n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)       # (batch, n_tokens, embed_dim)
        _, h = self.rnn(x)                  # h: (n_layers, batch, hidden)
        return self.decoder(h[-1])          # (batch, n_classes)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
