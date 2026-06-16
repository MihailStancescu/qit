import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """
    Minimal classical Transformer classifier.

    Uses full self-attention over all token positions — the classical
    mechanism that QIT's interference-based attention aims to replace.
    Mean-pools the encoder output to produce a fixed-size representation.

    Even at minimum viable size (d_model=4, nhead=2, ffn=8, 1 layer),
    the Transformer has ~206 params — nearly 3× QIT's 78 — due to the
    overhead of four separate projection matrices per attention head.

    Default config:
        Embedding(2,4)=8  +  PosEmb(4,4)=16  +  EncoderLayer≈172  +  Linear(4→2)=10  =  206 params
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_tokens: int,
        d_model: int = 4,
        nhead: int = 2,
        n_layers: int = 1,
        dim_feedforward: int = 8,
    ):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(n_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = token_ids.shape
        pos = torch.arange(n_tokens, device=token_ids.device).unsqueeze(0)

        x = self.embedding(token_ids) + self.pos_embedding(pos)  # (batch, n_tokens, d_model)
        x = self.encoder(x)                                       # (batch, n_tokens, d_model)
        x = x.mean(dim=1)                                         # mean pool → (batch, d_model)
        return self.decoder(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
