import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

### Classification Heads


class LinearHead(nn.Module):
    """Linear probe — fast baseline with no hidden layers."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    """Two-layer MLP with LayerNorm, GELU activation, and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPHead(nn.Module):
    """Three-layer MLP with a residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(input_dim, hidden_dim)  # residual projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        r = self.proj(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = F.gelu(self.fc2(x) + r)
        x = self.drop(x)
        return self.fc3(x)


# Registry — maps head name (used in configs/CLI) to class
HEAD_REGISTRY: dict = {
    "mlp": MLPHead,
    "linear": LinearHead,
    "deep_mlp": DeepMLPHead,
}

### BiLSTM encoder


class BiLSTMEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out  # (B, T, 2H)


### ESIM model


class ESIMLSTMModel(nn.Module):
    """
    ESIM-style model with a BiLSTM encoder and swappable classification head.

    Args:
        embedding_size: input embedding dimensionality (e.g. 100 for GloVe-100)
        hidden_size:    BiLSTM hidden size per direction
        num_layers:     number of LSTM layers
        p_dropout:      dropout probability
        head:           key into HEAD_REGISTRY, or an already-instantiated nn.Module
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        p_dropout: float = 0.2,
        head: str = "mlp",
    ):
        super().__init__()
        D = 2 * hidden_size  

        self.encoder = BiLSTMEncoder(embedding_size, hidden_size, num_layers, p_dropout)
        self.projection = nn.Sequential(
            nn.Linear(4 * D, D), nn.GELU(), nn.Dropout(p_dropout)
        )
        self.composition = BiLSTMEncoder(D, hidden_size, num_layers, p_dropout)

        # head can be a string key or a pre-built module for full flexibility
        if isinstance(head, str):
            head_cls = HEAD_REGISTRY[head]
            self.classifier = head_cls(
                input_dim=4 * D, hidden_dim=hidden_size, dropout=p_dropout
            )
        else:
            self.classifier = head  

    def _padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        return (
            torch.arange(max_len, device=lengths.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

    def _mean_max_pool(
        self, h: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        mask = (
            torch.arange(h.size(1), device=h.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        mean = (h * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).float()
        max_ = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        return torch.cat([mean, max_], dim=-1)  # (B, 2D)

    def _co_attention(self, h_c, h_e, mask_c, mask_e):
        e = torch.bmm(h_c, h_e.transpose(1, 2))  # (B, T_c, T_e)

        attn_c = F.softmax(e.masked_fill(mask_e.unsqueeze(1), float("-inf")), dim=2)
        ctx_c = torch.bmm(attn_c, h_e)

        attn_e = F.softmax(
            e.transpose(1, 2).masked_fill(mask_c.unsqueeze(1), float("-inf")), dim=2
        )
        ctx_e = torch.bmm(attn_e, h_c)

        inter_c = torch.cat([h_c, ctx_c, h_c - ctx_c, h_c * ctx_c], dim=-1)
        inter_e = torch.cat([h_e, ctx_e, h_e - ctx_e, h_e * ctx_e], dim=-1)
        return inter_c, inter_e

    def forward(self, X: dict) -> torch.Tensor:
        claim, evidence = X["claim"], X["evidence"]
        c_lens, e_lens = X["claim_lens"], X["evidence_lens"]

        mask_c = self._padding_mask(c_lens, claim.size(1))
        mask_e = self._padding_mask(e_lens, evidence.size(1))

        h_c = self.encoder(claim, c_lens)
        h_e = self.encoder(evidence, e_lens)

        inter_c, inter_e = self._co_attention(h_c, h_e, mask_c, mask_e)

        comp_c = self.composition(self.projection(inter_c), c_lens)
        comp_e = self.composition(self.projection(inter_e), e_lens)

        pooled = torch.cat(
            [self._mean_max_pool(comp_c, c_lens), self._mean_max_pool(comp_e, e_lens)],
            dim=-1,
        )  # (B, 4D)

        return self.classifier(pooled)  # (B, 1)

## For ensembles 

class EnsembleModel(nn.Module):
    """
    Averages logits from multiple models. Weights default to uniform.

    Usage:
        ensemble = EnsembleModel([model_a, model_b, model_c])
        # or with custom weights:
        ensemble = EnsembleModel([model_a, model_b], weights=[0.6, 0.4])
    """

    def __init__(self, models: list, weights: list = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, X: dict) -> torch.Tensor:
        logits = torch.stack([m(X) for m in self.models], dim=0)  # (n_models, B, 1)
        return (logits * self.weights.view(-1, 1, 1)).sum(0)


def build_model(config: dict) -> nn.Module:
    """
    Build an ESIMLSTMModel from a config dict.

    Expected keys: embedding_size, hidden_size, num_layers
    Optional keys: dropout (default 0.2), head (default "mlp")
    """
    return ESIMLSTMModel(
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        p_dropout=config.get("dropout", 0.2),
        head=config.get("head", "mlp"),
    )
