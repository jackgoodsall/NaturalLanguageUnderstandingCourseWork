"""
models.py — Neural network architectures for Solution B.

Contains:
  - Classification heads (Linear, MLP, DeepMLP) with a registry for easy swapping
  - BiLSTM encoder that handles variable-length sequences via packing
  - ESIMLSTMModel: full ESIM-style model with co-attention and composition
  - EnsembleModel: averages logits from multiple models
  - build_model(): factory function to construct a model from a config dict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ===========================================================================
# Classification Heads
# ===========================================================================
# Each head takes a fixed-size pooled vector and outputs a single logit (B, 1).
# They share a common __init__ signature so they can be swapped via the registry.


class LinearHead(nn.Module):
    """Linear probe — single linear layer, no hidden layers.
    Useful as a fast baseline to check if the encoder representations are
    already linearly separable."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    """Two-layer MLP with LayerNorm, GELU activation, and dropout.
    Default classification head — provides a good balance of capacity and
    regularisation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),      # normalise the pooled representation
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),                    # smooth activation (works well with LN)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),     # final logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPHead(nn.Module):
    """Three-layer MLP with a residual (skip) connection.
    The residual path projects the input to hidden_dim and is added after the
    second linear layer, which helps gradient flow in deeper heads."""

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
        r = self.proj(x)                  # residual branch
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = F.gelu(self.fc2(x) + r)       # add residual after second layer
        x = self.drop(x)
        return self.fc3(x)


# Registry — maps head name (used in configs / CLI flags) to its class.
# To add a new head, define the class above and register it here.
HEAD_REGISTRY: dict = {
    "mlp": MLPHead,
    "linear": LinearHead,
    "deep_mlp": DeepMLPHead,
}


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM that handles variable-length sequences.

    Uses pack_padded_sequence / pad_packed_sequence so the LSTM only
    processes real tokens and ignores padding, which is both more efficient
    and avoids the model learning from pad values.

    Args:
        input_size:  dimensionality of each input timestep (e.g. embedding dim).
        hidden_size: LSTM hidden size per direction (output is 2 * hidden_size).
        num_layers:  number of stacked LSTM layers.
        dropout:     dropout between LSTM layers (only active when num_layers > 1).
    """

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
        """
        Args:
            x:       (B, T, input_size) padded input tensor.
            lengths: (B,) actual sequence lengths.

        Returns:
            (B, T, 2*hidden_size) — output hidden states at every timestep.
        """
        # Pack to skip padding positions inside the LSTM
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        # Unpack back to padded tensor form
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out  # (B, T, 2H)


class ESIMLSTMModel(nn.Module):
    """ESIM-style model for claim-evidence pair classification.

    Architecture (following Chen et al., 2017 — "Enhanced LSTM for NLI"):
      1. **Input encoding**: a shared BiLSTM encodes claim and evidence separately.
      2. **Co-attention**: soft-aligns claim and evidence, producing interaction
         features [h, ctx, h-ctx, h*ctx] for each side.
      3. **Projection**: a linear layer reduces the 4xD interaction vectors to D.
      4. **Composition**: a second BiLSTM reads the projected interactions.
      5. **Pooling**: mean + max pooling over both composed sequences → (B, 4D).
      6. **Classification**: a swappable head maps the pooled vector to a logit.

    Args:
        embedding_size: input embedding dimensionality (e.g. 300 for FastText-300).
        hidden_size:    BiLSTM hidden size per direction (D = 2 * hidden_size).
        num_layers:     number of LSTM layers in each encoder.
        p_dropout:      dropout probability applied in projection and head.
        head:           key into HEAD_REGISTRY, or an already-instantiated nn.Module.
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
        D = 2 * hidden_size  # BiLSTM output dim (forward + backward)

        # Stage 1: shared encoder for both claim and evidence
        self.encoder = BiLSTMEncoder(embedding_size, hidden_size, num_layers, p_dropout)

        # Stage 3: project the 4D interaction features back down to D
        self.projection = nn.Sequential(
            nn.Linear(4 * D, D), nn.GELU(), nn.Dropout(p_dropout)
        )

        # Stage 4: composition encoder (reads projected interactions)
        self.composition = BiLSTMEncoder(D, hidden_size, num_layers, p_dropout)

        # Stage 6: classification head (string key → look up in registry)
        if isinstance(head, str):
            head_cls = HEAD_REGISTRY[head]
            self.classifier = head_cls(
                input_dim=4 * D, hidden_dim=hidden_size, dropout=p_dropout
            )
        else:
            self.classifier = head  # allow pre-built module for full flexibility

    # ----- helper methods -----

    def _padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create a boolean mask that is True for padding positions.
        Used by co-attention to prevent attending to pad tokens."""
        return (
            torch.arange(max_len, device=lengths.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

    def _mean_max_pool(
        self, h: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute both mean-pool and max-pool over the time dimension (ignoring
        padding) and concatenate them.  Returns (B, 2D)."""
        mask = (
            torch.arange(h.size(1), device=h.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        # Mean pool: sum real positions, divide by length
        mean = (h * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).float()
        # Max pool: fill padding with -inf so it's never the max
        max_ = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        return torch.cat([mean, max_], dim=-1)  # (B, 2D)

    def _co_attention(self, h_c, h_e, mask_c, mask_e):
        """Compute soft co-attention between claim (h_c) and evidence (h_e).

        For each claim token, produces an attended evidence context (and vice versa).
        Returns enhanced representations: [original, context, difference, product]
        for both claim and evidence sides.

        Args:
            h_c:    (B, T_c, D) encoded claim hidden states.
            h_e:    (B, T_e, D) encoded evidence hidden states.
            mask_c: (B, T_c) boolean padding mask for claim.
            mask_e: (B, T_e) boolean padding mask for evidence.

        Returns:
            inter_c: (B, T_c, 4D) enhanced claim representation.
            inter_e: (B, T_e, 4D) enhanced evidence representation.
        """
        # Raw attention scores: dot product between all claim–evidence token pairs
        e = torch.bmm(h_c, h_e.transpose(1, 2))  # (B, T_c, T_e)

        # Claim-to-evidence attention: for each claim token, softmax over evidence
        attn_c = F.softmax(e.masked_fill(mask_e.unsqueeze(1), float("-inf")), dim=2)
        ctx_c = torch.bmm(attn_c, h_e)  # weighted evidence context for each claim token

        # Evidence-to-claim attention: for each evidence token, softmax over claim
        attn_e = F.softmax(
            e.transpose(1, 2).masked_fill(mask_c.unsqueeze(1), float("-inf")), dim=2
        )
        ctx_e = torch.bmm(attn_e, h_c)  # weighted claim context for each evidence token

        # Build interaction features: [original, context, difference, element-wise product]
        # These four components capture different aspects of the alignment
        inter_c = torch.cat([h_c, ctx_c, h_c - ctx_c, h_c * ctx_c], dim=-1)
        inter_e = torch.cat([h_e, ctx_e, h_e - ctx_e, h_e * ctx_e], dim=-1)
        return inter_c, inter_e

    # ----- forward pass -----

    def forward(self, X: dict) -> torch.Tensor:
        """
        Args:
            X: dict with keys 'claim', 'evidence' (padded tensors),
               'claim_lens', 'evidence_lens' (original lengths).

        Returns:
            (B, 1) raw logits (apply sigmoid for probabilities).
        """
        claim, evidence = X["claim"], X["evidence"]
        c_lens, e_lens = X["claim_lens"], X["evidence_lens"]

        # Build padding masks for attention masking
        mask_c = self._padding_mask(c_lens, claim.size(1))
        mask_e = self._padding_mask(e_lens, evidence.size(1))

        # Stage 1: encode claim and evidence independently
        h_c = self.encoder(claim, c_lens)
        h_e = self.encoder(evidence, e_lens)

        # Stage 2: co-attention to capture cross-sequence interactions
        inter_c, inter_e = self._co_attention(h_c, h_e, mask_c, mask_e)

        # Stage 3+4: project interaction features and compose with second BiLSTM
        comp_c = self.composition(self.projection(inter_c), c_lens)
        comp_e = self.composition(self.projection(inter_e), e_lens)

        # Stage 5: pool both sequences and concatenate → fixed-size vector (B, 4D)
        pooled = torch.cat(
            [self._mean_max_pool(comp_c, c_lens), self._mean_max_pool(comp_e, e_lens)],
            dim=-1,
        )  # (B, 4D)

        # Stage 6: classify
        return self.classifier(pooled)  # (B, 1)


class EnsembleModel(nn.Module):
    """Averages logits from multiple independently trained models.

    Useful for reducing variance and improving robustness by combining
    predictions from models trained with different hyperparameters or seeds.

    Args:
        models:  list of nn.Module instances (e.g. several ESIMLSTMModel).
        weights: optional list of floats for weighted averaging.
                 Defaults to uniform weights (1/N each).

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
        # Store weights as a buffer (not a parameter — no gradients needed)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, X: dict) -> torch.Tensor:
        # Stack logits from all models: (n_models, B, 1)
        logits = torch.stack([m(X) for m in self.models], dim=0)
        # Weighted average across models
        return (logits * self.weights.view(-1, 1, 1)).sum(0)


def build_model(config: dict) -> nn.Module:
    """Construct an ESIMLSTMModel from a configuration dictionary.

    Args:
        config: dict with required keys:
            - embedding_size (int): input embedding dimension
            - hidden_size (int): BiLSTM hidden size per direction
            - num_layers (int): number of LSTM layers
          and optional keys:
            - dropout (float): dropout probability (default 0.2)
            - head (str): classification head type (default "mlp")

    Returns:
        An ESIMLSTMModel ready for training.
    """
    return ESIMLSTMModel(
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        p_dropout=config.get("dropout", 0.2),
        head=config.get("head", "mlp"),
    )
