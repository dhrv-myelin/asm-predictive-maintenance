"""
models/architectures/mamba.py
==============================
Pure PyTorch model definitions. No training logic, no data handling.
Import these classes in trainers and inferencers.

Classes
-------
SimplifiedMambaBlock   Core SSM block with conv + gated SSM scan.
Mamba_TS               Full time-series model: embedding → N blocks →
                       reconstruction head + prediction head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedMambaBlock(nn.Module):
    """
    A single Mamba (State Space Model) block.

    Architecture:
        input → in_proj (split into x_ssm, x_gate)
              → depthwise Conv1d on x_ssm
              → SiLU activation
              → SSM scan (discretised A, B, C matrices)
              → gate multiplication with SiLU(x_gate)
              → out_proj
              → LayerNorm + residual

    Parameters
    ----------
    d_model : int   Model / embedding dimension.
    d_state : int   SSM state dimension (higher = more memory capacity).
    d_conv  : int   Depthwise conv kernel size (local context window).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)

        # Depthwise conv — groups=d_model means each channel is independent
        self.conv1d = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )

        # Input-dependent B and C projections (selective SSM)
        self.x_to_BC = nn.Linear(d_model, d_state * 2)
        self.x_to_dt = nn.Linear(d_model, d_model)

        # Learnable SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        out : (batch, seq_len, d_model)   same shape, residual applied.
        """
        batch, seq_len, _ = x.shape

        # Project and split into SSM path and gate path
        x_ssm, x_gate = self.in_proj(x).chunk(2, dim=-1)

        # Depthwise conv: transpose to (batch, d_model, seq_len)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters (input-dependent)
        BC = self.x_to_BC(x_conv)
        B, C = BC.chunk(2, dim=-1)  # (batch, seq_len, d_state) each
        dt = F.softplus(self.x_to_dt(x_conv))  # (batch, seq_len, d_model)

        # Continuous-to-discrete A matrix
        A = -torch.exp(self.A_log)  # (d_model, d_state) — negative = stable

        y = self._ssm_scan(x_conv, dt, A, B, C)

        # Gate + project
        y = y * F.silu(x_gate)
        out = self.out_proj(y)

        return self.norm(out + x)  # residual connection

    def _ssm_scan(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        dt: torch.Tensor,  # (batch, seq_len, d_model)
        A: torch.Tensor,  # (d_model, d_state)
        B: torch.Tensor,  # (batch, seq_len, d_state)
        C: torch.Tensor,  # (batch, seq_len, d_state)
    ) -> torch.Tensor:
        """
        Sequential SSM recurrence.
        h_t = dA * h_{t-1} + dB * x_t
        y_t = C_t · h_t + D * x_t
        """
        batch, seq_len, d_model = x.shape
        h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # Discretise: Euler zero-order hold
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A)  # (batch, d_model, d_state)
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(
                1
            )  # (batch, d_model, d_state)

            # State update
            h = h * dA + dB * x[:, t, :].unsqueeze(-1)

            # Output projection
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (batch, d_model)
            y_t = y_t + x[:, t, :] * self.D
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class Mamba_TS(nn.Module):
    """
    Full Mamba model for time-series forecasting.

    Architecture:
        scalar input → linear embedding → N × MambaBlock → dropout
            ├── reconstruction head : predicts input sequence (self-supervised aux task)
            └── prediction head     : predicts next step value

    The dual-head design allows the model to learn both representation
    quality (reconstruction) and forecasting (prediction), weighted by
    loss_weight_recon during training.

    Parameters
    ----------
    input_size   : int   Number of input features (width of X).
    seq_len      : int   Input sequence length.
    hidden_dim   : int   Internal model dimension.
    d_state      : int   SSM state dimension.
    d_conv       : int   Conv kernel size.
    num_layers   : int   Number of stacked MambaBlocks.
    dropout      : float Dropout probability.
    forecast_horizon : int   Number of future steps to predict (multi-step).
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.name = "Mamba-SSM"
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Project multi-feature input into hidden_dim
        self.input_proj = nn.Linear(input_size, hidden_dim)

        self.layers = nn.ModuleList(
            [
                SimplifiedMambaBlock(d_model=hidden_dim, d_state=d_state, d_conv=d_conv)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Reconstruction head (auxiliary self-supervised task)
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_size),  # reconstructs all features
        )

        # Prediction head (primary task)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        recon : (batch, seq_len, input_size)   reconstruction of input
        pred  : (batch, forecast_horizon)       predicted future values
        """
        embedded = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        hidden = embedded
        for layer in self.layers:
            hidden = layer(hidden)
            hidden = self.dropout(hidden)

        recon = self.recon_head(hidden)  # (batch, seq_len, input_size)
        pred = self.pred_head(hidden[:, -1, :])  # (batch, forecast_horizon)

        return recon, pred

    def get_config(self) -> dict:
        """Return architecture config dict — logged as MLflow artifact."""
        return {
            "model_class": "Mamba_TS",
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "forecast_horizon": self.forecast_horizon,
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
