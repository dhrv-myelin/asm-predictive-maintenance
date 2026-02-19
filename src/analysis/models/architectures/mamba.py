"""
models/architectures/mamba.py
───────────────────────────────
Pure PyTorch nn.Module definitions for the Mamba SSM architecture.
No training loop, no data handling, no MLflow — just the model graph.

Classes
-------
SimplifiedMambaBlock
    Single Mamba layer: input projection → depthwise conv → SSM scan → gate → output proj.
    Implements the selective state-space mechanism from the Mamba paper
    (Gu & Dao, 2023) in a simplified but faithful form.

Mamba_TS
    Full time-series model stacking N SimplifiedMambaBlocks.
    Has two output heads:
      recon_head : reconstructs the input sequence (auxiliary loss)
      pred_head  : predicts the next single step (primary loss)

Both classes are directly ported from the validated Colab implementation,
with minor cleanups (no print statements, device-agnostic).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedMambaBlock(nn.Module):
    """
    Single Mamba SSM block.

    Parameters
    ----------
    d_model : int
        Model dimension (= hidden_dim in Mamba_TS).
    d_state : int
        SSM state dimension. Higher = richer memory, more compute.
    d_conv  : int
        Depthwise conv kernel size. Must be ≥ 2.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Split projection: half for SSM path, half for gate
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # Depthwise causal conv along sequence dimension
        self.conv1d = nn.Conv1d(
            d_model, d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )

        # Input-dependent SSM parameters (selective mechanism)
        self.x_to_BC = nn.Linear(d_model, d_state * 2)   # B and C matrices
        self.x_to_dt = nn.Linear(d_model, d_model)        # timestep ∆

        # Learnable state-space matrices
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))  # log(-A)
        self.D     = nn.Parameter(torch.ones(d_model))             # skip connection

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        out : (batch, seq_len, d_model)  — same shape, residual added and normalised
        """
        batch, seq_len, _ = x.shape

        # Gate split
        x_proj          = self.in_proj(x)
        x_ssm, x_gate   = x_proj.chunk(2, dim=-1)

        # Causal depthwise conv (trim to seq_len to remove acausal padding)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Input-selective SSM parameters
        BC      = self.x_to_BC(x_conv)
        B, C    = BC.chunk(2, dim=-1)                   # (batch, seq_len, d_state)
        dt      = F.softplus(self.x_to_dt(x_conv))      # (batch, seq_len, d_model)

        A = -torch.exp(self.A_log)                       # (d_model, d_state)

        # Sequential SSM scan (exact discretised ZOH)
        y = self._ssm_scan(x_conv, dt, A, B, C)

        # Gated output
        y   = y * F.silu(x_gate)
        out = self.out_proj(y)

        return self.norm(out + x)

    def _ssm_scan(
        self,
        x  : torch.Tensor,   # (batch, seq_len, d_model)
        dt : torch.Tensor,   # (batch, seq_len, d_model)
        A  : torch.Tensor,   # (d_model, d_state)
        B  : torch.Tensor,   # (batch, seq_len, d_state)
        C  : torch.Tensor,   # (batch, seq_len, d_state)
    ) -> torch.Tensor:
        """
        Discretised zero-order-hold (ZOH) SSM scan.
        Runs sequentially over time steps; each step updates the hidden state h.

        State update:  h_t = h_{t-1} * exp(∆_t ⊙ A) + ∆_t ⊙ B_t ⊙ x_t
        Output:        y_t = (h_t · C_t) + D ⊙ x_t

        Returns
        -------
        torch.Tensor : (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        h       = torch.zeros(batch, d_model, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A)          # (batch, d_model, d_state)
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (batch, d_model, d_state)

            h = h * dA + dB * x[:, t, :].unsqueeze(-1)

            y = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)           # (batch, d_model)
            y = y + x[:, t, :] * self.D
            outputs.append(y)

        return torch.stack(outputs, dim=1)                           # (batch, seq_len, d_model)


class Mamba_TS(nn.Module):
    """
    Mamba time-series model for manufacturing metric forecasting.

    Architecture
    ------------
    input (batch, seq_len)
        → Linear embedding to (batch, seq_len, hidden_dim)
        → N × SimplifiedMambaBlock
        → Dropout
        → recon_head : (batch, seq_len, 1) → squeeze → (batch, seq_len)
        → pred_head  : last hidden state (batch, hidden_dim) → (batch, 1)

    The dual-head design lets the model learn to reconstruct the input
    sequence (regularisation) while also predicting the next step.
    At inference time only pred_head output is used.

    Parameters
    ----------
    seq_len    : int   — input sequence length (number of past cycles)
    hidden_dim : int   — d_model for all Mamba blocks
    d_state    : int   — SSM state dimension
    d_conv     : int   — depthwise conv kernel size
    num_layers : int   — number of stacked Mamba blocks
    dropout    : float — applied after each block
    """

    def __init__(
        self,
        seq_len    : int,
        hidden_dim : int   = 64,
        d_state    : int   = 16,
        d_conv     : int   = 4,
        num_layers : int   = 2,
        dropout    : float = 0.2,
    ) -> None:
        super().__init__()
        self.name = "Mamba-SSM"

        self.input_proj = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(d_model=hidden_dim, d_state=d_state, d_conv=d_conv)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Reconstruction head (auxiliary loss — full sequence)
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Prediction head (primary — next step from last hidden state)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len) — scaled target-feature values

        Returns
        -------
        recon : (batch, seq_len)  — reconstruction of input
        pred  : (batch, 1)        — next-step prediction
        """
        # Embed scalar sequence to hidden_dim
        hidden = self.input_proj(x.unsqueeze(-1))   # (batch, seq_len, hidden_dim)

        for layer in self.layers:
            hidden = layer(hidden)
            hidden = self.dropout(hidden)

        recon = self.recon_head(hidden).squeeze(-1)  # (batch, seq_len)
        pred  = self.pred_head(hidden[:, -1, :])     # (batch, 1)

        return recon, pred

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
