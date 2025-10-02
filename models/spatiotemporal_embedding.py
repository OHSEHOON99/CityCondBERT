import math
import torch
from torch import nn
import numpy as np


class AbsoluteEncoding(nn.Module):
    """
    Standard absolute sinusoidal positional encoding with adjustable scaling factor.
    """
    def __init__(self, model_dim, max_len=5000, scaling_factor=2.0):
        super().__init__()
        self.model_dim = model_dim
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) * scaling_factor / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, D)
        Returns:
            Tensor with absolute positional encoding added: (B, L, D)
        """
        return x + self.pe[:, :x.size(1)]


class PeriodicEncoding(nn.Module):
    """
    Generalized sinusoidal encoding for periodic signals.
    Encoding: θ(t) = 2π * t / period
    """
    def __init__(self, model_dim, period, scaling_factor=8.0):
        super().__init__()
        self.model_dim = model_dim
        self.period = period
        self.scaling_factor = scaling_factor

        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) * scaling_factor / model_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, t):
        """
        Args:
            t: (B, L) tensor of time indices
        Returns:
            (B, L, D) tensor of sinusoidal embeddings
        """
        theta = 2 * math.pi * t / self.period
        theta = theta.unsqueeze(-1)
        scaled_theta = theta * self.div_term

        pe = torch.zeros(t.size(0), t.size(1), self.model_dim, device=t.device)
        pe[:, :, 0::2] = torch.sin(scaled_theta)
        pe[:, :, 1::2] = torch.cos(scaled_theta)
        return pe


class LearnableFourierFeatures(nn.Module):
    """
    Learnable Fourier features for multi-dimensional positional encoding.

    Args:
        pos_dim: input positional dimension (e.g., 1 for scalar)
        f_dim: number of Fourier features (must be even)
        h_dim: hidden dimension in MLP
        d_dim: output dimension
        g_dim: number of groups (for grouped MLPs)
        gamma: scaling factor for initialization
    """
    def __init__(self, pos_dim, f_dim, h_dim, d_dim, g_dim=1, gamma=1.0):
        super().__init__()
        assert f_dim % 2 == 0, "f_dim must be divisible by 2."
        assert d_dim % g_dim == 0, "d_dim must be divisible by g_dim."
        enc_f_dim = f_dim // 2

        self.pos_dim = pos_dim
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.d_dim = d_dim
        self.g_dim = g_dim
        self.dg_dim = d_dim // g_dim

        self.Wr = nn.Parameter(torch.randn(enc_f_dim, pos_dim) * (gamma ** 2))
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, self.dg_dim)
        )
        self.div_term = float(np.sqrt(f_dim))

    def forward(self, pos):
        """
        Args:
            pos: (B, L, M) or (B, L, G, M)
        Returns:
            (B, L, D) where D = d_dim = g_dim * dg_dim
        """
        if pos.dim() == 3:
            pos = pos.unsqueeze(-2)
        elif pos.dim() != 4:
            raise ValueError(f"Expected pos to be (B,L,M) or (B,L,G,M), got {pos.shape}")

        B, L, G, M = pos.shape
        if M != self.pos_dim:
            raise ValueError(f"Expected pos_dim={self.pos_dim}, got {M}")

        # (B, L, G, M) x (M, enc_f_dim) -> (B, L, G, enc_f_dim)
        XWr = torch.matmul(pos, self.Wr.T)

        # Fourier basis (cos, sin)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term

        # MLP per group
        Y = self.mlp(F)

        # Flatten groups: (B, L, G, dg_dim) -> (B, L, D)
        Y = Y.reshape(B, L, G * self.dg_dim)
        return Y