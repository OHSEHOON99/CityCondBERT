import math
import torch
from torch import nn
import numpy as np
from einops import rearrange



class AbsoluteEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000, scaling_factor=2.0):
        super(AbsoluteEncoding, self).__init__()
        self.model_dim = model_dim
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * - (math.log(10000.0) * scaling_factor / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class PeriodicEncoding(nn.Module):
    """
    Generalized sinusoidal time encoding module for DE (daily) and WE (weekly) patterns.
    θ(t) = 2π * t / period 에 따라 주기성을 조절하며, scaling_factor를 통해 주파수 변화를 줄 수 있음.
    """
    def __init__(self, model_dim, period, scaling_factor=8.0):
        super(PeriodicEncoding, self).__init__()
        self.model_dim = model_dim
        self.period = period  # ex) 48 or 336
        self.scaling_factor = scaling_factor

        # (D/2,) => 2i for sin/cos pair
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) * scaling_factor / model_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, t):
        """
        Args:
            t: (B, L) 형태의 시각 인덱스. 각 시점 t에 대해 θ(t)를 계산함.
        
        Returns:
            (B, L, D) 형태의 sin-cos 임베딩
        """
        # θ(t) = 2π t / period
        theta = 2 * math.pi * t / self.period  # (B, L)

        # 차원 확장하여 broadcasting 준비
        theta = theta.unsqueeze(-1)  # (B, L, 1)
        scaled_theta = theta * self.div_term  # (B, L, D/2)

        pe = torch.zeros(t.size(0), t.size(1), self.model_dim, device=t.device)
        pe[:, :, 0::2] = torch.sin(scaled_theta)
        pe[:, :, 1::2] = torch.cos(scaled_theta)
        return pe
    

# Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding
class LearnableFourierFeatures(nn.Module):
    def __init__(self, pos_dim, f_dim, h_dim, d_dim, g_dim=1, gamma=1.0):
        super().__init__()
        assert f_dim % 2 == 0, 'number of fourier feature dimensions must be divisible by 2.'
        assert d_dim % g_dim == 0, 'number of D dimension must be divisible by the number of G dimension.'
        enc_f_dim = f_dim // 2
        self.pos_dim = pos_dim
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.d_dim = d_dim
        self.g_dim = g_dim
        self.dg_dim = d_dim // g_dim

        # (enc_f_dim, pos_dim)
        self.Wr = nn.Parameter(torch.randn(enc_f_dim, pos_dim) * (gamma ** 2))
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, self.dg_dim)  # per-group output
        )
        self.div_term = float(np.sqrt(f_dim))

    def forward(self, pos):
        """
        Input:
          - pos: (B, L, M)  or  (B, L, G, M)
        Output:
          - (B, L, D) where D = d_dim = g_dim * dg_dim
        """
        # 1) 입력 모양 정규화: (B, L, G, M)
        if pos.dim() == 3:
            # (B, L, M) -> (B, L, 1, M)
            pos = pos.unsqueeze(-2)
        elif pos.dim() == 4:
            pass
        else:
            raise ValueError(f"pos must be (B,L,M) or (B,L,G,M), got {pos.shape}")

        B, L, G, M = pos.shape
        if M != self.pos_dim:
            raise ValueError(f"Expected last dim(pos_dim)={self.pos_dim}, got {M}")

        # 2) (B, L, G, M) x (M, enc_f_dim) -> (B, L, G, enc_f_dim)
        XWr = torch.matmul(pos, self.Wr.T)

        # 3) Fourier basis: (cos, sin) -> (B, L, G, f_dim)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term

        # 4) Per-group MLP: (B, L, G, f_dim) -> (B, L, G, dg_dim)
        Y = self.mlp(F)

        # 5) 그룹 결합: (B, L, G, dg_dim) -> (B, L, G*dg_dim) = (B, L, D)
        Y = Y.reshape(B, L, G * self.dg_dim)

        return Y
    

def interleave_bits_tensor(x, y):
    """
    x, y: LongTensor of shape (B, L)
    returns: interleaved Morton/Z-order index as LongTensor of shape (B, L)
    """
    def spread_bits(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    x_spread = spread_bits(x)
    y_spread = spread_bits(y)
    return (y_spread << 1) | x_spread


class ZCurveLocationEmbedding(nn.Module):
    def __init__(self, grid_size_x, grid_size_y, embedding_dim):
        super().__init__()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        # Compute max possible z-order index
        max_x = grid_size_x - 1
        max_y = grid_size_y - 1
        max_index = interleave_bits_tensor(
            torch.tensor(max_x), torch.tensor(max_y)
        ).item() + 1

        self.embedding = nn.Embedding(max_index, embedding_dim)

    def forward(self, location_id):
        """
        location_id: LongTensor of shape (B, L)
        Assumes location_id was flattened using row-major: id = y * grid_size_x + x
        """
        x = location_id % self.grid_size_x
        y = location_id // self.grid_size_x
        z_index = interleave_bits_tensor(x, y)
        return self.embedding(z_index)