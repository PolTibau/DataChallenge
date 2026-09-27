"""
lstm/src/model.py — v4
----------------------
Temporal Masked Autoencoder basat en Transformer.

v4: l'entrada és (B, T, C) — el valor de la corba més els canals de context
(veure context.py). PATCH_SIZE es configura a config.py (2 = un token per
hora, 1 = un token per mitja hora).

  Input (B, T, C) + Profile (B, PROFILE_DIM)
    → Patch Embedding (patch_size × C → D)       → (B, P, D)
    + Joint Time Embedding (hora × tipus de dia)  → (1, P, D)
    + Profile Embedding                           → (B, 1, D)
    → Encoder Transformer (tokens emmascarats → mask token)
    → Decoder Transformer
    → Projecció de sortida → (B, T)
"""

import math

import torch
import torch.nn as nn

from lstm.src.config import (
    D_MODEL, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, D_FF, DROPOUT, MAX_SEQ_LEN,
    SEQ_LEN, PROFILE_DIM, PATCH_SIZE, STEPS_PER_DAY, START_WEEKDAY,
)
from lstm.src.context import N_CHANNELS


def default_model_kwargs() -> dict:
    """Hiperparàmetres actuals; es desen amb cada entrenament per reconstruir el model."""
    return dict(seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
                n_heads=N_HEADS, n_enc=N_ENC_LAYERS, n_dec=N_DEC_LAYERS,
                d_ff=D_FF, dropout=DROPOUT, n_channels=N_CHANNELS,
                profile_dim=PROFILE_DIM)


class PatchEmbedding(nn.Module):
    """(B, T, C) → (B, n_patches, D)"""

    def __init__(self, seq_len: int, patch_size: int, d_model: int, n_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = math.ceil(seq_len / patch_size)
        self.pad        = self.n_patches * patch_size - seq_len
        self.proj       = nn.Linear(patch_size * n_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.pad > 0:
            x = nn.functional.pad(x, (0, 0, 0, self.pad))
        return self.proj(x.reshape(B, self.n_patches, self.patch_size * C))


class JointTimeEmbedding(nn.Module):
    """Embedding (franja horària × laborable/festiu) + dia de la setmana, per patch."""

    def __init__(self, d_model: int, patch_size: int, seq_len: int):
        super().__init__()
        n_patches = math.ceil(seq_len / patch_size)
        self.hour_x_type_embed = nn.Embedding(STEPS_PER_DAY * 2, d_model)
        self.weekday_embed     = nn.Embedding(7, d_model)

        starts     = torch.arange(n_patches) * patch_size
        slots      = (starts % STEPS_PER_DAY).long()
        weekdays   = ((starts // STEPS_PER_DAY + START_WEEKDAY) % 7).long()
        is_workday = (weekdays < 5).long()
        self.register_buffer("hour_x_type", slots * 2 + is_workday)
        self.register_buffer("weekdays", weekdays)

    def forward(self, n_patches: int) -> torch.Tensor:
        ht = self.hour_x_type_embed(self.hour_x_type[:n_patches])
        wd = self.weekday_embed(self.weekdays[:n_patches])
        return (ht + wd).unsqueeze(0)


class ProfileEmbedding(nn.Module):
    """Fitxa del consumidor → (B, 1, D), sumada a tots els tokens."""

    def __init__(self, profile_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(profile_dim, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        return self.proj(profile).unsqueeze(1)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TemporalMaskedAutoencoder(nn.Module):

    def __init__(self, seq_len: int = SEQ_LEN, patch_size: int = PATCH_SIZE,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_enc: int = N_ENC_LAYERS, n_dec: int = N_DEC_LAYERS,
                 d_ff: int = D_FF, dropout: float = DROPOUT,
                 n_channels: int = N_CHANNELS, profile_dim: int = PROFILE_DIM):
        super().__init__()
        self.patch_size = patch_size

        self.patch_embed   = PatchEmbedding(seq_len, patch_size, d_model, n_channels)
        self.time_embed    = JointTimeEmbedding(d_model, patch_size, seq_len)
        self.profile_embed = ProfileEmbedding(profile_dim, d_model, dropout)
        self.n_patches     = self.patch_embed.n_patches
        self.pad           = self.patch_embed.pad

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc_pos = SinusoidalPE(d_model, MAX_SEQ_LEN, dropout)
        self.dec_pos = SinusoidalPE(d_model, MAX_SEQ_LEN, dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                       dropout=dropout, batch_first=True, norm_first=True),
            num_layers=n_enc, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                       dropout=dropout, batch_first=True, norm_first=True),
            num_layers=n_dec, norm=nn.LayerNorm(d_model))

        self.output_proj = nn.Linear(d_model, patch_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.mask_token, std=0.02)

    def _to_patch_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """(B, T) bool → (B, n_patches) bool: patch majoritàriament desconegut."""
        B = mask.shape[0]
        m = mask.float()
        if self.pad > 0:
            m = nn.functional.pad(m, (0, self.pad))
        return m.reshape(B, self.n_patches, self.patch_size).mean(-1) > 0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                profile: torch.Tensor) -> torch.Tensor:
        """x (B, T, C), mask (B, T) bool, profile (B, PROFILE_DIM) → (B, T)"""
        B = x.size(0)
        tokens = self.patch_embed(x) + self.time_embed(self.n_patches) + self.profile_embed(profile)

        patch_mask = self._to_patch_mask(mask)
        x_enc = torch.where(patch_mask.unsqueeze(-1),
                            self.mask_token.expand(B, self.n_patches, -1), tokens)

        memory  = self.encoder(self.enc_pos(x_enc))
        dec_out = self.decoder(self.dec_pos(tokens), memory)

        out = self.output_proj(dec_out).reshape(B, -1)
        if self.pad > 0:
            out = out[:, :-self.pad]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
