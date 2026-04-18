"""
lstm/src/model.py
-----------------
Temporal Masked Autoencoder basat en Transformer.

Arquitectura v2 — millores:
  - PATCH_SIZE=4 (2h per patch, era 4h) → més resolució temporal
  - Embeddings temporals: hora del dia + dia de la setmana
    El model ara sap "és dimarts a les 8h" → millor imputació

Arquitectura
------------
  Input (T,)
    → Patch Embedding (patch_size timesteps → D_MODEL)
    + Time Embedding (hora_dia + dia_setmana → D_MODEL)
    → Positional Encoding sinusoidal
    → Encoder Transformer (N_ENC_LAYERS capes, N_HEADS caps)
    → Decoder Transformer (N_DEC_LAYERS capes, N_HEADS caps)
    → Projecció de sortida → (T,)
"""

import math
import torch
import torch.nn as nn
from lstm.src.config import (
    D_MODEL, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS,
    D_FF, DROPOUT, MAX_SEQ_LEN, SEQ_LEN,
)

PATCH_SIZE       = 4    # 4 timesteps = 2 hores (era 8 = 4h)
STEPS_PER_DAY   = 48   # 48 timesteps de 30min per dia
STEPS_PER_HOUR  = 2    # 2 timesteps per hora
START_WEEKDAY   = 0    # 2023-01-09 és dilluns (0=dilluns)


class PatchEmbedding(nn.Module):
    """(B, T) → (B, n_patches, D_MODEL)"""

    def __init__(self, seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = math.ceil(seq_len / patch_size)
        self.pad        = self.n_patches * patch_size - seq_len
        self.proj       = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if self.pad > 0:
            x = torch.nn.functional.pad(x, (0, self.pad))
        x = x.reshape(B, self.n_patches, self.patch_size)
        return self.proj(x)


class TimeEmbedding(nn.Module):
    """
    Embedding temporal per patch: hora del dia + dia de la setmana.

    Per cada patch calculem quina hora i quin dia correspon al seu
    primer timestep. Això dona al model context temporal explícit:
    sap que "és dimarts a les 8h del matí" → patrons de consum típics.

    Paràmetres
    ----------
    d_model    : dimensió de sortida (igual que el model)
    patch_size : timesteps per patch
    seq_len    : longitud total de la seqüència
    """

    def __init__(self, d_model=D_MODEL, patch_size=PATCH_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        self.patch_size = patch_size
        n_patches       = math.ceil(seq_len / patch_size)

        # Embedding aprenible per hora (0-47) i dia (0-6)
        self.hour_embed    = nn.Embedding(STEPS_PER_DAY, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)

        # Precalcular hora i dia per cada patch (fix, no après)
        patch_starts = torch.arange(n_patches) * patch_size  # primer timestep de cada patch
        hours    = (patch_starts % STEPS_PER_DAY).long()
        weekdays = ((patch_starts // STEPS_PER_DAY + START_WEEKDAY) % 7).long()

        # Registrar com a buffers (no són paràmetres, no es gradientegen)
        self.register_buffer("hours",    hours)
        self.register_buffer("weekdays", weekdays)

    def forward(self, n_patches: int) -> torch.Tensor:
        """Retorna embedding temporal (1, n_patches, D_MODEL) per sumar als patches."""
        h = self.hour_embed(self.hours[:n_patches])        # (P, D)
        w = self.weekday_embed(self.weekdays[:n_patches])  # (P, D)
        return (h + w).unsqueeze(0)                        # (1, P, D)


class SinusoidalPE(nn.Module):
    """Positional encoding sinusoidal (fix)."""

    def __init__(self, d_model=D_MODEL, max_len=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TemporalMaskedAutoencoder(nn.Module):
    """Transformer Masked Autoencoder per a imputació de sèries temporals."""

    def __init__(self,
                 seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
                 n_heads=N_HEADS, n_enc=N_ENC_LAYERS, n_dec=N_DEC_LAYERS,
                 d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.seq_len    = seq_len
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(seq_len, patch_size, d_model)
        self.time_embed  = TimeEmbedding(d_model, patch_size, seq_len)
        self.n_patches   = self.patch_embed.n_patches
        self.pad         = self.patch_embed.pad

        self.mask_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.enc_pos = SinusoidalPE(d_model, MAX_SEQ_LEN, dropout)
        self.dec_pos = SinusoidalPE(d_model, MAX_SEQ_LEN, dropout)

        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc, norm=nn.LayerNorm(d_model))

        dec_layer    = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_dec, norm=nn.LayerNorm(d_model))

        self.output_proj = nn.Linear(d_model, patch_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _to_patch_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """(B, T) bool → (B, n_patches) bool."""
        B, T = mask.shape
        if self.pad > 0:
            mask = torch.nn.functional.pad(mask.float(), (0, self.pad)).bool()
        return mask.reshape(B, self.n_patches, self.patch_size).float().mean(-1) > 0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, T) valors d'entrada (0.0 a posicions emmascarades)
        mask : (B, T) bool, True = emmascarat
        →      (B, T) reconstrucció completa
        """
        B = x.size(0)

        # Patch embedding + time embedding
        x_patches  = self.patch_embed(x)                              # (B, P, D)
        time_emb   = self.time_embed(self.n_patches)                  # (1, P, D)
        x_patches  = x_patches + time_emb                             # (B, P, D)

        patch_mask = self._to_patch_mask(mask)                        # (B, P)
        mask_tok   = self.mask_token.expand(B, self.n_patches, -1)
        x_enc      = torch.where(patch_mask.unsqueeze(-1), mask_tok, x_patches)

        memory  = self.encoder(self.enc_pos(x_enc))                   # (B, P, D)
        dec_out = self.decoder(self.dec_pos(x_patches), memory)       # (B, P, D)

        out = self.output_proj(dec_out).reshape(B, -1)                # (B, T+pad)
        if self.pad > 0:
            out = out[:, :-self.pad]
        return out                                                     # (B, T)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
