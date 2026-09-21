"""
lstm/src/model.py — v3
----------------------
Temporal Masked Autoencoder basat en Transformer.

Canvis v3 respecte v2:
  - PATCH_SIZE=2 (1h per token) — màxima resolució temporal
  - JointTimeEmbedding: hora × tipus_dia (96 entrades) en lloc de
    sumar hour + weekday + workday per separat.
    Captura interaccions: "8h dilluns laborable" ≠ "8h diumenge"
  - ProfileEmbedding ampliat a 9 dimensions amb estadístiques de forma
  - TTA a inferència: soroll als punts observats (no shifts cíclics)

Arquitectura (mides a lstm/src/config.py; actualment D=192, 4+4 capes)
------------
  Input (B, T) + Profile (B, 9)
    → Patch Embedding (patch_size=2)        → (B, P, D)
    + Joint Time Embedding (hora × diatipus) → (1, P, D)
    + Profile Embedding (9 features)         → (B, 1, D)
    → Positional Encoding sinusoidal
    → Encoder Transformer (N_ENC_LAYERS, N_HEADS, D_MODEL)
    → Decoder Transformer (N_DEC_LAYERS, N_HEADS, D_MODEL)
    → Projecció de sortida → (B, T)
"""

import math
import torch
import torch.nn as nn
from lstm.src.config import (
    D_MODEL, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS,
    D_FF, DROPOUT, MAX_SEQ_LEN, SEQ_LEN, PROFILE_DIM,
)

# NOU v3: PATCH_SIZE=2 (1h per patch)
PATCH_SIZE     = 2
STEPS_PER_DAY  = 48
START_WEEKDAY  = 0   # 2023-01-09 és dilluns


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


class JointTimeEmbedding(nn.Module):
    """
    NOU v3: Embedding conjunt hora × tipus_dia.

    En lloc de sumar embeddings separats (hora + dia + laborable), aquí
    cada combinació (hora_del_dia, laborable/cap_de_setmana) té el seu
    propi embedding. Això captura interaccions no-lineals:

      - "19h laborable" — pic vespre fort (clàssic domèstic després feina)
      - "19h cap setmana" — patró diferent (sopar més tard, més variabilitat)
      - "10h laborable" — pic baix (ningú a casa, no n'és consum)
      - "10h cap setmana" — pic mig (gent llevada, esmorzant)

    Total: 48 hores × 2 tipus = 96 entrades.

    A més afegim un embedding aprenible de dia de la setmana per capturar
    diferències entre dilluns i divendres dins de laborables, dissabte
    i diumenge dins de cap de setmana.
    """

    def __init__(self, d_model=D_MODEL, patch_size=PATCH_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        n_patches = math.ceil(seq_len / patch_size)

        # NOU v3: embedding conjunt (hora × tipus_dia) = 96 entrades
        # Index: hora * 2 + is_workday  (0..95)
        self.hour_x_type_embed = nn.Embedding(STEPS_PER_DAY * 2, d_model)

        # Mantenim weekday separat per capturar dilluns vs divendres
        self.weekday_embed = nn.Embedding(7, d_model)

        # Precalcular índexs per cada patch
        patch_starts = torch.arange(n_patches) * patch_size
        hours        = (patch_starts % STEPS_PER_DAY).long()
        weekdays     = ((patch_starts // STEPS_PER_DAY + START_WEEKDAY) % 7).long()
        is_workday   = (weekdays < 5).long()

        hour_x_type  = (hours * 2 + is_workday).long()  # 0..95

        self.register_buffer("hour_x_type", hour_x_type)
        self.register_buffer("weekdays",    weekdays)

    def forward(self, n_patches: int) -> torch.Tensor:
        """Retorna (1, n_patches, D_MODEL)."""
        ht = self.hour_x_type_embed(self.hour_x_type[:n_patches])
        wd = self.weekday_embed(self.weekdays[:n_patches])
        return (ht + wd).unsqueeze(0)


class ProfileEmbedding(nn.Module):
    """
    Embedding de perfil enriquit (9 dimensions).

    v3: 9 features [mu, mad, cv, weekend_ratio, peak_ratio, skew,
                    autocorr, p95, iqr]

    Projecta a D_MODEL via xarxa de 3 capes amb activació GELU.
    Resultat (B, 1, D) es suma a tots els patches com a condició global.
    """

    def __init__(self, profile_dim=PROFILE_DIM, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        # NOU v3: xarxa més profunda per processar les 9 features
        self.proj = nn.Sequential(
            nn.Linear(profile_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        """
        profile : (B, 9)
        retorna : (B, 1, D_MODEL)
        """
        return self.proj(profile).unsqueeze(1)


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
    """Transformer Masked Autoencoder per a imputació de sèries temporals — v3."""

    def __init__(self,
                 seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
                 n_heads=N_HEADS, n_enc=N_ENC_LAYERS, n_dec=N_DEC_LAYERS,
                 d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.seq_len    = seq_len
        self.patch_size = patch_size

        self.patch_embed   = PatchEmbedding(seq_len, patch_size, d_model)
        self.time_embed    = JointTimeEmbedding(d_model, patch_size, seq_len)
        self.profile_embed = ProfileEmbedding(PROFILE_DIM, d_model, dropout)

        self.n_patches = self.patch_embed.n_patches
        self.pad       = self.patch_embed.pad

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                profile: torch.Tensor | None = None) -> torch.Tensor:
        """
        x       : (B, T) valors d'entrada
        mask    : (B, T) bool
        profile : (B, 9)
        retorna : (B, T)
        """
        B = x.size(0)

        x_patches = self.patch_embed(x)                # (B, P, D)
        time_emb  = self.time_embed(self.n_patches)    # (1, P, D)
        x_patches = x_patches + time_emb               # (B, P, D)

        if profile is not None:
            prof_emb  = self.profile_embed(profile)    # (B, 1, D)
            x_patches = x_patches + prof_emb           # broadcast

        patch_mask = self._to_patch_mask(mask)
        mask_tok   = self.mask_token.expand(B, self.n_patches, -1)
        x_enc      = torch.where(patch_mask.unsqueeze(-1), mask_tok, x_patches)

        memory  = self.encoder(self.enc_pos(x_enc))
        dec_out = self.decoder(self.dec_pos(x_patches), memory)

        out = self.output_proj(dec_out).reshape(B, -1)
        if self.pad > 0:
            out = out[:, :-self.pad]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
