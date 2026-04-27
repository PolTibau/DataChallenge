"""
lstm/src/model.py
-----------------
Temporal Masked Autoencoder basat en Transformer.

Arquitectura v3 — millores sobre v2:
  - PATCH_SIZE=4 (2h per patch) → més resolució temporal
  - TimeEmbedding: hora del dia + dia setmana + flag laborable/cap de setmana
  - ProfileEmbedding: mu + sigma + cv de la corba → el model sap si és
    un client industrial o domèstic sense que li ho diguem explícitament

Arquitectura
------------
  Input (B, T) + Profile (B, 4)
    → Patch Embedding  (patch_size timesteps → D_MODEL)
    + Time Embedding   (hora + dia + laborable → D_MODEL)
    + Profile Embedding (mu_norm, sigma_norm, cv_norm, weekend → D_MODEL)
    → Positional Encoding sinusoidal
    → Encoder Transformer (N_ENC_LAYERS capes, N_HEADS caps)
    → Decoder Transformer (N_DEC_LAYERS capes, N_HEADS caps)
    → Projecció de sortida → (B, T)
"""

import math
import torch
import torch.nn as nn
from lstm.src.config import (
    D_MODEL, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS,
    D_FF, DROPOUT, MAX_SEQ_LEN, SEQ_LEN,
)

PATCH_SIZE     = 4    # 4 timesteps = 2 hores
STEPS_PER_DAY = 48   # timesteps de 30min per dia
START_WEEKDAY  = 0   # 2023-01-09 és dilluns (0=dilluns, 6=diumenge)
PROFILE_DIM    = 4   # [mu_norm, sigma_norm, cv_norm, weekend_ratio]


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
    Embedding temporal per patch.

    Tres components:
      - Hora del dia     (0-47 slots de 30min) → aprenible
      - Dia de la setmana (0-6)                → aprenible
      - Flag laborable/cap de setmana (0 o 1)  → aprenible (2 valors)

    El model aprèn que "dilluns 8h laborable" té un patró de consum
    molt diferent de "dissabte 8h cap de setmana".
    """

    def __init__(self, d_model=D_MODEL, patch_size=PATCH_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        n_patches = math.ceil(seq_len / patch_size)

        self.hour_embed     = nn.Embedding(STEPS_PER_DAY, d_model)
        self.weekday_embed  = nn.Embedding(7, d_model)
        self.is_work_embed  = nn.Embedding(2, d_model)  # 0=cap de setmana, 1=laborable

        # Precalcular índexs per cada patch
        patch_starts = torch.arange(n_patches) * patch_size
        hours        = (patch_starts % STEPS_PER_DAY).long()
        weekdays     = ((patch_starts // STEPS_PER_DAY + START_WEEKDAY) % 7).long()
        is_workday   = (weekdays < 5).long()  # 0=dissabte/diumenge, 1=dilluns-divendres

        self.register_buffer("hours",      hours)
        self.register_buffer("weekdays",   weekdays)
        self.register_buffer("is_workday", is_workday)

    def forward(self, n_patches: int) -> torch.Tensor:
        """Retorna (1, n_patches, D_MODEL)."""
        h = self.hour_embed(self.hours[:n_patches])
        w = self.weekday_embed(self.weekdays[:n_patches])
        d = self.is_work_embed(self.is_workday[:n_patches])
        return (h + w + d).unsqueeze(0)


class ProfileEmbedding(nn.Module):
    """
    Embedding de perfil de corba: [mu_norm, sigma_norm, cv_norm, weekend_ratio]

    Projecta 4 valors continus a D_MODEL dimensions via una xarxa de 2 capes.
    El model aprèn quins valors de mu/sigma/cv corresponen a clients industrials
    vs domèstics, i adapta la imputació en conseqüència.

    L'embedding resultant (B, 1, D_MODEL) es suma a TOTS els patches de la
    seqüència — és una condició global que afecta tota la reconstrucció.
    """

    def __init__(self, profile_dim=PROFILE_DIM, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(profile_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        """
        profile : (B, 4) — mu_norm, sigma_norm, cv_norm, weekend_ratio
        retorna : (B, 1, D_MODEL) — sumar a tots els patches
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
    """Transformer Masked Autoencoder per a imputació de sèries temporals."""

    def __init__(self,
                 seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
                 n_heads=N_HEADS, n_enc=N_ENC_LAYERS, n_dec=N_DEC_LAYERS,
                 d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.seq_len    = seq_len
        self.patch_size = patch_size

        self.patch_embed   = PatchEmbedding(seq_len, patch_size, d_model)
        self.time_embed    = TimeEmbedding(d_model, patch_size, seq_len)
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
        x       : (B, T) valors d'entrada (0.0 a posicions emmascarades)
        mask    : (B, T) bool, True = emmascarat
        profile : (B, 4) opcional — mu_norm, sigma_norm, cv_norm, weekend_ratio
        →         (B, T) reconstrucció completa
        """
        B = x.size(0)

        # Patch embedding + time embedding
        x_patches = self.patch_embed(x)               # (B, P, D)
        time_emb  = self.time_embed(self.n_patches)   # (1, P, D)
        x_patches = x_patches + time_emb              # (B, P, D)

        # Profile embedding — condició global sumada a tots els patches
        if profile is not None:
            prof_emb  = self.profile_embed(profile)   # (B, 1, D)
            x_patches = x_patches + prof_emb          # broadcast: (B, P, D)

        patch_mask = self._to_patch_mask(mask)                      # (B, P)
        mask_tok   = self.mask_token.expand(B, self.n_patches, -1)
        x_enc      = torch.where(patch_mask.unsqueeze(-1), mask_tok, x_patches)

        memory  = self.encoder(self.enc_pos(x_enc))                 # (B, P, D)
        dec_out = self.decoder(self.dec_pos(x_patches), memory)     # (B, P, D)

        out = self.output_proj(dec_out).reshape(B, -1)              # (B, T+pad)
        if self.pad > 0:
            out = out[:, :-self.pad]
        return out                                                   # (B, T)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
