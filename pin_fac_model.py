# pin_fac_model.py
# ------------------------------------------------------------
# PIN-FAC voxel-wise inference model
#
# Concept (fixed):
#   - Input tokens: x_val = [Re,Im] per echo (B,NE,2)  (precomputed)
#   - TE_norm embedding kept (protocol-invariant representation)
#   - Δt penalty in attention computed from TE_norm (not physical TE)
#   - Temporal differencing prior (WN): diff(x_e, x_{e+1}) injected
#       as gated residual right after token embedding (before CLS)
#   - Anchor: anchor_log = log(|S(te0)|) injected into CLS only (gated)
#   - CLS TE_norm = 0 (TE-free global latent)
#   - Output: 9 parameters (B,9,1,1)
#
# Forward signature:
#   preds = model(x_val, te_norm, anchor)
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, te_norm):  # (B,T,1)
        return self.mlp(te_norm)


class TEAwareSelfAttention(nn.Module):
    """
    TE-aware self-attention with RBF penalty (Δt_norm):
        score -= softplus(alpha_h) * (Δt_norm)^2
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

        self.alpha = nn.Parameter(torch.zeros(n_heads))  # per-head

    def forward(self, x, te_norm, attn_mask=None):
        """
        x: (B,T,D)
        te_norm: (B,T,1) in [0,1]
        """
        B, T, D = x.shape
        h, dk = self.h, self.dk

        q = self.q(x).view(B, T, h, dk).transpose(1, 2)  # (B,h,T,dk)
        k = self.k(x).view(B, T, h, dk).transpose(1, 2)  # (B,h,T,dk)
        v = self.v(x).view(B, T, h, dk).transpose(1, 2)  # (B,h,T,dk)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)  # (B,h,T,T)

        # Δt_norm
        assert te_norm.dim() == 3 and te_norm.size(-1) == 1, f"te_norm must be (B,T,1), got {te_norm.shape}"
        t = te_norm[..., 0]                           # (B,T)
        dt_2d = t.unsqueeze(1) - t.unsqueeze(2)       # (B,T,T)
        dt = dt_2d.unsqueeze(1).expand(B, h, T, T)    # (B,h,T,T)

        alpha = F.softplus(self.alpha).view(1, h, 1, 1)
        scores = scores - alpha * (dt.abs() ** 2)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)          # (B,h,T,T)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TEAwareSelfAttention(d_model, n_heads)
        self.do = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, te_norm):
        x = x + self.do(self.attn(self.ln1(x), te_norm))
        x = x + self.do(self.ff(self.ln2(x)))
        return x


class PINFAC(nn.Module):
    """
    Inputs:
      x_val:   (B,NE,2)  [Re,Im]
      te_norm: (B,NE,1)  in [0,1]
      anchor:  (B,1)     log(|S(te0)|)

    Output:
      preds: (B,9,1,1)
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        wn_gate_init: float = 0.0,
        anchor_gate_init: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # base token embedding
        self.val_proj = nn.Linear(2, d_model)
        self.time_emb = TimestepEmbedding(d_model)

        # WN residual embedding (diff in input space)
        self.wn_proj = nn.Linear(2, d_model)
        self.wn_gate = nn.Parameter(torch.tensor(float(wn_gate_init)))  # scalar

        # CLS + anchor conditioning
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.anchor_proj = nn.Linear(1, d_model)
        self.anchor_gate = nn.Parameter(torch.tensor(float(anchor_gate_init)))  # scalar

        # CLS TE_norm = 0 (fixed)
        self.register_buffer("te_cls_zero", torch.zeros(1, 1, 1), persistent=False)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, 4, dropout)
            for _ in range(n_layers)
        ])

        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 9),
        )

    @staticmethod
    def _temporal_diff(x_val: torch.Tensor) -> torch.Tensor:
        """
        x_val: (B,T,2)
        diff_e = x_e - x_{e+1}, last=0
        """
        diff = x_val[:, :-1, :] - x_val[:, 1:, :]
        last = torch.zeros_like(x_val[:, :1, :])
        return torch.cat([diff, last], dim=1)

    def forward(self, x_val: torch.Tensor, te_norm: torch.Tensor, anchor: torch.Tensor):
        """
        x_val:   (B,NE,2)
        te_norm: (B,NE,1)
        anchor:  (B,1)
        """
        B, T, _ = x_val.shape
        assert te_norm.shape == (B, T, 1), f"te_norm must be (B,T,1), got {te_norm.shape}"
        assert anchor.shape == (B, 1), f"anchor must be (B,1), got {anchor.shape}"

        # (1) base token embedding (temporal evidence + te_norm coordinate)
        z_tok = self.val_proj(x_val) + self.time_emb(te_norm)  # (B,T,d)

        # (2) WN prior: gated residual right after token embedding
        diff = self._temporal_diff(x_val)                       # (B,T,2)
        wn_emb = self.wn_proj(diff)                             # (B,T,d)
        g_wn = torch.sigmoid(self.wn_gate)                      # 0..1
        z_tok = z_tok + g_wn * wn_emb

        # (3) CLS token + anchor injected into CLS only
        cls = self.cls_token.expand(B, 1, self.d_model)         # (B,1,d)
        a = self.anchor_proj(anchor).unsqueeze(1)               # (B,1,d)
        g_a = torch.sigmoid(self.anchor_gate)                   # 0..1
        cls = cls + g_a * a

        # (4) concat tokens
        z = torch.cat([cls, z_tok], dim=1)                      # (B,T+1,d)

        # (5) concat TE_norm with CLS(0)
        te_cls = self.te_cls_zero.expand(B, 1, 1).to(te_norm.dtype).to(te_norm.device)
        te_cat = torch.cat([te_cls, te_norm], dim=1)            # (B,T+1,1)

        # (6) encoder
        for blk in self.layers:
            z = blk(z, te_cat)

        # (7) readout from CLS
        cls_out = z[:, 0, :]                                    # (B,d)
        y = self.out_head(cls_out)                              # (B,9)
        return y.view(B, 9, 1, 1)
