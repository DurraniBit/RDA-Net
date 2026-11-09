"""RDA-Net architecture (Section 3, Fig. 1)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class RiemannianBranch(nn.Module):
    """Covariance (Eq. 1) -> regularisation (Eqs. 6-7) -> SPD map (Eq. 9 or Eq. 8) -> MLP token."""

    def __init__(self, n_channels, emb_size, cfg):
        super().__init__()
        self.C = n_channels
        self.spd_map = cfg.spd_map
        self.cov_reg = cfg.cov_reg
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2
        C = n_channels
        dim = C * (C + 1) // 2
        self.register_buffer('tril_strict', torch.tril_indices(C, C, offset=-1), persistent=False)
        self.register_buffer('triu_strict', torch.triu_indices(C, C, offset=1), persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(dim, cfg.riemann_hidden), nn.ELU(), nn.Dropout(cfg.dropout_riemann),
            nn.Linear(cfg.riemann_hidden, emb_size))
        self.n_chol_fail = 0  # number of trials whose Cholesky factorisation failed (Section 4.6)

    def covariance(self, x):
        """x: (B, C, T) -> regularised SPD matrices (B, C, C), computed per trial."""
        B, C, T = x.shape
        xc = x - x.mean(dim=-1, keepdim=True)
        cov = xc @ xc.transpose(1, 2) / (T - 1)                                  # Eq. (1)
        eye = torch.eye(C, device=x.device, dtype=x.dtype).expand(B, C, C)
        if self.cov_reg in ('tikhonov', 'dual'):
            cov = cov + self.lambda1 * eye                                       # Eq. (6)
        if self.cov_reg == 'dual':
            tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)                   # tr(C_reg1), per trial
            cov = cov + self.lambda2 * tr.view(B, 1, 1) * eye                    # Eq. (7)
        return cov

    def _cholesky(self, cov):
        L, info = torch.linalg.cholesky_ex(cov)
        if bool((info > 0).any()):
            self.n_chol_fail += int((info > 0).sum())
            eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
            L = torch.linalg.cholesky(cov + 1e-6 * cov.diagonal(dim1=-2, dim2=-1).mean() * eye)
        return L

    def log_cholesky_vector(self, cov):
        """Log-Cholesky map of Lin (2019), Eq. (9): phi(C) = strict_lower(L) + log diag(L), C = L L^T.
        Returns the C(C+1)/2 entries [log diag(L), strictly lower-triangular part of L]."""
        L = self._cholesky(cov)
        log_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-10))
        strict = L[:, self.tril_strict[0], self.tril_strict[1]]
        return torch.cat([log_diag, strict], dim=1)

    def log_euclidean_vector(self, cov):
        """Log-Euclidean map, Eq. (8): Log(C) = U diag(log mu) U^T, isometric upper-triangle vectorisation.
        Used only as the comparison mapping of Table 7."""
        w, U = torch.linalg.eigh(cov)
        logC = U @ torch.diag_embed(torch.log(w.clamp_min(1e-10))) @ U.transpose(-1, -2)
        diag = torch.diagonal(logC, dim1=-2, dim2=-1)
        off = logC[:, self.triu_strict[0], self.triu_strict[1]] * math.sqrt(2.0)
        return torch.cat([diag, off], dim=1)

    def spd_vector(self, x):
        cov = self.covariance(x)
        if self.spd_map == 'log_cholesky':
            return self.log_cholesky_vector(cov)
        return self.log_euclidean_vector(cov)

    def forward(self, x):                                     # x: (B, C, T)
        return self.proj(self.spd_vector(x)).unsqueeze(1)     # F_RG: (B, 1, d_emb)


class LightweightCNN(nn.Module):
    """Temporal conv -> depthwise spatial conv -> depthwise-separable conv -> multi-scale residual (Section 3.4)."""

    def __init__(self, n_channels, emb_size, cfg):
        super().__init__()
        F1, F2 = cfg.F1, cfg.F1 * cfg.D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, cfg.kernel1), padding=(0, cfg.kernel1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),           # Eq. (10)
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, cfg.pool1)), nn.Dropout(cfg.dropout_cnn))
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, cfg.kernel2), padding=(0, cfg.kernel2 // 2), groups=F2, bias=False),
            nn.BatchNorm2d(F2),
            nn.Conv2d(F2, emb_size, (1, 1), bias=False),
            nn.BatchNorm2d(emb_size), nn.ELU(),
            nn.AvgPool2d((1, cfg.pool2)), nn.Dropout(cfg.dropout_cnn))
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(emb_size, emb_size // 2, 3, padding=1, groups=emb_size // 2),
            nn.BatchNorm1d(emb_size // 2), nn.GELU(),
            nn.Conv1d(emb_size // 2, emb_size, 1))

    def forward(self, x):                      # x: (B, 1, C, T)
        x = self.block2(self.block1(x))        # (B, d_emb, 1, S_CNN)
        x = x.flatten(2).transpose(1, 2)       # (B, S_CNN, d_emb)
        return x + self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)


class LocalWindowAttention(nn.Module):
    """Multi-head attention restricted to the band |i - j| <= w (Eq. 14).

    impl='banded': for each query only the 2w+1 keys/values of its window are gathered and
                   scored, O(N (2w+1) d_emb); the N x N score matrix is never formed.
    impl='masked': dense scores plus the additive -inf mask Omega, O(N^2 d_emb).
    Both give identical outputs and gradients.
    """

    def __init__(self, emb_size, heads, window, attn_drop, impl='banded'):
        super().__init__()
        assert emb_size % heads == 0
        self.h, self.d, self.w, self.impl = heads, emb_size // heads, window, impl
        self.scale = self.d ** -0.5
        self.qkv = nn.Linear(emb_size, 3 * emb_size)
        self.proj = nn.Linear(emb_size, emb_size)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, E = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)   # (B, h, N, d_k)
        w = self.w
        if self.impl == 'masked':
            i = torch.arange(N, device=x.device)
            outside = (i[:, None] - i[None, :]).abs() > w                               # Omega = -inf here
            s = (q @ k.transpose(-2, -1)) * self.scale
            a = self.attn_drop(F.softmax(s.masked_fill(outside, float('-inf')), dim=-1))
            out = a @ v
        else:
            K = 2 * w + 1
            kw = F.pad(k, (0, 0, w, w)).unfold(2, K, 1)                                 # (B, h, N, d_k, K)
            vw = F.pad(v, (0, 0, w, w)).unfold(2, K, 1)
            s = torch.einsum('bhnd,bhndk->bhnk', q, kw) * self.scale                     # (B, h, N, K)
            pos = torch.arange(N, device=x.device)[:, None] + torch.arange(K, device=x.device)[None, :] - w
            invalid = (pos < 0) | (pos >= N)                                             # padded slots
            a = self.attn_drop(F.softmax(s.masked_fill(invalid, float('-inf')), dim=-1))
            out = torch.einsum('bhnk,bhndk->bhnd', a, vw)
        return self.proj(out.transpose(1, 2).reshape(B, N, E))


class EncoderLayer(nn.Module):
    """Pre-norm Transformer layer: LN -> local-window attention -> residual; LN -> FFN (4 d_emb, GELU) -> residual."""

    def __init__(self, emb_size, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = LocalWindowAttention(emb_size, cfg.heads, cfg.window, cfg.dropout_attn, cfg.attn_impl)
        self.drop1 = nn.Dropout(cfg.dropout_transformer)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size), nn.GELU(), nn.Dropout(cfg.dropout_transformer),
            nn.Linear(4 * emb_size, emb_size), nn.Dropout(cfg.dropout_transformer))

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        return x + self.ffn(self.norm2(x))


class GradientReversal(Function):
    """GRL (Eqs. 15-16): identity forward, gradient multiplied by -alpha backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


class DomainDiscriminator(nn.Module):
    """d_emb -> 64 -> M domain logits (ELU, dropout 0.3); softmax CE, equivalent to sigmoid BCE for M = 2."""

    def __init__(self, emb_size, n_domains, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_size, cfg.disc_hidden), nn.ELU(),
                                 nn.Dropout(cfg.dropout_disc), nn.Linear(cfg.disc_hidden, n_domains))

    def forward(self, z):
        return self.net(z)


class RDANet(nn.Module):
    """RDA-Net. The domain discriminator is instantiated only when n_domains >= 2 (LOSO-UDA)."""

    def __init__(self, cfg, n_channels, n_samples, n_classes, n_domains=0):
        super().__init__()
        assert cfg.use_riemann or cfg.use_cnn
        E = cfg.emb_size
        self.cfg = cfg
        self.riemann = RiemannianBranch(n_channels, E, cfg) if cfg.use_riemann else None
        self.cnn = LightweightCNN(n_channels, E, cfg) if cfg.use_cnn else None
        S = 0
        if self.cnn is not None:
            self.cnn.eval()
            with torch.no_grad():
                S = self.cnn(torch.zeros(1, 1, n_channels, n_samples)).shape[1]
            self.cnn.train()
        self.n_tokens = S + (1 if self.riemann is not None else 0)            # N = 1 + S_CNN
        self.W_fusion = nn.Parameter(torch.empty(E, E)) if cfg.fusion == 'adaptive' else None
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, E))         # P_pos
        self.blocks = nn.ModuleList([EncoderLayer(E, cfg) for _ in range(cfg.depth)]) \
            if cfg.use_transformer else nn.ModuleList()
        self.classifier = nn.Sequential(nn.LayerNorm(E), nn.Linear(E, n_classes))
        self.domain_disc = DomainDiscriminator(E, n_domains, cfg) if n_domains >= 2 else None
        self._init_weights()

    def _init_weights(self):
        """Glorot-uniform for all linear and convolutional weights; P_pos ~ truncated N(0, 0.02^2)."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.W_fusion is not None:
            nn.init.xavier_uniform_(self.W_fusion)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def fusion_reg(self):
        """L_reg = ||W_fusion||_F^2 (Eq. 4); zero when the fusion projection is ablated."""
        if self.W_fusion is None:
            return self.pos_emb.new_zeros(())
        return (self.W_fusion ** 2).sum()

    def features(self, x):                     # x: (B, 1, C, T) -> z: (B, d_emb)
        tokens = []
        if self.riemann is not None:
            tokens.append(self.riemann(x.squeeze(1)))       # F_RG
        if self.cnn is not None:
            tokens.append(self.cnn(x))                       # F_CNN
        f = torch.cat(tokens, dim=1)                         # F_concat: (B, N, d_emb)
        if self.W_fusion is not None:
            f = f @ self.W_fusion                            # F_projected
        f = f + self.pos_emb                                 # F_fused
        for blk in self.blocks:
            f = blk(f)
        return f.mean(dim=1)                                 # global average pooling

    def forward(self, x):
        z = self.features(x)
        return self.classifier(z), z

    def domain_logits(self, z, alpha):
        return self.domain_disc(GradientReversal.apply(z, alpha))

    def n_inference_params(self):
        """Parameters of the inference network (the training-only discriminator excluded)."""
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith('domain_disc'))
