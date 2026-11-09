"""Model complexity and latency (Table 6, Section 4.5).

  python scripts/complexity.py                     # RDA-Net, BCI-IV 2a input (22 x 1000)
  python scripts/complexity.py --attn_impl masked  # dense masked attention, for comparison
  python scripts/complexity.py --seq_scan          # attention cost versus sequence length N

Reports
  * inference parameters (the training-only domain discriminator is excluded),
  * MACs per trial from operator-level counting (torch.utils.flop_counter; 1 MAC = 2 FLOPs) plus
    the Cholesky factorisation (~C^3/3), which the counter does not see,
  * the MACs of the attention score and aggregation products,
  * latency: median and interquartile range of 300 CUDA-synchronised forward passes after 50
    warm-up passes, at batch size 1 and 72.

Any other model can be measured with the same procedure through profile(model, input_shape).
Requires PyTorch >= 2.1 for torch.utils.flop_counter.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rdanet.config import RDAConfig  # noqa: E402
from rdanet.model import RDANet  # noqa: E402


def count_macs(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2


@torch.no_grad()
def latency(model, x, warmup=50, reps=300):
    model.eval()
    cuda = x.device.type == 'cuda'
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return np.median(ts), np.percentile(ts, 25), np.percentile(ts, 75)


def profile(model, input_shape, device, extra_macs=0.0, batch_sizes=(1, 72)):
    """Parameters, MACs per trial and latency of any nn.Module taking a tensor of input_shape (batch first)."""
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, torch.randn(1, *input_shape[1:], device=device)) + extra_macs
    lat = {bs: latency(model, torch.randn(bs, *input_shape[1:], device=device)) for bs in batch_sizes}
    return n_params, macs, lat


def attention_core_macs(N, E, w, impl):
    keys = N if impl == 'masked' else (2 * w + 1)
    return 2 * N * keys * E                       # Q K^T and A V, summed over heads


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--channels', type=int, default=22)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--attn_impl', default='banded', choices=['banded', 'masked'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seq_scan', action='store_true')
    a = p.parse_args()
    dev = torch.device(a.device)
    cfg = RDAConfig(attn_impl=a.attn_impl)
    m = RDANet(cfg, a.channels, a.samples, a.classes, n_domains=0)   # inference network
    C, N, E = a.channels, m.n_tokens, cfg.emb_size
    chol = C ** 3 / 3
    n_params, macs, lat = profile(m, (1, 1, C, a.samples), dev, extra_macs=chol)
    att = cfg.depth * attention_core_macs(N, E, cfg.window, a.attn_impl)
    print(f'device: {torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}')
    print(f'tokens N = {N}, window w = {cfg.window} ({2 * cfg.window + 1} of {N} positions visible)')
    print(f'inference parameters   : {n_params:,} ({n_params / 1e6:.3f} M)')
    print(f'MACs per trial         : {macs / 1e9:.4f} G (= {2 * macs / 1e9:.4f} GFLOPs), '
          f'incl. Cholesky {chol / 1e3:.1f} K')
    print(f'attention QK^T + AV    : {att / 1e6:.4f} M MACs = {100 * att / macs:.2f}% of total ({a.attn_impl})')
    for bs, (med, q1, q3) in lat.items():
        print(f'latency batch {bs:3d}      : median {med:.2f} ms, IQR [{q1:.2f}, {q3:.2f}] ms -> {med / bs:.3f} ms/trial')
    if a.seq_scan:
        print(f'\nattention core MACs per layer versus N (d_emb = {E}, w = {cfg.window}):')
        for n in (16, 64, 256, 1024, 4096):
            d = attention_core_macs(n, E, cfg.window, 'masked')
            b = attention_core_macs(n, E, cfg.window, 'banded')
            print(f'  N = {n:5d}: dense {d / 1e6:9.3f} M   banded {b / 1e6:9.3f} M   ratio {d / b:6.1f}x')


if __name__ == '__main__':
    main()
