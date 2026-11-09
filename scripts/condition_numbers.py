"""Condition numbers and Cholesky failures for Table 7 (BCI-IV 2a).

  python scripts/condition_numbers.py --dataset bci2a

For every covariance-regularisation setting, the regularised covariance matrices of all
standardised session-T (training-session) trials (9 x 288 = 2592 for BCI-IV 2a) are computed
with the same code as in the model. The condition number is computed in double precision and
therefore does not depend on the SPD mapping; Cholesky failures are counted in single
precision, as used during training. With dual regularisation cond <= 1 + 1/lambda2.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rdanet.config import RDAConfig  # noqa: E402
from rdanet.data import load_subject  # noqa: E402
from rdanet.model import RiemannianBranch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='bci2a')
    p.add_argument('--data_dir', default='')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    for reg in ('none', 'tikhonov', 'dual'):
        cfg = RDAConfig(dataset=a.dataset, data_dir=a.data_dir, synthetic=a.synthetic, cov_reg=reg)
        conds, fails32, n = [], 0, 0
        for s in cfg.subject_list():
            Xt, _, _, _ = load_subject(cfg, s)
            Xt = (Xt - Xt.mean()) / Xt.std()
            R = RiemannianBranch(Xt.shape[1], cfg.emb_size, cfg)
            w = torch.linalg.eigvalsh(R.covariance(torch.from_numpy(Xt).double()))
            conds.append((w[:, -1] / w[:, 0]).numpy())
            _, info = torch.linalg.cholesky_ex(R.covariance(torch.from_numpy(Xt).float()))
            fails32 += int((info > 0).sum())
            n += len(Xt)
        c = np.concatenate(conds)
        bound = f'  (bound 1 + 1/lambda2 = {1 + 1 / cfg.lambda2:.0f})' if reg == 'dual' else ''
        print(f'{reg:9s}: mean cond(C) {c.mean():12.2f}  max {c.max():12.2f}  '
              f'float32 Cholesky failures {fails32}/{n}{bound}')


if __name__ == '__main__':
    main()
