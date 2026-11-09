"""Paired significance tests (Tables 2 and 9).

  # Table 2: RDA-Net vs. the baselines of Table 1 (BCI-IV 2a, subject-dependent)
  python scripts/significance.py baselines runs/T1_sd_bci2a

  # Table 9: effect of domain adaptation (LOSO, target session E)
  python scripts/significance.py da runs/T5_loso_si runs/T5_loso_uda

RDA-Net per-subject accuracy is the mean over seeds, read from per_run.csv. Baseline
per-subject accuracies are the values of Table 1, quoted from the cited publications.
Two-sided paired t-test and exact two-sided Wilcoxon signed-rank test; with nine subjects the
smallest attainable Wilcoxon p-value is 0.0039.
"""
import csv
import sys

import numpy as np
from scipy import stats

BASELINES = {  # S01..S09, Table 1
    'CSP':           [78.3, 44.7, 82.2, 59.1, 39.7, 50.1, 68.5, 77.4, 84.2],
    'TSLDA':         [80.5, 51.3, 87.5, 59.3, 45.0, 55.3, 82.1, 84.8, 86.1],
    'MCNN':          [90.2, 63.4, 89.4, 71.2, 62.8, 47.7, 90.9, 83.7, 82.3],
    'EEG-TCNet':     [85.8, 65.0, 94.5, 64.9, 75.4, 61.4, 87.4, 83.8, 78.0],
    'Tensor-CSPNet': [80.5, 70.1, 87.8, 63.5, 74.6, 53.7, 89.6, 85.4, 70.2],
    'CTNet':         [90.97, 73.61, 94.53, 84.72, 77.08, 64.24, 86.11, 84.38, 85.07],
    'Conformer':     [88.2, 61.5, 93.4, 78.1, 52.1, 65.3, 92.3, 88.2, 88.9],
    'RGN':           [89.3, 67.3, 88.3, 74.8, 72.7, 75.9, 87.7, 85.5, 83.0],
}


def per_subject(run_dir, test='E'):
    rows = [r for r in csv.DictReader(open(f'{run_dir}/per_run.csv')) if r['test'] == test]
    subs = sorted({int(r['subject']) for r in rows})
    acc = np.array([np.mean([float(r['acc']) for r in rows if int(r['subject']) == s]) for s in subs])
    return subs, acc


def paired(a, b):
    d = a - b
    t_p = stats.ttest_rel(a, b).pvalue
    w_p = stats.wilcoxon(a, b, method='exact').pvalue if np.any(d != 0) else 1.0
    return d.mean(), d.mean() / d.std(ddof=1), t_p, w_p


def table2(run_dir):
    subs, ours = per_subject(run_dir)
    print('RDA-Net per subject:', np.round(ours, 2).tolist(), f' mean {ours.mean():.2f}')
    print(f"{'Baseline':14s} {'Acc.':>7s} {'Diff.':>7s} {'d_z':>6s} {'t-test p':>9s} {'Wilcoxon p':>11s}")
    for name, b in BASELINES.items():
        b = np.array(b)[[s - 1 for s in subs]]
        g, dz, tp, wp = paired(ours, b)
        print(f'{name:14s} {b.mean():7.2f} {g:+7.2f} {dz:6.2f} {tp:9.4f} {wp:11.4f}')


def table9(run_no_da, run_da):
    subs_a, a = per_subject(run_no_da, 'E')
    subs_b, b = per_subject(run_da, 'E')
    assert subs_a == subs_b
    print('subject   w/o DA   with DA')
    for s, x, y in zip(subs_a, a, b):
        print(f'S{s:02d}     {x:7.2f}  {y:7.2f}')
    g, dz, tp, wp = paired(b, a)
    print(f'mean     {a.mean():7.2f}  {b.mean():7.2f} (+/- {b.std(ddof=1):.2f})   gain {g:+.2f}, '
          f'improved {int((b > a).sum())}/{len(a)}, Wilcoxon p {wp:.3f}, t-test p {tp:.3f}')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'baselines':
        table2(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == 'da':
        table9(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
