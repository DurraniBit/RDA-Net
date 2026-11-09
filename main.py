"""Command-line entry point of RDA-Net.

Examples
--------
  # subject-dependent, BCI-IV 2a, five seeds (Table 1)
  python main.py --dataset bci2a --protocol sd --tag sd_bci2a

  # strict subject-independent LOSO (Table 5, top) and transductive LOSO-UDA (Table 5, bottom)
  python main.py --dataset bci2a --protocol loso_si  --seeds 0 --tag loso_si
  python main.py --dataset bci2a --protocol loso_uda --seeds 0 --tag loso_uda

Every option is a field of rdanet.config.RDAConfig; unspecified fields keep the defaults of
Table A.1. A JSON file with RDAConfig fields can be given with --config.
"""
import argparse
import json
from dataclasses import fields

from rdanet.config import RDAConfig
from rdanet.train import run_experiment


def _bool(s):
    if s.lower() in ('1', 'true', 'yes'):
        return True
    if s.lower() in ('0', 'false', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {s!r}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='RDA-Net training and evaluation')
    p.add_argument('--config', type=str, default=None, help='JSON file with RDAConfig fields')
    for fdef in fields(RDAConfig):
        t = _bool if fdef.type is bool else fdef.type
        p.add_argument(f'--{fdef.name}', type=t, default=None, help=f'(default: {fdef.default})')
    a = p.parse_args(argv)
    kw = {}
    if a.config:
        with open(a.config) as f:
            kw.update(json.load(f))
    for fdef in fields(RDAConfig):
        v = getattr(a, fdef.name)
        if v is not None:
            kw[fdef.name] = v
    return RDAConfig(**kw)


if __name__ == '__main__':
    run_experiment(parse_args())
