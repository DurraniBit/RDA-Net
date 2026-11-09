"""RDA-Net: Riemannian Domain Adaptation Network for motor-imagery EEG classification."""
from .config import RDAConfig, DATASETS
from .model import RDANet, RiemannianBranch, LightweightCNN, LocalWindowAttention, GradientReversal
from .data import build_split, load_subject, sr_augment, SealedLabels
from .train import train_one, run_experiment, set_seed, metrics

__version__ = '1.0.0'

__all__ = ['RDAConfig', 'DATASETS', 'RDANet', 'RiemannianBranch', 'LightweightCNN', 'LocalWindowAttention',
           'GradientReversal', 'build_split', 'load_subject', 'sr_augment', 'SealedLabels', 'train_one',
           'run_experiment', 'set_seed', 'metrics']
