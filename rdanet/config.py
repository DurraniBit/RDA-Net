"""Single configuration of RDA-Net.

Every experiment in the paper uses the defaults below (Appendix A, Table A.1). Each table row
of the paper changes only the fields that define that row (protocol, ablated component, SPD
mapping, covariance regularisation or the swept hyperparameter); see scripts/reproduce_all.sh.
"""
from dataclasses import dataclass

# Dataset presets. The data directory holds one file per subject and session (see data.py).
DATASETS = {
    'bci2a': dict(prefix='A', n_subjects=9, n_channels=22, n_classes=4, data_dir='data/BCICIV_2a'),
    'bci2b': dict(prefix='B', n_subjects=9, n_channels=3, n_classes=2, data_dir='data/BCICIV_2b'),
    'hgd': dict(prefix='H', n_subjects=14, n_channels=44, n_classes=4, data_dir='data/HGD'),
}
DATASET_ALIASES = {'A': 'bci2a', 'B': 'bci2b', 'HGD': 'hgd', '2a': 'bci2a', '2b': 'bci2b'}

PROTOCOLS = ('sd', 'loso_si', 'loso_uda')


@dataclass
class RDAConfig:
    # ---------------- data / protocol ----------------
    dataset: str = 'bci2a'             # 'bci2a' | 'bci2b' | 'hgd'
    data_dir: str = ''                 # '' -> preset directory of the dataset
    subjects: str = 'all'              # 'all' or a comma-separated list, e.g. '1,3'
    protocol: str = 'sd'               # 'sd' | 'loso_si' | 'loso_uda' (Section 4.1)
    synthetic: bool = False            # synthetic data; for unit and smoke tests only
    # ---------------- architecture (Sections 3.3-3.7) ----------------
    emb_size: int = 40                 # d_emb
    depth: int = 6                     # Transformer layers
    heads: int = 10                    # h
    window: int = 5                    # w: token i attends to token j iff |i - j| <= w
    attn_impl: str = 'banded'          # 'banded' O(N(2w+1)d) | 'masked' dense O(N^2 d), identical output
    F1: int = 16
    D: int = 2
    kernel1: int = 64                  # k_1
    kernel2: int = 16                  # k_2
    pool1: int = 8
    pool2: int = 8
    riemann_hidden: int = 256
    disc_hidden: int = 64
    dropout_cnn: float = 0.3
    dropout_transformer: float = 0.3
    dropout_attn: float = 0.1
    dropout_riemann: float = 0.2
    dropout_disc: float = 0.3
    # ---------------- Riemannian branch (Section 3.3) ----------------
    spd_map: str = 'log_cholesky'      # 'log_cholesky' (Eq. 9) | 'log_euclidean' (Eq. 8, comparison only)
    cov_reg: str = 'dual'              # 'none' | 'tikhonov' (Eq. 6) | 'dual' (Eqs. 6-7)
    lambda1: float = 1e-5              # Tikhonov
    lambda2: float = 1e-3              # per-trial trace-scaled shrinkage; cond <= 1 + 1/lambda2
    # ---------------- ablation switches (Table 8) ----------------
    use_riemann: bool = True
    use_cnn: bool = True
    use_transformer: bool = True
    fusion: str = 'adaptive'           # 'adaptive' (W_fusion + L_reg) | 'concat' (no W_fusion, no L_reg)
    use_aug: bool = True
    use_da: bool = True                # effective only under 'loso_uda' (Table 9)
    # ---------------- optimisation (Section 4.3, Table A.1) ----------------
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 72
    epochs: int = 1000                 # SD
    patience: int = 300                # SD early stopping (validation loss)
    loso_epochs: int = 300             # LOSO
    loso_patience: int = 100           # LOSO early stopping (source validation loss)
    grad_clip: float = 1.0
    lambda_reg: float = 1e-4           # weight of ||W_fusion||_F^2
    lambda_da: float = 0.3             # weight of L_da
    grl_schedule: str = 'dann'         # 'dann': alpha_p = 2 / (1 + exp(-10 p)) - 1 | 'const'
    grl_alpha: float = 1.0             # maximum (dann) or constant GRL coefficient
    n_aug: int = 2                     # S&R: N_aug * floor(B / K) synthetic trials per class and batch
    n_seg: int = 8                     # S&R: number of segments N_s
    val_ratio: float = 0.2             # SD: stratified fraction of session T held out for validation
    # ---------------- reproducibility / output ----------------
    seeds: str = '0,1,2,3,4'           # per-run seed = 1000 * seed + subject
    deterministic: bool = True
    device: str = 'auto'
    out_dir: str = 'runs'
    tag: str = ''
    save_model: bool = False           # store the selected checkpoint (needed for Fig. 3)

    def __post_init__(self):
        self.dataset = DATASET_ALIASES.get(self.dataset, self.dataset)
        self.validate()

    def validate(self):
        assert self.dataset in DATASETS, f'unknown dataset {self.dataset!r}; choose from {list(DATASETS)}'
        assert self.protocol in PROTOCOLS, f'unknown protocol {self.protocol!r}; choose from {PROTOCOLS}'
        assert self.spd_map in ('log_cholesky', 'log_euclidean'), self.spd_map
        assert self.cov_reg in ('none', 'tikhonov', 'dual'), self.cov_reg
        assert self.attn_impl in ('banded', 'masked'), self.attn_impl
        assert self.fusion in ('adaptive', 'concat'), self.fusion
        assert self.grl_schedule in ('dann', 'const'), self.grl_schedule
        assert self.emb_size % self.heads == 0, 'heads must divide emb_size'
        assert self.use_riemann or self.use_cnn, 'at least one feature branch is required'
        assert 0.0 < self.val_ratio < 1.0, 'a validation set is required for model selection'

    @property
    def preset(self):
        return DATASETS[self.dataset]

    @property
    def n_subjects(self):
        return self.preset['n_subjects']

    @property
    def n_classes(self):
        return self.preset['n_classes']

    def resolved_data_dir(self):
        return self.data_dir or self.preset['data_dir']

    def seed_list(self):
        return [int(s) for s in str(self.seeds).split(',') if s.strip() != '']

    def subject_list(self):
        if self.subjects == 'all':
            return list(range(1, self.n_subjects + 1))
        return [int(s) for s in str(self.subjects).split(',') if s.strip() != '']

    @property
    def da_active(self):
        """Domain adaptation is used only under the transductive protocol (Section 3.7)."""
        return self.protocol == 'loso_uda' and self.use_da
