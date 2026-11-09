"""Data loading, evaluation protocols and leakage guards (Sections 4.1-4.2).

File format
-----------
One MATLAB file per subject and session in ``cfg.resolved_data_dir()``:

    BCI-IV 2a : A01T.mat, A01E.mat, ..., A09E.mat
    BCI-IV 2b : B01T.mat, B01E.mat, ...
    HGD       : H01T.mat, H01E.mat, ..., H14E.mat

``T`` is the training session (2a: session T; 2b: sessions 01T-03T; HGD: training set) and
``E`` the evaluation session (2a: session E; 2b: 04E-05E; HGD: test set). Each file holds
``data`` (trials x channels x samples, or the transposed layout samples x channels x trials
produced by common MATLAB preprocessing scripts) and ``label`` (one label per trial, 1..K).
"""
import os

import numpy as np
import scipy.io


# =============================================================================
# Loading
# =============================================================================
def _canonical_layout(X, n_trials, n_channels):
    """Return X as (trials, channels, samples)."""
    X = np.squeeze(np.asarray(X))
    if X.ndim != 3:
        raise ValueError(f'expected a 3-D data array, got shape {X.shape}')
    if X.shape[0] == n_trials and X.shape[1] == n_channels:
        return X
    if X.shape[2] == n_trials and X.shape[1] == n_channels:          # (samples, channels, trials)
        return X.transpose(2, 1, 0)
    t_ax = [i for i, s in enumerate(X.shape) if s == n_trials]
    c_ax = [i for i, s in enumerate(X.shape) if s == n_channels and i not in t_ax]
    if len(t_ax) != 1 or len(c_ax) != 1:
        raise ValueError(f'cannot infer the layout of data with shape {X.shape} '
                         f'({n_trials} trials, {n_channels} channels expected)')
    s_ax = ({0, 1, 2} - {t_ax[0], c_ax[0]}).pop()
    return X.transpose(t_ax[0], c_ax[0], s_ax)


def load_session_file(path, n_channels):
    """Load one session file; returns X (trials, channels, samples) float32 and y (trials,) in 0..K-1."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found; '
                                f'set --data_dir to the folder with the prepared files.')
    m = scipy.io.loadmat(path)
    y = np.asarray(m['label']).reshape(-1).astype(np.int64)
    X = _canonical_layout(m['data'], len(y), n_channels).astype(np.float32)
    if y.min() < 1:
        raise ValueError(f'{path}: labels are expected to be 1..K')
    return X, y - 1


def _synthetic_subject(sub, n_channels, n_classes, n=96, T=1000):
    """Synthetic SPD-structured trials with a subject-specific spatial shift (tests only)."""
    rng = np.random.RandomState(1000 + sub)
    C, K = n_channels, n_classes
    mix = [rng.randn(C, C) * 0.3 + np.eye(C) * (1 + 0.5 * c) for c in range(K)]
    shift = rng.randn(C, C) * 0.2 + np.eye(C)

    def make(n_):
        y = np.repeat(np.arange(K), n_ // K)
        X = np.stack([shift @ mix[c] @ rng.randn(C, T) for c in y]).astype(np.float32)
        return X, y
    Xt, yt = make(n)
    Xe, ye = make(n)
    return Xt, yt, Xe, ye


_SUBJECT_CACHE = {}


def load_subject(cfg, sub):
    """Returns (X_T, y_T, X_E, y_E) of one subject; labels are 0-based."""
    p = cfg.preset
    key = (cfg.synthetic, cfg.resolved_data_dir(), cfg.dataset, sub)
    if key not in _SUBJECT_CACHE:
        if cfg.synthetic:
            out = _synthetic_subject(sub, p['n_channels'], p['n_classes'])
        else:
            d = cfg.resolved_data_dir()
            Xt, yt = load_session_file(os.path.join(d, f"{p['prefix']}{sub:02d}T.mat"), p['n_channels'])
            Xe, ye = load_session_file(os.path.join(d, f"{p['prefix']}{sub:02d}E.mat"), p['n_channels'])
            out = (Xt, yt, Xe, ye)
        _SUBJECT_CACHE[key] = out
    return _SUBJECT_CACHE[key]


# =============================================================================
# Protocols
# =============================================================================
class SealedLabels:
    """Holds test labels. They are opened exactly once, by the final evaluation after training;
    run_experiment() asserts this for every run."""

    def __init__(self, y):
        self._y = np.asarray(y)
        self.n_open = 0

    def open(self):
        self.n_open += 1
        return self._y


def stratified_split(y, ratio, rng):
    val = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val.extend(idx[:int(round(ratio * len(idx)))])
    val = np.sort(np.array(val, dtype=np.int64))
    train = np.setdiff1d(np.arange(len(y)), val)
    return train, val


def validation_subject(target, subjects):
    """LOSO validation subject: the subject preceding the target in cyclic order (a source subject)."""
    return subjects[(subjects.index(target) - 1) % len(subjects)]


def build_split(cfg, target, seed):
    """Data split of one run.

    sd       : train / validation = stratified split of the target's session T; test = session E.
    loso_si  : train = 7 source subjects (T+E), validation = 1 held-out source subject;
               no data of the target are used; test = target T+E (and E alone).
    loso_uda : as loso_si, plus the target's session T WITHOUT labels for adaptation;
               test = target session E (disjoint from the adaptation trials).
    Normalisation statistics come from the training trials only. Test labels are sealed.
    """
    rng = np.random.RandomState(seed)
    subs = list(range(1, cfg.n_subjects + 1))
    audit = {'protocol': cfg.protocol, 'target': target}
    X_unl = None
    if cfg.protocol == 'sd':
        Xt, yt, Xe, ye = load_subject(cfg, target)
        tr, va = stratified_split(yt, cfg.val_ratio, rng)
        Xtr, ytr = Xt[tr], yt[tr]
        Xva, yva = Xt[va], yt[va]
        tests = {'E': (Xe, SealedLabels(ye))}
        audit.update(train_idx_sessionT=tr.tolist(), val_idx_sessionT=va.tolist())
    else:
        sources = [s for s in subs if s != target]
        val_sub = validation_subject(target, subs)
        train_subs = [s for s in sources if s != val_sub]
        Xvt, yvt, Xve, yve = load_subject(cfg, val_sub)
        Xva, yva = np.concatenate([Xvt, Xve]), np.concatenate([yvt, yve])
        Xs, ys = [], []
        for s in train_subs:
            Xt, yt, Xe, ye = load_subject(cfg, s)
            Xs.append(np.concatenate([Xt, Xe]))
            ys.append(np.concatenate([yt, ye]))
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        Xt_tgt, yt_tgt, Xe_tgt, ye_tgt = load_subject(cfg, target)
        audit.update(train_subjects=train_subs, val_subject=val_sub)
        if cfg.protocol == 'loso_si':
            tests = {'T+E': (np.concatenate([Xt_tgt, Xe_tgt]), SealedLabels(np.concatenate([yt_tgt, ye_tgt]))),
                     'E': (Xe_tgt, SealedLabels(ye_tgt))}
        else:
            tests = {'E': (Xe_tgt, SealedLabels(ye_tgt))}
            if cfg.da_active:
                X_unl = Xt_tgt                       # target session T; its labels are never read
                audit['target_unlabelled'] = 'session T (labels discarded)'
        del yt_tgt

    mu, sd = float(Xtr.mean()), float(Xtr.std())
    norm = lambda a: ((a - mu) / sd).astype(np.float32)
    return dict(Xtr=norm(Xtr), ytr=ytr, Xva=norm(Xva), yva=yva,
                Xunl=None if X_unl is None else norm(X_unl),
                tests={k: (norm(v[0]), v[1]) for k, v in tests.items()},
                n_domains=2 if X_unl is not None else 0, audit=audit, norm=(mu, sd))


# =============================================================================
# Segmentation and reconstruction (S&R) augmentation (Section 4.2)
# =============================================================================
def sr_augment(X, y, n_classes, n_per_class, n_seg, rng):
    """Synthetic trials whose N_s segments come from random TRAINING trials of the same class,
    kept in chronological order. X: (n, C, T)."""
    n, C, T = X.shape
    bounds = np.linspace(0, T, n_seg + 1).astype(int)
    Xa, ya = [], []
    for c in range(n_classes):
        pool = X[y == c]
        if len(pool) == 0:
            continue
        A = np.empty((n_per_class, C, T), np.float32)
        for r in range(n_per_class):
            src = rng.randint(0, len(pool), n_seg)
            for j in range(n_seg):
                A[r, :, bounds[j]:bounds[j + 1]] = pool[src[j], :, bounds[j]:bounds[j + 1]]
        Xa.append(A)
        ya.append(np.full(n_per_class, c, np.int64))
    return np.concatenate(Xa), np.concatenate(ya)
