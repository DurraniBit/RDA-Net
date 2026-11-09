"""Training, model selection and evaluation (Section 4.3, Algorithm 1)."""
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import build_split, sr_augment
from .model import RDANet


def set_seed(seed, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(cfg):
    if cfg.device != 'auto':
        return torch.device(cfg.device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def metrics(y, p, K):
    """Accuracy (%), Cohen's kappa (Eq. 18), macro precision / recall / F1 (%)."""
    y, p = np.asarray(y), np.asarray(p)
    cm = np.zeros((K, K), np.int64)
    np.add.at(cm, (y, p), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(acc=100 * po, kappa=(po - pe) / (1 - pe), precision=100 * prec.mean(),
                recall=100 * rec.mean(), f1=100 * f1.mean())


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device))[0].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def extract_features(model, X, device, bs=256):
    """Pooled representation z (input of the classifier and of the discriminator)."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device))[1].float().cpu())
    return torch.cat(out).numpy()


def grl_alpha(cfg, progress):
    """alpha_p = 2 / (1 + exp(-10 p)) - 1, p in [0, 1] the training progress (DANN schedule)."""
    if cfg.grl_schedule == 'dann':
        return cfg.grl_alpha * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
    return cfg.grl_alpha


def train_one(cfg, split, seed, device, log_every=50):
    """Trains one model and evaluates the selected checkpoint once on the sealed test data."""
    set_seed(seed, cfg.deterministic)
    rng = np.random.RandomState(seed)
    Xtr, ytr = split['Xtr'], split['ytr']
    n, C, T = Xtr.shape
    K = cfg.n_classes
    is_loso = cfg.protocol != 'sd'
    da_on = split['n_domains'] >= 2                      # LOSO-UDA with use_da only
    epochs = cfg.loso_epochs if is_loso else cfg.epochs
    patience = cfg.loso_patience if is_loso else cfg.patience

    model = RDANet(cfg, C, T, K, split['n_domains']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))   # no weight decay
    ce = nn.CrossEntropyLoss()
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total_steps, step = epochs * steps_per_epoch, 0
    Xva, yva = split['Xva'], torch.from_numpy(split['yva'])
    n_aug_pc = cfg.n_aug * (cfg.batch_size // K)
    best, best_state, best_epoch, best_val_acc, bad, hist = float('inf'), None, -1, float('nan'), 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        run = dict(loss=0.0, cls=0.0, da=0.0, reg=0.0, correct=0, count=0)
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            parts_x, parts_y = [xb], [yb]
            if cfg.use_aug:                                  # S&R from training trials only
                xa, ya = sr_augment(Xtr, ytr, K, n_aug_pc, cfg.n_seg, rng)
                parts_x.append(xa); parts_y.append(ya)
            n_lab = sum(len(p) for p in parts_y)             # N_c: real + augmented labelled trials
            if da_on:                                        # unlabelled target trials (session T)
                ui = rng.randint(0, len(split['Xunl']), len(xb))
                parts_x.append(split['Xunl'][ui])
            x = torch.from_numpy(np.concatenate(parts_x)).unsqueeze(1).to(device)
            y = torch.from_numpy(np.concatenate(parts_y)).to(device)
            logits, z = model(x)
            loss_cls = ce(logits[:n_lab], y)                 # Eq. (3)
            loss_reg = model.fusion_reg()                    # Eq. (4)
            loss = loss_cls + cfg.lambda_reg * loss_reg
            loss_da = torch.zeros((), device=device)
            if da_on:                                        # Eq. (17): source (0) vs. target (1), N_d trials
                alpha = grl_alpha(cfg, step / max(total_steps - 1, 1))
                zd = torch.cat([z[:len(xb)], z[n_lab:]])     # augmented trials excluded
                dlab = torch.cat([torch.zeros(len(xb)), torch.ones(len(z) - n_lab)]).long().to(device)
                loss_da = ce(model.domain_logits(zd, alpha), dlab)
                loss = loss + cfg.lambda_da * loss_da
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            run['loss'] += loss.item(); run['cls'] += loss_cls.item()
            run['da'] += loss_da.item(); run['reg'] += loss_reg.item()
            run['correct'] += (logits[:len(xb)].argmax(1) == y[:len(xb)]).sum().item()
            run['count'] += len(xb)
        # ---- model selection on the held-out (source) validation set only ----
        val_logits = predict(model, Xva, device)
        val_loss = F.cross_entropy(val_logits, yva).item()
        val_acc = 100 * (val_logits.argmax(1) == yva).float().mean().item()
        rec = dict(epoch=ep, train_loss=run['loss'] / steps_per_epoch, train_cls=run['cls'] / steps_per_epoch,
                   train_da=run['da'] / steps_per_epoch, train_reg=run['reg'] / steps_per_epoch,
                   train_acc=100 * run['correct'] / max(run['count'], 1), val_loss=val_loss, val_acc=val_acc)
        hist.append(rec)
        if val_loss < best:
            best, best_epoch, bad, best_val_acc = val_loss, ep, 0, val_acc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if log_every and (ep % log_every == 0 or bad == 0):
            print(f"  ep {ep:4d} loss {rec['train_loss']:.4f} tr_acc {rec['train_acc']:.1f} "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.1f}{' *' if bad == 0 else ''}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    # ---- single final evaluation: test labels are opened here and only here ----
    results = {}
    for name, (Xte, sealed) in split['tests'].items():
        pred = predict(model, Xte, device).argmax(1).numpy()
        yte = sealed.open()
        results[name] = dict(metrics(yte, pred, K), pred=pred.tolist(), true=yte.tolist())
    info = dict(best_epoch=best_epoch, best_val_loss=best, best_val_acc=best_val_acc, epochs_run=len(hist),
                seconds=time.time() - t0, n_params=sum(p.numel() for p in model.parameters()),
                n_params_inference=model.n_inference_params(),
                chol_fail=model.riemann.n_chol_fail if model.riemann is not None else 0)
    return results, hist, info, model


def run_experiment(cfg):
    """Runs all seeds and subjects of one configuration and writes runs/<tag>/."""
    device = get_device(cfg)
    name = cfg.tag or f'{cfg.dataset}_{cfg.protocol}_{int(time.time())}'
    out = os.path.join(cfg.out_dir, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    rows = []
    for seed_base in cfg.seed_list():
        for sub in cfg.subject_list():
            seed = 1000 * seed_base + sub
            split = build_split(cfg, sub, seed)
            print(f"[{name}] subject {sub} seed {seed} train {split['Xtr'].shape} val {split['Xva'].shape} "
                  f"tests { {k: v[0].shape for k, v in split['tests'].items()} }")
            res, hist, info, model = train_one(cfg, split, seed, device)
            assert all(s.n_open == 1 for _, s in split['tests'].values()), 'test labels opened != once'
            run_dir = os.path.join(out, f'seed{seed_base}_sub{sub}')
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, 'result.json'), 'w') as f:
                json.dump(dict(subject=sub, seed=seed, results=res, info=info, audit=split['audit']), f)
            with open(os.path.join(run_dir, 'history.json'), 'w') as f:
                json.dump(hist, f)
            if cfg.save_model:
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(run_dir, 'model.pt'))
            for tname, r in res.items():
                rows.append(dict(seed=seed_base, subject=sub, test=tname,
                                 **{k: r[k] for k in ('acc', 'kappa', 'precision', 'recall', 'f1')},
                                 val_loss=info['best_val_loss'], val_acc=info['best_val_acc'],
                                 best_epoch=info['best_epoch'], params=info['n_params_inference']))
                print(f"  -> subject {sub} [{tname}] acc {r['acc']:.2f} kappa {r['kappa']:.4f} "
                      f"(selected epoch {info['best_epoch']})")
    summarise(rows, out)
    return rows


def summarise(rows, out):
    """per_run.csv (one row per seed, subject and test set) and summary.txt."""
    with open(os.path.join(out, 'per_run.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = []
    for test in sorted({r['test'] for r in rows}):
        rs = [r for r in rows if r['test'] == test]
        seeds = sorted({r['seed'] for r in rs})
        subs = sorted({r['subject'] for r in rs})
        per_sub = {s: np.mean([r['acc'] for r in rs if r['subject'] == s]) for s in subs}
        per_seed_mean = [np.mean([r['acc'] for r in rs if r['seed'] == sd]) for sd in seeds]
        accs = list(per_sub.values())
        lines.append(f'test={test}  subjects={subs}  seeds={seeds}')
        lines.append('  per-subject acc (mean over seeds): ' +
                     ', '.join(f'S{s:02d} {a:.2f}' for s, a in per_sub.items()))
        lines.append(f"  mean acc {np.mean(accs):.2f}  (std across subjects "
                     f"{np.std(accs, ddof=1) if len(accs) > 1 else 0:.2f}; std across seeds "
                     f"{np.std(per_seed_mean, ddof=1) if len(seeds) > 1 else 0:.2f})  "
                     f"kappa {np.mean([r['kappa'] for r in rs]):.4f}")
        lines.append(f"  model selection: mean validation loss {np.mean([r['val_loss'] for r in rs]):.4f}, "
                     f"mean validation acc {np.mean([r['val_acc'] for r in rs]):.2f}")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(out, 'summary.txt'), 'w') as f:
        f.write(txt + '\n')
