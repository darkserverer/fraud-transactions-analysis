import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from vbank.common import MODELS, SEED, STAND, save_json

STEP = ['log_amt', 'rel_amt', 'gap_h', 'hour_sin', 'hour_cos', 'chip0', 'chip1', 'chip2', 'err_any', 'mcc_risk',
        'merchant_risk', 'out_of_state']
LENGTH = 12


class Net(nn.Module):
    def __init__(self, dim, hidden=48):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], x[:, -1]], 1)).squeeze(1)


def prepare():
    f = pd.read_parquet(STAND / 'features.parquet', columns=['op_id', 'ts', 'period', 'card_id', 'cls', 'y', 'y_known'] +
                        ['log_amt', 'rel_amt', 'gap_h', 'hour_sin', 'hour_cos', 'chip', 'err_any', 'mcc_risk', 'merchant_risk',
                         'out_of_state'])
    f = f.sort_values(['card_id', 'ts', 'op_id'], kind='stable').reset_index(drop=True)
    for k in range(3):
        f[f'chip{k}'] = (f.chip == k).astype(np.float32)
    f['rel_amt'] = np.sign(f.rel_amt) * np.log1p(f.rel_amt.abs())
    f['gap_h'] = np.log1p(f.gap_h)
    f['mcc_risk'] = np.log(f.mcc_risk)
    f['merchant_risk'] = np.log(f.merchant_risk)
    train = f[f.period == 'train']
    mu, sd = train[STEP].mean(), train[STEP].std().replace(0, 1)
    X = ((f[STEP] - mu) / sd).clip(-8, 8).astype(np.float32).values
    start = f.groupby('card_id').cumcount().values
    return f, X, start, mu, sd


def windows(X, start, idx):
    out = np.zeros((len(idx), LENGTH, X.shape[1]), np.float32)
    for back in range(LENGTH):
        ok = start[idx] >= back
        out[ok, LENGTH - 1 - back] = X[idx[ok] - back]
    return out


def score(model, X, start, idx, batch=8192):
    model.eval()
    res = []
    with torch.no_grad():
        for a in range(0, len(idx), batch):
            res.append(torch.sigmoid(model(torch.from_numpy(windows(X, start, idx[a:a + batch])))).numpy())
    return np.concatenate(res)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(8)
    f, X, start, mu, sd = prepare()
    lab = (f.cls >= 0).values
    tr = np.flatnonzero(lab & (f.period == 'train').values)
    va = np.flatnonzero(lab & (f.period == 'valid').values)
    te = np.flatnonzero(lab & (f.period == 'test').values)
    ytr = f.y_known.values[tr]
    pos, neg = tr[ytr == 1], tr[ytr == 0]
    model = Net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ratio = 30
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(ratio)))
    best, best_state, history = -1, None, []
    yva = f.y.values[va]
    for epoch in range(10):
        model.train()
        idx = np.concatenate([pos, rng.choice(neg, ratio * len(pos), replace=False)])
        rng.shuffle(idx)
        y = torch.from_numpy(f.y_known.values[idx].astype(np.float32))
        for a in range(0, len(idx), 1024):
            xb = torch.from_numpy(windows(X, start, idx[a:a + 1024]))
            opt.zero_grad()
            loss = loss_fn(model(xb), y[a:a + 1024])
            loss.backward()
            opt.step()
        ap = average_precision_score(yva, score(model, X, start, va))
        history.append(round(ap, 4))
        print(epoch, round(ap, 4), round(time.time() - t0))
        if ap > best:
            best, best_state = ap, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    s = score(model, X, start, te)
    yte = f.y.values[te]
    test = f.iloc[te][['op_id']].assign(score_gru=s)
    scores = pd.read_parquet(STAND / 'test_scores.parquet').merge(test, on='op_id')
    def rank(v):
        return pd.Series(v).rank(pct=True).values
    blend = (rank(scores.score) + rank(scores.score_gru)) / 2
    k = int(0.01 * len(scores))
    top_h = set(scores.op_id.values[np.argsort(-scores.score.values)[:k]])
    top_g = set(scores.op_id.values[np.argsort(-scores.score_gru.values)[:k]])
    fraud = set(scores.op_id[scores.y == 1])
    R = {'valid_history': history, 'valid_best': round(best, 4),
         'test': {'roc_auc': round(roc_auc_score(yte, s), 4), 'pr_auc': round(average_precision_score(yte, s), 4)},
         'blend_with_hgb': {'roc_auc': round(roc_auc_score(scores.y, blend), 4), 'pr_auc': round(average_precision_score(scores.y, blend), 4)},
         'hgb_pr_auc': round(average_precision_score(scores.y, scores.score), 4),
         'at_1pct': {'hgb_only': len((top_h - top_g) & fraud), 'gru_only': len((top_g - top_h) & fraud),
                     'both': len(top_h & top_g & fraud), 'fraud_total': len(fraud)},
         'recall_by_class_at_1pct': {int(c): round(100 * scores[(scores.cls == c)].op_id.isin(top_g).mean(), 2)
                                     for c in sorted(scores.cls.unique()) if c != 0},
         'seconds': round(time.time() - t0)}
    scores.to_parquet(STAND / 'test_scores.parquet', index=False)
    torch.save({'state': best_state, 'mu': mu.to_dict(), 'sd': sd.to_dict(), 'step': STEP, 'length': LENGTH}, MODELS / 'gru.pt')
    save_json(R, 'gru_results.json')
    print(R)


if __name__ == '__main__':
    main()
