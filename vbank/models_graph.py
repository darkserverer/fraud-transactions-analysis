import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from vbank.common import MODELS, SEED, STAND, save_json
from vbank.figures import typologies

HOURS = np.timedelta64(1, 'h')
EDGE = ['log_amt', 'rel_amt', 'cross_currency', 'same_bank', 'self_loop', 'hour_sin', 'hour_cos', 'dow', 'format',
        'currency', 'to_next_out_h', 'next_out_ratio', 'since_prev_in_h', 'prev_in_ratio']
NODE = ['out_deg', 'in_deg', 'uniq_out', 'uniq_in', 'log_out_sum', 'log_in_sum', 'pass_ratio', 'reciprocal',
        'formats', 'currencies', 'active_days', 'max_out_day', 'max_in_day']


def edge_features(t):
    med = t.groupby('currency_received').amount_received.transform('median')
    t['rel'] = t.amount_received / med
    t['log_amt'] = np.log1p(t.amount_received)
    t['rel_amt'] = np.log1p(t.rel)
    t['cross_currency'] = (t.currency_received != t.currency_paid).astype(int)
    t['same_bank'] = (t.from_bank == t.to_bank).astype(int)
    t['self_loop'] = (t.src == t.dst).astype(int)
    h = t.ts.dt.hour + t.ts.dt.minute / 60
    t['hour_sin'] = np.sin(2 * np.pi * h / 24)
    t['hour_cos'] = np.cos(2 * np.pi * h / 24)
    t['dow'] = t.ts.dt.dayofweek
    t['format'] = t.format.astype('category').cat.codes
    t['currency'] = t.currency_received.astype('category').cat.codes
    t['row'] = np.arange(len(t))
    out = t[['src', 'ts', 'rel', 'row']].rename(columns={'src': 'acct', 'ts': 'out_ts', 'rel': 'out_rel', 'row': 'out_row'})
    inc = t[['dst', 'ts', 'rel', 'row']].rename(columns={'dst': 'acct', 'ts': 'in_ts', 'rel': 'in_rel', 'row': 'in_row'})
    q = inc.rename(columns={'in_ts': 'ts'}).sort_values('ts')
    nxt = pd.merge_asof(q, out.rename(columns={'out_ts': 'ts_out'}).assign(ts=lambda d: d.ts_out).sort_values('ts'),
                        on='ts', by='acct', direction='forward', allow_exact_matches=False)
    nxt = nxt.set_index('in_row')
    t['to_next_out_h'] = ((nxt.ts_out - nxt.ts) / HOURS).reindex(t.row).fillna(24 * 30).values
    t['next_out_ratio'] = (nxt.out_rel / nxt.in_rel).reindex(t.row).fillna(0).clip(0, 20).values
    q = out.rename(columns={'out_ts': 'ts'}).sort_values('ts')
    prv = pd.merge_asof(q, inc.rename(columns={'in_ts': 'ts_in'}).assign(ts=lambda d: d.ts_in).sort_values('ts'),
                        on='ts', by='acct', direction='backward', allow_exact_matches=False)
    prv = prv.set_index('out_row')
    t['since_prev_in_h'] = ((prv.ts - prv.ts_in) / HOURS).reindex(t.row).fillna(24 * 30).values
    t['prev_in_ratio'] = (prv.out_rel / prv.in_rel).reindex(t.row).fillna(0).clip(0, 20).values
    return t


def node_features(t):
    day = t.ts.dt.floor('D')
    out = t.groupby('src').agg(out_deg=('dst', 'size'), uniq_out=('dst', 'nunique'), out_sum=('rel', 'sum'))
    inc = t.groupby('dst').agg(in_deg=('src', 'size'), uniq_in=('src', 'nunique'), in_sum=('rel', 'sum'))
    n = out.join(inc, how='outer').fillna(0)
    n['log_out_sum'] = np.log1p(n.out_sum)
    n['log_in_sum'] = np.log1p(n.in_sum)
    n['pass_ratio'] = np.minimum(n.out_sum, n.in_sum) / np.maximum(np.maximum(n.out_sum, n.in_sum), 1e-9)
    pairs = t[t.src != t.dst][['src', 'dst']].drop_duplicates()
    rev = pairs.merge(pairs.rename(columns={'src': 'dst', 'dst': 'src'}), on=['src', 'dst'])
    n['reciprocal'] = rev.groupby('src').size().reindex(n.index).fillna(0)
    both = pd.concat([t[['src', 'format', 'currency_received', 'ts']].rename(columns={'src': 'acct'}),
                      t[['dst', 'format', 'currency_received', 'ts']].rename(columns={'dst': 'acct'})])
    g = both.groupby('acct')
    n['formats'] = g.format.nunique().reindex(n.index).fillna(0)
    n['currencies'] = g.currency_received.nunique().reindex(n.index).fillna(0)
    n['active_days'] = both.assign(d=both.ts.dt.floor('D')).groupby('acct').d.nunique().reindex(n.index).fillna(0)
    n['max_out_day'] = t.assign(d=day).groupby(['src', 'd']).dst.nunique().groupby(level=0).max().reindex(n.index).fillna(0)
    n['max_in_day'] = t.assign(d=day).groupby(['dst', 'd']).src.nunique().groupby(level=0).max().reindex(n.index).fillna(0)
    return n[NODE]


class Sage(nn.Module):
    def __init__(self, dim, edim, hidden=48):
        super().__init__()
        self.l1 = nn.Linear(3 * dim, hidden)
        self.l2 = nn.Linear(3 * hidden, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden + edim, 64), nn.ReLU(), nn.Linear(64, 1))

    def embed(self, x, a_in, a_out):
        h = torch.relu(self.l1(torch.cat([x, torch.sparse.mm(a_in, x), torch.sparse.mm(a_out, x)], 1)))
        return torch.relu(self.l2(torch.cat([h, torch.sparse.mm(a_in, h), torch.sparse.mm(a_out, h)], 1)))

    def forward(self, h, s, d, e):
        return self.head(torch.cat([h[s], h[d], e], 1)).squeeze(1)


def mean_adj(rows, cols, n):
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    vals = 1.0 / np.maximum(deg[rows], 1)
    idx = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals.astype(np.float32)), (n, n)).coalesce()


def budget(y, s, days):
    order = np.argsort(-s)
    out = []
    for share in [0.001, 0.005, 0.01]:
        k = int(round(share * len(y)))
        tp = int(y[order[:k]].sum())
        out.append({'share_pct': share * 100, 'tp': tp, 'fp': k - tp, 'recall_pct': round(100 * tp / y.sum(), 2),
                    'precision_pct': round(100 * tp / k, 2), 'fp_per_day': round((k - tp) / days, 1)})
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    t = pd.read_parquet(STAND / 'transfers.parquet')
    t = t.sort_values('ts', kind='stable').reset_index(drop=True)
    t = edge_features(t)
    nodes = node_features(t)
    print('features', round(time.time() - t0))
    q1, q2 = t.ts.quantile(0.6), t.ts.quantile(0.8)
    t['split'] = np.where(t.ts < q1, 'train', np.where(t.ts < q2, 'valid', 'test'))
    X = pd.concat([t[EDGE].reset_index(drop=True),
                   nodes.reindex(t.src).add_prefix('s_').reset_index(drop=True),
                   nodes.reindex(t.dst).add_prefix('d_').reset_index(drop=True)], axis=1)
    y = t.laundering.values
    tr, va, te = (t.split == s for s in ['train', 'valid', 'test'])
    days = (t.ts[te].max() - t.ts[te].min()).total_seconds() / 86400
    R = {'edges': len(t), 'nodes': len(nodes), 'laundering': int(y.sum()),
         'split': {s: {'edges': int((t.split == s).sum()), 'laundering': int(y[(t.split == s).values].sum())} for s in ['train', 'valid', 'test']},
         'split_bounds': [str(q1), str(q2)]}
    hgb_edge = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, class_weight='balanced',
                                              categorical_features=['format', 'currency'], early_stopping=False, random_state=SEED)
    hgb_edge.fit(X.loc[tr.values, EDGE[:10]], y[tr.values])
    s0 = hgb_edge.predict_proba(X.loc[te.values, EDGE[:10]])[:, 1]
    R['hgb_edge_only'] = {'roc_auc': round(roc_auc_score(y[te.values], s0), 4), 'pr_auc': round(average_precision_score(y[te.values], s0), 4)}
    hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, class_weight='balanced',
                                         categorical_features=['format', 'currency'], early_stopping=False, random_state=SEED)
    hgb.fit(X[tr.values], y[tr.values])
    s1 = hgb.predict_proba(X[te.values])[:, 1]
    R['hgb_graph_features'] = {'roc_auc': round(roc_auc_score(y[te.values], s1), 4), 'pr_auc': round(average_precision_score(y[te.values], s1), 4),
                               'valid_pr_auc': round(average_precision_score(y[va.values], hgb.predict_proba(X[va.values])[:, 1]), 4),
                               'budget': budget(y[te.values], s1, days)}
    print('hgb', R['hgb_graph_features'], round(time.time() - t0))

    acct = pd.Index(nodes.index)
    s_idx = acct.get_indexer(t.src)
    d_idx = acct.get_indexer(t.dst)
    n = len(acct)
    a_in = mean_adj(d_idx, s_idx, n)
    a_out = mean_adj(s_idx, d_idx, n)
    nf = np.log1p(nodes.values.astype(np.float32))
    nf = (nf - nf.mean(0)) / np.maximum(nf.std(0), 1e-6)
    xn = torch.from_numpy(nf.astype(np.float32))
    ef = t[EDGE].astype(np.float32).values
    ef[:, [0, 1, 10, 12]] = np.log1p(np.abs(ef[:, [0, 1, 10, 12]]))
    ef = (ef - ef[tr.values].mean(0)) / np.maximum(ef[tr.values].std(0), 1e-6)
    ef = torch.from_numpy(ef.astype(np.float32))
    st, dt = torch.from_numpy(s_idx), torch.from_numpy(d_idx)
    model = Sage(xn.shape[1], ef.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    pos = np.flatnonzero(tr.values & (y == 1))
    neg = np.flatnonzero(tr.values & (y == 0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
    va_idx = torch.from_numpy(np.flatnonzero(va.values))
    best, best_state, hist = -1, None, []
    for epoch in range(200):
        model.train()
        idx = torch.from_numpy(np.concatenate([pos, rng.choice(neg, 100 * len(pos), replace=False)]))
        h = model.embed(xn, a_in, a_out)
        logit = model(h, st[idx], dt[idx], ef[idx])
        loss = loss_fn(logit, torch.from_numpy(y[idx.numpy()].astype(np.float32)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                h = model.embed(xn, a_in, a_out)
                sv = torch.sigmoid(model(h, st[va_idx], dt[va_idx], ef[va_idx])).numpy()
            ap = average_precision_score(y[va.values], sv)
            hist.append(round(ap, 4))
            print('sage', epoch, round(ap, 4), round(time.time() - t0))
            if ap > best:
                best, best_state = ap, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    te_idx = torch.from_numpy(np.flatnonzero(te.values))
    with torch.no_grad():
        h = model.embed(xn, a_in, a_out)
        s2 = torch.sigmoid(model(h, st[te_idx], dt[te_idx], ef[te_idx])).numpy()
    R['graphsage'] = {'valid_history': hist, 'roc_auc': round(roc_auc_score(y[te.values], s2), 4),
                      'pr_auc': round(average_precision_score(y[te.values], s2), 4), 'budget': budget(y[te.values], s2, days)}
    rank = (pd.Series(s1).rank(pct=True) + pd.Series(s2).rank(pct=True)).values
    R['blend'] = {'pr_auc': round(average_precision_score(y[te.values], rank), 4)}
    torch.save(best_state, MODELS / 'graphsage.pt')
    print('sage', R['graphsage']['pr_auc'], round(time.time() - t0))

    lab = t[t.pattern.notna()].copy()
    Xl = X.loc[lab.index]
    yl = lab.pattern.values
    pred = np.empty(len(lab), dtype=object)
    for a, b in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xl, yl, lab.attack.values):
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, class_weight='balanced',
                                           categorical_features=['format', 'currency'], early_stopping=False,
                                           random_state=SEED).fit(Xl.iloc[a], yl[a])
        pred[b] = m.predict(Xl.iloc[b])
    names = sorted(set(yl))
    R['typology'] = {'edges': len(lab), 'attacks': int(lab.attack.nunique()),
                     'macro_f1': round(f1_score(yl, pred, average='macro'), 4),
                     'per_class_f1': dict(zip(names, np.round(f1_score(yl, pred, labels=names, average=None), 3))),
                     'labels': names, 'confusion': confusion_matrix(yl, pred, labels=names).tolist()}
    typologies(t)
    R['seconds'] = round(time.time() - t0)
    save_json(R, 'graph_results.json')
    print(R['typology']['macro_f1'], R['seconds'])


if __name__ == '__main__':
    main()
