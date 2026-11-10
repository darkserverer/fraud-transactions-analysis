import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn

from vbank.common import CLASSES, MODELS, OUT, SEED, STAND, TRAIN_END, VALID_END, save_json
from vbank.features import CATEGORICAL, COUNTERS, FINAL
from vbank.models_cards import FEATURES, budget, load, metrics
from vbank.models_gru import LENGTH, STEP, Net, score as gru_score, windows

VARIANTS = [
    ('лічильники', True, True, None),
    ('лічильники, без копій Sparkov', True, False, None),
    ('лічильники, без копій, останні 12 місяців', True, False, 12),
    ('лічильники, без копій, останні 6 місяців', True, False, 6),
]
MULTI = [0, 1, 2, 4, 5]


def hgb(n, weight='balanced'):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=n, max_leaf_nodes=63, min_samples_leaf=80,
                                          l2_regularization=1.0, class_weight=weight, categorical_features=CATEGORICAL,
                                          warm_start=True, early_stopping=False, random_state=SEED)


def columns(counters):
    return FEATURES[:-1] + (COUNTERS if counters else []) + CATEGORICAL


def matrix(f, cols, final):
    X = f[cols].copy()
    if final:
        for base, new in FINAL.items():
            X[base] = f[new].values
    return X


def rows(f, end, copies, months):
    keep = np.array(f.ts < end)
    if not copies:
        keep &= (f['copy'] == 0).values
    if months:
        keep &= (f.ts >= end - pd.DateOffset(months=months)).values
    return keep


def search(Xtr, ytr, Xva, yva, Xte, max_rounds=48):
    model = hgb(25)
    best, best_iter, stale, best_test = -1, 25, 0, None
    for r in range(1, max_rounds + 1):
        model.max_iter = 25 * r
        model.fit(Xtr, ytr)
        ap = average_precision_score(yva, model.predict_proba(Xva)[:, 1])
        if ap > best + 1e-4:
            best, best_iter, stale = ap, model.max_iter, 0
            best_test = model.predict_proba(Xte)[:, 1]
        else:
            stale += 1
            if stale >= 4:
                break
    return best_iter, round(best, 4), best_test


def place(f):
    state = f.merchant_state.fillna('')
    return np.where(f.merchant_city == 'ONLINE', 'online', np.where(state.str.len() == 2, 'home', 'abroad'))


def recall_by(groups, y, s, share):
    flag = np.zeros(len(s), bool)
    flag[np.argsort(-s)[:int(share * len(s))]] = True
    return {str(g): round(100 * flag[(groups == g) & (y == 1)].mean(), 2) for g in pd.unique(groups[y == 1])}


def gru_frame(final):
    cols = ['log_amt', 'rel_amt', 'gap_h', 'hour_sin', 'hour_cos', 'chip', 'err_any', 'mcc_risk', 'merchant_risk', 'out_of_state']
    f = pd.read_parquet(STAND / 'features.parquet', columns=['op_id', 'ts', 'period', 'card_id', 'cls', 'y', 'y_known', 'y_known_f',
                                                             'copy'] + cols + list(FINAL.values()))
    if final:
        for base, new in FINAL.items():
            f[base] = f[new]
    f = f.sort_values(['card_id', 'ts', 'op_id'], kind='stable').reset_index(drop=True)
    for k in range(3):
        f[f'chip{k}'] = (f.chip == k).astype(np.float32)
    f['rel_amt'] = np.sign(f.rel_amt) * np.log1p(f.rel_amt.abs())
    f['gap_h'] = np.log1p(f.gap_h)
    f['mcc_risk'] = np.log(f.mcc_risk)
    f['merchant_risk'] = np.log(f.merchant_risk)
    fit = f[f.period.isin(['train', 'valid'] if final else ['train'])]
    mu, sd = fit[STEP].mean(), fit[STEP].std().replace(0, 1)
    X = ((f[STEP] - mu) / sd).clip(-8, 8).astype(np.float32).values
    start = f.groupby('card_id').cumcount().values
    return f, X, start, mu, sd


def train_gru(f, X, start, idx, labels, epochs, va=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    y = labels[idx]
    pos, neg = idx[y == 1], idx[y == 0]
    model = Net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ratio = 30
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(ratio)))
    best, best_state, best_epoch, history = -1, None, epochs, []
    for epoch in range(epochs):
        model.train()
        batch = np.concatenate([pos, rng.choice(neg, ratio * len(pos), replace=False)])
        rng.shuffle(batch)
        yb = torch.from_numpy(labels[batch].astype(np.float32))
        for a in range(0, len(batch), 1024):
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(windows(X, start, batch[a:a + 1024]))), yb[a:a + 1024])
            loss.backward()
            opt.step()
        if va is not None:
            ap = average_precision_score(f.y.values[va], gru_score(model, X, start, va))
            history.append(round(ap, 4))
            if ap > best:
                best, best_state, best_epoch = ap, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1
    if va is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, round(best, 4), history


def main():
    t0 = time.time()
    torch.set_num_threads(8)
    f = load()
    tr_all = (f.period == 'train').values
    va = (f.period == 'valid').values
    te = (f.period == 'test').values
    yte = f.y.values[te]
    base = json.load(open(OUT / 'cards_results.json', encoding='utf-8'))['hgb']
    R = {'base': {k: base[k] for k in ['valid_pr_auc', 'iterations', 'roc_auc', 'pr_auc']}}
    stage = []
    for name, counters, copies, months in VARIANTS:
        cols = columns(counters)
        keep = rows(f, TRAIN_END, copies, months)
        X = matrix(f, cols, False)
        it, ap_va, s = search(X[keep], f.y_known.values[keep], X[va], f.y.values[va], X[te])
        stage.append({'name': name, 'counters': counters, 'copies': copies, 'months': months, 'rows': int(keep.sum()),
                      'positives': int(f.y_known.values[keep].sum()), 'iterations': it, 'valid_pr_auc': ap_va,
                      'test': metrics(yte, s)})
        print(stage[-1], round(time.time() - t0))
    R['stage1'] = stage
    best = max(stage, key=lambda v: v['valid_pr_auc'])
    if best['valid_pr_auc'] <= base['valid_pr_auc']:
        best = {'name': 'базова', 'counters': False, 'copies': True, 'months': None, 'iterations': base['iterations']}
    R['chosen'] = best['name']

    s_ref = {}
    for label, cfg in [('base_refit', {'counters': False, 'copies': True, 'months': None, 'iterations': base['iterations']}),
                       ('final', best)]:
        cols = columns(cfg['counters'])
        keep = rows(f, VALID_END, cfg['copies'], cfg['months'])
        X = matrix(f, cols, True)
        model = hgb(cfg['iterations']).fit(X[keep], f.y_known_f.values[keep])
        s_ref[label] = model.predict_proba(X[te])[:, 1]
        R[label] = {'rows': int(keep.sum()), 'positives': int(f.y_known_f.values[keep].sum()), 'iterations': cfg['iterations'],
                    **metrics(yte, s_ref[label]), 'budget': budget(yte, s_ref[label], f.ts[te])}
        print(label, R[label]['pr_auc'], round(time.time() - t0))
        if label == 'final':
            final_model, final_cols, final_keep = model, cols, keep
    joblib.dump(final_model, MODELS / 'hgb_final.joblib')
    json.dump(final_cols, open(MODELS / 'final_features.json', 'w'))

    Xf = matrix(f, final_cols, True)
    multi = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, min_samples_leaf=40,
                                           class_weight='balanced', categorical_features=CATEGORICAL, random_state=SEED,
                                           early_stopping=False)
    mk = final_keep & f.cls.isin(MULTI).values
    multi.fit(Xf[mk], f.cls.values[mk])
    mte = te & f.cls.isin(MULTI).values
    pred = multi.predict(Xf[mte])
    R['multiclass'] = {'macro_f1': round(f1_score(f.cls.values[mte], pred, labels=MULTI, average='macro'), 4),
                       'per_class_f1': {CLASSES[c]: round(v, 4) for c, v in zip(MULTI, f1_score(f.cls.values[mte], pred, labels=MULTI, average=None))},
                       'confusion': confusion_matrix(f.cls.values[mte], pred, labels=MULTI).tolist()}
    joblib.dump(multi, MODELS / 'hgb_multi_final.joblib')
    print('multi', R['multiclass']['macro_f1'], round(time.time() - t0))

    import shap
    sample = Xf[te].sample(4000, random_state=SEED)
    sv = shap.TreeExplainer(final_model).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    R['shap_top'] = pd.Series(np.abs(sv).mean(0), index=final_cols).sort_values(ascending=False).head(12).round(4).to_dict()

    g_stage = {}
    gf, GX, gstart, _, _ = gru_frame(False)
    lab = (gf.cls >= 0).values
    gva = np.flatnonzero(lab & (gf.period == 'valid').values)
    for name, copies in [('з копіями', True), ('без копій', False)]:
        idx = np.flatnonzero(lab & (gf.period == 'train').values & ((gf['copy'] == 0).values | copies))
        _, epoch, ap, hist = train_gru(gf, GX, gstart, idx, gf.y_known.values, 10, gva)
        g_stage[name] = {'best_epoch': epoch, 'valid_pr_auc': ap, 'history': hist}
        print('gru', name, ap, epoch, round(time.time() - t0))
    g_name = max(g_stage, key=lambda k: g_stage[k]['valid_pr_auc'])
    R['gru_stage1'] = g_stage
    R['gru_chosen'] = g_name
    gf, GX, gstart, mu, sd = gru_frame(True)
    lab = (gf.cls >= 0).values
    idx = np.flatnonzero(lab & gf.period.isin(['train', 'valid']).values & ((gf['copy'] == 0).values | (g_name == 'з копіями')))
    model, _, _, _ = train_gru(gf, GX, gstart, idx, gf.y_known_f.values, g_stage[g_name]['best_epoch'])
    gte = np.flatnonzero(lab & (gf.period == 'test').values)
    g_scores = pd.Series(gru_score(model, GX, gstart, gte), index=gf.op_id.values[gte])
    torch.save({'state': model.state_dict(), 'mu': mu.to_dict(), 'sd': sd.to_dict(), 'step': STEP, 'length': LENGTH},
               MODELS / 'gru_final.pt')

    scores = pd.read_parquet(STAND / 'test_scores.parquet')
    for col in ['score_final', 'score_gru_final']:
        if col in scores:
            scores = scores.drop(columns=col)
    scores = scores.merge(pd.DataFrame({'op_id': f.op_id.values[te], 'score_final': s_ref['final']}), on='op_id')
    scores['score_gru_final'] = g_scores.reindex(scores.op_id).values
    blend = (scores.score_final.rank(pct=True) + scores.score_gru_final.rank(pct=True)).values / 2
    y = scores.y.values
    R['gru_final'] = {**metrics(y, scores.score_gru_final.values), 'budget': budget(y, scores.score_gru_final.values, scores.ts)}
    R['blend_final'] = {**metrics(y, blend), 'budget': budget(y, blend, scores.ts)}
    pl = place(f[te].set_index('op_id').reindex(scores.op_id))
    R['recall_by_place_at_05pct'] = {name: recall_by(pl, y, v, 0.005) for name, v in
                                     [('бустинг', scores.score.values), ('бустинг, остаточний', scores.score_final.values),
                                      ('середнє рангів, остаточне', blend)]}
    R['recall_by_class_at_05pct'] = {name: recall_by(scores.cls.map(CLASSES).values, y, v, 0.005) for name, v in
                                     [('бустинг, остаточний', scores.score_final.values), ('GRU, остаточна', scores.score_gru_final.values),
                                      ('середнє рангів, остаточне', blend)]}
    scores.to_parquet(STAND / 'test_scores.parquet', index=False)
    R['seconds'] = round(time.time() - t0)
    save_json(R, 'improve_results.json')
    print(json.dumps({k: v for k, v in R.items() if k not in ('stage1',)}, ensure_ascii=False, default=str)[:3000])


if __name__ == '__main__':
    main()
