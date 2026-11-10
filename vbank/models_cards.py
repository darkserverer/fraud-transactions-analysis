import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from vbank.common import CLASSES, MODELS, OUT, SEED, STAND, save_json
from vbank.features import CATEGORICAL, NUMERIC

plt.rcParams.update({'figure.dpi': 220, 'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
MODEL_NUMERIC = [c for c in NUMERIC if not c.startswith('cpp')]
FEATURES = MODEL_NUMERIC + CATEGORICAL
SHARES = [0.001, 0.005, 0.01, 0.02]
NBU_RATE = 27e-6


def load():
    f = pd.read_parquet(STAND / 'features.parquet')
    f = f[(f.period != 'history') & (f.cls >= 0)].reset_index(drop=True)
    codes = sorted(int(m) for m in f.mcc.unique())
    MODELS.mkdir(parents=True, exist_ok=True)
    json.dump(codes, open(MODELS / 'mcc_codes.json', 'w'))
    f['mcc'] = f.mcc.astype(int).map({m: i for i, m in enumerate(codes)})
    return f


def metrics(y, s):
    return {'roc_auc': round(roc_auc_score(y, s), 4), 'pr_auc': round(average_precision_score(y, s), 4)}


def budget(y, s, ts):
    days = (ts.max() - ts.min()).total_seconds() / 86400
    order = np.argsort(-s)
    neg = int((y == 0).sum())
    pos = int(y.sum())
    rows = []
    for share in SHARES:
        k = int(round(share * len(y)))
        tp = int(y[order[:k]].sum())
        fp = k - tp
        rec = tp / pos
        fpr = fp / neg
        prec = tp / k
        nbu = rec * NBU_RATE / (rec * NBU_RATE + fpr * (1 - NBU_RATE)) if rec or fpr else 0
        rows.append({'share_pct': share * 100, 'alerts': k, 'tp': tp, 'fp': fp, 'recall_pct': round(100 * rec, 2),
                     'precision_pct': round(100 * prec, 2), 'fpr_pct': round(100 * fpr, 4), 'fp_per_day': round(fp / days, 2),
                     'precision_nbu_pct': round(100 * nbu, 3)})
    return rows


def fit_hgb(Xtr, ytr, Xva, yva, weight='balanced', max_rounds=40, lr=0.05):
    model = HistGradientBoostingClassifier(learning_rate=lr, max_iter=25, max_leaf_nodes=63, min_samples_leaf=80,
                                           l2_regularization=1.0, class_weight=weight, categorical_features=CATEGORICAL,
                                           warm_start=True, early_stopping=False, random_state=SEED)
    best, best_iter, stale = -1, 25, 0
    for r in range(1, max_rounds + 1):
        model.max_iter = 25 * r
        model.fit(Xtr, ytr)
        ap = average_precision_score(yva, model.predict_proba(Xva)[:, 1])
        if ap > best + 1e-4:
            best, best_iter, stale = ap, model.max_iter, 0
        else:
            stale += 1
            if stale >= 4:
                break
    final = HistGradientBoostingClassifier(learning_rate=lr, max_iter=best_iter, max_leaf_nodes=63, min_samples_leaf=80,
                                           l2_regularization=1.0, class_weight=weight, categorical_features=CATEGORICAL,
                                           random_state=SEED).fit(Xtr, ytr)
    return final, best_iter, round(best, 4)


def undersample(X, y, ratio, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = rng.choice(neg, min(len(neg), ratio * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return X.iloc[idx], y[idx]


def plot_pr(curves, name):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f'{label} (PR-AUC {average_precision_score(y, s):.3f})', lw=1.3)
    ax.set_xlabel('повнота (recall)')
    ax.set_ylabel('точність (precision)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def plot_confusion(cm, labels, name):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n{100 * norm[i, j]:.1f} %', ha='center', va='center', fontsize=6.5,
                    color='white' if norm[i, j] > .55 else 'black')
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right', fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel('передбачений клас')
    ax.set_ylabel('справжній клас')
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def plot_budget(y, s, ts, name):
    days = (ts.max() - ts.min()).total_seconds() / 86400
    order = np.argsort(-s)
    ys = y[order]
    shares = np.linspace(0.0005, 0.03, 120)
    rec, fpd = [], []
    for sh in shares:
        k = int(sh * len(y))
        tp = ys[:k].sum()
        rec.append(100 * tp / y.sum())
        fpd.append((k - tp) / days)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(100 * shares, rec, color='#c0392b', label='повнота, %')
    ax2 = ax.twinx()
    ax2.plot(100 * shares, fpd, color='#2c6fbb', label='хибні тривоги за добу')
    ax.set_xlabel('частка операцій, відправлених на перевірку, %')
    ax.set_ylabel('повнота, %')
    ax2.set_ylabel('FP за добу')
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    f = load()
    tr, va, te = f[f.period == 'train'], f[f.period == 'valid'], f[f.period == 'test']
    Xtr, Xva, Xte = tr[FEATURES], va[FEATURES], te[FEATURES]
    ytr, yva, yte = tr.y_known.values, va.y.values, te.y.values
    R = {'rows': {p: int(len(g)) for p, g in f.groupby('period')},
         'positives': {p: int(g.y.sum()) for p, g in f.groupby('period')},
         'positives_known_train': int(ytr.sum()),
         'prevalence_test_pct': round(100 * yte.mean(), 4)}
    print(R, round(time.time() - t0))

    amount_cut = float(np.percentile(tr.amount, 99))
    R['baseline_rule'] = {'threshold': round(amount_cut, 2), **metrics(yte, (te.amount > amount_cut).astype(float).values),
                          'budget': budget(yte, (te.amount > amount_cut).astype(float).values + 1e-9 * te.amount.values, te.ts)}

    hgb, it, ap_va = fit_hgb(Xtr, ytr, Xva, yva)
    s_hgb = hgb.predict_proba(Xte)[:, 1]
    R['hgb'] = {'iterations': it, 'valid_pr_auc': ap_va, **metrics(yte, s_hgb), 'budget': budget(yte, s_hgb, te.ts),
                'train': metrics(ytr, hgb.predict_proba(Xtr)[:, 1])}
    joblib.dump(hgb, MODELS / 'hgb_binary.joblib')
    print('hgb', R['hgb']['pr_auc'], round(time.time() - t0))
    social = pd.read_parquet(STAND / 'features.parquet')
    social = social[social.source == 'scenario_social'].copy()
    social['mcc'] = social.mcc.astype(int).map({m: i for i, m in enumerate(json.load(open(MODELS / 'mcc_codes.json')))}).fillna(0).astype(int)
    s_soc = hgb.predict_proba(social[FEATURES])[:, 1]
    ranked = np.sort(s_hgb)[::-1]
    R['social_engineering'] = {'ops': int(len(social)), 'score_median': round(float(np.median(s_soc)), 4),
                               'flagged_pct': {f'{100 * sh:g} %': round(100 * float((s_soc >= ranked[int(sh * len(ranked)) - 1]).mean()), 1) for sh in SHARES},
                               'rel_amt_median': round(float(social.rel_amt.median()), 2)}

    num = MODEL_NUMERIC
    scaler = StandardScaler().fit(Xtr[num])
    Ztr, Zva, Zte = (pd.DataFrame(scaler.transform(X[num]).clip(-8, 8), columns=num) for X in (Xtr, Xva, Xte))
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.5).fit(Ztr, ytr)
    s_lr = lr.predict_proba(Zte)[:, 1]
    R['lr'] = {**metrics(yte, s_lr), 'budget': budget(yte, s_lr, te.ts),
               'coef': dict(sorted(zip(num, np.round(lr.coef_[0], 3)), key=lambda kv: -abs(kv[1]))[:12])}
    joblib.dump({'scaler': scaler, 'model': lr, 'features': num}, MODELS / 'lr.joblib')
    print('lr', R['lr']['pr_auc'], round(time.time() - t0))

    Xu, yu = undersample(Xtr, ytr, 20, rng)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features='sqrt', n_jobs=-1,
                                class_weight='balanced_subsample', random_state=SEED).fit(Xu, yu)
    s_rf = rf.predict_proba(Xte)[:, 1]
    R['rf'] = {'train_rows': int(len(yu)), **metrics(yte, s_rf), 'budget': budget(yte, s_rf, te.ts),
               'importance': dict(sorted(zip(FEATURES, np.round(rf.feature_importances_, 4)), key=lambda kv: -kv[1])[:12])}
    print('rf', R['rf']['pr_auc'], round(time.time() - t0))

    normal = Xtr[ytr == 0][num].sample(300_000, random_state=SEED)
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=SEED, n_jobs=-1).fit(normal)
    s_if = -iso.score_samples(Xte[num])
    R['iforest'] = {**metrics(yte, s_if), 'budget': budget(yte, s_if, te.ts)}
    sample = te.sample(200_000, random_state=SEED)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(scaler.transform(normal.sample(60_000, random_state=SEED)).clip(-8, 8))
    s_lof = -lof.score_samples(scaler.transform(sample[num]).clip(-8, 8))
    R['lof'] = {'test_sample': len(sample), 'sample_positives': int(sample.y.sum()), **metrics(sample.y.values, s_lof)}
    print('unsupervised', R['iforest']['pr_auc'], R['lof']['pr_auc'], round(time.time() - t0))

    bal = {}
    variants = {'ваги класів': (Xtr, ytr, 'balanced'), 'зменшення більшого класу 20:1': (*undersample(Xtr, ytr, 20, rng), None)}
    Xs, ys = undersample(Xtr, ytr, 60, rng)
    sm = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=SEED)
    Xsm, ysm = sm.fit_resample(Xs[num], ys)
    for name, (X, y, w) in variants.items():
        m, _, _ = fit_hgb(X, y, Xva, yva, weight=w)
        s = m.predict_proba(Xte)[:, 1]
        l = LogisticRegression(class_weight=w, max_iter=2000, C=0.5).fit(scaler.transform(X[num]).clip(-8, 8), y)
        bal[name] = {'hgb': metrics(yte, s), 'lr': metrics(yte, l.predict_proba(Zte)[:, 1]), 'train_rows': int(len(y))}
    hgb_num = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=63, min_samples_leaf=80,
                                             random_state=SEED).fit(Xsm, ysm)
    l = LogisticRegression(max_iter=2000, C=0.5).fit(scaler.transform(Xsm).clip(-8, 8), ysm)
    bal['SMOTE до 1:10'] = {'hgb': metrics(yte, hgb_num.predict_proba(Xte[num])[:, 1]),
                            'lr': metrics(yte, l.predict_proba(Zte)[:, 1]), 'train_rows': int(len(ysm))}
    R['balancing'] = bal
    print('balancing', round(time.time() - t0))

    no_copy = tr['copy'] == 0
    m, _, _ = fit_hgb(Xtr[no_copy.values], ytr[no_copy.values], Xva, yva)
    R['without_sparkov_copies'] = metrics(yte, m.predict_proba(Xte)[:, 1])
    m, _, _ = fit_hgb(Xtr, tr.y.values, Xva, yva)
    R['oracle_labels'] = metrics(yte, m.predict_proba(Xte)[:, 1])
    print('ablations', round(time.time() - t0))

    order = np.argsort(-s_hgb)
    flagged = np.zeros(len(te), bool)
    flagged[order[:int(0.01 * len(te))]] = True
    R['recall_by_class_at_1pct'] = {CLASSES[c]: round(100 * flagged[(te.cls == c).values].mean(), 2)
                                    for c in sorted(te.cls.unique()) if c != 0}
    flagged5 = np.zeros(len(te), bool)
    flagged5[order[:int(0.005 * len(te))]] = True
    R['recall_by_class_at_05pct'] = {CLASSES[c]: round(100 * flagged5[(te.cls == c).values].mean(), 2)
                                     for c in sorted(te.cls.unique()) if c != 0}
    missed = te[(te.y == 1) & ~flagged]
    caught = te[(te.y == 1) & flagged]
    R['missed_vs_caught'] = {col: {'missed': round(float(missed[col].median()), 3), 'caught': round(float(caught[col].median()), 3)}
                             for col in ['rel_amt', 'gap_h', 'merchant_risk', 'mcc_risk', 'chip', 'client_disputes', 'cpp_exposed']}

    multi_classes = [0, 1, 2, 4, 5]
    mtr = tr[tr.cls.isin(multi_classes)]
    multi = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, min_samples_leaf=40,
                                           class_weight='balanced', categorical_features=CATEGORICAL, random_state=SEED,
                                           early_stopping=False).fit(mtr[FEATURES], mtr.cls.values)
    mte = te[te.cls.isin(multi_classes)]
    pred = multi.predict(mte[FEATURES])
    cm = confusion_matrix(mte.cls.values, pred, labels=multi_classes)
    rep = classification_report(mte.cls.values, pred, labels=multi_classes, output_dict=True, zero_division=0)
    R['multiclass'] = {'macro_f1': round(f1_score(mte.cls.values, pred, labels=multi_classes, average='macro'), 4),
                       'per_class': {CLASSES[c]: {k: round(v, 4) for k, v in rep[str(c)].items()} for c in multi_classes},
                       'confusion': cm.tolist()}
    fr_te = mte[mte.cls > 0]
    fr_tr = mtr[mtr.cls > 0]
    typer = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
                                           class_weight='balanced', categorical_features=CATEGORICAL, random_state=SEED,
                                           early_stopping=False).fit(fr_tr[FEATURES], fr_tr.cls.values)
    tp = typer.predict(fr_te[FEATURES])
    R['type_given_fraud'] = {'macro_f1': round(f1_score(fr_te.cls.values, tp, average='macro'), 4),
                             'confusion': confusion_matrix(fr_te.cls.values, tp, labels=[1, 2, 4, 5]).tolist()}
    joblib.dump(multi, MODELS / 'hgb_multi.joblib')
    plot_confusion(cm, [CLASSES[c] for c in multi_classes], 'рис_confusion.png')
    print('multiclass', R['multiclass']['macro_f1'], round(time.time() - t0))

    import shap
    sample_x = Xte.sample(4000, random_state=SEED)
    explainer = shap.TreeExplainer(hgb)
    sv = explainer.shap_values(sample_x)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    R['shap_top'] = imp.head(12).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    imp.head(12)[::-1].plot.barh(ax=ax, color='#2c6fbb')
    ax.set_xlabel('середнє |SHAP|')
    fig.tight_layout()
    fig.savefig(OUT / 'рис_shap.png')
    plt.close(fig)

    plot_pr({'бустинг': (yte, s_hgb), 'випадковий ліс': (yte, s_rf), 'логістична регресія': (yte, s_lr),
             'Isolation Forest': (yte, s_if)}, 'рис_pr.png')
    plot_budget(yte, s_hgb, te.ts, 'рис_budget.png')
    te[['op_id', 'ts', 'cls', 'y', 'scenario']].assign(score=s_hgb, score_lr=s_lr, score_rf=s_rf, score_if=s_if).to_parquet(STAND / 'test_scores.parquet', index=False)
    R['seconds'] = round(time.time() - t0)
    save_json(R, 'cards_results.json')
    print('done', R['seconds'])


if __name__ == '__main__':
    main()
