import json
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vbank.common import CLASSES, OUT, SEED, STAND, save_json
from vbank.features import CATEGORICAL, NUMERIC

plt.rcParams.update({'figure.dpi': 220, 'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
COLS = [c for c in NUMERIC if not c.startswith('cpp') and not c.startswith('client_')] + CATEGORICAL


def c2st(a, b):
    X = pd.concat([a[COLS], b[COLS]])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, categorical_features=CATEGORICAL, random_state=SEED)
    auc = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='roc_auc')
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    model.fit(Xa, ya)
    imp = permutation_importance(model, Xb, yb, scoring='roc_auc', n_repeats=5, random_state=SEED)
    top = pd.Series(imp.importances_mean, index=COLS).sort_values(ascending=False).head(5).round(4).to_dict()
    return {'roc_auc': round(float(auc.mean()), 4), 'roc_auc_sd': round(float(auc.std()), 4), 'top_features': top}


def compare(a, b):
    out = {}
    for col, fn in [('rel_amt', np.log1p), ('gap_h', np.log1p), ('hour', None)]:
        x = a[col] if fn is None else fn(a[col].clip(lower=0))
        z = b[col] if fn is None else fn(b[col].clip(lower=0))
        ks = ks_2samp(x, z)
        out[col] = {'median_a': round(float(a[col].median()), 3), 'median_b': round(float(b[col].median()), 3),
                    'ks': round(float(ks.statistic), 3), 'p': float(ks.pvalue)}
    return out


def cpp_report(f):
    meta = pd.DataFrame(json.load(open(OUT / 'stand_summary.json', encoding='utf-8'))['merchant_scenarios'])
    flags = pd.read_parquet(STAND / 'cpp_flags.parquet')
    attacks = f[f.cls == 4]
    rows = []
    for m in meta.itertuples():
        start = pd.Timestamp(m.window_start)
        own = flags[(flags.merchant_id == m.merchant) & (flags.flag_ts >= start)].flag_ts
        first = own.min() if len(own) else pd.NaT
        hits = attacks[attacks.scenario == m.scenario]
        after = float((hits.ts >= first).mean()) if pd.notna(first) and len(hits) else 0.0
        rows.append({'scenario': m.scenario, 'period': m.period, 'flagged': bool(pd.notna(first)),
                     'delay_days': round((first - start).total_seconds() / 86400, 1) if pd.notna(first) else None,
                     'attacks': int(len(hits)), 'attacks_after_flag_pct': round(100 * after, 1)})
    r = pd.DataFrame(rows)
    scen_merchants = set(meta.merchant)
    return {'scenarios': len(r), 'flagged': int(r.flagged.sum()),
            'delay_median_days': float(r.delay_days.median()) if r.flagged.any() else None,
            'attacks_after_flag_pct': round(float(100 * (r.attacks * r.attacks_after_flag_pct / 100).sum() / r.attacks.sum()), 1),
            'flags_total': len(flags), 'flags_other_merchants': int((~flags.merchant_id.isin(scen_merchants)).sum()),
            'flags_by_year': flags.flag_ts.dt.year.value_counts().sort_index().to_dict(), 'rows': rows}


def timeline(f, name):
    test = f[(f.period == 'test') & (f.cls == 2)]
    card = test.groupby('card_id').size().sort_values().index[-1]
    day = test[test.card_id == card].ts.min().normalize()
    g = f[(f.card_id == card) & (f.ts >= day - pd.Timedelta(days=6)) & (f.ts < day + pd.Timedelta(days=4))]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    colors = {0: '#7f8c8d', 1: '#e67e22', 2: '#c0392b', 4: '#8e44ad', 5: '#16a085', -1: '#bdc3c7'}
    for c, part in g.groupby('cls'):
        ax.scatter(part.ts, part.amount, s=14, color=colors[c], label=CLASSES.get(c, 'без мітки'))
    ax.set_ylabel('сума, $')
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    return {'card': int(card), 'ops': len(g), 'fraud_ops': int((g.cls == 2).sum())}


def main():
    t0 = time.time()
    f = pd.read_parquet(STAND / 'features.parquet')
    f = f[f.period != 'history'].copy()
    f['hour'] = f.ts.dt.hour
    f['mcc'] = f.mcc.astype('category').cat.codes
    real_series = f[(f.cls == 2) & (f.source == 'ftd') & (f.period == 'train')]
    copies = f[f.source == 'sparkov']
    merchant = f[f.source == 'scenario_merchant']
    real_fraud = f[f.cls.isin([1, 2]) & (f.source == 'ftd')]
    R = {'copies_vs_real_series': {'n': [len(copies), len(real_series)], 'distributions': compare(copies, real_series),
                                   'c2st': c2st(copies, real_series)},
         'merchant_vs_real_fraud': {'n': [len(merchant), len(real_fraud)], 'distributions': compare(merchant, real_fraud),
                                    'c2st': c2st(merchant, real_fraud)},
         'cpp': cpp_report(f), 'timeline': timeline(f, 'рис_timeline.png'), 'seconds': round(time.time() - t0)}
    save_json(R, 'realism_results.json')
    print(json.dumps({k: v for k, v in R.items() if k != 'cpp'}, ensure_ascii=False, default=str)[:2000])
    print({k: v for k, v in R['cpp'].items() if k != 'rows'})


if __name__ == '__main__':
    main()
