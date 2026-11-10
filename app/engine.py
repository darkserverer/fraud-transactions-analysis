import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vbank.common import CLASSES, MODELS, STAND
from vbank.features import CHIP, ERRORS
from vbank.models_cards import NBU_RATE

DATA = Path(__file__).resolve().parent / 'data'
MULTI = [0, 1, 2, 4, 5]


@lru_cache
def artifacts():
    cards = pd.read_parquet(DATA / 'cards.parquet').set_index('card_id')
    card_ts = pd.read_parquet(DATA / 'card_ts.parquet')
    groups = {c: (g.ts.values, g.amount.clip(lower=0).values) for c, g in card_ts.groupby('card_id')}
    merchant = pd.read_parquet(DATA / 'merchant_risk.parquet').set_index('merchant_id').merchant_risk
    mcc = pd.read_parquet(DATA / 'mcc_risk.parquet').set_index('mcc').mcc_risk
    disputes = pd.read_parquet(DATA / 'disputes.parquet').sort_values('dispute_ts')
    codes = {m: i for i, m in enumerate(json.load(open(MODELS / 'mcc_codes.json')))}
    names = json.load(open(DATA / 'mcc_names.json', encoding='utf-8'))
    return {'cards': cards, 'ts': groups, 'merchant_risk': merchant, 'mcc_risk': mcc, 'disputes': disputes,
            'codes': codes, 'mcc_names': names, 'prior_merchant': float(merchant.median()), 'prior_mcc': float(mcc.median())}


@lru_cache
def models():
    return {'binary': joblib.load(MODELS / 'hgb_final.joblib'), 'multi': joblib.load(MODELS / 'hgb_multi_final.joblib'),
            'features': json.load(open(MODELS / 'final_features.json'))}


def featurize(raw):
    a = artifacts()
    raw = raw.copy()
    raw['ts'] = pd.to_datetime(raw.ts)
    raw['card_id'] = raw.card_id.astype(int)
    raw['mcc'] = raw.mcc.astype(int)
    raw['errors'] = raw.errors.fillna('').astype(str)
    online = raw.merchant_id.astype(str).str.lower().eq('online') | raw.chip.eq('Online Transaction')
    info = a['cards'].reindex(raw.card_id)
    out = pd.DataFrame(index=raw.index)
    gaps, prev_gaps, hour_cnt, day_cnt, day_spent = [], [], [], [], []
    empty = (np.array([], dtype='datetime64[ns]'), np.array([]))
    for card, t in zip(raw.card_id, raw.ts.values):
        times, spent = a['ts'].get(card, empty)
        before = times < t
        times, spent = times[before], spent[before]
        gaps.append((t - times[-1]) / np.timedelta64(1, 'h') if len(times) else 24 * 30)
        prev_gaps.append((times[-1] - times[-2]) / np.timedelta64(1, 'h') if len(times) > 1 else 24 * 30)
        hour_cnt.append(int((times >= t - np.timedelta64(1, 'h')).sum()))
        last_day = times >= t - np.timedelta64(1, 'D')
        day_cnt.append(int(last_day.sum()))
        day_spent.append(float(spent[last_day].sum()))
    out['log_amt'] = np.sign(raw.amount) * np.log1p(raw.amount.abs())
    out['rel_amt'] = (raw.amount / info['median'].values).clip(-50, 200).values
    out['gap_h'] = np.clip(gaps, 0, 24 * 90)
    out['gap_prev_h'] = prev_gaps
    out['cnt_1h'] = hour_cnt
    out['cnt_24h'] = day_cnt
    out['log_spent_24h'] = np.log1p(day_spent)
    hour = raw.ts.dt.hour + raw.ts.dt.minute / 60
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['dow'] = raw.ts.dt.dayofweek
    out['chip'] = raw.chip.map(CHIP).fillna(2).astype(int)
    out['err_any'] = (raw.errors != '').astype(int)
    out['mcc_risk'] = raw.mcc.map(a['mcc_risk']).fillna(a['prior_mcc'])
    mid = pd.to_numeric(raw.merchant_id, errors='coerce')
    out['merchant_risk'] = mid.map(a['merchant_risk']).fillna(a['prior_merchant'])
    out['out_of_state'] = ((~online) & (raw.merchant_state.values != info.home_state.values)).astype(int)
    for col in ['card_credit', 'card_prepaid', 'card_has_chip', 'log_limit', 'log_income', 'credit_score', 'debt_ratio']:
        out[col] = info[col].values
    out['acct_age_y'] = (raw.ts.values - info.acct_open.values) / np.timedelta64(365, 'D')
    out['age'] = raw.ts.dt.year.values - info.birth_year.values
    d = a['disputes']
    clients = info.client_id.values
    out['client_disputes'] = [int(((d.client_id == c) & (d.dispute_ts < t)).sum()) for c, t in zip(clients, raw.ts)]
    out['client_disputes_online'] = [int(((d.client_id == c) & (d.dispute_ts < t) & (d.online == 1)).sum()) for c, t in zip(clients, raw.ts)]
    for e in ERRORS:
        out['err_' + e.split()[-1].lower()] = raw.errors.str.contains(e, regex=False).astype(int)
    out['mcc'] = raw.mcc.map(a['codes']).fillna(0).astype(int)
    return out[models()['features']]


def score(raw):
    a = models()
    X = featurize(raw)
    p = a['binary'].predict_proba(X)[:, 1]
    multi = a['multi'].predict_proba(X)
    labels = a['multi'].classes_
    best = labels[np.argmax(multi[:, 1:], axis=1) + 1]
    res = raw.copy()
    res['ймовірність_шахрайства'] = np.round(p, 4)
    res['найімовірніший_тип'] = [CLASSES[int(c)] for c in best]
    return res, X, multi, labels


def explain(X, top=8):
    import shap
    a = models()
    explainer = shap.TreeExplainer(a['binary'])
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    rows = pd.DataFrame({'ознака': list(X.columns), 'значення': X.iloc[0].values, 'внесок': sv[0]})
    return rows.reindex(rows.внесок.abs().sort_values(ascending=False).index).head(top)


@lru_cache
def test_scores():
    return pd.read_parquet(STAND / 'test_scores.parquet')


def budget_row(y, s, ts_days, share):
    order = np.argsort(-s)
    k = max(1, int(round(share * len(y))))
    tp = int(y[order[:k]].sum())
    fp = k - tp
    rec = tp / max(int(y.sum()), 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    nbu = rec * NBU_RATE / (rec * NBU_RATE + fpr * (1 - NBU_RATE)) if (rec or fpr) else 0
    return {'сигналів': k, 'TP': tp, 'FP': fp, 'повнота, %': round(100 * rec, 2), 'точність, %': round(100 * tp / k, 2),
            'FPR, %': round(100 * fpr, 4), 'FP за добу': round(fp / ts_days, 2), 'точність при частоті НБУ, %': round(100 * nbu, 3)}


