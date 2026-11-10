import json
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson

from vbank.common import OUT, STAND, TRAIN_END, VALID_END, save_json

ERRORS = ['Insufficient Balance', 'Bad PIN', 'Technical Glitch', 'Bad Card Number', 'Bad Expiration', 'Bad CVV', 'Bad Zipcode']
CHIP = {'Swipe Transaction': 0, 'Chip Transaction': 1, 'Online Transaction': 2}
CPP_LOOKBACK = pd.Timedelta(days=60)
CPP_REPORT_LOOKBACK = pd.Timedelta(days=120)
CPP_EXPOSURE = pd.Timedelta(days=90)
CPP_MIN_REPORTS = 6
CPP_P_VALUE = 1e-3
FRAUD = [1, 2, 4, 5]

NUMERIC = ['log_amt', 'rel_amt', 'gap_h', 'gap_prev_h', 'hour_sin', 'hour_cos', 'dow', 'chip', 'err_any',
           'mcc_risk', 'merchant_risk', 'out_of_state', 'card_credit', 'card_prepaid', 'card_has_chip', 'log_limit',
           'acct_age_y', 'age', 'log_income', 'credit_score', 'debt_ratio', 'cpp_exposed', 'cpp_days',
           'client_disputes', 'client_disputes_online'] + ['err_' + e.split()[-1].lower() for e in ERRORS]
CATEGORICAL = ['mcc']
COUNTERS = ['cnt_1h', 'cnt_24h', 'log_spent_24h']
FINAL = {'rel_amt': 'rel_amt_f', 'mcc_risk': 'mcc_risk_f', 'merchant_risk': 'merchant_risk_f'}


def smooth_rate(keys, y, k=50):
    prior = y.mean()
    grouped = pd.DataFrame({'k': keys.values, 'y': y.values}).groupby('k').y.agg(['sum', 'size'])
    return (grouped['sum'] + k * prior) / (grouped['size'] + k), prior


def known_labels(ops, disputes, cutoff):
    known = disputes[disputes.dispute_ts < cutoff].op_id
    return (ops.op_id.isin(known)).astype(int)


def cpp_flags(ops, disputes):
    fr = disputes[disputes.cls.isin([1, 2, 4])].sort_values(['card_id', 'ts'])
    fr['episode'] = ((fr.card_id != fr.card_id.shift()) | (fr.ts - fr.ts.shift() > pd.Timedelta(days=30))).cumsum()
    first = fr.groupby('episode').agg(card_id=('card_id', 'first'), first_fraud=('ts', 'min'), report=('dispute_ts', 'min')).reset_index()
    visits = ops[(ops.merchant_city != 'ONLINE') & ops.card_id.isin(first.card_id)][['card_id', 'merchant_id', 'ts']]
    visits = visits.merge(first, on='card_id')
    visits = visits[(visits.ts < visits.first_fraud) & (visits.ts >= visits.first_fraud - CPP_REPORT_LOOKBACK)]
    reports = visits.drop_duplicates(['episode', 'merchant_id'])[['merchant_id', 'card_id', 'report']].sort_values('report')
    pos = ops[ops.merchant_city != 'ONLINE'][['merchant_id', 'card_id', 'ts']].sort_values(['merchant_id', 'ts'], kind='stable')
    m_arr = pos.merchant_id.values
    ts_arr = pos.ts.values
    card_arr = pos.card_id.values
    look = np.timedelta64(CPP_LOOKBACK)

    def window(merchant, t, spans=1):
        a, b = np.searchsorted(m_arr, merchant, 'left'), np.searchsorted(m_arr, merchant, 'right')
        lo = a + np.searchsorted(ts_arr[a:b], t - spans * look, 'right')
        hi = a + np.searchsorted(ts_arr[a:b], t, 'right')
        return card_arr[lo:hi]

    flags = []
    first_seen = pos.groupby('merchant_id').ts.min()
    counts = reports.groupby('merchant_id').size()
    for merchant in counts[counts >= CPP_MIN_REPORTS].index:
        times = reports.loc[reports.merchant_id == merchant, 'report'].values
        start = first_seen[merchant].to_datetime64()
        ready = times[0]
        for t in times:
            if t < ready:
                continue
            recent = int(((times <= t) & (times > t - look)).sum())
            windows = max(1.0, ((t - look) - start) / look)
            past = int((times <= t - look).sum()) / windows
            if recent >= CPP_MIN_REPORTS and poisson.sf(recent - 1, max(past, 0.5)) < CPP_P_VALUE:
                flags.append((merchant, pd.Timestamp(t), recent, round(past, 2), len(np.unique(window(merchant, t)))))
                ready = t + look
    flags = pd.DataFrame(flags, columns=['merchant_id', 'flag_ts', 'reported_cards', 'past_rate', 'visiting_cards'])
    exposed = []
    for f in flags.itertuples():
        exposed += [(c, f.flag_ts, f.merchant_id) for c in np.unique(window(f.merchant_id, f.flag_ts.to_datetime64(), 2))]
    exposed = pd.DataFrame(exposed, columns=['card_id', 'exposed_ts', 'merchant_id'])
    return flags, exposed.sort_values('exposed_ts')


def build():
    t0 = time.time()
    scenario_merchants = [m['merchant'] for m in json.load(open(OUT / 'stand_summary.json', encoding='utf-8'))['merchant_scenarios']]
    ops = pd.read_parquet(STAND / 'ops.parquet')
    disputes = pd.read_parquet(STAND / 'disputes.parquet')
    cards = pd.read_parquet(STAND / 'cards.parquet').set_index('card_id')
    clients = pd.read_parquet(STAND / 'clients.parquet').set_index('client_id')
    ops = ops.sort_values(['card_id', 'ts', 'op_id'], kind='stable').reset_index(drop=True)
    prev = ops.groupby('card_id').ts.shift()
    ops['gap_h'] = ((ops.ts - prev).dt.total_seconds() / 3600).fillna(24 * 30).clip(upper=24 * 90)
    ops['gap_prev_h'] = ops.groupby('card_id').gap_h.shift().fillna(24 * 30)
    key = ops.card_id.values.astype(np.int64) * 10 ** 11 + ops.ts.values.astype('datetime64[s]').astype(np.int64)
    first = np.searchsorted(key, key, 'left')
    spent = np.r_[0, np.cumsum(ops.amount.clip(lower=0).values)]
    hour_ago = np.searchsorted(key, key - 3600, 'left')
    day_ago = np.searchsorted(key, key - 86400, 'left')
    ops['cnt_1h'] = first - hour_ago
    ops['cnt_24h'] = first - day_ago
    ops['log_spent_24h'] = np.log1p(spent[first] - spent[day_ago])
    train_normal = ops[(ops.ts < TRAIN_END) & (ops.cls == 0) & (ops.amount > 0)]
    med = train_normal.groupby('card_id').amount.median()
    client_med = train_normal.groupby('client_id').amount.median()
    base = ops.card_id.map(med).fillna(ops.client_id.map(client_med)).fillna(train_normal.amount.median())
    ops['rel_amt'] = (ops.amount / base).clip(-50, 200)
    ops['log_amt'] = np.sign(ops.amount) * np.log1p(ops.amount.abs())
    hour = ops.ts.dt.hour + ops.ts.dt.minute / 60
    ops['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    ops['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    ops['dow'] = ops.ts.dt.dayofweek
    ops['chip'] = ops.chip.map(CHIP).astype(int)
    ops['err_any'] = (ops.errors != '').astype(int)
    for e in ERRORS:
        ops['err_' + e.split()[-1].lower()] = ops.errors.str.contains(e, regex=False).astype(int)
    train = ops[(ops.ts < TRAIN_END) & (ops.cls >= 0) & (ops['copy'] == 0)]
    y_train = known_labels(train, disputes, TRAIN_END)
    rate, prior = smooth_rate(train.mcc, y_train)
    ops['mcc_risk'] = ops.mcc.map(rate).fillna(prior)
    rate, prior = smooth_rate(train.merchant_id, y_train)
    ops['merchant_risk'] = ops.merchant_id.map(rate).fillna(prior)
    normal_f = ops[(ops.ts < VALID_END) & (ops.cls == 0) & (ops.amount > 0)]
    med_f = normal_f.groupby('card_id').amount.median()
    client_med_f = normal_f.groupby('client_id').amount.median()
    base_f = ops.card_id.map(med_f).fillna(ops.client_id.map(client_med_f)).fillna(normal_f.amount.median())
    ops['rel_amt_f'] = (ops.amount / base_f).clip(-50, 200)
    fit = ops[(ops.ts < VALID_END) & (ops.cls >= 0) & (ops['copy'] == 0)]
    y_fit = known_labels(fit, disputes, VALID_END)
    rate, prior = smooth_rate(fit.mcc, y_fit)
    ops['mcc_risk_f'] = ops.mcc.map(rate).fillna(prior)
    rate, prior = smooth_rate(fit.merchant_id, y_fit)
    ops['merchant_risk_f'] = ops.merchant_id.map(rate).fillna(prior)
    home = ops[ops.merchant_city != 'ONLINE'].groupby('client_id').merchant_state.agg(lambda s: s.mode().iat[0])
    ops['out_of_state'] = ((ops.merchant_city != 'ONLINE') & (ops.merchant_state != ops.client_id.map(home))).astype(int)
    c = cards.reindex(ops.card_id)
    ops['card_credit'] = (c.card_type.values == 'Credit').astype(int)
    ops['card_prepaid'] = (c.card_type.values == 'Debit (Prepaid)').astype(int)
    ops['card_has_chip'] = (c.has_chip.values == 'YES').astype(int)
    ops['log_limit'] = np.log1p(c.credit_limit.values)
    opened = pd.to_datetime(c.acct_open_date.values, format='%m/%Y')
    ops['acct_age_y'] = (ops.ts.values - opened.values) / np.timedelta64(365, 'D')
    u = clients.reindex(ops.client_id)
    ops['age'] = ops.ts.dt.year.values - u.birth_year.values
    ops['log_income'] = np.log1p(u.yearly_income.values)
    ops['credit_score'] = u.credit_score.values
    ops['debt_ratio'] = u.total_debt.values / np.maximum(u.yearly_income.values, 1)
    flags, exposed = cpp_flags(ops, disputes)
    left = ops[['ts', 'card_id']].reset_index().assign(ts=lambda d: d.ts.astype('datetime64[ns]')).sort_values('ts')
    right = exposed[['card_id', 'exposed_ts']].assign(exposed_ts=lambda d: d.exposed_ts.astype('datetime64[ns]')).sort_values('exposed_ts')
    hit = pd.merge_asof(left, right, left_on='ts', right_on='exposed_ts', by='card_id', direction='backward')
    ex = hit.set_index('index').exposed_ts.reindex(ops.index)
    since = (ops.ts.astype('datetime64[ns]') - ex).dt.total_seconds() / 86400
    ops['cpp_exposed'] = ((since >= 0) & (since <= CPP_EXPOSURE.days)).astype(int)
    ops['cpp_days'] = since.where(ops.cpp_exposed == 1, -1).fillna(-1)
    d = disputes.sort_values('dispute_ts')
    online_ops = set(ops.loc[ops.chip == 2, 'op_id'])
    d['online'] = d.op_id.isin(online_ops).astype(int)
    ops = ops.sort_values('ts', kind='stable').reset_index(drop=True)
    for col, sub in [('client_disputes', d), ('client_disputes_online', d[d.online == 1])]:
        ev = sub[['client_id', 'dispute_ts']].rename(columns={'dispute_ts': 'ts'}).sort_values('ts')
        ev['ts'] = ev.ts.astype('datetime64[ns]')
        ev['n'] = ev.groupby('client_id').cumcount() + 1
        left = ops[['ts', 'client_id']].reset_index()
        left['ts'] = left.ts.astype('datetime64[ns]')
        merged = pd.merge_asof(left, ev, on='ts', by='client_id', allow_exact_matches=False)
        ops[col] = merged.set_index('index').n.reindex(ops.index).fillna(0).values
    ops['y'] = ops.cls.isin(FRAUD).astype(int)
    ops['y_known'] = 0
    known = set(disputes[disputes.dispute_ts < TRAIN_END].op_id)
    ops.loc[ops.op_id.isin(known), 'y_known'] = 1
    ops.loc[ops.period != 'train', 'y_known'] = ops.loc[ops.period != 'train', 'y']
    ops['y_known_f'] = ops.op_id.isin(set(disputes[disputes.dispute_ts < VALID_END].op_id)).astype(int)
    ops.loc[ops.period == 'test', 'y_known_f'] = ops.loc[ops.period == 'test', 'y']
    keep = ['op_id', 'ts', 'period', 'client_id', 'card_id', 'merchant_id', 'merchant_city', 'merchant_state', 'amount',
            'cls', 'y', 'y_known', 'y_known_f', 'source', 'copy', 'scenario'] + NUMERIC + CATEGORICAL + COUNTERS + list(FINAL.values())
    ops[keep].to_parquet(STAND / 'features.parquet', index=False)
    flags.to_parquet(STAND / 'cpp_flags.parquet', index=False)
    lab = ops[(ops.period != 'history') & (ops.cls >= 0)]
    summary = {
        'rows': len(ops), 'labelled_in_window': len(lab), 'features': len(NUMERIC) + len(CATEGORICAL),
        'cpp_flags': len(flags), 'cpp_flags_scenario': int(flags.merchant_id.isin(scenario_merchants).sum()),
        'scenario_merchants': len(scenario_merchants),
        'cpp_exposed_cards': int(exposed.card_id.nunique()),
        'cpp_exposed_by_class': lab.groupby('cls').cpp_exposed.mean().round(4).to_dict(),
        'delayed_train_positives': int(((ops.period == 'train') & (ops.y == 1) & (ops.y_known == 0)).sum()),
        'delayed_final_positives': int(((ops.period.isin(['train', 'valid'])) & (ops.y == 1) & (ops.y_known_f == 0)).sum()),
        'final_positives': int(((ops.period.isin(['train', 'valid'])) & (ops.y == 1)).sum()),
        'counters_median_by_class': {c: lab.groupby('cls')[c].median().round(3).to_dict() for c in COUNTERS},
        'train_positives': int(((ops.period == 'train') & (ops.y == 1)).sum()),
        'rel_amt_median_by_class': lab.groupby('cls').rel_amt.median().round(3).to_dict(),
        'gap_h_median_by_class': lab.groupby('cls').gap_h.median().round(3).to_dict(),
        'client_disputes_mean_by_class': lab.groupby('cls').client_disputes.mean().round(3).to_dict(),
        'seconds': round(time.time() - t0),
    }
    save_json(summary, 'features_summary.json')
    print(summary)


if __name__ == '__main__':
    build()
