import time

import numpy as np
import pandas as pd

from vbank.common import (CFPB, END, SEED, SPARKOV_MCC, STAND, START, TRAIN_END, VALID_END, load_aml,
                          load_aml_patterns, load_cards, load_ftd, load_sparkov, load_users, period_of, save_json)

HISTORY = pd.Timedelta(days=92)
EPISODE_GAP = pd.Timedelta(days=3)
WEEK = pd.Timedelta(days=7)
AML_SHIFT_WEEKS = 156


def episodes(ts, card):
    frame = pd.DataFrame({'ts': ts.values, 'card': card.values}, index=ts.index).sort_values(['card', 'ts'])
    new = (frame.card != frame.card.shift()) | (frame.ts - frame.ts.shift() > EPISODE_GAP)
    return new.cumsum().reindex(ts.index)


def base_ops(rng):
    tx = load_ftd()
    tx = tx[(tx.ts >= START - HISTORY) & (tx.ts < END)].copy()
    tx['y'] = tx.label.map({'Yes': 1, 'No': 0})
    fraud = tx[tx.y == 1]
    per_day = fraud.groupby([fraud.card_id, fraud.ts.dt.floor('D')]).op_id.transform('size')
    tx['cls'] = np.where(tx.y.isna(), -1, 0)
    tx.loc[fraud.index, 'cls'] = np.where(per_day >= 3, 2, 1)
    tx['source'] = 'ftd'
    tx['copy'] = 0
    tx['scenario'] = ''
    keep = ['op_id', 'ts', 'client_id', 'card_id', 'amount', 'chip', 'merchant_id', 'merchant_city', 'merchant_state',
            'zip', 'mcc', 'errors', 'cls', 'source', 'copy', 'scenario']
    return tx[keep].reset_index(drop=True)


def card_medians(ops, until):
    hist = ops[(ops.ts < until) & (ops.amount > 0) & (ops.cls == 0)]
    return hist.groupby('card_id').amount.median()


def merchant_pool(ops):
    online = ops[ops.merchant_city == 'ONLINE'].groupby('mcc').merchant_id.unique()
    pos = ops[ops.merchant_city != 'ONLINE'].groupby(['mcc', 'merchant_state']).merchant_id.unique()
    anywhere = ops[ops.merchant_city != 'ONLINE'].groupby('mcc').merchant_id.unique()
    city = ops.drop_duplicates('merchant_id').set_index('merchant_id')[['merchant_city', 'merchant_state', 'zip']]
    return online, pos, anywhere, city


def shift_to_week(src_ts, target_week_start):
    first = src_ts.min()
    weeks = int(np.floor((target_week_start - first.normalize()) / WEEK))
    return src_ts + weeks * WEEK


def transplant_sparkov(ops, cards, users, rng, n_episodes=300):
    sp = load_sparkov()
    sp['ep'] = -1
    fr = sp[sp.is_fraud == 1]
    sp.loc[fr.index, 'ep'] = episodes(fr.ts, fr.card)
    sizes = sp[sp.ep >= 0].groupby('ep').size()
    chosen = rng.choice(sizes[sizes >= 3].index.values, n_episodes, replace=False)
    src_med = sp[sp.is_fraud == 0].groupby('card').amount.median()
    train_ops = ops[(ops.ts >= START) & (ops.ts < TRAIN_END)]
    activity = train_ops.groupby('card_id').size()
    fraud_cards = set(ops.loc[ops.cls > 0, 'card_id'])
    targets = [c for c in activity[activity >= 100].index if c not in fraud_cards]
    targets = rng.choice(targets, n_episodes, replace=False)
    med = card_medians(ops, TRAIN_END)
    online, pos, anywhere, city = merchant_pool(train_ops)
    card_client = cards.set_index('card_id').client_id
    card_chip = cards.set_index('card_id').has_chip
    home_state = ops[ops.merchant_city != 'ONLINE'].groupby('client_id').merchant_state.agg(lambda s: s.mode().iat[0])
    span_days = (TRAIN_END - START).days - 60
    rows = []
    for k, (ep, card) in enumerate(zip(chosen, targets)):
        src = sp[sp.ep == ep]
        week = START + pd.Timedelta(days=28 + int(rng.integers(0, span_days)))
        ts = shift_to_week(src.ts, week.normalize() - pd.Timedelta(days=week.dayofweek))
        ratio = src.amount.values / src_med.get(src.card.iat[0], src.amount.median())
        amount = np.round(ratio * med.get(card, med.median()), 2)
        client = card_client[card]
        for t, cat, a in zip(ts, src.category.values, amount):
            mcc, is_online = SPARKOV_MCC[cat]
            if mcc not in anywhere.index:
                is_online = True
            if is_online and mcc in online.index:
                merchant = int(rng.choice(online[mcc]))
                chip = 'Online Transaction'
            else:
                key = (mcc, home_state.get(client, 'CA'))
                choices = pos[key] if key in pos.index else anywhere[mcc]
                merchant = int(rng.choice(choices))
                chip = 'Chip Transaction' if card_chip[card] == 'YES' else 'Swipe Transaction'
            m_city, m_state, m_zip = city.loc[merchant]
            rows.append((t, client, card, a, chip, merchant, m_city, m_state, m_zip, mcc, np.nan, 2, 'sparkov', 1, f'sp{k}'))
    cols = ['ts', 'client_id', 'card_id', 'amount', 'chip', 'merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc',
            'errors', 'cls', 'source', 'copy', 'scenario']
    return pd.DataFrame(rows, columns=cols)


def ftd_episode_pool(ops):
    fr = ops[ops.cls.isin([1, 2]) & (ops.ts >= START)].copy()
    fr['ep'] = episodes(fr.ts, fr.card_id)
    med = card_medians(ops, END)
    fr['ratio'] = fr.amount / fr.card_id.map(med).fillna(med.median())
    sizes = fr.groupby('ep').size()
    fr = fr[fr.ep.isin(sizes[sizes >= 2].index)]
    return fr


def merchant_scenarios(ops, rng, plan=(('train', 14), ('valid', 4), ('test', 8)), window_days=10, share=0.3):
    pool = ftd_episode_pool(ops)
    pool['period'] = period_of(pool.ts)
    med = card_medians(ops, END)
    pos = ops[(ops.merchant_city != 'ONLINE') & (ops.ts >= START)]
    fraud_cards = set(ops.loc[ops.cls > 0, 'card_id'])
    bounds = {'train': (START + pd.Timedelta(days=30), TRAIN_END - pd.Timedelta(days=110)),
              'valid': (TRAIN_END, VALID_END - pd.Timedelta(days=110)),
              'test': (VALID_END, END - pd.Timedelta(days=110))}
    rows, meta, used_cards = [], [], set()
    for period, count in plan:
        lo, hi = bounds[period]
        templates = pool[pool.period == ('test' if period == 'test' else 'train')].ep.unique()
        made = 0
        while made < count:
            start = lo + pd.Timedelta(days=int(rng.integers(0, (hi - lo).days)))
            win = pos[(pos.ts >= start) & (pos.ts < start + pd.Timedelta(days=window_days))]
            counts = win.groupby('merchant_id').card_id.nunique()
            cand = counts[(counts >= 25) & (counts <= 120)]
            if cand.empty:
                continue
            merchant = int(rng.choice(cand.index.values))
            visits = win[win.merchant_id == merchant].groupby('card_id').ts.max()
            visits = visits[[c not in fraud_cards and c not in used_cards for c in visits.index]]
            n = max(3, int(round(share * len(visits))))
            hit = rng.choice(visits.index.values, min(n, len(visits)), replace=False)
            sid = f'm_{period}_{made}'
            for card in hit:
                used_cards.add(card)
                tpl = pool[pool.ep == rng.choice(templates)].sort_values('ts')
                begin = visits[card] + pd.Timedelta(days=int(rng.integers(5, 91)))
                weeks = int(np.ceil((begin - tpl.ts.iat[0]) / WEEK))
                ts = tpl.ts + weeks * WEEK
                amount = np.round(tpl.ratio.values * med.get(card, med.median()), 2)
                client = int(ops.loc[ops.card_id == card, 'client_id'].iat[0])
                for t, a, (_, s) in zip(ts, amount, tpl.iterrows()):
                    rows.append((t, client, card, a, s.chip, s.merchant_id, s.merchant_city, s.merchant_state, s.zip, s.mcc,
                                 np.nan, 4, 'scenario_merchant', 0, sid))
            meta.append({'scenario': sid, 'period': period, 'merchant': merchant, 'window_start': str(start.date()),
                         'cards_seen': int(len(visits)), 'cards_hit': int(len(hit))})
            made += 1
    cols = ['ts', 'client_id', 'card_id', 'amount', 'chip', 'merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc',
            'errors', 'cls', 'source', 'copy', 'scenario']
    return pd.DataFrame(rows, columns=cols), meta


def owner_abuse(ops, rng, n_clients=24):
    online = ops[(ops.chip == 'Online Transaction') & (ops.cls == 0) & (ops.ts >= START) & (ops.amount > 0)]
    fraud_clients = set(ops.loc[ops.cls > 0, 'client_id'])
    clients = [c for c in online.client_id.unique() if c not in fraud_clients]
    chosen = rng.choice(clients, n_clients, replace=False)
    picked = []
    for k, client in enumerate(chosen):
        own = online[online.client_id == client]
        n = int(rng.integers(3, 9))
        idx = rng.choice(own.index.values, min(n, len(own)), replace=False)
        picked.append((idx, f'o{k}'))
    return picked


def social_engineering(ops, rng, n_clients=40):
    window = ops[(ops.ts >= VALID_END) & (ops.ts < END - pd.Timedelta(days=7)) & (ops.cls == 0)]
    recent = ops[(ops.ts >= VALID_END - pd.Timedelta(days=365)) & (ops.ts < END)]
    transfer_users = set(recent.loc[recent.mcc == 4829, 'client_id'])
    pool = [c for c in window.client_id.unique() if c not in transfer_users]
    clients = rng.choice(pool, min(n_clients, len(pool)), replace=False)
    merchant = int(ops.loc[ops.mcc == 4829, 'merchant_id'].mode().iat[0])
    med = card_medians(ops, END)
    rows = []
    for k, client in enumerate(clients):
        own = window[window.client_id == client]
        anchor = own.loc[rng.choice(own.index.values)]
        ts = anchor.ts + pd.Timedelta(minutes=int(rng.integers(30, 48 * 60)))
        amount = round(float(rng.uniform(10, 30) * med.get(anchor.card_id, med.median())), 2)
        rows.append((ts, client, anchor.card_id, amount, 'Online Transaction', merchant, 'ONLINE', np.nan, np.nan, 4829,
                     np.nan, -2, 'scenario_social', 0, f'social_{k}'))
    cols = ['ts', 'client_id', 'card_id', 'amount', 'chip', 'merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc',
            'errors', 'cls', 'source', 'copy', 'scenario']
    return pd.DataFrame(rows, columns=cols)


def disputes(ops, rng):
    fr = ops[ops.cls.isin([1, 2, 4, 5])].copy()
    groups = fr.groupby([fr.card_id, fr.ts.dt.floor('D')]).ngroup()
    delay = np.clip(rng.lognormal(np.log(10), 0.6, groups.max() + 1), 1, 120)
    fr['dispute_ts'] = fr.ts + pd.to_timedelta(delay[groups.values], unit='D')
    cf = pd.read_csv(CFPB, usecols=['issue', 'consumer_complaint_narrative'])
    cf = cf.dropna()
    theft = cf[cf.issue == 'Identity theft / Fraud / Embezzlement'].consumer_complaint_narrative.values
    billing = cf[cf.issue == 'Billing disputes'].consumer_complaint_narrative.values
    fr['text'] = [rng.choice(billing if c == 5 else theft)[:600] for c in fr.cls]
    return fr[['op_id', 'card_id', 'client_id', 'ts', 'dispute_ts', 'cls', 'text']]


def transfers(users, rng):
    aml = load_aml()
    aml['ts'] = aml.ts - AML_SHIFT_WEEKS * WEEK
    pat = load_aml_patterns()
    pat['ts'] = pat.ts - AML_SHIFT_WEEKS * WEEK
    aml = aml.merge(pat[['ts', 'src', 'dst', 'amount_received', 'pattern', 'attack']].drop_duplicates(['ts', 'src', 'dst', 'amount_received']),
                    on=['ts', 'src', 'dst', 'amount_received'], how='left')
    usd = aml[aml.currency_received == 'US Dollar']
    activity = pd.concat([usd.src, usd.dst]).value_counts()
    median_amount = pd.concat([usd[['src', 'amount_received']].rename(columns={'src': 'acct'}),
                               usd[['dst', 'amount_received']].rename(columns={'dst': 'acct'})]).groupby('acct').amount_received.median()
    pool = median_amount[activity.reindex(median_amount.index) >= 3]
    pool = pool.loc[rng.choice(pool.index.values, len(users), replace=False)].sort_values()
    clients = users.sort_values('yearly_income').client_id.values
    accounts = pd.DataFrame({'account': pool.index.values, 'client_id': clients})
    aml['src_client'] = aml.src.map(accounts.set_index('account').client_id)
    aml['dst_client'] = aml.dst.map(accounts.set_index('account').client_id)
    aml['cls'] = np.where(aml.laundering == 1, 3, 0)
    return aml, accounts


def build():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    STAND.mkdir(parents=True, exist_ok=True)
    cards, users = load_cards(), load_users()
    ops = base_ops(rng)
    print('base', len(ops), round(time.time() - t0))
    sp = transplant_sparkov(ops, cards, users, rng)
    print('sparkov', len(sp), round(time.time() - t0))
    mc, meta = merchant_scenarios(ops, rng)
    print('merchant', len(mc), round(time.time() - t0))
    se = social_engineering(ops, rng)
    print('social', len(se), round(time.time() - t0))
    new = pd.concat([sp, mc, se], ignore_index=True)
    new['op_id'] = np.arange(len(new)) + 30_000_000
    ops = pd.concat([ops, new[ops.columns]], ignore_index=True)
    for idx, sid in owner_abuse(ops, rng):
        ops.loc[idx, 'cls'] = 5
        ops.loc[idx, 'scenario'] = sid
    ops = ops.sort_values(['ts', 'op_id'], kind='stable').reset_index(drop=True)
    ops['period'] = np.where(ops.ts < START, 'history', period_of(ops.ts))
    ops['errors'] = ops.errors.fillna('')
    dsp = disputes(ops, rng)
    print('disputes', len(dsp), round(time.time() - t0))
    tr, accounts = transfers(users, rng)
    print('transfers', len(tr), round(time.time() - t0))
    ops.to_parquet(STAND / 'ops.parquet', index=False)
    dsp.to_parquet(STAND / 'disputes.parquet', index=False)
    tr.to_parquet(STAND / 'transfers.parquet', index=False)
    accounts.to_parquet(STAND / 'accounts.parquet', index=False)
    cards.to_parquet(STAND / 'cards.parquet', index=False)
    users.to_parquet(STAND / 'clients.parquet', index=False)
    inwin = ops[ops.period != 'history']
    summary = {
        'ops_total': len(ops), 'ops_window': len(inwin), 'labelled_window': int((inwin.cls >= 0).sum()),
        'by_period_class': {p: g.cls.value_counts().sort_index().to_dict() for p, g in inwin.groupby('period')},
        'by_source': inwin.groupby('source').size().to_dict(),
        'sparkov_episodes': int(sp.scenario.nunique()), 'sparkov_ops': len(sp),
        'merchant_scenarios': meta, 'merchant_ops': len(mc),
        'owner_clients': int(ops.loc[ops.cls == 5, 'client_id'].nunique()), 'owner_ops': int((ops.cls == 5).sum()),
        'social_ops': int((ops.source == 'scenario_social').sum()),
        'disputes': len(dsp), 'dispute_delay_days_median': float((dsp.dispute_ts - dsp.ts).dt.total_seconds().median() / 86400),
        'transfers': len(tr), 'laundering': int(tr.laundering.sum()), 'own_accounts': len(accounts),
        'transfers_period': [str(tr.ts.min()), str(tr.ts.max())],
        'cards': int(inwin.card_id.nunique()), 'clients': int(inwin.client_id.nunique()), 'merchants': int(inwin.merchant_id.nunique()),
        'seconds': round(time.time() - t0),
    }
    save_json(summary, 'stand_summary.json')
    print(summary)


if __name__ == '__main__':
    build()
