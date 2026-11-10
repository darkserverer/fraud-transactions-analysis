import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vbank.common import STAND, VALID_END
from vbank.features import CHIP

APP = Path(__file__).resolve().parent / 'data'
INPUT = ['card_id', 'ts', 'amount', 'chip', 'merchant_id', 'merchant_state', 'mcc', 'errors']


def main():
    APP.mkdir(exist_ok=True)
    ops = pd.read_parquet(STAND / 'ops.parquet')
    f = pd.read_parquet(STAND / 'features.parquet')
    cards = pd.read_parquet(STAND / 'cards.parquet').set_index('card_id')
    clients = pd.read_parquet(STAND / 'clients.parquet').set_index('client_id')
    disputes = pd.read_parquet(STAND / 'disputes.parquet')
    normal = ops[(ops.ts < VALID_END) & (ops.cls == 0) & (ops.amount > 0)]
    card_med = normal.groupby('card_id').amount.median()
    client_med = normal.groupby('client_id').amount.median()
    home = ops[ops.merchant_city != 'ONLINE'].groupby('client_id').merchant_state.agg(lambda s: s.mode().iat[0])
    info = pd.DataFrame({'client_id': ops.groupby('card_id').client_id.first()})
    info['median'] = info.index.map(card_med)
    info['median'] = info['median'].fillna(info.client_id.map(client_med)).fillna(normal.amount.median())
    c = cards.reindex(info.index)
    u = clients.reindex(info.client_id)
    info['card_credit'] = (c.card_type.values == 'Credit').astype(int)
    info['card_prepaid'] = (c.card_type.values == 'Debit (Prepaid)').astype(int)
    info['card_has_chip'] = (c.has_chip.values == 'YES').astype(int)
    info['log_limit'] = np.log1p(c.credit_limit.values)
    info['acct_open'] = pd.to_datetime(c.acct_open_date.values, format='%m/%Y')
    info['card_type'] = c.card_type.values
    info['card_brand'] = c.card_brand.values
    info['birth_year'] = u.birth_year.values
    info['log_income'] = np.log1p(u.yearly_income.values)
    info['credit_score'] = u.credit_score.values
    info['debt_ratio'] = u.total_debt.values / np.maximum(u.yearly_income.values, 1)
    info['home_state'] = info.client_id.map(home).values
    info.reset_index().to_parquet(APP / 'cards.parquet', index=False)
    ops[['card_id', 'ts', 'amount']].sort_values(['card_id', 'ts']).to_parquet(APP / 'card_ts.parquet', index=False)
    risk = f.drop_duplicates('merchant_id')[['merchant_id', 'merchant_risk_f']].rename(columns={'merchant_risk_f': 'merchant_risk'})
    risk.to_parquet(APP / 'merchant_risk.parquet', index=False)
    f.drop_duplicates('mcc')[['mcc', 'mcc_risk_f']].rename(columns={'mcc_risk_f': 'mcc_risk'}).to_parquet(APP / 'mcc_risk.parquet', index=False)
    online_ops = set(ops.loc[ops.chip == 'Online Transaction', 'op_id'])
    disputes.assign(online=disputes.op_id.isin(online_ops).astype(int))[['client_id', 'dispute_ts', 'online', 'cls']].to_parquet(APP / 'disputes.parquet', index=False)
    merchants = ops.drop_duplicates('merchant_id')[['merchant_id', 'merchant_city', 'merchant_state', 'mcc']]
    merchants.to_parquet(APP / 'merchants.parquet', index=False)
    test = ops[(ops.period == 'test') & (ops.cls >= 0)]
    rng = np.random.default_rng(7)
    pick = np.r_[rng.choice(test.index[test.cls == 0], 40, replace=False),
                 rng.choice(test.index[test.cls == 1], 4, replace=False), rng.choice(test.index[test.cls == 2], 3, replace=False),
                 rng.choice(test.index[test.cls == 4], 2, replace=False), rng.choice(test.index[test.cls == 5], 1, replace=False)]
    sample = test.loc[np.sort(pick)].copy()
    sample['ts'] = sample.ts.dt.strftime('%Y-%m-%d %H:%M')
    sample[INPUT].to_csv(APP / 'приклад_вхідних_даних.csv', index=False)
    sample[['op_id', 'cls']].to_csv(APP / 'приклад_справжніх_класів.csv', index=False)
    lab = f[(f.period != 'history') & (f.cls >= 0)]
    lab = lab.assign(hour=lab.ts.dt.hour)
    overview = {
        'ops': int(len(lab)), 'cards': int(lab.card_id.nunique()), 'clients': int(lab.client_id.nunique()),
        'merchants': int(lab.merchant_id.nunique()),
        'by_class': {p: g.cls.value_counts().sort_index().to_dict() for p, g in lab.groupby('period')},
        'hour_rate': lab.groupby('hour').y.mean().round(5).to_dict(),
        'hour_volume': lab.groupby('hour').size().to_dict(),
        'state_fraud': lab[lab.y == 1].merchant_state.fillna('ONLINE').value_counts().to_dict(),
        'chip_rate': lab.groupby('chip').y.mean().round(5).rename({v: k for k, v in CHIP.items()}).to_dict(),
        'mcc_rate': lab.groupby('mcc').y.agg(['mean', 'size']).query('size > 2000')['mean'].sort_values(ascending=False).head(12).round(5).to_dict(),
    }
    json.dump(overview, open(APP / 'overview.json', 'w', encoding='utf-8'), ensure_ascii=False, default=int)
    mcc_names = json.load(open(Path(__file__).resolve().parent.parent / 'data/kh/datasets/computingvictor/transactions-fraud-datasets/versions/1/mcc_codes.json'))
    json.dump(mcc_names, open(APP / 'mcc_names.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('prepared', APP, len(info), len(sample))


if __name__ == '__main__':
    main()
