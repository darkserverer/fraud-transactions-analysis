import numpy as np

from vbank.common import load_ftd, save_json


def main():
    tx = load_ftd()
    med = tx[tx.amount > 0].groupby('card_id').amount.median()
    fraud = tx[tx.label == 'Yes'].copy()
    per_day = fraud.groupby([fraud.card_id, fraud.ts.dt.floor('D')]).op_id.transform('size')
    rel = fraud.amount / fraud.card_id.map(med)
    kind = np.where(per_day >= 3, 'series', np.where(rel > 5, 'anomalous', 'other'))
    counts = dict(zip(*np.unique(kind, return_counts=True)))
    R = {'fraud_ops': int(len(fraud)), 'counts': {k: int(v) for k, v in counts.items()},
         'shares_pct': {k: round(100 * v / len(fraud), 1) for k, v in counts.items()},
         'card_days_by_size': per_day.groupby([fraud.card_id, fraud.ts.dt.floor('D')]).first().value_counts().sort_index().head(6).to_dict(),
         'series_with_rel_above_5_pct': round(100 * float((rel[kind == 'series'] > 5).mean()), 1)}
    save_json(R, 'ftd_types.json')
    print(R)


if __name__ == '__main__':
    main()
