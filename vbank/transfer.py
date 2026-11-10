import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from vbank.common import SEED, SPARKOV, ULB, save_json

COLS = ['log_amt', 'hour_sin', 'hour_cos']


def prep(amount, hour, y):
    return pd.DataFrame({'log_amt': np.log1p(amount), 'hour_sin': np.sin(2 * np.pi * hour / 24),
                         'hour_cos': np.cos(2 * np.pi * hour / 24), 'y': y})


def fit_eval(train, test):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=SEED)
    m.fit(train[COLS], train.y)
    s = m.predict_proba(test[COLS])[:, 1]
    return {'roc_auc': round(roc_auc_score(test.y, s), 4), 'pr_auc': round(average_precision_score(test.y, s), 4),
            'prevalence': round(float(test.y.mean()), 5)}


def main():
    cols = ['trans_date_trans_time', 'amt', 'is_fraud']
    sp = {k: pd.read_csv(SPARKOV / f, usecols=cols, parse_dates=['trans_date_trans_time']) for k, f in
          [('train', 'fraudTrain.csv'), ('test', 'fraudTest.csv')]}
    sp = {k: prep(d.amt, d.trans_date_trans_time.dt.hour + d.trans_date_trans_time.dt.minute / 60, d.is_fraud) for k, d in sp.items()}
    ulb = pd.read_csv(ULB).drop_duplicates()
    ulb = ulb.sort_values('Time')
    cut = int(0.7 * len(ulb))
    hours = (ulb.Time / 3600) % 24
    u = prep(ulb.Amount.values, hours.values, ulb.Class.values)
    ut, uv = u.iloc[:cut], u.iloc[cut:]
    R = {'features': COLS,
         'sparkov_to_sparkov': fit_eval(sp['train'], sp['test']),
         'ulb_to_ulb': fit_eval(ut, uv),
         'sparkov_to_ulb': fit_eval(sp['train'], uv),
         'ulb_to_sparkov': fit_eval(ut, sp['test'])}
    save_json(R, 'transfer_results.json')
    print(R)


if __name__ == '__main__':
    main()
