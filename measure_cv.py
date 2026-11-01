import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

CV = 'data/kh/datasets/computingvictor/transactions-fraud-datasets/versions/1/'
tx = pd.read_csv(CV + 'transactions_data.csv', parse_dates=['date'])
labels = json.load(open(CV + 'train_fraud_labels.json'))
labels = labels.get('target', labels)
tx['y'] = tx.id.astype(str).map(labels)
tx['amt'] = tx.amount.str.replace('$', '', regex=False).astype(float)
tx = tx.sort_values('date', kind='stable').reset_index(drop=True)
tx['gap'] = tx.groupby('card_id').date.diff().dt.total_seconds() / 3600
tx['hour'] = tx.date.dt.hour
tx['dow'] = tx.date.dt.dayofweek
tx['chip'] = tx.use_chip.astype('category').cat.codes
tx['online'] = (tx.use_chip == 'Online Transaction').astype(int)
tx['err'] = tx.errors.notna().astype(int)
tx['state'] = tx.merchant_state.astype('category').cat.codes
data = tx[tx.y.notna()].copy()
data['y'] = (data.y == 'Yes').astype(int)
fraud = data[data.y == 1]
per_day = fraud.groupby(['card_id', fraud.date.dt.floor('D')]).size()
R = {'period': [str(tx.date.min()), str(tx.date.max())], 'rows': len(tx), 'labelled': len(data), 'fraud': len(fraud),
     'share_pct': round(100 * data.y.mean(), 4), 'cards': int(tx.card_id.nunique()), 'clients': int(tx.client_id.nunique()),
     'merchants': int(tx.merchant_id.nunique()), 'ops_per_card_median': int(tx.groupby('card_id').size().median()),
     'fraud_cards': int(fraud.card_id.nunique()), 'fraud_clients': int(fraud.client_id.nunique()),
     'fraud_merchants': int(fraud.merchant_id.nunique()), 'card_days': int(len(per_day)),
     'card_days_ge3': int((per_day >= 3).sum()), 'fraud_by_year': fraud.groupby(fraud.date.dt.year).size().to_dict(),
     'online_share': data.groupby('y').online.mean().round(3).to_dict(), 'amount_median': data.groupby('y').amt.median().to_dict(),
     'top_states': fraud.merchant_state.fillna('ONLINE').value_counts().head(4).to_dict()}
BASE = ['amt', 'gap', 'hour', 'dow', 'chip', 'mcc', 'err', 'state']


def smooth(fit, column, k=50):
    prior = fit.y.mean()
    grouped = fit.groupby(column).y.agg(['sum', 'size'])
    return (grouped['sum'] + k * prior) / (grouped['size'] + k), prior


def score(fit, test, name):
    fit, test = fit.copy(), test.copy()
    medians = fit[fit.y == 0].groupby('card_id').amt.median()
    for part in (fit, test):
        part['ratio'] = part.amt / part.card_id.map(medians).replace(0, np.nan)
    out = {'test_rate_pct': round(100 * test.y.mean(), 4), 'fit_fraud': int(fit.y.sum()), 'test_fraud': int(test.y.sum())}
    features = BASE + ['ratio']
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, early_stopping=False, random_state=0)
    proba = model.fit(fit[features], fit.y).predict_proba(test[features])[:, 1]
    out['base'] = [round(roc_auc_score(test.y, proba), 4), round(average_precision_score(test.y, proba), 4)]
    encoded = features + ['merchant_rate', 'mcc_rate', 'city_rate']
    for column, name_rate in (('merchant_id', 'merchant_rate'), ('mcc', 'mcc_rate'), ('merchant_city', 'city_rate')):
        rates, prior = smooth(fit, column)
        for part in (fit, test):
            part[name_rate] = part[column].map(rates).fillna(prior)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, early_stopping=False, random_state=0)
    proba = model.fit(fit[encoded], fit.y).predict_proba(test[encoded])[:, 1]
    out['encoded'] = [round(roc_auc_score(test.y, proba), 4), round(average_precision_score(test.y, proba), 4)]
    R[name] = out
    print(name, out, flush=True)


cut = int(len(data) * .7)
score(data.iloc[:cut], data.iloc[cut:], 'time_70_30')
shuffled = data.sample(frac=1, random_state=0)
score(shuffled.iloc[:cut], shuffled.iloc[cut:], 'random_70_30')
score(data[(data.date >= '2015-01-01') & (data.date < '2018-01-01')], data[data.date >= '2018-01-01'], 'window_2015_2017')

fit, test = data.iloc[:cut].copy(), data.iloc[cut:].copy()
medians = fit[fit.y == 0].groupby('card_id').amt.median()
for part in (fit, test):
    part['ratio'] = part.amt / part.card_id.map(medians).replace(0, np.nan)
features = BASE + ['ratio']
pos, neg = np.where(fit.y.values == 1)[0], np.where(fit.y.values == 0)[0]
curve = []
for share in (.01, .05, .1, .25, .5, 1):
    runs = []
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        rows = np.r_[rng.choice(pos, int(len(pos) * share), replace=False), rng.choice(neg, int(len(neg) * share), replace=False)]
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, early_stopping=False, random_state=seed)
        proba = model.fit(fit.iloc[rows][features], fit.iloc[rows].y).predict_proba(test[features])[:, 1]
        runs.append(average_precision_score(test.y, proba))
    curve.append({'fraud': int(len(pos) * share), 'pr_auc': round(float(np.mean(runs)), 4)})
    print(curve[-1], flush=True)
R['curve'] = curve
json.dump(R, open('out/cv_results.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, default=str))
