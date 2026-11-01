import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

SPARKOV = os.path.expanduser('~/.cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1/')
USERS = 'data/kh/datasets/computingvictor/transactions-fraud-datasets/versions/1/users_data.csv'
cols = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'gender', 'lat', 'long', 'dob', 'is_fraud']
train = pd.read_csv(SPARKOV + 'fraudTrain.csv', usecols=cols, parse_dates=['trans_date_trans_time', 'dob'])
test = pd.read_csv(SPARKOV + 'fraudTest.csv', usecols=cols, parse_dates=['trans_date_trans_time', 'dob'])
df = pd.concat([train, test], ignore_index=True).sort_values('trans_date_trans_time', kind='stable').reset_index(drop=True)
R = {}

moment = pd.Timestamp('2022-09-01 00:20')
shifted = moment - pd.Timedelta(weeks=156)
R['date_shift'] = [str(moment), moment.day_name(), str(shifted), shifted.day_name()]

CARD = 3534718226968689
source = df[(df.cc_num == CARD) & (df.is_fraud == 1)].sort_values('trans_date_trans_time')
history = df[df.is_fraud == 0].groupby('cc_num').amt.agg(['median', 'size'])
history = history[history['size'] > 500]
target_card = history['median'].idxmax()
target_median = history.loc[target_card, 'median']
source_median = df[(df.cc_num == CARD) & (df.is_fraud == 0)].amt.median()
start = pd.Timestamp('2020-03-18 01:01:29')
moved = source.assign(trans_date_trans_time=start + (source.trans_date_trans_time - source.trans_date_trans_time.iloc[0]),
                      amt=(source.amt / source_median * target_median).round(2), origin='вставлена')
own = df[(df.cc_num == target_card) & (df.is_fraud == 0)].assign(origin='своя')
own = own[own.trans_date_trans_time.between(start - pd.Timedelta(days=1), moved.trans_date_trans_time.max() + pd.Timedelta(hours=12))]
mixed = pd.concat([own, moved]).sort_values('trans_date_trans_time')
R['transplant'] = {'source_median': round(float(source_median), 2), 'target_median': round(float(target_median), 2),
                   'rows': [[str(t), c, a, o] for t, c, a, o in zip(mixed.trans_date_trans_time, mixed.category, mixed.amt, mixed.origin)][:14]}

users = pd.read_csv(USERS)
row = df[df.cc_num == CARD].iloc[0]
age = (row.trans_date_trans_time - row.dob).days / 365.25
users['distance'] = (np.hypot(users.latitude - row.lat, users.longitude - row.long) + abs(users.current_age - age) / 5
                     + (users.gender.str[0] != row.gender) * 100)
match = users.sort_values('distance').iloc[0]
R['matching'] = {'card': [row.gender, round(age, 1), row.lat, row.long],
                 'client': {k: str(match[k]) for k in ['id', 'gender', 'current_age', 'latitude', 'longitude', 'yearly_income', 'total_debt', 'credit_score']}}

merchants = df.groupby('merchant').cc_num.nunique().sort_values(ascending=False)
merchant = merchants.index[100]
begin = pd.Timestamp('2019-07-01')
window = df[(df.merchant == merchant) & df.trans_date_trans_time.between(begin, begin + pd.Timedelta(days=10)) & (df.is_fraud == 0)]
chosen = window.groupby('cc_num').trans_date_trans_time.min().sample(frac=.3, random_state=7)
R['merchant_scenario'] = {'merchant': merchant, 'category': window.category.iloc[0], 'cards': int(window.cc_num.nunique()),
                          'compromised': int(len(chosen))}
R['graph_density'] = {'merchants_per_card_median': int(df.groupby('cc_num').merchant.nunique().median()),
                      'cards_per_merchant_median': int(merchants.median()), 'merchants': int(len(merchants))}

fraud = df[df.is_fraud == 1]
per_day = fraud.groupby(['cc_num', fraud.trans_date_trans_time.dt.floor('D')]).size()
R['sparkov_card_days_ge3'] = int((per_day >= 3).sum())

df['gap'] = df.groupby('cc_num').trans_date_trans_time.diff().dt.total_seconds() / 3600
df['hour'] = df.trans_date_trans_time.dt.hour
df['dow'] = df.trans_date_trans_time.dt.dayofweek
df['cat'] = df.category.astype('category').cat.codes
cut = len(train)
base = df.iloc[:cut]
df['ratio'] = df.amt / df.cc_num.map(base[base.is_fraud == 0].groupby('cc_num').amt.median())
features = ['amt', 'ratio', 'gap', 'hour', 'dow', 'cat']
fit, hold = df.iloc[:cut], df.iloc[cut:]
pos, neg = np.where(fit.is_fraud.values == 1)[0], np.where(fit.is_fraud.values == 0)[0]
curve = []
for share in (.01, .05, .1, .25, .5, 1):
    runs = []
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        rows = np.r_[rng.choice(pos, int(len(pos) * share), replace=False), rng.choice(neg, int(len(neg) * share), replace=False)]
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, early_stopping=False, random_state=seed)
        proba = model.fit(fit.iloc[rows][features], fit.iloc[rows].is_fraud).predict_proba(hold[features])[:, 1]
        runs.append(average_precision_score(hold.is_fraud, proba))
    curve.append({'fraud': int(len(pos) * share), 'pr_auc': round(float(np.mean(runs)), 4)})
R['sparkov_curve'] = curve
json.dump(R, open('out/examples_results.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=0, default=str))
