import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SPARKOV = os.path.expanduser('~/.cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1/')
OUT = 'out/'
plt.rcParams.update({'figure.dpi': 220, 'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
LEGIT_COLOR, FRAUD_COLOR = '#4878a8', '#c44e52'
COMMA = FuncFormatter(lambda value, _: f'{value:g}'.replace('.', ','))


def ua(value):
    return f'{value:.2f}'.replace('.', ',')

cols = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'city', 'state', 'zip', 'lat', 'long',
        'city_pop', 'job', 'dob', 'merch_lat', 'merch_long', 'gender', 'is_fraud']
train = pd.read_csv(SPARKOV + 'fraudTrain.csv', usecols=cols, parse_dates=['trans_date_trans_time', 'dob'])
test = pd.read_csv(SPARKOV + 'fraudTest.csv', usecols=cols, parse_dates=['trans_date_trans_time', 'dob'])
df = pd.concat([train, test], ignore_index=True).sort_values('trans_date_trans_time', kind='stable')
df = df.reset_index(drop=True)

cut = len(train)
val_start = int(cut * 0.85)
fit_idx, val_idx, test_idx = df.index[:val_start], df.index[val_start:cut], df.index[cut:]
y_test = df.loc[test_idx, 'is_fraud'].values
R = {'rows': len(df), 'fraud': int(df.is_fraud.sum()), 'share_pct': round(100 * df.is_fraud.mean(), 4),
     'train_rows': cut, 'train_fraud': int(train.is_fraud.sum()), 'train_share_pct': round(100 * train.is_fraud.mean(), 4),
     'test_rows': len(test), 'test_fraud': int(test.is_fraud.sum()), 'test_share_pct': round(100 * test.is_fraud.mean(), 4),
     'fit_rows': int(val_start), 'val_rows': int(cut - val_start),
     'period': [str(df.trans_date_trans_time.min()), str(df.trans_date_trans_time.max())],
     'train_end': str(train.trans_date_trans_time.max()), 'test_start': str(test.trans_date_trans_time.min()),
     'missing': int(df.isna().sum().sum()), 'duplicates': int(df.duplicated().sum())}

legit, fraud = df[df.is_fraud == 0], df[df.is_fraud == 1]
R['amount'] = {name: {'median': round(float(part.amt.median()), 2), 'p75': round(float(part.amt.quantile(.75)), 2),
                      'p99': round(float(part.amt.quantile(.99)), 2), 'max': round(float(part.amt.max()), 2),
                      'mean': round(float(part.amt.mean()), 2)} for name, part in (('legit', legit), ('fraud', fraud))}
R['skew'] = [round(float(df.amt.skew()), 2), round(float(np.log1p(df.amt).skew()), 2)]
R['bands'] = {f'{a}-{b}': [int(((fraud.amt >= a) & (fraud.amt < b)).sum()),
                           round(100 * float(((fraud.amt >= a) & (fraud.amt < b)).mean()), 1)]
              for a, b in ((0, 30), (200, 400), (700, 1100))}
fraud_cap = float(fraud.amt.max())
R['legit_above_fraud_max'] = [int((legit.amt > fraud_cap).sum()), round(100 * float((df.amt > fraud_cap).mean()), 3)]
R['legit_p99_fraud_above_pct'] = round(100 * float((fraud.amt > legit.amt.quantile(.99)).mean()), 1)

q1, q3 = df.amt.quantile(.25), df.amt.quantile(.75)
fence = float(q3 + 1.5 * (q3 - q1))
outside = df.amt > fence
R['iqr'] = {'q1': round(float(q1), 2), 'q3': round(float(q3), 2), 'fence': round(fence, 2),
            'outside_rows': int(outside.sum()), 'outside_pct': round(100 * float(outside.mean()), 2),
            'rate_outside_pct': round(100 * float(df.loc[outside, 'is_fraud'].mean()), 3),
            'rate_inside_pct': round(100 * float(df.loc[~outside, 'is_fraud'].mean()), 3),
            'fraud_outside': int(df.loc[outside, 'is_fraud'].sum()),
            'fraud_outside_pct': round(100 * float(df.loc[outside, 'is_fraud'].sum() / df.is_fraud.sum()), 2)}

df['hour'] = df.trans_date_trans_time.dt.hour
df['dow'] = df.trans_date_trans_time.dt.dayofweek
rate_hour = df.groupby('hour').is_fraud.agg(['mean', 'size'])
rate_hour['mean'] *= 100
R['hour_rate_pct'] = {int(k): round(float(v), 3) for k, v in rate_hour['mean'].items()}
R['gender_rate_pct'] = {k: round(100 * float(v), 3) for k, v in df.groupby('gender').is_fraud.mean().items()}
df['age'] = (df.trans_date_trans_time - df.dob).dt.days / 365.25
R['age_median'] = [round(float(df.loc[df.is_fraud == 1, 'age'].median()), 1),
                   round(float(df.loc[df.is_fraud == 0, 'age'].median()), 1)]

by_cat = df.groupby('category').is_fraud.agg(['mean', 'size', 'sum'])
by_cat['mean'] *= 100
R['category'] = {k: {'rate_pct': round(float(v['mean']), 3), 'rows': int(v['size']), 'fraud': int(v['sum'])}
                 for k, v in by_cat.sort_values('mean', ascending=False).iterrows()}
bands = {}
for name, part in df.groupby('category'):
    f, n = part[part.is_fraud == 1].amt, part[part.is_fraud == 0].amt
    bands[name] = {'fraud_min': round(float(f.min()), 2), 'fraud_max': round(float(f.max()), 2),
                   'fraud_median': round(float(f.median()), 2), 'legit_median': round(float(n.median()), 2),
                   'legit_max': round(float(n.max()), 2)}
R['category_amounts'] = bands

base = df.loc[df.index[:cut]]
card_med = base[base.is_fraud == 0].groupby('cc_num').amt.median().rename('card_med')
global_med = float(base[base.is_fraud == 0].amt.median())
df = df.join(card_med, on='cc_num')
R['cards_total'] = int(df.cc_num.nunique())
R['cards_fraud'] = int(df.loc[df.is_fraud == 1, 'cc_num'].nunique())
R['cards_without_history'] = int(df.loc[df.card_med.isna(), 'cc_num'].nunique())
df['ratio'] = df.amt / df.card_med.fillna(global_med)
df['prev_gap_h'] = df.groupby('cc_num').trans_date_trans_time.diff().dt.total_seconds() / 3600
isf = df.is_fraud == 1
R['ratio'] = {'median': [round(float(df.loc[isf, 'ratio'].median()), 2), round(float(df.loc[~isf, 'ratio'].median()), 2)],
              'gt5_pct': [round(100 * float((df.loc[isf, 'ratio'] > 5).mean()), 1),
                          round(100 * float((df.loc[~isf, 'ratio'] > 5).mean()), 1)],
              'gt10_pct': [round(100 * float((df.loc[isf, 'ratio'] > 10).mean()), 1),
                           round(100 * float((df.loc[~isf, 'ratio'] > 10).mean()), 1)]}
R['gap_h'] = {'median': [round(float(df.loc[isf, 'prev_gap_h'].median()), 2),
                         round(float(df.loc[~isf, 'prev_gap_h'].median()), 2)],
              'p25_fraud': round(float(df.loc[isf, 'prev_gap_h'].quantile(.25)), 2)}
hour_bucket = df.trans_date_trans_time.dt.floor('h')
day_bucket = df.trans_date_trans_time.dt.floor('D')
df['cnt_hour'] = df.groupby(['cc_num', hour_bucket]).amt.transform('size')
df['cnt_day'] = df.groupby(['cc_num', day_bucket]).amt.transform('size')
R['count_median'] = {'hour': [float(df.loc[isf, 'cnt_hour'].median()), float(df.loc[~isf, 'cnt_hour'].median())],
                     'day': [float(df.loc[isf, 'cnt_day'].median()), float(df.loc[~isf, 'cnt_day'].median())]}
fr = df[isf].copy()
per_day = fr.groupby(['cc_num', fr.trans_date_trans_time.dt.floor('D')]).size()
fr['per_day'] = fr.groupby(['cc_num', fr.trans_date_trans_time.dt.floor('D')]).amt.transform('size')
in_series = fr.per_day >= 3
R['types'] = {'series': int(in_series.sum()), 'anomalous_amount': int((~in_series & (fr.ratio > 5)).sum()),
              'other': int((~in_series & (fr.ratio <= 5)).sum())}
fraud_span = fr.groupby('cc_num').trans_date_trans_time.agg(['min', 'max'])
R['fraud_span_median_days'] = round(float(((fraud_span['max'] - fraud_span['min']).dt.total_seconds() / 86400).median()), 2)
R['series'] = {'card_days': int(len(per_day)), 'per_card_day_median': float(per_day.median()),
               'per_card_day_max': int(per_day.max()), 'card_days_ge3_pct': round(100 * float((per_day >= 3).mean()), 1)}
by_m = df.groupby('merchant').is_fraud.agg(['mean', 'sum'])
R['merchant'] = {'total': int(len(by_m)), 'with_fraud': int((by_m['sum'] > 0).sum()),
                 'max_rate_pct': round(100 * float(by_m['mean'].max()), 2),
                 'median_rate_pct': round(100 * float(by_m['mean'].median()), 3)}
R['geo'] = {'cities': int(df.city.nunique()), 'states': int(df.state.nunique()), 'zips': int(df.zip.nunique()),
            'jobs': int(df.job.nunique()), 'categories': int(df.category.nunique()),
            'pop_median': float(df.city_pop.median()), 'pop_max': float(df.city_pop.max()),
            'pop_skew': round(float(df.city_pop.skew()), 2)}

lat1, lon1, lat2, lon2 = map(np.radians, (df.lat, df.long, df.merch_lat, df.merch_long))
hav = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
df['dist_km'] = 6371 * 2 * np.arcsin(np.sqrt(hav))
R['dist_km'] = {'legit': [round(float(df.loc[~isf, 'dist_km'].quantile(q)), 1) for q in (.25, .5, .75, .95)],
                'fraud': [round(float(df.loc[isf, 'dist_km'].quantile(q)), 1) for q in (.25, .5, .75, .95)]}

df['amt_log'] = np.log1p(df.amt)
df['cat'] = df.category.astype('category').cat.codes
df['male'] = (df.gender == 'M').astype(int)
FULL = ['amt', 'amt_log', 'ratio', 'hour', 'dow', 'age', 'dist_km', 'city_pop', 'prev_gap_h', 'cat', 'male']
NO_DIST = [f for f in FULL if f != 'dist_km']


def fit(features, rows_fit, rows_val, seed, rate):
    model = HistGradientBoostingClassifier(early_stopping=True, scoring='average_precision', n_iter_no_change=15,
                                           max_iter=300, learning_rate=rate, random_state=seed,
                                           categorical_features=['cat'] if 'cat' in features else None)
    model.fit(df.loc[rows_fit, features], df.loc[rows_fit, 'is_fraud'],
              X_val=df.loc[rows_val, features], y_val=df.loc[rows_val, 'is_fraud'])
    return model


def runs(features, rows_fit, rows_val, rows_test, rate):
    out = []
    for seed in (0, 1, 2):
        proba = fit(features, rows_fit, rows_val, seed, rate).predict_proba(df.loc[rows_test, features])[:, 1]
        target = df.loc[rows_test, 'is_fraud']
        out.append((roc_auc_score(target, proba), average_precision_score(target, proba)))
    out = np.array(out)
    return {'roc': [round(float(out[:, 0].mean()), 4), round(float(out[:, 0].std()), 4)],
            'pr': [round(float(out[:, 1].mean()), 4), round(float(out[:, 1].std()), 4)]}


val_pr = {}
for rate in (0.1, 0.05):
    model = fit(NO_DIST, fit_idx, val_idx, 0, rate)
    val_pr[rate] = average_precision_score(df.loc[val_idx, 'is_fraud'],
                                           model.predict_proba(df.loc[val_idx, NO_DIST])[:, 1])
RATE = max(val_pr, key=val_pr.get)
R['rate_choice'] = {str(k): round(float(v), 4) for k, v in val_pr.items()}
R['rate'] = RATE
R['model_amt'] = runs(['amt'], fit_idx, val_idx, test_idx, RATE)
R['model_amt_hour_cat'] = runs(['amt', 'hour', 'cat'], fit_idx, val_idx, test_idx, RATE)
R['model_no_dist'] = runs(NO_DIST, fit_idx, val_idx, test_idx, RATE)
R['model_full'] = runs(FULL, fit_idx, val_idx, test_idx, RATE)
perm = np.random.default_rng(0).permutation(len(df))
rand_fit, rand_val, rand_test = df.index[perm[:val_start]], df.index[perm[val_start:cut]], df.index[perm[cut:]]
R['model_random_split'] = runs(FULL, rand_fit, rand_val, rand_test, RATE)

model = fit(NO_DIST, fit_idx, val_idx, 0, RATE)
proba = model.predict_proba(df.loc[test_idx, NO_DIST])[:, 1]
order = np.argsort(-proba)
positives = int(y_test.sum())
budget = []
for share in (0.001, 0.005, 0.01, 0.02):
    k = int(len(y_test) * share)
    hit = int(y_test[order[:k]].sum())
    budget.append({'share_pct': share * 100, 'alerts': k, 'hit': hit, 'recall_pct': round(100 * hit / positives, 2),
                   'precision_pct': round(100 * hit / k, 2)})
threshold = float(np.percentile(df.loc[df.index[:cut], 'amt'], 99))
rule = df.loc[test_idx, 'amt'].values > threshold
R['budget'] = budget
R['rule'] = {'threshold': round(threshold, 2), 'alerts': int(rule.sum()), 'hit': int(y_test[rule].sum()),
             'recall_pct': round(100 * y_test[rule].sum() / positives, 2),
             'precision_pct': round(100 * y_test[rule].mean(), 2)}

numeric = ['amt', 'amt_log', 'ratio', 'hour', 'dow', 'age', 'city_pop', 'prev_gap_h', 'dist_km', 'is_fraud']
corr = df[numeric].corr(method='spearman')
R['corr_label_max'] = round(float(corr['is_fraud'].drop('is_fraud').abs().max()), 3)
R['corr_amt_ratio'] = round(float(corr.loc['amt', 'ratio']), 2)

bins = np.logspace(np.log10(0.5), np.log10(30000), 60)
fig, ax = plt.subplots(figsize=(6.2, 3.1))
ax.hist(legit.amt, bins=bins, weights=np.full(len(legit), 1 / len(legit)), label='легальні', color=LEGIT_COLOR)
ax.hist(fraud.amt, bins=bins, weights=np.full(len(fraud), 1 / len(fraud)), label='шахрайські', color=FRAUD_COLOR, alpha=.75)
ax.axvline(fence, ls='--', c='k', lw=1)
ax.text(fence * 1.07, .105, f'межа IQR {ua(fence)}', fontsize=7.5)
ax.axvline(fraud_cap, ls=':', c=FRAUD_COLOR, lw=1.2)
ax.text(fraud_cap * 1.07, .08, f'максимум шахрайських\n{ua(fraud_cap)}', fontsize=7.5, color=FRAUD_COLOR)
ax.set_xscale('log')
ax.set_xlabel('сума операції, умовних одиниць (логарифмічна шкала)')
ax.set_ylabel('частка операцій класу')
ax.yaxis.set_major_formatter(COMMA)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT + 'рис1.png')

fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.bar(rate_hour.index, rate_hour['mean'], color=FRAUD_COLOR)
ax.plot(rate_hour.index, rate_hour['size'] / rate_hour['size'].max() * 2.6, color=LEGIT_COLOR, lw=1.4,
        label='обсяг операцій (масштабовано)')
ax.set_xticks(range(24))
ax.set_xlabel('година доби')
ax.set_ylabel('частка шахрайських, %')
ax.yaxis.set_major_formatter(COMMA)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT + 'рис2.png')

order_cat = by_cat['mean'].sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.barh(order_cat.index, order_cat.values, color=[FRAUD_COLOR if v > 1 else '#8b9dc3' for v in order_cat.values])
mean_rate = 100 * df.is_fraud.mean()
ax.axvline(mean_rate, ls='--', c='k', lw=1)
ax.text(mean_rate + 0.04, 0.2, f'середнє {mean_rate:.3f} %'.replace('.', ','), fontsize=7.5)
ax.set_xlabel('частка шахрайських операцій у категорії, %')
ax.xaxis.set_major_formatter(COMMA)
fig.tight_layout()
fig.savefig(OUT + 'рис3.png')

bins = np.logspace(np.log10(0.01), np.log10(200), 60)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.hist(df.loc[~isf, 'ratio'].clip(0.01, 200), bins=bins, weights=np.full(int((~isf).sum()), 1 / int((~isf).sum())),
        label='легальні', color=LEGIT_COLOR)
ax.hist(df.loc[isf, 'ratio'].clip(0.01, 200), bins=bins, weights=np.full(int(isf.sum()), 1 / int(isf.sum())),
        label='шахрайські', color=FRAUD_COLOR, alpha=.75)
ax.axvline(5, ls='--', c='k', lw=1)
ax.text(5.4, .07, 'п’ятикратне\nперевищення', fontsize=7.5)
ax.set_xscale('log')
ax.set_xlabel('сума операції, поділена на медіанну суму цієї ж картки')
ax.set_ylabel('частка операцій класу')
ax.yaxis.set_major_formatter(COMMA)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT + 'рис4.png')

shares = np.linspace(0.0002, 0.05, 120)
recall, precision = [], []
for share in shares:
    k = max(1, int(len(y_test) * share))
    hit = y_test[order[:k]].sum()
    recall.append(100 * hit / positives)
    precision.append(100 * hit / k)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.plot(shares * 100, recall, color=FRAUD_COLOR, label='повнота (recall)')
ax.plot(shares * 100, precision, color=LEGIT_COLOR, label='точність (precision)')
rule_share, rule_recall = 100 * rule.sum() / len(y_test), 100 * y_test[rule].sum() / positives
ax.scatter([rule_share], [rule_recall], color='k', zorder=5, s=18)
ax.annotate(f'правило «сума понад {ua(threshold)}»\nповнота {rule_recall:.1f} %, точність {100 * y_test[rule].mean():.1f} %'.replace('.', ','),
            (rule_share, rule_recall), textcoords='offset points', xytext=(12, -28), fontsize=7.5)
ax.axvline(0.5, ls=':', c='k', lw=1)
ax.set_xlabel('частка операцій, відправлених на перевірку, %')
ax.set_ylabel('%')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT + 'рис5.png')

names = ['сума', 'лог суми', 'до медіани картки', 'година', 'день тижня', 'вік', 'населення', 'інтервал, год',
         'відстань, км', 'мітка']
matrix = corr.values
fig, ax = plt.subplots(figsize=(5.4, 4.6))
image = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7.5)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, ua(matrix[i, j]), ha='center', va='center', fontsize=6,
                color='white' if abs(matrix[i, j]) > .6 else 'black')
ax.grid(False)
fig.colorbar(image, shrink=.8)
fig.tight_layout()
fig.savefig(OUT + 'рис6.png')

sample = df.sample(120_000, random_state=0)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.scatter(sample.long, sample.lat, s=.6, alpha=.12, color=LEGIT_COLOR, label='легальні (вибірка)')
ax.scatter(fraud.long, fraud.lat, s=2.2, alpha=.5, color=FRAUD_COLOR, label='шахрайські (усі)')
ax.set_xlim(-127, -66)
ax.set_ylim(23, 51)
ax.set_xlabel('довгота')
ax.set_ylabel('широта')
ax.legend(frameon=False, markerscale=4)
fig.tight_layout()
fig.savefig(OUT + 'рис7.png')

json.dump(R, open(OUT + 'sparkov_results.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
