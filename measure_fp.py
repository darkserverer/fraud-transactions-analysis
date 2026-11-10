import json
import os
import pandas as pd

SPARKOV = os.path.expanduser('~/.cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1/')
OUT = 'out/'

R = json.load(open(OUT + 'sparkov_results.json', encoding='utf-8'))
test = pd.read_csv(SPARKOV + 'fraudTest.csv', usecols=['trans_date_trans_time', 'is_fraud'], parse_dates=['trans_date_trans_time'])
days = (test.trans_date_trans_time.max() - test.trans_date_trans_time.min()).total_seconds() / 86400
negatives = int(len(test) - test.is_fraud.sum())

rows = [dict(b, name=f"{b['share_pct']} %") for b in R['budget']]
rows.append({'name': 'rule', 'alerts': R['rule']['alerts'], 'hit': R['rule']['hit']})
out = []
for r in rows:
    fp = r['alerts'] - r['hit']
    out.append({'name': r['name'], 'alerts': r['alerts'], 'tp': r['hit'], 'fp': fp,
                'fpr_pct': round(100 * fp / negatives, 4), 'fp_per_day': round(fp / days, 2),
                'fp_per_tp': round(fp / r['hit'], 2)})
res = {'test_days': round(days, 2), 'negatives': negatives, 'rows': out}
json.dump(res, open(OUT + 'fp_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
