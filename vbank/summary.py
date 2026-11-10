import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from vbank.common import STAND, save_json
from vbank.models_cards import budget

COLUMNS = {'бустинг': 'score', 'GRU': 'score_gru', 'випадковий ліс': 'score_rf', 'логістична регресія': 'score_lr',
           'Isolation Forest': 'score_if'}


def main():
    s = pd.read_parquet(STAND / 'test_scores.parquet')
    s['score_blend'] = (s.score.rank(pct=True) + s.score_gru.rank(pct=True)) / 2
    cols = dict(COLUMNS, **{'бустинг + GRU': 'score_blend'})
    if 'score_final' in s:
        s['score_blend_final'] = (s.score_final.rank(pct=True) + s.score_gru_final.rank(pct=True)) / 2
        cols.update({'бустинг, остаточний': 'score_final', 'GRU, остаточна': 'score_gru_final',
                     'бустинг + GRU, остаточні': 'score_blend_final'})
    R = {}
    for name, col in cols.items():
        y, v = s.y.values, s[col].values
        R[name] = {'roc_auc': round(roc_auc_score(y, v), 4), 'pr_auc': round(average_precision_score(y, v), 4),
                   'budget': budget(y, v, s.ts)}
        order = np.argsort(-v)
        flag = np.zeros(len(s), bool)
        flag[order[:int(0.005 * len(s))]] = True
        R[name]['recall_by_class_at_05pct'] = {int(c): round(100 * flag[(s.cls == c).values].mean(), 2) for c in sorted(s.cls.unique()) if c}
    R['test_days'] = round((s.ts.max() - s.ts.min()).total_seconds() / 86400, 2)
    save_json(R, 'models_summary.json')
    for k, v in R.items():
        print(k, v if not isinstance(v, dict) else (v['pr_auc'], v['budget'][1]))


if __name__ == '__main__':
    main()
