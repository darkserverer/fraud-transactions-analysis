import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

from engine import CLASSES, ROOT, budget_row, test_scores

st.set_page_config(page_title='Моделі й поріг', layout='wide')
st.title('Порівняння моделей і вибір порога')
s = test_scores()
days = (s.ts.max() - s.ts.min()).total_seconds() / 86400
models = {}
if 'score_final' in s:
    s['score_blend_final'] = (s.score_final.rank(pct=True) + s.score_gru_final.rank(pct=True)) / 2
    models.update({'бустинг і GRU, остаточні': 'score_blend_final', 'бустинг, остаточний': 'score_final', 'GRU, остаточна': 'score_gru_final'})
models.update({'бустинг': 'score', 'GRU': 'score_gru', 'випадковий ліс': 'score_rf', 'логістична регресія': 'score_lr',
               'Isolation Forest': 'score_if'})
models = {k: v for k, v in models.items() if v in s}
st.caption(f'Тестовий період: {s.ts.min():%d.%m.%Y}–{s.ts.max():%d.%m.%Y}, {len(s):,} операцій, з них шахрайських {int(s.y.sum())}.'.replace(',', ' '))
fig = go.Figure()
for name, col in models.items():
    p, r, _ = precision_recall_curve(s.y, s[col])
    step = max(1, len(p) // 3000)
    fig.add_trace(go.Scatter(x=r[::step], y=p[::step], name=f'{name} (PR-AUC {average_precision_score(s.y, s[col]):.3f})', mode='lines'))
fig.update_layout(xaxis_title='повнота', yaxis_title='точність', height=420)
st.plotly_chart(fig, width='stretch')

st.subheader('Бюджет перевірок')
model = st.selectbox('Модель', list(models))
share = st.slider('Частка операцій, що йдуть на перевірку, %', 0.05, 3.0, 0.5, 0.05) / 100
row = budget_row(s.y.values, s[models[model]].values, days, share)
cols = st.columns(4)
for c, key in zip(cols, ['повнота, %', 'точність, %', 'FP за добу', 'FPR, %']):
    c.metric(key, row[key])
st.dataframe(pd.DataFrame([row]), hide_index=True, width='stretch')
order = np.argsort(-s[models[model]].values)
flag = np.zeros(len(s), bool)
flag[order[:row['сигналів']]] = True
per = pd.DataFrame([{'клас': CLASSES[c], 'операцій': int((s.cls == c).sum()), 'виявлено, %': round(100 * flag[(s.cls == c).values].mean(), 1)}
                    for c in sorted(s.cls.unique()) if c != 0])
st.subheader('Яку частку кожного типу знаходить модель за цього бюджету')
st.dataframe(per, hide_index=True, width='stretch')
img = ROOT / 'out/r3/рис_confusion_final.png'
if img.exists():
    st.subheader('Багатокласова модель: матриця помилок на тесті')
    st.image(str(img), width=640)
