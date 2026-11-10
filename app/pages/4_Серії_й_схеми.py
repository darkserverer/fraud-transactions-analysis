import json

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import CLASSES, ROOT, STAND

st.set_page_config(page_title='Серії й схеми', layout='wide')
st.title('Серії на картці й схеми відмивання')


@st.cache_data
def ops_of_fraud_cards():
    f = pd.read_parquet(STAND / 'features.parquet', columns=['ts', 'card_id', 'amount', 'cls', 'period', 'mcc', 'chip', 'scenario'])
    test = f[(f.period == 'test') & (f.cls > 0)]
    cards = test.groupby('card_id').size().sort_values(ascending=False)
    return f[f.card_id.isin(cards.index[:60])], cards


f, cards = ops_of_fraud_cards()
card = st.selectbox('Картка з шахрайством у тестовому періоді', list(cards.index[:60]),
                    format_func=lambda c: f'{c} · шахрайських операцій: {cards[c]}')
g = f[f.card_id == card]
fraud_day = g[g.cls > 0].ts.min()
days = st.slider('Днів до й після першої шахрайської операції', 2, 30, 7)
w = g[(g.ts >= fraud_day - pd.Timedelta(days=days)) & (g.ts <= fraud_day + pd.Timedelta(days=days))].copy()
w['клас'] = w.cls.map(lambda c: CLASSES.get(int(c), 'без мітки'))
st.plotly_chart(px.scatter(w, x='ts', y='amount', color='клас', hover_data=['mcc', 'scenario'],
                           labels={'ts': 'час', 'amount': 'сума, $'}), width='stretch')
st.subheader('Типології відмивання IBM AML')
img = ROOT / 'out/r3/рис_typologies.png'
if img.exists():
    st.image(str(img), width='stretch')
res = ROOT / 'out/r3/graph_results.json'
if res.exists():
    g = json.load(open(res, encoding='utf-8'))
    st.dataframe(pd.DataFrame({'типологія': list(g['typology']['per_class_f1']), 'F1': list(g['typology']['per_class_f1'].values())}),
                 hide_index=True)
