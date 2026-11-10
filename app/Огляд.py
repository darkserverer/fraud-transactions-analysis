import json

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import CLASSES, DATA

st.set_page_config(page_title='Віртуальний банк: моніторинг операцій', layout='wide')
st.title('Віртуальний банк: огляд даних')
st.caption('Стенд зібрано з Financial Transactions Dataset, серій Sparkov, сценаріїв компрометації торговця й зловживання власника, переказів IBM AML.')

o = json.load(open(DATA / 'overview.json', encoding='utf-8'))
c1, c2, c3, c4 = st.columns(4)
c1.metric('Розмічених операцій', f"{o['ops']:,}".replace(',', ' '))
c2.metric('Карток', o['cards'])
c3.metric('Клієнтів', o['clients'])
c4.metric('Торговців', f"{o['merchants']:,}".replace(',', ' '))

rows = []
for period, counts in o['by_class'].items():
    for cls, n in counts.items():
        rows.append({'період': period, 'клас': CLASSES[int(cls)], 'операцій': n})
table = pd.DataFrame(rows).pivot(index='клас', columns='період', values='операцій').fillna(0).astype(int)
table = table.loc[[CLASSES[c] for c in sorted(CLASSES) if CLASSES[c] in table.index]]
st.subheader('Склад стенда за класами й періодами')
st.dataframe(table[['train', 'valid', 'test']].style.format(lambda v: f'{v:,}'.replace(',', ' ')), width='stretch')

left, right = st.columns(2)
hours = pd.DataFrame({'година': [int(h) for h in o['hour_rate']], 'частка шахрайства, %': [100 * v for v in o['hour_rate'].values()],
                      'операцій': list(o['hour_volume'].values())})
left.subheader('Коли трапляється шахрайство')
left.plotly_chart(px.bar(hours, x='година', y='частка шахрайства, %', hover_data=['операцій']), width='stretch')

names = json.load(open(DATA / 'mcc_names.json', encoding='utf-8'))
mcc = pd.DataFrame({'категорія': [names.get(str(k), str(k)) for k in o['mcc_rate']], 'частка шахрайства, %': [100 * v for v in o['mcc_rate'].values()]})
right.subheader('Найризикованіші категорії торговців')
right.plotly_chart(px.bar(mcc.sort_values('частка шахрайства, %'), x='частка шахрайства, %', y='категорія', orientation='h'), width='stretch')

states = pd.DataFrame({'штат': list(o['state_fraud']), 'шахрайських операцій': list(o['state_fraud'].values())})
online = int(states.loc[states.штат == 'ONLINE', 'шахрайських операцій'].sum())
st.subheader('Де відбуваються шахрайські операції')
st.caption(f'Окремо онлайн: {online} операцій. На карті — операції в магазинах за штатом торговця.')
fig = px.choropleth(states[states.штат.str.len() == 2], locations='штат', locationmode='USA-states', color='шахрайських операцій',
                    scope='usa', color_continuous_scale='Reds')
st.plotly_chart(fig, width='stretch')
