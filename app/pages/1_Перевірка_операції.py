import pandas as pd
import plotly.express as px
import streamlit as st

from engine import CLASSES, DATA, artifacts, explain, score

st.set_page_config(page_title='Перевірка операції', layout='wide')
st.title('Перевірка однієї операції')
a = artifacts()
sample = pd.read_csv(DATA / 'приклад_вхідних_даних.csv')
names = a['mcc_names']
mccs = sorted(int(m) for m in a['mcc_risk'].index)

with st.form('op'):
    c1, c2, c3 = st.columns(3)
    card = c1.selectbox('Картка', sorted(a['cards'].index), index=sorted(a['cards'].index).index(int(sample.card_id.iloc[0])))
    date = c2.date_input('Дата', pd.Timestamp('2019-06-15'))
    time = c3.time_input('Час', pd.Timestamp('2019-06-15 21:30').time())
    c4, c5, c6 = st.columns(3)
    amount = c4.number_input('Сума, $', value=250.0, step=10.0)
    chip = c5.selectbox('Спосіб', ['Online Transaction', 'Chip Transaction', 'Swipe Transaction'])
    mcc = c6.selectbox('Категорія торговця', mccs, format_func=lambda m: f'{m} · {names.get(str(m), "")}', index=mccs.index(5311))
    c7, c8, c9 = st.columns(3)
    merchant = c7.text_input('Торговець (номер або online)', 'online')
    state = c8.text_input('Штат торговця', '')
    errors = c9.selectbox('Помилка авторизації', ['', 'Insufficient Balance', 'Bad PIN', 'Technical Glitch', 'Bad CVV', 'Bad Expiration'])
    threshold = st.slider('Поріг тривоги', 0.0, 1.0, 0.5, 0.01)
    go = st.form_submit_button('Перевірити')

if go:
    raw = pd.DataFrame([{'card_id': card, 'ts': pd.Timestamp.combine(date, time), 'amount': amount, 'chip': chip,
                         'merchant_id': merchant, 'merchant_state': state or None, 'mcc': mcc, 'errors': errors}])
    res, X, multi, labels = score(raw)
    p = float(res['ймовірність_шахрайства'].iat[0])
    info = a['cards'].loc[card]
    left, right = st.columns([1, 2])
    left.metric('Ймовірність шахрайства', f'{100 * p:.1f} %')
    if p >= threshold:
        left.error(f'Тривога: операцію відправлено на перевірку. Найімовірніший тип: {res["найімовірніший_тип"].iat[0]}')
    else:
        left.success('Операцію пропущено')
    left.write(f'Картка {card}: {info.card_brand}, {info.card_type}; звичайна сума {info["median"]:.2f} $; домашній штат {info.home_state}')
    types = pd.DataFrame({'тип': [CLASSES[int(c)] for c in labels], 'ймовірність': multi[0]})
    left.dataframe(types.sort_values('ймовірність', ascending=False), hide_index=True, width='stretch')
    contrib = explain(X)
    contrib['напрям'] = ['підвищує ризик' if v > 0 else 'знижує ризик' for v in contrib.внесок]
    right.subheader('Чому така оцінка: внески ознак за SHAP')
    right.plotly_chart(px.bar(contrib[::-1], x='внесок', y='ознака', color='напрям', orientation='h',
                              color_discrete_map={'підвищує ризик': '#c0392b', 'знижує ризик': '#2c6fbb'},
                              hover_data=['значення']), width='stretch')
    right.dataframe(contrib.round(4), hide_index=True, width='stretch')
