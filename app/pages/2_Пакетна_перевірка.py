import pandas as pd
import streamlit as st

from engine import DATA, score

st.set_page_config(page_title='Пакетна перевірка', layout='wide')
st.title('Пакетна перевірка операцій')
st.write('Файл CSV з колонками: card_id, ts, amount, chip, merchant_id, merchant_state, mcc, errors. '
         'Для інтернет-операцій chip дорівнює Online Transaction, а merchant_state залишається порожнім; '
         'якщо номер торговця невідомий, у merchant_id можна написати online.')
with open(DATA / 'приклад_вхідних_даних.csv', 'rb') as f:
    st.download_button('Завантажити приклад вхідного файла', f, 'приклад_вхідних_даних.csv')
uploaded = st.file_uploader('Файл з операціями', type='csv')
use_sample = st.checkbox('Перевірити приклад', value=uploaded is None)
threshold = st.slider('Поріг тривоги', 0.0, 1.0, 0.5, 0.01)
raw = pd.read_csv(uploaded) if uploaded is not None else (pd.read_csv(DATA / 'приклад_вхідних_даних.csv') if use_sample else None)
if raw is not None:
    res, _, _, _ = score(raw)
    if 'errors' in res:
        res['errors'] = res['errors'].fillna('').astype(str).replace('None', '')
    res['рішення'] = ['на перевірку' if p >= threshold else 'пропустити' for p in res['ймовірність_шахрайства']]
    c1, c2, c3 = st.columns(3)
    c1.metric('Операцій', len(res))
    c2.metric('На перевірку', int((res.рішення == 'на перевірку').sum()))
    c3.metric('Частка тривог', f"{100 * (res.рішення == 'на перевірку').mean():.1f} %")
    st.dataframe(res.sort_values('ймовірність_шахрайства', ascending=False).style.apply(
        lambda r: ['background-color: #fde2e1' if r['рішення'] == 'на перевірку' else '' for _ in r], axis=1).format(
        {'amount': '{:.2f}', 'ймовірність_шахрайства': '{:.4f}'}, na_rep=''),
        width='stretch', hide_index=True)
    st.download_button('Завантажити результат', res.to_csv(index=False).encode('utf-8'), 'результат_перевірки.csv')
