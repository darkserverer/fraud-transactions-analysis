# Аналіз (візуалізація, прогноз) незаконних транзакцій за картковими рахунками

Код вимірювань до проєкту з дисципліни «Проєкт з аналізу та візуалізації даних», група ШІ-33, НУ «Львівська політехніка». Він відтворює числа й рисунки розділів 2–4 записки.

## Розділ 2: вимірювання на відкритих наборах

| Файл | Що вимірює |
|---|---|
| `measure_sparkov.py` | Sparkov: розподіли, викиди, ознаки, базові рівні бустингу, рисунки 1–7 |
| `measure_aml.py` | IBM AML HI-Small: граф переказів, рисунок 8 |
| `measure_other.py` | ULB, SAML-D, Financial Transactions Dataset, Return Abuse |
| `measure_cv.py` | Financial Transactions Dataset: період, хвилі шахрайства, вплив поділу, обсяг даних |
| `measure_examples.py` | контрольні приклади перетворень і залежність якості від обсягу на Sparkov |
| `measure_fp.py` | хибні тривоги на Sparkov за часткою перевірених операцій, після `measure_sparkov.py` |
| `out/` | збережені результати й рисунки |

## Дані

Набори не входять до репозиторію, їх потрібно завантажити самостійно:

- [Sparkov](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- [IBM AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- [SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- [Financial Transactions Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
- [ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [E-Commerce Return Abuse](https://www.kaggle.com/datasets/sarveshchhetri/e-commerce-return-abuse-detection-dataset)
- [Bank and Credit Card Complaints (CFPB)](https://www.kaggle.com/datasets/mexwell/bank-and-credit-card-complaints), лише для розділів 3–4

Sparkov скрипти читають зі стандартного кешу `kagglehub` (`~/.cache/kagglehub`). Решта наборів очікується в `data/kh/datasets/`, для цього під час завантаження задайте `KAGGLEHUB_CACHE=data/kh`. `measure_aml.py` читає `single/HI-Small_Trans.csv` з архіву `data/kh/single.zip`.

## Запуск

```bash
pip install -r requirements.txt
mkdir -p out
python measure_sparkov.py
python measure_aml.py
python measure_other.py
python measure_cv.py
python measure_examples.py
python measure_fp.py
```

## Розділи 3 і 4: віртуальний банк, моделі й застосунок

| Шлях | Що робить |
|---|---|
| `vbank/common.py` | шляхи до наборів, завантаження, межі періодів |
| `vbank/stand.py` | збирає віртуальний банк у `data/stand/` |
| `vbank/features.py` | обчислює ознаки й правило спільної точки компрометації |
| `vbank/ftd_types.py` | розподіл шахрайства Financial Transactions Dataset за типами |
| `vbank/models_cards.py` | логістична регресія, випадковий ліс, бустинг, Isolation Forest, LOF, балансування, бюджет перевірок, SHAP |
| `vbank/models_gru.py` | рекурентна модель історії картки |
| `vbank/improve.py` | остаточні моделі: лічильники, вибір варіанта на валідації, перенавчання разом із валідаційним періодом |
| `vbank/realism.py` | перевірка реалістичності сценаріїв |
| `vbank/transfer.py` | переносимість між Sparkov і ULB |
| `vbank/models_graph.py` | графові ознаки, GraphSAGE і типології відмивання на IBM AML |
| `vbank/summary.py` | зведення метрик за бюджетами перевірок |
| `vbank/figures.py` | рисунки |
| `app/` | застосунок Streamlit, `app/screenshots.py` знімає рисунки його сторінок |
| `out/r3/` | збережені результати й рисунки розділів 3–4 |

```bash
python -m vbank.stand
python -m vbank.features
python -m vbank.ftd_types
python -m vbank.models_cards
python -m vbank.models_gru
python -m vbank.improve
python -m vbank.realism
python -m vbank.transfer
python -m vbank.models_graph
python -m vbank.summary
python -m vbank.figures
python app/prepare.py
streamlit run app/Огляд.py
```

Моделі зберігаються в `data/models/`, дані застосунку в `app/data/`. Приклади вхідного й вихідного файлів пакетної перевірки лежать у `app/data/приклад_*.csv`. Увесь ланцюжок на ноутбуці з Apple M4 Max і 36 ГБ пам'яті триває близько години.

Частина наборів є синтетичною, тому значення характеризують дані, а не якість виявлення в реальному банку.
