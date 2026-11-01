# Аналіз (візуалізація, прогноз) незаконних транзакцій за картковими рахунками

Код вимірювань до проєкту з дисципліни «Проєкт з аналізу та візуалізації даних», група ШІ-33, НУ «Львівська політехніка». Він відтворює числа й рисунки розділу 2 записки.

## Файли

| Файл | Що вимірює | Додаток записки |
|---|---|---|
| `measure_sparkov.py` | Sparkov: розподіли, викиди, ознаки, базові рівні бустингу, рисунки 1–7 | А |
| `measure_aml.py` | IBM AML HI-Small: граф переказів, рисунок 8 | Б |
| `measure_other.py` | ULB, SAML-D, Financial Transactions Dataset, Return Abuse | В |
| `measure_cv.py` | Financial Transactions Dataset: період, хвилі шахрайства, вплив поділу, обсяг даних | Г |
| `measure_examples.py` | контрольні приклади перетворень і залежність якості від обсягу на Sparkov | Д |
| `out/` | збережені результати й рисунки | |

## Дані

Набори не входять до репозиторію, їх потрібно завантажити самостійно:

- [Sparkov](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- [IBM AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- [SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- [Financial Transactions Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
- [ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [E-Commerce Return Abuse](https://www.kaggle.com/datasets/sarveshchhetri/e-commerce-return-abuse-detection-dataset)

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
```

Частина наборів є синтетичною, тому значення характеризують дані, а не якість виявлення в реальному банку.
