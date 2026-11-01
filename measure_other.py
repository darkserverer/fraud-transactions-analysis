import json
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

K = 'data/kh/datasets/'
SAML = K + 'berkanoztas/synthetic-transaction-monitoring-dataset-aml/versions/2/SAML-D.csv'
IBM_CV = K + 'computingvictor/transactions-fraud-datasets/versions/1/'
ULB = K + 'mlg-ulb/creditcardfraud/versions/3/creditcard.csv'
RETURNS = K + 'sarveshchhetri/e-commerce-return-abuse-detection-dataset/versions/1/ecommerce_return_abuse_dataset.csv'


def audit(frame, label):
    numeric = frame.drop(columns=[label]).select_dtypes('number')
    x_fit, x_test, y_fit, y_test = train_test_split(numeric, frame[label], test_size=.3, stratify=frame[label],
                                                    random_state=0)
    model = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(x_fit, y_fit)
    return roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])


ulb = pd.read_csv(ULB)
v = ulb.loc[:, 'V1':'V28']
print('рядків', len(ulb), '| пропусків', int(ulb.isna().sum().sum()), '| повних дублікатів', int(ulb.duplicated().sum()),
      '| серед шахрайських', int(ulb[ulb.Class == 1].duplicated().sum()))
print('шахрайських', int(ulb.Class.sum()), '| частка %', round(100 * ulb.Class.mean(), 4))
print('Amount: стд', round(ulb.Amount.std(), 2), '| V1..V28: середнє стд', round(v.std().mean(), 3))

saml = pd.read_csv(SAML, parse_dates=['Date'], usecols=['Date', 'Amount', 'Is_laundering', 'Laundering_type'])
print('переказів', len(saml), '| пропусків', int(saml.isna().sum().sum()), '| відмивання', int(saml.Is_laundering.sum()),
      '| частка %', round(100 * saml.Is_laundering.mean(), 4), '| класів', saml.Laundering_type.nunique())
print('період', saml.Date.min().date(), '—', saml.Date.max().date())
print(saml.Laundering_type.value_counts().head(12))
schemes = saml[saml.Is_laundering == 1]
print(schemes.groupby('Laundering_type').Amount.median().round(2).sort_values().to_string())

cards = pd.read_csv(IBM_CV + 'cards_data.csv')
users = pd.read_csv(IBM_CV + 'users_data.csv')
mcc = json.load(open(IBM_CV + 'mcc_codes.json'))
labels = json.load(open(IBM_CV + 'train_fraud_labels.json'))
labels = labels.get('target', labels)
tx = pd.read_csv(IBM_CV + 'transactions_data.csv', usecols=['id', 'client_id', 'card_id', 'mcc'])
seen_cards = pd.Series(tx.card_id.unique())
seen_clients = pd.Series(tx.client_id.unique())
seen_mcc = pd.Series(tx.mcc.unique())
print('операцій', len(tx), '| пропусків у ключах', int(tx.isna().sum().sum()))
print('карток: у довіднику', len(cards), '| в операціях', len(seen_cards),
      '| покриття %', round(100 * seen_cards.isin(cards.id).mean(), 2))
print('клієнтів: у довіднику', len(users), '| в операціях', len(seen_clients),
      '| покриття %', round(100 * seen_clients.isin(users.id).mean(), 2))
print('кодів: покриття %', round(100 * seen_mcc.isin([int(code) for code in mcc]).mean(), 2))
print('розмічено', len(labels), '| частка рядків %', round(100 * len(labels) / len(tx), 1))

returns = pd.read_csv(RETURNS)
print('рядків', len(returns), '| колонок', returns.shape[1], '| пропусків', int(returns.isna().sum().sum()),
      '| дублікатів', int(returns.duplicated().sum()))
print(returns.abuse_type.value_counts())
