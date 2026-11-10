import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / 'data/kh/datasets/computingvictor/transactions-fraud-datasets/versions/1'
SPARKOV = Path(os.path.expanduser('~/.cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1'))
AML_ZIP = ROOT / 'data/kh/single.zip'
CFPB = ROOT / 'data/kh/datasets/mexwell/bank-and-credit-card-complaints/versions/1/credit_card_complaints.csv'
ULB = ROOT / 'data/kh/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv'
STAND = ROOT / 'data/stand'
OUT = ROOT / 'out/r3'
MODELS = ROOT / 'data/models'

START = pd.Timestamp('2016-01-01')
TRAIN_END = pd.Timestamp('2018-07-01')
VALID_END = pd.Timestamp('2019-01-01')
END = pd.Timestamp('2019-11-01')
SEED = 2026

CLASSES = {0: 'Норма', 1: 'Поодинока атака', 2: 'Серія після компрометації', 3: 'Відмивання',
           4: 'Компрометація торговця', 5: 'Зловживання власника рахунку'}

SPARKOV_MCC = {
    'entertainment': (7832, False), 'food_dining': (5812, False), 'gas_transport': (5541, False),
    'grocery_net': (5311, True), 'grocery_pos': (5411, False), 'health_fitness': (7802, False),
    'home': (5719, False), 'kids_pets': (5947, False), 'misc_net': (5815, True), 'misc_pos': (5310, False),
    'personal_care': (7230, False), 'shopping_net': (5311, True), 'shopping_pos': (5311, False),
    'travel': (4722, False),
}


def money(series):
    return series.astype(str).str.replace('$', '', regex=False).astype(float)


def period_of(ts):
    return np.where(ts < TRAIN_END, 'train', np.where(ts < VALID_END, 'valid', 'test'))


def load_ftd():
    tx = pd.read_csv(CV / 'transactions_data.csv', parse_dates=['date'])
    labels = json.load(open(CV / 'train_fraud_labels.json'))
    labels = labels.get('target', labels)
    tx['label'] = tx.id.astype(str).map(labels)
    tx['amount'] = money(tx.amount)
    tx = tx.rename(columns={'id': 'op_id', 'date': 'ts', 'use_chip': 'chip'})
    return tx.sort_values(['ts', 'op_id'], kind='stable').reset_index(drop=True)


def load_cards():
    cards = pd.read_csv(CV / 'cards_data.csv')
    cards['credit_limit'] = money(cards.credit_limit)
    return cards.rename(columns={'id': 'card_id'})


def load_users():
    users = pd.read_csv(CV / 'users_data.csv')
    for col in ['per_capita_income', 'yearly_income', 'total_debt']:
        users[col] = money(users[col])
    return users.rename(columns={'id': 'client_id'})


def load_sparkov():
    cols = ['trans_date_trans_time', 'cc_num', 'category', 'amt', 'is_fraud']
    parts = [pd.read_csv(SPARKOV / f, usecols=cols, parse_dates=['trans_date_trans_time']) for f in ['fraudTrain.csv', 'fraudTest.csv']]
    sp = pd.concat(parts, ignore_index=True).rename(columns={'trans_date_trans_time': 'ts', 'cc_num': 'card', 'amt': 'amount'})
    return sp.sort_values('ts', kind='stable').reset_index(drop=True)


def load_aml():
    z = zipfile.ZipFile(AML_ZIP)
    aml = pd.read_csv(z.open('single/HI-Small_Trans.csv'), parse_dates=['Timestamp'], dtype={'From Bank': str, 'To Bank': str})
    aml.columns = ['ts', 'from_bank', 'src', 'to_bank', 'dst', 'amount_received', 'currency_received',
                   'amount_paid', 'currency_paid', 'format', 'laundering']
    aml['src'] = aml.from_bank.astype(int).astype(str) + '_' + aml.src.astype(str)
    aml['dst'] = aml.to_bank.astype(int).astype(str) + '_' + aml.dst.astype(str)
    return aml.sort_values('ts', kind='stable').reset_index(drop=True)


def load_aml_patterns():
    z = zipfile.ZipFile(AML_ZIP)
    rows = []
    attack = None
    attack_no = 0
    for raw in z.open('single/HI-Small_Patterns.txt').read().decode().splitlines():
        line = raw.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT'):
            name = line.split('-', 1)[1].strip()
            attack = (attack_no, name.split(':')[0].strip())
            attack_no += 1
        elif line.startswith('END LAUNDERING ATTEMPT'):
            attack = None
        elif attack and line:
            p = line.split(',')
            rows.append((attack[0], attack[1], pd.Timestamp(p[0].replace('/', '-')), str(int(p[1])) + '_' + p[2], str(int(p[3])) + '_' + p[4], float(p[5])))
    return pd.DataFrame(rows, columns=['attack', 'pattern', 'ts', 'src', 'dst', 'amount_received'])


def save_json(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(OUT / name, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
