import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 220, 'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False})
z = zipfile.ZipFile('data/kh/single.zip')
aml = pd.read_csv(z.open('single/HI-Small_Trans.csv'), parse_dates=['Timestamp'],
                  usecols=['Timestamp', 'Account', 'Account.1', 'Amount Received', 'Receiving Currency',
                           'Payment Format', 'Is Laundering'])
out_deg = aml['Account'].value_counts()
in_deg = aml['Account.1'].value_counts()
R = {'rows': len(aml), 'laundering': int(aml['Is Laundering'].sum()),
     'share_pct': round(100 * aml['Is Laundering'].mean(), 4),
     'period': [str(aml.Timestamp.min()), str(aml.Timestamp.max())],
     'currencies': int(aml['Receiving Currency'].nunique()),
     'senders': int(len(out_deg)), 'receivers': int(len(in_deg)),
     'formats': aml['Payment Format'].value_counts().to_dict(),
     'top_currencies': aml['Receiving Currency'].value_counts().head(2).to_dict(),
     'out_deg': [int(out_deg.median()), int(out_deg.quantile(.9)), int(out_deg.quantile(.99)), int(out_deg.max())],
     'in_deg': [int(in_deg.median()), int(in_deg.quantile(.9)), int(in_deg.quantile(.99)), int(in_deg.max())],
     'median_amount': {str(k): round(float(v), 2) for k, v in aml.groupby('Is Laundering')['Amount Received'].median().items()}}
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for degrees, label, color in ((out_deg, 'вихідний ступінь', '#c44e52'), (in_deg, 'вхідний ступінь', '#4878a8')):
    counts = degrees.value_counts().sort_index()
    ax.scatter(counts.index, counts.values, s=4, alpha=.6, label=label, color=color)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('кількість переказів у рахунку')
ax.set_ylabel('кількість рахунків')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('out/рис8.png')
json.dump(R, open('out/aml_results.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False))
main = aml.Timestamp < '2022-09-11'
print('основний період', int(main.sum()), 'відмивання', int(aml.loc[main, 'Is Laundering'].sum()),
      '| після', int((~main).sum()), 'відмивання', int(aml.loc[~main, 'Is Laundering'].sum()))
