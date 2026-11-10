import json

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vbank.common import OUT, STAND

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
MODELS = [('бустинг', 'score', '#1f77b4'), ('GRU', 'score_gru', '#9467bd'), ('випадковий ліс', 'score_rf', '#ff7f0e'),
          ('логістична регресія', 'score_lr', '#2ca02c'), ('Isolation Forest', 'score_if', '#d62728')]

BOXES = {
    'src': (0.02, 0.30, 0.17, 0.40, 'Джерела\n\nFinancial Transactions\nDataset\nSparkov\nIBM AML\nскарги CFPB', '#eef2f5'),
    'stand': (0.23, 0.30, 0.17, 0.40, 'Віртуальний банк\nstand.py\n\nоперації, картки,\nклієнти, спори,\nперекази', '#dce8f0'),
    'feat': (0.44, 0.30, 0.15, 0.40, 'Ознаки\nfeatures.py\n\nвідносна сума,\nінтервали, ризик\nторговця, спори', '#dce8f0'),
    'cards': (0.63, 0.62, 0.17, 0.25, 'Карткові моделі\nбустинг, ліс, регресія,\nIsolation Forest, LOF, GRU', '#f6dfc9'),
    'graph': (0.63, 0.13, 0.17, 0.25, 'Граф переказів\nграфові ознаки,\nGraphSAGE, типології', '#f6dfc9'),
    'app': (0.84, 0.30, 0.14, 0.40, 'Застосунок\nStreamlit\n\nогляд, перевірка,\nпакет, поріг,\nсерії й схеми', '#e3f1e3'),
}
ARROWS = [('src', 'stand', 0.5), ('stand', 'feat', 0.5), ('feat', 'cards', 0.5), ('cards', 'app', 0.72), ('graph', 'app', 0.28)]


def architecture(name='рис_arch.png'):
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    centers = {}
    for key, (x, y, w, h, text, color) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008', fc=color, ec='#2f5d7c', lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6)
        centers[key] = (x, y, w, h)
    for a, b, level in ARROWS:
        xa, ya, wa, ha = centers[a]
        xb, yb, wb, hb = centers[b]
        start = (xa + wa + 0.006, ya + ha / 2)
        end = (xb - 0.006, yb + hb * level)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10, color='#2f5d7c', lw=1))
    xs, ys, ws, hs = centers['stand']
    xg, yg, wg, hg = centers['graph']
    ax.annotate('', xy=(xg - 0.006, yg + hg / 2), xytext=(xs + ws / 2, ys - 0.008),
                arrowprops=dict(arrowstyle='-|>', color='#2f5d7c', lw=1, mutation_scale=10, connectionstyle='angle,angleA=-90,angleB=180,rad=0'))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def pr_curves(name='рис_pr.png'):
    s = pd.read_parquet(STAND / 'test_scores.parquet')
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(alpha=.3)
    for label, col, color in MODELS:
        if col not in s:
            continue
        p, r, _ = precision_recall_curve(s.y, s[col])
        ax.plot(r, p, lw=1.3, color=color, label=f'{label} (PR-AUC {average_precision_score(s.y, s[col]):.3f})')
    ax.set_xlabel('повнота (recall)')
    ax.set_ylabel('точність (precision)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(loc='lower center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def budget_curve(name='рис_budget.png'):
    s = pd.read_parquet(STAND / 'test_scores.parquet')
    days = (s.ts.max() - s.ts.min()).total_seconds() / 86400
    y = s.y.values[np.argsort(-s.score.values)]
    shares = np.linspace(0.0005, 0.03, 120)
    rec, fpd = [], []
    for sh in shares:
        k = int(sh * len(y))
        tp = y[:k].sum()
        rec.append(100 * tp / y.sum())
        fpd.append((k - tp) / days)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.grid(alpha=.3)
    l1, = ax.plot(100 * shares, rec, color='#c0392b', label='повнота, % (ліва вісь)')
    ax2 = ax.twinx()
    l2, = ax2.plot(100 * shares, fpd, color='#2c6fbb', label='хибні тривоги за добу (права вісь)')
    ax.axvline(0.5, color='#555555', ls=':', lw=1)
    ax.set_xlabel('частка операцій, відправлених на перевірку, %')
    ax.set_ylabel('повнота, %')
    ax2.set_ylabel('FP за добу')
    fig.legend(handles=[l1, l2], loc='lower center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def improve_curves(name='рис_improve.png'):
    s = pd.read_parquet(STAND / 'test_scores.parquet')
    s['blend_final'] = (s.score_final.rank(pct=True) + s.score_gru_final.rank(pct=True)) / 2
    curves = [('бустинг, базовий', 'score', '#1f77b4', ':'), ('GRU, базова', 'score_gru', '#9467bd', ':'),
              ('бустинг, остаточний', 'score_final', '#1f77b4', '-'), ('GRU, остаточна', 'score_gru_final', '#9467bd', '-'),
              ('середнє рангів, остаточне', 'blend_final', '#c0392b', '-')]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(alpha=.3)
    for label, col, color, style in curves:
        p, r, _ = precision_recall_curve(s.y, s[col])
        ax.plot(r, p, lw=1.3, color=color, ls=style, label=f'{label} (PR-AUC {average_precision_score(s.y, s[col]):.3f})')
    ax.set_xlabel('повнота (recall)')
    ax.set_ylabel('точність (precision)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(loc='lower center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def confusion_final(name='рис_confusion_final.png'):
    from vbank.models_cards import plot_confusion
    r = json.load(open(OUT / 'improve_results.json', encoding='utf-8'))['multiclass']
    plot_confusion(np.array(r['confusion']), list(r['per_class_f1']), name)


def layout(G, pattern):
    if pattern == 'CYCLE':
        return nx.circular_layout(G)
    if pattern not in ('FAN-IN', 'FAN-OUT') and nx.is_directed_acyclic_graph(G):
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for n in nodes:
                G.nodes[n]['layer'] = layer
        return nx.multipartite_layout(G, subset_key='layer')
    return nx.kamada_kawai_layout(G)


def typologies(t=None, name='рис_typologies.png'):
    if t is None:
        t = pd.read_parquet(STAND / 'transfers.parquet', columns=['src', 'dst', 'pattern', 'attack'])
    labelled = t[t.pattern.notna()]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for ax, (pattern, g) in zip(axes.flat, labelled.groupby('pattern')):
        sizes = g.groupby('attack').size()
        good = sizes[(sizes >= 8) & (sizes <= 16)]
        attack = good.index[0] if len(good) else sizes.idxmax()
        sub = g[(g.attack == attack) & (g.src != g.dst)]
        G = nx.DiGraph()
        G.add_edges_from(zip(sub.src, sub.dst))
        pos = layout(G, pattern)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=28, node_color='#2c6fbb')
        nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=7, width=0.7, edge_color='#555555', node_size=28)
        ax.set_title(f'{pattern}: {len(sub)} переказів', fontsize=8)
        ax.margins(0.12)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    architecture()
    pr_curves()
    budget_curve()
    typologies()
    if (OUT / 'improve_results.json').exists():
        improve_curves()
        confusion_final()
