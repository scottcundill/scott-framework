"""
paper1_results.py
=================
Produces:
  1. Console output — Table 1 (bilateral symmetry) and Table 2 (secondary metrics)
     with all numbers needed for the paper.
  2. figure1_violin.png — publication-quality violin plots for Figure 1.
  3. figure2_secondary.png — bar chart for Table 2 secondary metrics.

Data sources (raw=export7, edge=export6, topographic=export8)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── file paths ──────────────────────────────────────────────────────────────
FILES = {
    'raw':         '/mnt/user-data/uploads/2026-06-03T05-53_export7.csv',
    'edge':        '/mnt/user-data/uploads/2026-06-03T05-53_export6.csv',
    'topographic': '/mnt/user-data/uploads/2026-06-03T09-26_export8.csv',
}
OUT = '/mnt/user-data/outputs/'
os.makedirs(OUT, exist_ok=True)

REAL = 'real_faces_1k'
AI   = 'ai_faces_1k'

# ── helper ───────────────────────────────────────────────────────────────────
def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pool  = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    d     = (b.mean() - a.mean()) / pool          # positive = AI > real
    return t, p, d

# ── load dataframes ──────────────────────────────────────────────────────────
dfs = {k: pd.read_csv(v) for k, v in FILES.items()}

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 1 — bilateral symmetry across three filters
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("TABLE 1 — Bilateral Symmetry: real_faces_1k vs ai_faces_1k")
print("=" * 90)
hdr = f"{'Filter':<14}{'n_real':>7}{'n_ai':>7}{'mean_real':>11}{'mean_ai':>11}"
hdr += f"{'sd_real':>9}{'sd_ai':>9}{'t':>9}{'p':>14}{'d':>9}"
print(hdr)
print("-" * 90)

table1_data = {}
for fname in ['raw', 'topographic', 'edge']:
    df   = dfs[fname]
    real = df[df['category'] == REAL]['bilateral_symmetry'].dropna()
    ai   = df[df['category'] == AI  ]['bilateral_symmetry'].dropna()
    t, p, d = welch(real, ai)
    table1_data[fname] = {'real': real, 'ai': ai}
    print(f"{fname:<14}{len(real):>7}{len(ai):>7}{real.mean():>11.4f}{ai.mean():>11.4f}"
          f"{real.std():>9.4f}{ai.std():>9.4f}{t:>9.3f}{p:>14.3e}{d:>9.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 2 — secondary metrics (raw filter only)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TABLE 2 — Secondary Metrics (raw filter only)")
print("=" * 90)
hdr2 = f"{'Metric':<26}{'mean_real':>11}{'sd_real':>9}{'mean_ai':>10}{'sd_ai':>9}{'t':>9}{'p':>14}{'d':>9}{'direction':>14}"
print(hdr2)
print("-" * 90)

df_raw = dfs['raw']
SECONDARY = ['multifractal_width', 'entropy', 'junctions']
table2_data = {}
for metric in SECONDARY:
    real = df_raw[df_raw['category'] == REAL][metric].dropna()
    ai   = df_raw[df_raw['category'] == AI  ][metric].dropna()
    t, p, d = welch(real, ai)
    direction = 'real > AI' if real.mean() > ai.mean() else 'AI > real'
    table2_data[metric] = {'real': real, 'ai': ai}
    print(f"{metric:<26}{real.mean():>11.4f}{real.std():>9.4f}{ai.mean():>10.4f}"
          f"{ai.std():>9.4f}{t:>9.3f}{p:>14.3e}{d:>9.3f}{direction:>14}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — violin plots, bilateral symmetry, three filters
# ═══════════════════════════════════════════════════════════════════════════
REAL_COL = '#4878CF'   # muted blue
AI_COL   = '#D65F5F'   # muted red
ALPHA    = 0.85

fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=False)
filter_labels = {'raw': 'Raw', 'topographic': 'Topographic', 'edge': 'Edge'}

for ax, fname in zip(axes, ['raw', 'topographic', 'edge']):
    real_vals = table1_data[fname]['real'].values
    ai_vals   = table1_data[fname]['ai'].values

    parts = ax.violinplot(
        [real_vals, ai_vals],
        positions=[1, 2],
        showmedians=False,
        showextrema=False,
        widths=0.6
    )
    colours = [REAL_COL, AI_COL]
    for pc, col in zip(parts['bodies'], colours):
        pc.set_facecolor(col)
        pc.set_alpha(ALPHA)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.8)

    # mean lines
    for pos, vals, col in zip([1, 2], [real_vals, ai_vals], colours):
        ax.hlines(np.mean(vals), pos - 0.25, pos + 0.25,
                  color=col, linewidth=2.2, zorder=5)

    # p and d annotation
    _, p_val, d_val = welch(pd.Series(real_vals), pd.Series(ai_vals))
    p_str = f"p = {p_val:.2e}\nd = {d_val:.2f}"
    ax.text(1.5, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else real_vals.max(),
            p_str, ha='center', va='bottom', fontsize=8,
            color='#333333')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'GAN'], fontsize=10)
    ax.set_title(filter_labels[fname], fontsize=11, fontweight='normal', pad=8)
    ax.set_ylabel('Bilateral Symmetry (BS)' if fname == 'raw' else '', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)

# re-annotate with correct ylim after first draw
plt.tight_layout(rect=[0, 0.05, 1, 1])

# re-add p/d text now axes are scaled
for ax, fname in zip(axes, ['raw', 'topographic', 'edge']):
    real_vals = table1_data[fname]['real'].values
    ai_vals   = table1_data[fname]['ai'].values
    _, p_val, d_val = welch(pd.Series(real_vals), pd.Series(ai_vals))
    ymax = ax.get_ylim()[1]
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    for txt in ax.texts:
        txt.set_visible(False)
    ax.text(1.5, ymax - yrange * 0.04,
            f"p = {p_val:.2e},  d = {d_val:.2f}",
            ha='center', va='top', fontsize=8, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.8))

real_patch = mpatches.Patch(color=REAL_COL, alpha=ALPHA, label='Real (FFHQ)')
ai_patch   = mpatches.Patch(color=AI_COL,   alpha=ALPHA, label='GAN (StyleGAN)')
fig.legend(handles=[real_patch, ai_patch], loc='lower center',
           ncol=2, fontsize=9, frameon=False)

fig.suptitle('Figure 1. Bilateral symmetry: real vs GAN-generated face images',
             fontsize=10, y=1.01, color='#333333')

out1 = os.path.join(OUT, 'figure1_bilateral_symmetry_violin.png')
plt.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure 1 saved → {out1}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — bar chart, secondary metrics
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(11, 4))
metric_labels = {
    'multifractal_width': 'Multifractal Width',
    'entropy':            'Shannon Entropy',
    'junctions':          'Junction Count'
}

for ax, metric in zip(axes2, SECONDARY):
    real_mean = table2_data[metric]['real'].mean()
    ai_mean   = table2_data[metric]['ai'].mean()
    real_se   = table2_data[metric]['real'].sem()
    ai_se     = table2_data[metric]['ai'].sem()

    bars = ax.bar([1, 2], [real_mean, ai_mean],
                  color=[REAL_COL, AI_COL], alpha=ALPHA,
                  yerr=[real_se, ai_se], capsize=4,
                  error_kw={'linewidth': 1.2, 'color': '#555555'},
                  width=0.5)

    _, p_val, d_val = welch(table2_data[metric]['real'],
                            table2_data[metric]['ai'])
    ymax = max(real_mean, ai_mean)
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.text(1.5, ax.get_ylim()[1] * 0.97,
            f"p = {p_val:.2e},  d = {d_val:.2f}",
            ha='center', va='top', fontsize=8, color='#444444')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'GAN'], fontsize=10)
    ax.set_title(metric_labels[metric], fontsize=10, fontweight='normal', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)

fig2.suptitle('Figure 2. Secondary metrics (raw filter): real vs GAN-generated',
              fontsize=10, y=1.02, color='#333333')
fig2.legend(handles=[real_patch, ai_patch], loc='lower center',
            ncol=2, fontsize=9, frameon=False)
plt.tight_layout()

out2 = os.path.join(OUT, 'figure2_secondary_metrics_bar.png')
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figure 2 saved → {out2}")
print("\nDone. All numbers and figures ready for the paper.")
