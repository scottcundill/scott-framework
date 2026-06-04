"""
paper1_reproduce_results.py
============================
Reproduces all tables and figures from:
  Cundill, S. (2026). Bilateral Symmetry as a Filter-Robust Discriminator
  Between Real and GAN-Generated Face Images.

Usage:
  1. Clone the repo: git clone https://github.com/scottcundill/scott-framework
  2. Install deps:   pip install -r requirements.txt pandas scipy matplotlib
  3. Run:            python paper1_reproduce_results.py

All CSV files must be in the same directory as this script.
Figures are saved to the current directory.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── file paths (relative to script location) ─────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    'raw':         os.path.join(SCRIPT_DIR, 'paper1_measurements_raw.csv'),
    'edge':        os.path.join(SCRIPT_DIR, 'paper1_measurements_edge.csv'),
    'topographic': os.path.join(SCRIPT_DIR, 'paper1_measurements_topographic.csv'),
}
VALIDATION_FILE = os.path.join(SCRIPT_DIR, 'paper1_validation_results.csv')
DIFFUSION_FILE  = os.path.join(SCRIPT_DIR, 'paper1_diffusion_results.csv')

REAL = 'real_faces_1k'
AI   = 'ai_faces_1k'

# ── helper ───────────────────────────────────────────────────────────────────
def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pool = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    d = (b.mean() - a.mean()) / pool
    return t, p, d

# ── load dataframes ──────────────────────────────────────────────────────────
print("Loading data...")
dfs = {}
for k, v in FILES.items():
    if not os.path.exists(v):
        print("ERROR: " + v + " not found. Ensure CSVs are in the same folder.")
        exit(1)
    dfs[k] = pd.read_csv(v)

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 1 — bilateral symmetry across three filters
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TABLE 1 — Bilateral Symmetry: real_faces_1k vs ai_faces_1k")
print("=" * 90)
hdr = "{:<14}{:>7}{:>7}{:>11}{:>11}{:>9}{:>9}{:>9}{:>14}{:>9}".format(
    'Filter', 'n_real', 'n_ai', 'mean_real', 'mean_ai',
    'sd_real', 'sd_ai', 't', 'p', 'd')
print(hdr)
print("-" * 90)

for fname in ['raw', 'topographic', 'edge']:
    df = dfs[fname]
    real = df[df['category'] == REAL]['bilateral_symmetry'].dropna()
    ai   = df[df['category'] == AI]['bilateral_symmetry'].dropna()
    t, p, d = welch(real, ai)
    print("{:<14}{:>7}{:>7}{:>11.4f}{:>11.4f}{:>9.4f}{:>9.4f}{:>9.3f}{:>14.3e}{:>9.3f}".format(
        fname, len(real), len(ai), real.mean(), ai.mean(),
        real.std(), ai.std(), t, p, d))

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 2 — secondary metrics (raw filter only)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TABLE 2 — Secondary Metrics (raw filter only)")
print("=" * 90)

df_raw = dfs['raw']
SECONDARY = ['multifractal_width', 'entropy', 'junctions']
for metric in SECONDARY:
    real = df_raw[df_raw['category'] == REAL][metric].dropna()
    ai   = df_raw[df_raw['category'] == AI][metric].dropna()
    t, p, d = welch(real, ai)
    direction = 'real > AI' if real.mean() > ai.mean() else 'AI > real'
    print("  {:<22} real={:.4f}  ai={:.4f}  t={:.3f}  p={:.4e}  d={:.3f}  {}".format(
        metric, real.mean(), ai.mean(), t, p, d, direction))

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 3 — diffusion vs real
# ═══════════════════════════════════════════════════════════════════════════
if os.path.exists(DIFFUSION_FILE):
    diff = pd.read_csv(DIFFUSION_FILE)
    print("\n" + "=" * 90)
    print("TABLE 3 — Diffusion (SFHQ-T2I) vs Real Faces")
    print("=" * 90)
    for fname in ['raw', 'topographic', 'edge']:
        df = dfs[fname]
        real = df[df['category'] == REAL]['bilateral_symmetry'].dropna()
        d_vals = diff[diff['filter_type'] == fname]['bilateral_symmetry'].dropna()
        t, p, d = welch(real, d_vals)
        direction = 'diff > real' if d_vals.mean() > real.mean() else 'real > diff'
        print("  {:<14} real={:.4f}  diff={:.4f}  t={:.3f}  p={:.3e}  d={:.3f}  {}".format(
            fname, real.mean(), d_vals.mean(), t, p, d, direction))

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — three-way violin plot
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1...")

REAL_COL = '#4878CF'
GAN_COL  = '#D65F5F'
DIFF_COL = '#5BAD6F'
ALPHA    = 0.85

diff = pd.read_csv(DIFFUSION_FILE) if os.path.exists(DIFFUSION_FILE) else None

data = {}
for fname in ['raw', 'topographic', 'edge']:
    df = dfs[fname]
    data[fname] = {
        'real': df[df['category'] == REAL]['bilateral_symmetry'].dropna().values,
        'gan':  df[df['category'] == AI]['bilateral_symmetry'].dropna().values,
    }
    if diff is not None:
        data[fname]['diff'] = diff[diff['filter_type'] == fname]['bilateral_symmetry'].dropna().values

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
labels = {'raw': 'Raw', 'topographic': 'Topographic', 'edge': 'Edge'}

for ax, fname in zip(axes, ['raw', 'topographic', 'edge']):
    r = data[fname]['real']
    g = data[fname]['gan']
    d = data[fname].get('diff', np.array([]))
    vals = [r, g] + ([d] if len(d) > 0 else [])
    pos  = [1, 2] + ([3] if len(d) > 0 else [])
    cols = [REAL_COL, GAN_COL] + ([DIFF_COL] if len(d) > 0 else [])

    parts = ax.violinplot(vals, positions=pos,
                          showmedians=False, showextrema=False, widths=0.6)
    for pc, col in zip(parts['bodies'], cols):
        pc.set_facecolor(col)
        pc.set_alpha(ALPHA)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.8)

    for p, v, c in zip(pos, vals, cols):
        ax.hlines(np.mean(v), p - 0.25, p + 0.25, color=c, linewidth=2.2, zorder=5)

    xlabels = ['Real', 'StyleGAN'] + (['Diffusion'] if len(d) > 0 else [])
    ax.set_xticks(pos)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_title(labels[fname], fontsize=11, pad=8)
    ax.set_ylabel('Bilateral Symmetry (BS)' if fname == 'raw' else '', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0.06, 1, 1])
real_p = mpatches.Patch(color=REAL_COL, alpha=ALPHA, label='Real (FFHQ)')
gan_p  = mpatches.Patch(color=GAN_COL,  alpha=ALPHA, label='GAN (StyleGAN)')
diff_p = mpatches.Patch(color=DIFF_COL, alpha=ALPHA, label='Diffusion (SFHQ-T2I)')
fig.legend(handles=[real_p, gan_p, diff_p], loc='lower center',
           ncol=3, fontsize=8, frameon=False)

out1 = os.path.join(SCRIPT_DIR, 'figure1_threeway_violin.png')
plt.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: " + out1)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — secondary metrics bar chart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2...")
fig2, axes2 = plt.subplots(1, 3, figsize=(11, 4))
metric_labels = {
    'multifractal_width': 'Multifractal Width',
    'entropy':            'Shannon Entropy',
    'junctions':          'Junction Count'
}

for ax, metric in zip(axes2, SECONDARY):
    real = df_raw[df_raw['category'] == REAL][metric].dropna()
    ai   = df_raw[df_raw['category'] == AI][metric].dropna()

    bars = ax.bar([1, 2], [real.mean(), ai.mean()],
                  color=[REAL_COL, GAN_COL], alpha=ALPHA,
                  yerr=[real.sem(), ai.sem()], capsize=4,
                  error_kw={'linewidth': 1.2, 'color': '#555555'}, width=0.5)

    t, p, d = welch(real, ai)
    ax.text(1.5, ax.get_ylim()[1] * 0.97,
            "p = {:.2e},  d = {:.2f}".format(p, d),
            ha='center', va='top', fontsize=8, color='#444444')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'GAN'], fontsize=10)
    ax.set_title(metric_labels[metric], fontsize=10, pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig2.legend(handles=[real_p, mpatches.Patch(color=GAN_COL, alpha=ALPHA, label='GAN (StyleGAN)')],
            loc='lower center', ncol=2, fontsize=9, frameon=False)
plt.tight_layout()

out2 = os.path.join(SCRIPT_DIR, 'figure2_secondary_metrics_bar.png')
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: " + out2)

print("\nDone. All tables printed and figures saved.")
