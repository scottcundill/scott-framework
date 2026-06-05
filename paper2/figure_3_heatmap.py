"""
figure_3_heatmap.py
====================
Generates Figure 3 for Paper 2: a heatmap of separation rates across
metrics and categories, making the two-layer structure (near-universal
fractal/geometric vs category-dependent TDA) visually clear at a glance.

Uses the n=30 surrogate results CSV already in exports/.

Output: exports/figure_3_heatmap.png  (300 dpi)
        exports/figure_3_heatmap.pdf  (vector)

Usage:  python scripts/figure_3_heatmap.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# =============================================================================
CSV_PATH = "exports/surrogate_by_category_raw.csv"  # the n=30 results
# =============================================================================

# Categories to show (display names)
CAT_MAP = {
    "ai_faces_1k":     "AI Faces",
    "real_faces_1k":   "Real Faces",
    "faces (FG-NET)":  "FG-NET",
    "coral (healthy)": "Coral (H)",
    "coral (bleached)":"Coral (B)",
    "quasicrystal":    "Quasicrystal",
}
COL_ORDER = ["AI Faces","Real Faces","FG-NET","Coral (H)","Coral (B)","Quasicrystal"]

# Metrics to show — ordered by group, within-group by overall effect
METRIC_ROWS = [
    # (csv_name, display_name, group_label)
    ("f_dim",                  "Fractal dimension",     "UNIVERSAL"),
    ("lacunarity_mean",        "Lacunarity",            "UNIVERSAL"),
    ("junctions",              "Junctions",             "UNIVERSAL"),
    ("",                       "",                      "---"),  # spacer
    ("euler_std",              "Euler std",             "GEOMETRIC"),
    ("multiscale_entropy_slope","MS entropy slope",     "GEOMETRIC"),
    ("euler_range",            "Euler range",           "GEOMETRIC"),
    ("clustering",             "Clustering",            "GEOMETRIC"),
    ("",                       "",                      "---"),  # spacer
    ("persistence_entropy_h1", "Persistence entropy",  "TDA (H1)"),
    ("total_persistence_h1",   "Total persistence",    "TDA (H1)"),
    ("betti_1",                "Betti-1",              "TDA (H1)"),
    ("betti_0",                "Betti-0",              "TDA (H0)"),
    ("",                       "",                      "---"),  # spacer
    ("bilateral_symmetry",     "Bilateral symmetry",   "NON-SEP"),
    ("entropy",                "Entropy",              "VALIDATION"),
]


def compute_sep_rates(df, cat_map, col_order):
    """Build a DataFrame of separation rates (%) by metric × category."""
    data = {}
    for cat_raw, cat_label in cat_map.items():
        sub = df[df['category'] == cat_raw]
        col = {}
        for csv_name, display, group in METRIC_ROWS:
            if csv_name == "":
                col[display] = np.nan
                continue
            m = sub[sub['metric'] == csv_name]
            valid = m.dropna(subset=['z_score'])
            valid = valid[valid['verdict'] != 'zero-var']
            n = len(valid)
            if n == 0:
                col[display] = np.nan
            else:
                n_sep = (valid['verdict'] == 'SEPARATED').sum()
                col[display] = 100 * n_sep / n
        data[cat_label] = col

    return pd.DataFrame(data, columns=col_order)


def main():
    os.makedirs("exports", exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run scripts/surrogate_by_category.py first with N_IMAGES_PER_CAT=30.")
        return

    df = pd.read_csv(CSV_PATH)
    df_paper = df[df['category'].isin(CAT_MAP.keys())]

    sep_df = compute_sep_rates(df_paper, CAT_MAP, COL_ORDER)
    display_names = [row[1] for row in METRIC_ROWS]
    sep_df = sep_df.reindex(display_names)

    # Build mask for spacer rows
    spacer_rows = [i for i, row in enumerate(METRIC_ROWS) if row[0] == ""]

    # ---- PLOT ----
    n_rows = len(METRIC_ROWS)
    n_cols = len(COL_ORDER)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Custom colormap: white (0%) → light blue → deep blue (100%)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sep_rate",
        [(1,1,1), (0.85, 0.92, 1.0), (0.1, 0.35, 0.75)],
        N=256
    )

    # Build display matrix, inserting NaN rows for spacers
    matrix = sep_df.values.astype(float)

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=0, vmax=100,
        aspect='auto',
        interpolation='nearest'
    )

    # Cell text
    for i in range(n_rows):
        if i in spacer_rows:
            continue
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val >= 70 else 'black'
            ax.text(j, i, f"{val:.0f}%",
                    ha='center', va='center',
                    fontsize=7.5, color=text_color,
                    fontweight='bold' if val >= 80 else 'normal')

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COL_ORDER, fontsize=8.5, rotation=30, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.tick_params(length=0)

    # Draw spacer lines
    for i in spacer_rows:
        ax.axhline(i - 0.5, color='white', linewidth=6)
        ax.axhline(i + 0.5, color='white', linewidth=6)

    # Group bracket annotations on the right
    group_spans = [
        ("UNIVERSAL",   0, 2,  "#1a4fa0"),
        ("GEOMETRIC",   4, 7,  "#2e7d32"),
        ("TDA",         9, 13, "#c62828"),
        ("NON-SEP /\nVALIDATION", 15, 16, "#6a1b9a"),
    ]
    for label, r_start, r_end, color in group_spans:
        ax.annotate(
            label,
            xy=(n_cols - 0.5, (r_start + r_end) / 2),
            xytext=(n_cols + 0.35, (r_start + r_end) / 2),
            xycoords='data', textcoords='data',
            fontsize=7.5, color=color, fontweight='bold',
            ha='left', va='center',
            annotation_clip=False
        )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.22, shrink=0.7)
    cbar.set_label("Separation rate (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        "Figure 3. IAAFT surrogate separation rates across metrics and image categories.\n"
        "Bold = \u226580%. Two structural layers are visible: near-universal scaling metrics\n"
        "(top) and category-dependent topological metrics (middle).",
        fontsize=8, pad=10
    )

    plt.tight_layout()

    out_png = os.path.join("exports", "figure_3_heatmap.png")
    out_pdf = os.path.join("exports", "figure_3_heatmap.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
